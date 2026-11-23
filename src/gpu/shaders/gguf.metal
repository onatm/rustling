// Compute kernels for a GGUF file's block formats (see ggml.rs for the layouts): unpacking chosen matrices to
// float for the long-prompt path, and products of quantised rows with a float vector for generating a token.

#include <metal_stdlib>
using namespace metal;

static inline float read_half(const device uchar* p) {
    return float(as_type<half>(ushort(ushort(p[0]) | (ushort(p[1]) << 8))));
}

// ---------------------------------------------------------------------------------------------------------------
// The fast path for long prompts: chosen matrices of a stack are unpacked to float, for Apple's tuned matrix
// product. Out matrix `i` is stack matrix `order[i]`. The grid is (cols / 256, rounded up) x (count * rows).

struct WidenParams {
    // Rows per matrix.
    uint rows;
    uint cols;
    // How many matrices to unpack.
    uint count;
};

// A thread unpacks 256 values, eight runs of 32: starting a thread costs about as much as unpacking a run, so a
// thread to a run spends half its time starting.
#define WIDEN_KERNEL(name, values_per_block, block_bytes)                                                  \
    kernel void name(const device uchar* weights [[buffer(0)]],                                            \
                     const device uint* order [[buffer(1)]],                                               \
                     device float* out [[buffer(2)]],                                                      \
                     constant WidenParams& p [[buffer(3)]],                                                \
                     uint2 id [[thread_position_in_grid]]) {                                               \
        if (id.y >= p.count * p.rows) return;                                                              \
        uint slot = id.y / p.rows, row = id.y % p.rows;                                                    \
        ulong row_bytes = ulong(p.cols / (values_per_block)) * (block_bytes);                              \
        uint per_block = (values_per_block) / 32;                                                          \
        const device uchar* start = weights + (ulong(order[slot]) * p.rows + row) * row_bytes;             \
        device float* o = out + (ulong(id.y) * p.cols + id.x * 256);                                       \
        for (uint run = 0; run < 8; run++) {                                                               \
            uint at = id.x * 8 + run;                                                                      \
            if (at >= p.cols / 32) return;                                                                 \
            name##_part(start + ulong(at / per_block) * (block_bytes), at % per_block, o + run * 32);      \
        }                                                                                                  \
    }

// Each unpacks 32 values, four at a time: a thread gets through a loop slowly, so the fewer turns the better.

static inline void widen_q8_0_part(const device uchar* block, uint part, device float* o) {
    float d = read_half(block);
    const device packed_char4* q = (const device packed_char4*)(block + 2);
    device packed_float4* to = (device packed_float4*)o;
    for (uint i = 0; i < 8; i++) {
        to[i] = d * float4(char4(q[i]));
    }
}

static inline void widen_q5_0_part(const device uchar* block, uint part, device float* o) {
    float d = read_half(block);
    uint high = uint(block[2]) | (uint(block[3]) << 8) | (uint(block[4]) << 16) | (uint(block[5]) << 24);
    const device packed_uchar4* q = (const device packed_uchar4*)(block + 6);
    device packed_float4* to = (device packed_float4*)o;
    for (uint i = 0; i < 4; i++) {
        uint4 v = uint4(uchar4(q[i]));
        uint4 shifts = uint4(0, 1, 2, 3) + 4 * i;
        uint4 first = (v & 15u) | (((uint4(high) >> shifts) & 1u) << 4);
        uint4 second = (v >> 4) | (((uint4(high) >> (shifts + 16)) & 1u) << 4);
        to[i] = d * (float4(first) - 16.0f);
        to[i + 4] = d * (float4(second) - 16.0f);
    }
}

// Sub-block `part` of eight: its 6-bit scale and minimum, then 32 nibbles.
static inline void widen_q4k_part(const device uchar* block, uint part, device float* o) {
    float d = read_half(block), dmin = read_half(block + 2);
    const device uchar* packed = block + 4;
    uint scale, minimum;
    if (part < 4) {
        scale = packed[part] & 63;
        minimum = packed[part + 4] & 63;
    } else {
        scale = (packed[part + 4] & 15) | ((packed[part - 4] >> 6) << 4);
        minimum = (packed[part + 4] >> 4) | ((packed[part] >> 6) << 4);
    }
    float step = d * float(scale), low = dmin * float(minimum);
    const device packed_uchar4* q = (const device packed_uchar4*)(block + 16 + 32 * (part / 2));
    device packed_float4* to = (device packed_float4*)o;
    uint shift = 4 * (part % 2);
    for (uint i = 0; i < 8; i++) {
        to[i] = step * float4((uint4(uchar4(q[i])) >> shift) & 15u) - low;
    }
}

// Part `part` of eight: half-block `part / 4`, and within it one of the four runs of 32 values.
static inline void widen_q6k_part(const device uchar* block, uint part, device float* o) {
    float d = read_half(block + 208);
    uint half_block = part / 4, run = part % 4;
    const device packed_uchar4* low = (const device packed_uchar4*)(block + 64 * half_block + 32 * (run % 2));
    const device packed_uchar4* high = (const device packed_uchar4*)(block + 128 + 32 * half_block);
    const device char* scales = (const device char*)(block + 192 + 8 * half_block + 2 * run);
    device packed_float4* to = (device packed_float4*)o;
    uint shift = 4 * (run / 2);
    for (uint i = 0; i < 8; i++) {
        uint4 value = ((uint4(uchar4(low[i])) >> shift) & 15u) | (((uint4(uchar4(high[i])) >> (2 * run)) & 3u) << 4);
        to[i] = d * float(scales[i / 4]) * (float4(value) - 32.0f);
    }
}

WIDEN_KERNEL(widen_q8_0, 32, 34)
WIDEN_KERNEL(widen_q5_0, 32, 22)
WIDEN_KERNEL(widen_q4k, 256, 144)
WIDEN_KERNEL(widen_q6k, 256, 210)

// ---------------------------------------------------------------------------------------------------------------
// Products for generating token by token: one row per group of 32 threads. Each thread takes every 32nd unit of
// the row (a unit is 64 values, or 32 in the formats with 32-value blocks) and the group adds its sums up. A row
// is then finished in the time of two or three units, where a thread to a row takes the whole row's time however
// few rows there are, which is what a matrix of a few thousand rows costs.
//
// The grid is rows x matrices groups; out[matrix * rows + row].

static inline float unit_q4k(const device uchar* row, const device float* x, uint unit) {
    uint b = unit / 4, pair = unit % 4;
    const device uchar* block = row + b * 144;
    float d = read_half(block), dmin = read_half(block + 2);
    const device uchar* packed = block + 4;
    const device packed_uchar4* q = (const device packed_uchar4*)(block + 16 + pair * 32);
    const device packed_float4* xb = (const device packed_float4*)(x + b * 256 + pair * 64);
    uint j = 2 * pair;
    float s0, m0, s1, m1;
    if (j < 4) {
        s0 = float(packed[j] & 63); m0 = float(packed[j + 4] & 63);
        s1 = float(packed[j + 1] & 63); m1 = float(packed[j + 5] & 63);
    } else {
        s0 = float((packed[j + 4] & 15) | ((packed[j - 4] >> 6) << 4));
        m0 = float((packed[j + 4] >> 4) | ((packed[j] >> 6) << 4));
        s1 = float((packed[j + 5] & 15) | ((packed[j - 3] >> 6) << 4));
        m1 = float((packed[j + 5] >> 4) | ((packed[j + 1] >> 6) << 4));
    }
    float4 low = 0.0f, high = 0.0f, xlow = 0.0f, xhigh = 0.0f;
    for (uint i = 0; i < 8; i++) {
        uchar4 v = q[i];
        float4 a = xb[i], c = xb[8 + i];
        low += float4(v & uchar4(15)) * a;
        high += float4(v >> uchar4(4)) * c;
        xlow += a;
        xhigh += c;
    }
    float4 total = low * (d * s0) - xlow * (dmin * m0) + high * (d * s1) - xhigh * (dmin * m1);
    return total.x + total.y + total.z + total.w;
}

static inline float unit_q8_0(const device uchar* row, const device float* x, uint unit) {
    const device uchar* block = row + unit * 34;
    const device packed_char4* q = (const device packed_char4*)(block + 2);
    const device packed_float4* xb = (const device packed_float4*)(x + unit * 32);
    float4 acc = 0.0f;
    for (uint i = 0; i < 8; i++) {
        acc += float4(char4(q[i])) * float4(xb[i]);
    }
    return read_half(block) * (acc.x + acc.y + acc.z + acc.w);
}

static inline float unit_q5_0(const device uchar* row, const device float* x, uint unit) {
    const device uchar* block = row + unit * 22;
    uint high = uint(block[2]) | (uint(block[3]) << 8) | (uint(block[4]) << 16) | (uint(block[5]) << 24);
    const device packed_uchar4* q = (const device packed_uchar4*)(block + 6);
    const device packed_float4* xb = (const device packed_float4*)(x + unit * 32);
    float4 acc = 0.0f;
    for (uint i = 0; i < 4; i++) {
        uchar4 v = q[i];
        uint4 shifts = uint4(0, 1, 2, 3) + 4 * i;
        uint4 first = uint4(v & uchar4(15)) | (((uint4(high) >> shifts) & 1u) << 4);
        uint4 second = uint4(v >> uchar4(4)) | (((uint4(high) >> (shifts + 16)) & 1u) << 4);
        acc += (float4(first) - 16.0f) * float4(xb[i]) + (float4(second) - 16.0f) * float4(xb[4 + i]);
    }
    return read_half(block) * (acc.x + acc.y + acc.z + acc.w);
}

static inline float unit_q6k(const device uchar* row, const device float* x, uint unit) {
    uint b = unit / 4, part = (unit / 2) % 2, g = unit % 2;
    const device uchar* block = row + b * 210;
    float d = read_half(block + 208);
    const device packed_uchar4* low = (const device packed_uchar4*)(block + 64 * part);
    const device packed_uchar4* high = (const device packed_uchar4*)(block + 128 + 32 * part);
    const device char* scales = (const device char*)(block + 192 + 8 * part);
    const device packed_float4* xb = (const device packed_float4*)(x + b * 256 + 128 * part);
    float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint i = 0; i < 4; i++) {
        uint at = 4 * g + i;
        uchar4 l0 = low[at], l1 = low[8 + at], h = high[at];
        uchar4 three = uchar4(3), nibble = uchar4(15);
        float4 q1 = float4((l0 & nibble) | ((h & three) << uchar4(4))) - 32.0f;
        float4 q2 = float4((l1 & nibble) | (((h >> uchar4(2)) & three) << uchar4(4))) - 32.0f;
        float4 q3 = float4((l0 >> uchar4(4)) | (((h >> uchar4(4)) & three) << uchar4(4))) - 32.0f;
        float4 q4 = float4((l1 >> uchar4(4)) | ((h >> uchar4(6)) << uchar4(4))) - 32.0f;
        a0 += q1 * float4(xb[at]);
        a1 += q2 * float4(xb[8 + at]);
        a2 += q3 * float4(xb[16 + at]);
        a3 += q4 * float4(xb[24 + at]);
    }
    float4 total = a0 * float(scales[g]) + a1 * float(scales[2 + g]) + a2 * float(scales[4 + g]) + a3 * float(scales[6 + g]);
    return d * (total.x + total.y + total.z + total.w);
}

static inline float unit_f32(const device uchar* row, const device float* x, uint unit) {
    const device packed_float4* w = (const device packed_float4*)(row + unit * 128);
    const device packed_float4* xb = (const device packed_float4*)(x + unit * 32);
    float4 acc = 0.0f;
    for (uint i = 0; i < 8; i++) {
        acc += float4(w[i]) * float4(xb[i]);
    }
    return acc.x + acc.y + acc.z + acc.w;
}

struct RowParams {
    uint rows;
    uint cols;
    // Bytes from one matrix of a stack to the next.
    uint stride;
    // Rows a group of 32 threads shares: 1, 2, 4 or 8. Fewer threads to a row suit a large matrix, where
    // starting the threads is what costs; more suit a small one, which is over when its slowest thread is.
    uint together;
    // Floats from one matrix's input to the next: 0 when they all take the same vector.
    uint x_stride;
    // Floats from one token's inputs and outputs to the next, in a batch (the grid's third dimension), and
    // whether each token has its own order.
    uint token_stride;
    uint own_order;
};

// Grid row `y` multiplies matrix `order[y]` of the stack, so the matrices can be chosen by an earlier kernel.
#define ROW_KERNEL(name, unit_dot, unit_values, row_bytes)                                      \
    kernel void name(const device uchar* weights [[buffer(0)]],                                 \
                     const device float* x [[buffer(1)]],                                       \
                     device float* out [[buffer(2)]],                                           \
                     const device uint* order [[buffer(3)]],                                    \
                     constant RowParams& p [[buffer(4)]],                                       \
                     uint3 group [[threadgroup_position_in_grid]],                              \
                     ushort lane [[thread_index_in_threadgroup]]) {                             \
        uint lanes = 32 / p.together;                                                           \
        uint mine = lane / lanes;                                                               \
        uint r = group.x * p.together + mine;                                                   \
        uint shift = group.z * p.token_stride;                                                  \
        const device float* xs = x + shift + group.y * p.x_stride;                              \
        order += p.own_order != 0 ? shift : 0;                                                  \
        out += shift;                                                                           \
        float sum = 0.0f;                                                                       \
        if (r < p.rows) {                                                                       \
            const device uchar* row = weights + ulong(order[group.y]) * p.stride + ulong(r) * (row_bytes); \
            for (uint unit = lane % lanes; unit < p.cols / (unit_values); unit += lanes) {      \
                sum += unit_dot(row, xs, unit);                                                 \
            }                                                                                   \
        }                                                                                       \
        for (uint which = 0; which < p.together; which++) {                                     \
            float total = simd_sum(mine == which ? sum : 0.0f);                                 \
            if (lane == 0 && group.x * p.together + which < p.rows) {                           \
                out[group.y * p.rows + group.x * p.together + which] = total;                   \
            }                                                                                   \
        }                                                                                       \
    }

ROW_KERNEL(row_q4k, unit_q4k, 64, p.cols / 256 * 144)
ROW_KERNEL(row_q8_0, unit_q8_0, 32, p.cols / 32 * 34)
ROW_KERNEL(row_q5_0, unit_q5_0, 32, p.cols / 32 * 22)
ROW_KERNEL(row_q6k, unit_q6k, 64, p.cols / 256 * 210)
ROW_KERNEL(row_f32, unit_f32, 32, p.cols * 4)

// A short run of one expert's tokens. Read compressed weights directly rather than padding the
// input to an MPS matrix of 256 rows. Each SIMD group computes four output rows for one token.
#define TAIL_KERNEL(name, dot_unit, width, bytes)                                            \
    kernel void name(const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], \
                     device float* out [[buffer(2)]], constant uint2& shape [[buffer(3)]],     \
                     uint3 group [[threadgroup_position_in_grid]],                            \
                     ushort lane [[thread_index_in_threadgroup]]) {                           \
        uint rows = shape.x, cols = shape.y, row = group.x * 4 + lane / 8;                    \
        float sum = 0;                                                                        \
        for (uint u = lane % 8; u < cols / width; u += 8)                                     \
            sum += dot_unit(w + ulong(row) * (bytes), x + group.y * cols, u);                 \
        for (ushort d = 4; d; d /= 2) sum += simd_shuffle_xor(sum, d);                        \
        if (lane % 8 == 0) out[group.y * rows + row] = sum;                                   \
    }

TAIL_KERNEL(tail_q4k, unit_q4k, 64, cols / 256 * 144)
TAIL_KERNEL(tail_q5_0, unit_q5_0, 32, cols / 32 * 22)
TAIL_KERNEL(tail_q6k, unit_q6k, 64, cols / 256 * 210)
TAIL_KERNEL(tail_q8_0, unit_q8_0, 32, cols / 32 * 34)
