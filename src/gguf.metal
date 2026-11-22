// Compute kernels for generating from a GGUF file: products of quantised weight rows with one float vector.
//
// A thread takes one row. Rows are whole blocks in ggml's formats (see ggml.rs for the layouts).

#include <metal_stdlib>
using namespace metal;

struct MvParams {
    uint rows;
    uint cols;
    // Bytes from one matrix of a stack to the next.
    uint stride;
};

static inline float read_half(const device uchar* p) {
    return float(as_type<half>(ushort(ushort(p[0]) | (ushort(p[1]) << 8))));
}

static inline float dot_q4k(const device uchar* row, const device float* x, uint cols) {
    float sum = 0.0f;
    for (uint b = 0; b < cols / 256; b++) {
        const device uchar* block = row + b * 144;
        float d = read_half(block), dmin = read_half(block + 2);
        const device uchar* packed = block + 4;
        const device packed_uchar4* q = (const device packed_uchar4*)(block + 16);
        const device packed_float4* xb = (const device packed_float4*)(x + b * 256);
        for (uint pair = 0; pair < 4; pair++) {
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
                uchar4 v = q[pair * 8 + i];
                float4 a = xb[pair * 16 + i], c = xb[pair * 16 + 8 + i];
                low += float4(v & uchar4(15)) * a;
                high += float4(v >> uchar4(4)) * c;
                xlow += a;
                xhigh += c;
            }
            float4 total = low * (d * s0) - xlow * (dmin * m0) + high * (d * s1) - xhigh * (dmin * m1);
            sum += total.x + total.y + total.z + total.w;
        }
    }
    return sum;
}

static inline float dot_q8_0(const device uchar* row, const device float* x, uint cols) {
    float sum = 0.0f;
    for (uint b = 0; b < cols / 32; b++) {
        const device uchar* block = row + b * 34;
        const device packed_char4* q = (const device packed_char4*)(block + 2);
        const device packed_float4* xb = (const device packed_float4*)(x + b * 32);
        float4 acc = 0.0f;
        for (uint i = 0; i < 8; i++) {
            acc += float4(char4(q[i])) * float4(xb[i]);
        }
        sum += read_half(block) * (acc.x + acc.y + acc.z + acc.w);
    }
    return sum;
}

static inline float dot_q5_0(const device uchar* row, const device float* x, uint cols) {
    float sum = 0.0f;
    for (uint b = 0; b < cols / 32; b++) {
        const device uchar* block = row + b * 22;
        uint high = uint(block[2]) | (uint(block[3]) << 8) | (uint(block[4]) << 16) | (uint(block[5]) << 24);
        const device packed_uchar4* q = (const device packed_uchar4*)(block + 6);
        const device packed_float4* xb = (const device packed_float4*)(x + b * 32);
        float4 acc = 0.0f;
        for (uint i = 0; i < 4; i++) {
            uchar4 v = q[i];
            uint4 shifts = uint4(0, 1, 2, 3) + 4 * i;
            uint4 first = uint4(v & uchar4(15)) | (((uint4(high) >> shifts) & 1u) << 4);
            uint4 second = uint4(v >> uchar4(4)) | (((uint4(high) >> (shifts + 16)) & 1u) << 4);
            acc += (float4(first) - 16.0f) * float4(xb[i]) + (float4(second) - 16.0f) * float4(xb[4 + i]);
        }
        sum += read_half(block) * (acc.x + acc.y + acc.z + acc.w);
    }
    return sum;
}

static inline float dot_q6k(const device uchar* row, const device float* x, uint cols) {
    float sum = 0.0f;
    for (uint b = 0; b < cols / 256; b++) {
        const device uchar* block = row + b * 210;
        float d = read_half(block + 208);
        for (uint part = 0; part < 2; part++) {
            const device packed_uchar4* low = (const device packed_uchar4*)(block + 64 * part);
            const device packed_uchar4* high = (const device packed_uchar4*)(block + 128 + 32 * part);
            const device char* scales = (const device char*)(block + 192 + 8 * part);
            const device packed_float4* xb = (const device packed_float4*)(x + b * 256 + 128 * part);
            // Sixteen values at a time share a scale: `g` of them per half-block.
            for (uint g = 0; g < 2; g++) {
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
                float4 total = a0 * float(scales[g]) + a1 * float(scales[2 + g]) + a2 * float(scales[4 + g])
                    + a3 * float(scales[6 + g]);
                sum += d * (total.x + total.y + total.z + total.w);
            }
        }
    }
    return sum;
}

// out[matrix][row] = weights[matrix][row] . x[matrix], over a rows x matrices grid.
#define MV_KERNEL(name, dot, row_bytes)                                                         \
    kernel void name(const device uchar* weights [[buffer(0)]],                                 \
                     const device float* x [[buffer(1)]],                                       \
                     device float* out [[buffer(2)]],                                           \
                     constant MvParams& p [[buffer(3)]],                                        \
                     uint2 id [[thread_position_in_grid]]) {                                    \
        if (id.x >= p.rows) return;                                                             \
        const device uchar* row = weights + ulong(id.y) * p.stride + ulong(id.x) * (row_bytes); \
        out[id.y * p.rows + id.x] = dot(row, x + id.y * p.cols, p.cols);                        \
    }

MV_KERNEL(mv_q4k, dot_q4k, p.cols / 256 * 144)
MV_KERNEL(mv_q8_0, dot_q8_0, p.cols / 32 * 34)
MV_KERNEL(mv_q5_0, dot_q5_0, p.cols / 32 * 22)
MV_KERNEL(mv_q6k, dot_q6k, p.cols / 256 * 210)

// Variant: aligned word loads, sub-block sums of x precomputed (buffer 4), two rows per thread.
static inline float2 dot_q4k_2(const device uint* r0, const device uint* r1, const device float4* x,
                               const device float* xsum, uint cols) {
    float2 sum = 0.0f;
    for (uint b = 0; b < cols / 256; b++) {
        const device uint* b0 = r0 + b * 36;
        const device uint* b1 = r1 + b * 36;
        uint h0 = b0[0], h1 = b1[0];
        float2 d = float2(float(as_type<half>(ushort(h0 & 0xFFFF))), float(as_type<half>(ushort(h1 & 0xFFFF))));
        float2 dmin = float2(float(as_type<half>(ushort(h0 >> 16))), float(as_type<half>(ushort(h1 >> 16))));
        const device float4* xb = x + b * 64;
        for (uint j = 0; j < 8; j += 2) {
            uint2 s0, m0, s1, m1;
            uint3 p0 = uint3(b0[1], b0[2], b0[3]), p1 = uint3(b1[1], b1[2], b1[3]);
            // packed bytes k = 0..11 live in words 1..3
            #define PK(p, k) (((p)[(k) / 4] >> (8 * ((k) % 4))) & 0xFFu)
            if (j < 4) {
                s0 = uint2(PK(p0, j) & 63, PK(p1, j) & 63); m0 = uint2(PK(p0, j + 4) & 63, PK(p1, j + 4) & 63);
                s1 = uint2(PK(p0, j + 1) & 63, PK(p1, j + 1) & 63); m1 = uint2(PK(p0, j + 5) & 63, PK(p1, j + 5) & 63);
            } else {
                s0 = uint2((PK(p0, j + 4) & 15) | ((PK(p0, j - 4) >> 6) << 4), (PK(p1, j + 4) & 15) | ((PK(p1, j - 4) >> 6) << 4));
                m0 = uint2((PK(p0, j + 4) >> 4) | ((PK(p0, j) >> 6) << 4), (PK(p1, j + 4) >> 4) | ((PK(p1, j) >> 6) << 4));
                s1 = uint2((PK(p0, j + 5) & 15) | ((PK(p0, j - 3) >> 6) << 4), (PK(p1, j + 5) & 15) | ((PK(p1, j - 3) >> 6) << 4));
                m1 = uint2((PK(p0, j + 5) >> 4) | ((PK(p0, j + 1) >> 6) << 4), (PK(p1, j + 5) >> 4) | ((PK(p1, j + 1) >> 6) << 4));
            }
            float4 l0 = 0.0f, g0 = 0.0f, l1 = 0.0f, g1 = 0.0f;
            for (uint i = 0; i < 8; i++) {
                uint w0 = b0[4 + j * 4 + i], w1 = b1[4 + j * 4 + i];
                float4 a = xb[j * 8 + i], c = xb[j * 8 + 8 + i];
                uint4 sh = uint4(0, 8, 16, 24);
                l0 += float4((uint4(w0) >> sh) & 15u) * a;
                g0 += float4((uint4(w0) >> (sh + 4)) & 15u) * c;
                l1 += float4((uint4(w1) >> sh) & 15u) * a;
                g1 += float4((uint4(w1) >> (sh + 4)) & 15u) * c;
            }
            float2 lo = float2(l0.x + l0.y + l0.z + l0.w, l1.x + l1.y + l1.z + l1.w);
            float2 hi = float2(g0.x + g0.y + g0.z + g0.w, g1.x + g1.y + g1.z + g1.w);
            sum += d * float2(s0) * lo + d * float2(s1) * hi
                - dmin * (float2(m0) * xsum[b * 8 + j] + float2(m1) * xsum[b * 8 + j + 1]);
        }
    }
    return sum;
}

kernel void mv_q4k_fast(const device uchar* weights [[buffer(0)]],
                        const device float* x [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant MvParams& p [[buffer(3)]],
                        const device float* xsum [[buffer(4)]],
                        uint2 id [[thread_position_in_grid]]) {
    uint row = id.x * 2;
    if (row >= p.rows) return;
    uint row_bytes = p.cols / 256 * 144;
    const device uchar* base = weights + ulong(id.y) * p.stride + ulong(row) * row_bytes;
    float2 r = dot_q4k_2((const device uint*)base, (const device uint*)(base + row_bytes),
                         (const device float4*)(x + id.y * p.cols), xsum + id.y * (p.cols / 32), p.cols);
    out[id.y * p.rows + row] = r.x;
    out[id.y * p.rows + row + 1] = r.y;
}

// ---------------------------------------------------------------------------------------------------------------
// The fast path for long prompts: chosen matrices of a stack are unpacked to float, for Apple's tuned matrix
// product. Out matrix `i` is stack matrix `order[i]`. A thread unpacks 32 values; the grid is
// (cols / 32) x (count * rows).

struct WidenParams {
    // Rows per matrix.
    uint rows;
    uint cols;
    // How many matrices to unpack.
    uint count;
};

#define WIDEN_KERNEL(name, values_per_block, block_bytes)                                                  \
    kernel void name(const device uchar* weights [[buffer(0)]],                                            \
                     const device uint* order [[buffer(1)]],                                               \
                     device float* out [[buffer(2)]],                                                      \
                     constant WidenParams& p [[buffer(3)]],                                                \
                     uint2 id [[thread_position_in_grid]]) {                                               \
        if (id.x >= p.cols / 32 || id.y >= p.count * p.rows) return;                                       \
        uint slot = id.y / p.rows, row = id.y % p.rows;                                                    \
        ulong row_bytes = ulong(p.cols / (values_per_block)) * (block_bytes);                              \
        uint per_block = (values_per_block) / 32;                                                          \
        const device uchar* block = weights + (ulong(order[slot]) * p.rows + row) * row_bytes              \
            + ulong(id.x / per_block) * (block_bytes);                                                     \
        device float* o = out + (ulong(id.y) * p.cols + id.x * 32);                                        \
        name##_part(block, id.x % per_block, o);                                                           \
    }

static inline void widen_q8_0_part(const device uchar* block, uint part, device float* o) {
    float d = read_half(block);
    for (uint i = 0; i < 32; i++) {
        o[i] = d * float(char(block[2 + i]));
    }
}

static inline void widen_q5_0_part(const device uchar* block, uint part, device float* o) {
    float d = read_half(block);
    uint high = uint(block[2]) | (uint(block[3]) << 8) | (uint(block[4]) << 16) | (uint(block[5]) << 24);
    for (uint i = 0; i < 16; i++) {
        uint v = block[6 + i];
        uint first = (v & 15u) | (((high >> i) & 1u) << 4);
        uint second = (v >> 4) | (((high >> (i + 16)) & 1u) << 4);
        o[i] = d * (float(first) - 16.0f);
        o[i + 16] = d * (float(second) - 16.0f);
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
    const device uchar* q = block + 16 + 32 * (part / 2);
    uint shift = 4 * (part % 2);
    for (uint i = 0; i < 32; i++) {
        o[i] = step * float((q[i] >> shift) & 15u) - low;
    }
}

// Part `part` of eight: half-block `part / 4`, and within it one of the four runs of 32 values.
static inline void widen_q6k_part(const device uchar* block, uint part, device float* o) {
    float d = read_half(block + 208);
    uint half_block = part / 4, run = part % 4;
    const device uchar* low = block + 64 * half_block + 32 * (run % 2);
    const device uchar* high = block + 128 + 32 * half_block;
    const device char* scales = (const device char*)(block + 192 + 8 * half_block + 2 * run);
    uint shift = 4 * (run / 2);
    for (uint i = 0; i < 32; i++) {
        uint value = ((low[i] >> shift) & 15u) | (((high[i] >> (2 * run)) & 3u) << 4);
        o[i] = d * float(scales[i / 16]) * (float(value) - 32.0f);
    }
}

WIDEN_KERNEL(widen_q8_0, 32, 34)
WIDEN_KERNEL(widen_q5_0, 32, 22)
WIDEN_KERNEL(widen_q4k, 256, 144)
WIDEN_KERNEL(widen_q6k, 256, 210)
