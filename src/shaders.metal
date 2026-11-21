// Compute kernels for the two matrix products that dominate prompt processing.
//
// Each product of a weight row with a vector is computed with exactly the operations and summation order of the
// NEON kernels in quant.rs. With the library compiled in safe math mode that makes the results bit for bit the
// same as the CPU's, so the two backends can be mixed freely and tested against each other for equality.
//
// A thread handles one weight row against four vectors at once: unpacking the row is the expensive part, and this
// way it is paid once for the four.

#include <metal_stdlib>
using namespace metal;

// Vectors per thread.
#define SHARE 4

// Rounds to bfloat16 (nearest, ties to even) and widens back.
static inline float round_bf16(float value) {
    uint bits = as_type<uint>(value);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return as_type<float>(bits & 0xFFFF0000u);
}

struct DenseParams {
    uint rows;
    // A multiple of 16.
    uint cols;
    uint vectors;
};

// out[vector][row] = weights[row] . x[vector], for bfloat16 weights. Mirrors `dot_bf16_row`: four accumulators of
// four lanes, one per quarter of each 16-value step, combined pairwise at the end.
//
// The grid is rows x ceil(vectors / SHARE); a thread takes vectors SHARE * id.y onwards.
kernel void dense(const device uchar* weights [[buffer(0)]],
                  const device float* x [[buffer(1)]],
                  device float* out [[buffer(2)]],
                  constant DenseParams& p [[buffer(3)]],
                  uint2 id [[thread_position_in_grid]]) {
    uint first = id.y * SHARE;
    if (id.x >= p.rows || first >= p.vectors) {
        return;
    }
    // Eight bfloat16 values per load of four words. A bfloat16 is the top half of a float, so within each word
    // the first value is the word shifted up and the second is the word with its low half cleared.
    const device packed_uint4* words = (const device packed_uint4*)(weights + ulong(id.x) * p.cols * 2);
    const device packed_float4* v[SHARE];
    for (uint k = 0; k < SHARE; k++) {
        // Past the last vector, repeat it; the result is not stored.
        v[k] = (const device packed_float4*)(x + ulong(min(first + k, p.vectors - 1)) * p.cols);
    }
    float4 acc[SHARE][4];
    for (uint k = 0; k < SHARE; k++) {
        acc[k][0] = acc[k][1] = acc[k][2] = acc[k][3] = float4(0);
    }
    uint steps = p.cols / 16;
    for (uint i = 0; i < steps; i++) {
        uint4 a = words[2 * i], b = words[2 * i + 1];
        float4 a_first = as_type<float4>(a << 16), a_second = as_type<float4>(a & 0xFFFF0000u);
        float4 b_first = as_type<float4>(b << 16), b_second = as_type<float4>(b & 0xFFFF0000u);
        float4 w0 = float4(a_first.x, a_second.x, a_first.y, a_second.y);
        float4 w1 = float4(a_first.z, a_second.z, a_first.w, a_second.w);
        float4 w2 = float4(b_first.x, b_second.x, b_first.y, b_second.y);
        float4 w3 = float4(b_first.z, b_second.z, b_first.w, b_second.w);
        for (uint k = 0; k < SHARE; k++) {
            acc[k][0] = fma(w0, float4(v[k][4 * i]), acc[k][0]);
            acc[k][1] = fma(w1, float4(v[k][4 * i + 1]), acc[k][1]);
            acc[k][2] = fma(w2, float4(v[k][4 * i + 2]), acc[k][2]);
            acc[k][3] = fma(w3, float4(v[k][4 * i + 3]), acc[k][3]);
        }
    }
    for (uint k = 0; k < SHARE && first + k < p.vectors; k++) {
        float4 s = (acc[k][0] + acc[k][1]) + (acc[k][2] + acc[k][3]);
        out[ulong(first + k) * p.rows + id.x] = (s.x + s.y) + (s.z + s.w);
    }
}

struct ExpertParams {
    uint rows;
    // A multiple of 64.
    uint cols;
    uint groups;
};

// Up to SHARE activations to multiply by one expert's matrix.
struct Group {
    uint expert;
    // How many of `activations` are real.
    uint count;
    // The expert's outer weight scale, amax / denominator.
    float outer_scale;
    uint unused;
    uint activations[SHARE];
};

// Twice the E2M1 value of each 4-bit code: magnitudes 0, 1, 2, 3, 4, 6, 8, 12 for codes 0 to 7, negated when
// bit 3 is set. The magnitudes are the nibbles of 0xC8643210.
static inline int4 twice_e2m1(uint4 code) {
    int4 magnitude = int4((uint4(0xC8643210u) >> ((code & 7u) * 4u)) & 15u);
    int4 negative = int4(code >> 3);
    return (magnitude ^ -negative) + negative;
}

// out[SHARE * group + k][row] = round_bf16(expert weights[row] . activation k), both in NVFP4. Mirrors
// `dot_nvfp4_row_quantized_many`: exact integer sums per block of 16, scaled and accumulated four blocks at a time.
//
// The grid is rows x groups.
kernel void experts(const device uchar* values [[buffer(0)]],
                    const device uchar* scales [[buffer(1)]],
                    const device char* even [[buffer(2)]],
                    const device char* odd [[buffer(3)]],
                    const device float* x_scales [[buffer(4)]],
                    const device float* x_outer [[buffer(5)]],
                    const device Group* groups [[buffer(6)]],
                    const device float* e4m3 [[buffer(7)]],
                    device float* out [[buffer(8)]],
                    constant ExpertParams& p [[buffer(9)]],
                    uint2 id [[thread_position_in_grid]]) {
    if (id.x >= p.rows || id.y >= p.groups) {
        return;
    }
    Group group = groups[id.y];
    uint row_bytes = p.cols / 2;
    uint blocks = p.cols / 16;
    ulong row = ulong(group.expert) * p.rows + id.x;
    // Four packed bytes (eight weights) per load, with the matching four even-position and four odd-position
    // activation values. The sums within a block are integers, so their order does not matter.
    const device packed_uchar4* w = (const device packed_uchar4*)(values + row * row_bytes);
    const device uchar* w_scales = scales + row * blocks;
    const device packed_char4* e[SHARE];
    const device packed_char4* o[SHARE];
    const device packed_float4* xs[SHARE];
    for (uint k = 0; k < SHARE; k++) {
        ulong activation = group.activations[k];
        e[k] = (const device packed_char4*)(even + activation * row_bytes);
        o[k] = (const device packed_char4*)(odd + activation * row_bytes);
        xs[k] = (const device packed_float4*)(x_scales + activation * blocks);
    }

    float4 sum[SHARE];
    for (uint k = 0; k < SHARE; k++) {
        sum[k] = float4(0);
    }
    for (uint g = 0; g < blocks / 4; g++) {
        int4 block_sums[SHARE];
        for (uint b = 0; b < 4; b++) {
            uint base = (g * 4 + b) * 2;
            uint4 first = uint4(uchar4(w[base])), second = uint4(uchar4(w[base + 1]));
            int4 low0 = twice_e2m1(first & 15u), high0 = twice_e2m1(first >> 4);
            int4 low1 = twice_e2m1(second & 15u), high1 = twice_e2m1(second >> 4);
            for (uint k = 0; k < SHARE; k++) {
                int4 products = low0 * int4(char4(e[k][base])) + high0 * int4(char4(o[k][base]))
                              + low1 * int4(char4(e[k][base + 1])) + high1 * int4(char4(o[k][base + 1]));
                block_sums[k][b] = (products.x + products.y) + (products.z + products.w);
            }
        }
        float4 weight_scales = float4(e4m3[w_scales[g * 4]], e4m3[w_scales[g * 4 + 1]],
                                      e4m3[w_scales[g * 4 + 2]], e4m3[w_scales[g * 4 + 3]]);
        for (uint k = 0; k < SHARE; k++) {
            float4 block_scales = weight_scales * float4(xs[k][g]);
            sum[k] = fma(float4(block_sums[k]), block_scales, sum[k]);
        }
    }
    for (uint k = 0; k < group.count; k++) {
        float total = (sum[k].x + sum[k].y) + (sum[k].z + sum[k].w);
        float product = total * ((0.25f * group.outer_scale) * x_outer[group.activations[k]]);
        out[(ulong(id.y) * SHARE + k) * p.rows + id.x] = round_bf16(product);
    }
}
