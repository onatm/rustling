// Attention for a long batch on the GPU. Scores and outputs are matrix products, which Apple's routine does;
// between them every row's scores have to become weights: the bias for each key's distance added, the keys
// outside the row's window dropped, and a softmax taken over the rest and the head's sink.

#include <metal_stdlib>
using namespace metal;

struct WeightParams {
    // Keys in the table the scores were taken against: columns per row.
    uint keys;
    // Rows to a token: the query heads that share the key head.
    uint share;
    // The position of the first row's token, and of the first key in the table.
    uint first;
    uint lowest;
    // A sliding layer's rows see only the last `window` keys.
    uint local;
    uint window;
    // Distances the bias covers.
    uint extent;
    // Groups of 32 threads to a row.
    uint width;
};

// Scores to weights, in place. One group of `32 * width` threads per row, each taking every so-manyth key.
kernel void prompt_weights(device float* scores [[buffer(0)]],
                           const device float* bias [[buffer(1)]],
                           const device float* sinks [[buffer(2)]],
                           constant WeightParams& p [[buffer(3)]],
                           uint3 group [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_threadgroup]],
                           ushort simd_lane [[thread_index_in_simdgroup]],
                           ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float highs[32];
    threadgroup float sums[32];
    uint row = group.x;
    uint t = p.first + row / p.share;
    // The columns this row sees: [from, to).
    uint oldest = (p.local != 0 && t + 1 > p.window) ? t + 1 - p.window : 0;
    uint from = oldest > p.lowest ? oldest - p.lowest : 0;
    uint to = t + 1 - p.lowest;
    uint threads = 32 * p.width;
    device float* mine = scores + ulong(row) * p.keys;
    const device float* biased = bias + ulong(row) * p.extent;
    float sink = sinks[row];

    float highest = sink;
    for (uint c = lane; c < p.keys; c += threads) {
        if (c < from || c >= to) continue;
        uint back = to - 1 - c;
        float score = mine[c] + (back < p.extent ? biased[back] : 0.0f);
        mine[c] = score;
        highest = max(highest, score);
    }
    highest = simd_max(highest);
    if (simd_lane == 0) highs[simd_group] = highest;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 0; g < p.width; g++) highest = max(highest, highs[g]);

    float sum = 0.0f;
    for (uint c = lane; c < p.keys; c += threads) {
        if (c < from || c >= to) continue;
        float e = exp(mine[c] - highest);
        mine[c] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    if (simd_lane == 0) sums[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = exp(sink - highest);
    for (uint g = 0; g < p.width; g++) total += sums[g];

    for (uint c = lane; c < p.keys; c += threads) {
        mine[c] = (c < from || c >= to) ? 0.0f : mine[c] / total;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The experts of a long batch, kept on the GPU from the tokens' inputs to their outputs. The two matrix products
// are Apple's; these kernels do what lies between: laying each expert's inputs out as rows, the activation, and
// each token's weighted sum of its experts' outputs.

struct GatherParams {
    uint width;
};

// Row `r` of the products' input is token `token_of[r]`'s vector, or zeros for a padding row. The grid is
// (width / 4) x rows.
kernel void gather_rows(const device packed_float4* tokens [[buffer(0)]],
                        const device uint* token_of [[buffer(1)]],
                        device packed_float4* rows [[buffer(2)]],
                        constant GatherParams& p [[buffer(3)]],
                        uint2 id [[thread_position_in_grid]]) {
    uint quarter = p.width / 4;
    if (id.x >= quarter) return;
    uint token = token_of[id.y];
    rows[ulong(id.y) * quarter + id.x] = token == 0xFFFFFFFFu ? float4(0.0f) : float4(tokens[ulong(token) * quarter + id.x]);
}

struct ExpertActivationParams {
    uint units;
    float gate_cap;
    float up_cap;
    uint scaled;
};

// softcap(gate) * sigmoid(gate) * softcap(up) for every unit of every row. The grid is units x rows.
kernel void expert_activation(const device float* products [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              const device float* scales [[buffer(2)]],
                              constant ExpertActivationParams& p [[buffer(3)]],
                              uint2 id [[thread_position_in_grid]]) {
    if (id.x >= p.units) return;
    const device float* row = products + ulong(id.y) * 2 * p.units;
    float scale = p.scaled != 0 ? scales[id.y] : 1.f;
    float gate = row[id.x] * scale;
    float up = row[p.units + id.x] * scale;
    out[ulong(id.y) * p.units + id.x] =
        tanh(gate / p.gate_cap) * p.gate_cap / (1.0f + exp(-gate)) * tanh(up / p.up_cap) * p.up_cap;
}

struct RoutesParams {
    uint width;
    // Experts to a token.
    uint chosen;
};

// Each token's output: its experts' rows, weighted. `row_of` and `weight` hold `chosen` entries per token.
// The grid is (width / 4) x tokens.
kernel void combine_routes(const device packed_float4* rows [[buffer(0)]],
                           const device uint* row_of [[buffer(1)]],
                           const device float* weight [[buffer(2)]],
                           device packed_float4* out [[buffer(3)]],
                           constant RoutesParams& p [[buffer(4)]],
                           uint2 id [[thread_position_in_grid]]) {
    uint quarter = p.width / 4;
    if (id.x >= quarter) return;
    float4 sum = 0.0f;
    for (uint k = 0; k < p.chosen; k++) {
        uint at = id.y * p.chosen + k;
        sum += weight[at] * float4(rows[ulong(row_of[at]) * quarter + id.x]);
    }
    out[ulong(id.y) * quarter + id.x] = sum;
}
