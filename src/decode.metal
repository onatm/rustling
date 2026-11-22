// Everything in a layer visit that is not a matrix product, for generating a token entirely on the GPU: norms,
// the reads from the residual blocks, the short convolutions, attention over the cached keys and values, expert
// routing and the experts' activation. With the row kernels in gguf.metal these make up one token as a single
// submission, with nothing handed back to the CPU in between.
//
// Vectors live in one scratch buffer and constants in another; a kernel is given each of its operands as a
// buffer binding at the operand's offset. Kernels that need a sum over a vector run a group of 32 threads to a
// vector, each thread taking every 32nd element.

#include <metal_stdlib>
using namespace metal;

// What changes from token to token.
struct Token {
    // How many tokens came before this one.
    uint position;
    // The long-range scale of full-attention layers at this position.
    float tau;
};

struct NormParams {
    uint n;
    uint weighted;
    float eps;
    // Groups of 32 threads to a vector.
    uint width;
};

// out = in * weight / sqrt(mean(in^2) + eps), for each vector of `n`. One group of `32 * width` threads per
// vector, each taking every so-manyth four elements: a thread gets through a loop slowly, so a long vector is
// worth many threads.
kernel void rms_norm(const device float* in [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     const device float* weight [[buffer(2)]],
                     constant NormParams& p [[buffer(3)]],
                     uint2 group [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_threadgroup]],
                     ushort simd_lane [[thread_index_in_simdgroup]],
                     ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[32];
    const device packed_float4* x = (const device packed_float4*)(in + group.x * p.n);
    uint threads = 32 * p.width;
    float4 squares = 0.0f;
    for (uint i = lane; i < p.n / 4; i += threads) squares += float4(x[i]) * float4(x[i]);
    float sum = simd_sum(squares.x + squares.y + squares.z + squares.w);
    if (simd_lane == 0) partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint g = 0; g < p.width; g++) total += partial[g];
    float scale = 1.0f / sqrt(total / float(p.n) + p.eps);
    device packed_float4* to = (device packed_float4*)(out + group.x * p.n);
    const device packed_float4* w = (const device packed_float4*)weight;
    for (uint i = lane; i < p.n / 4; i += threads) {
        to[i] = float4(x[i]) * scale * (p.weighted != 0 ? float4(w[i]) : float4(1.0f));
    }
}

struct ClipParams {
    uint n;
    uint scaled;
    float threshold;
    uint long_range;
};

// Each vector of `n`: times a learned scale, shrunk to the threshold when its length exceeds it, then times the
// position's long-range scale.
kernel void scale_clip(const device float* in [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       const device float* scale [[buffer(2)]],
                       const device Token* token [[buffer(3)]],
                       constant ClipParams& p [[buffer(4)]],
                       uint2 group [[threadgroup_position_in_grid]],
                       ushort lane [[thread_index_in_threadgroup]]) {
    uint base = group.x * p.n;
    float sum = 0.0f;
    for (uint i = lane; i < p.n; i += 32) {
        float v = in[base + i] * (p.scaled != 0 ? scale[base + i] : 1.0f);
        sum += v * v;
    }
    float norm = sqrt(simd_sum(sum));
    float factor = min(p.threshold / (norm + 1e-6f), 1.0f) * (p.long_range != 0 ? token->tau : 1.0f);
    for (uint i = lane; i < p.n; i += 32) {
        out[base + i] = in[base + i] * (p.scaled != 0 ? scale[base + i] : 1.0f) * factor;
    }
}

struct ReadParams {
    uint h;
    uint blocks;
    float eps;
    // Groups of 32 threads to a query.
    uint width;
};

static inline float4 block_values(uint which, uint i, const device packed_float4* b0, const device packed_float4* b1,
                                  const device packed_float4* b2, const device packed_float4* b3,
                                  const device packed_float4* b4) {
    switch (which) {
        case 0: return float4(b0[i]);
        case 1: return float4(b1[i]);
        case 2: return float4(b2[i]);
        case 3: return float4(b3[i]);
        default: return float4(b4[i]);
    }
}

static inline float added(float4 v) {
    return v.x + v.y + v.z + v.w;
}

// A read from the residual stream: attention with a learned query over the finished blocks (normed already)
// and the block in progress (normed here). One group of `32 * width` threads per query; out holds the queries'
// results back to back.
kernel void residual_read(const device float* queries [[buffer(0)]],
                          const device packed_float4* partial [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          const device packed_float4* b0 [[buffer(3)]],
                          const device packed_float4* b1 [[buffer(4)]],
                          const device packed_float4* b2 [[buffer(5)]],
                          const device packed_float4* b3 [[buffer(6)]],
                          const device packed_float4* b4 [[buffer(7)]],
                          constant ReadParams& p [[buffer(8)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]],
                          ushort simd_lane [[thread_index_in_simdgroup]],
                          ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    // Per group of 32 threads: the square of the block in progress, then the query's product with it and with
    // each finished block.
    threadgroup float partials[32 * 7];
    const device packed_float4* query = (const device packed_float4*)(queries + group.x * p.h);
    uint threads = 32 * p.width;
    float4 square = 0.0f, current = 0.0f;
    float4 finished[5] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
    for (uint i = lane; i < p.h / 4; i += threads) {
        float4 x = float4(partial[i]), q = float4(query[i]);
        square += x * x;
        current += q * x;
        for (uint s = 0; s < p.blocks; s++) finished[s] += q * block_values(s, i, b0, b1, b2, b3, b4);
    }
    float sums[7];
    sums[0] = simd_sum(added(square));
    sums[1] = simd_sum(added(current));
    for (uint s = 0; s < p.blocks; s++) sums[2 + s] = simd_sum(added(finished[s]));
    if (simd_lane == 0) {
        for (uint s = 0; s < 2 + p.blocks; s++) partials[simd_group * 7 + s] = sums[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < 2 + p.blocks; s++) {
        sums[s] = 0.0f;
        for (uint g = 0; g < p.width; g++) sums[s] += partials[g * 7 + s];
    }

    float normed = 1.0f / sqrt(sums[0] / float(p.h) + p.eps);
    float scale = 1.0f / sqrt(float(p.h));
    // The block in progress last, as the CPU orders them.
    float logits[6];
    for (uint s = 0; s < p.blocks; s++) logits[s] = sums[2 + s] * scale;
    logits[p.blocks] = sums[1] * normed * scale;
    float highest = logits[0];
    for (uint s = 1; s <= p.blocks; s++) highest = max(highest, logits[s]);
    float total = 0.0f;
    for (uint s = 0; s <= p.blocks; s++) {
        logits[s] = exp(logits[s] - highest);
        total += logits[s];
    }
    device packed_float4* to = (device packed_float4*)(out + group.x * p.h);
    for (uint i = lane; i < p.h / 4; i += threads) {
        float4 value = logits[p.blocks] / total * normed * float4(partial[i]);
        for (uint s = 0; s < p.blocks; s++) value += logits[s] / total * block_values(s, i, b0, b1, b2, b3, b4);
        to[i] = value;
    }
}

// A message from the previous pass merged into a state: (1 - tanh(forget)) * hidden + tanh(gain) * message.
kernel void merge(const device float* hidden [[buffer(0)]],
                  const device float* message [[buffer(1)]],
                  const device float* forget [[buffer(2)]],
                  const device float* gain [[buffer(3)]],
                  device float* out [[buffer(4)]],
                  uint i [[thread_position_in_grid]]) {
    out[i] = (1.0f - tanh(forget[i])) * hidden[i] + tanh(gain[i]) * message[i];
}

struct ConvParams {
    uint channels;
    uint kernel_size;
    // Channels that share a tap in the dynamic form.
    uint head_dim;
    uint dynamic;
    float scale;
    uint add;
};

// The short causal convolution with its residual: (sum over taps + x) * scale, plus `residual` when asked.
// `history` holds the inputs of the `kernel_size - 1` tokens before, oldest first. The static form's weights
// are [channel][tap] with the last tap on the current token; the dynamic form's are [lag][channel] plus a tap
// per head and lag computed from the token.
kernel void conv(const device float* x [[buffer(0)]],
                 const device float* history [[buffer(1)]],
                 const device float* weight [[buffer(2)]],
                 const device float* taps [[buffer(3)]],
                 const device float* residual [[buffer(4)]],
                 device float* out [[buffer(5)]],
                 const device Token* token [[buffer(6)]],
                 constant ConvParams& p [[buffer(7)]],
                 uint c [[thread_position_in_grid]]) {
    if (c >= p.channels) return;
    uint k = p.kernel_size;
    uint reach = min(k - 1, token->position);
    float sum = 0.0f;
    for (uint lag = 0; lag <= reach; lag++) {
        float past = lag == 0 ? x[c] : history[(k - 1 - lag) * p.channels + c];
        float tap = p.dynamic != 0 ? weight[lag * p.channels + c] + taps[(c / p.head_dim) * k + lag]
                                   : weight[c * k + (k - 1 - lag)];
        sum += tap * past;
    }
    float value = (sum + x[c]) * p.scale;
    out[c] = p.add != 0 ? residual[c] + value : value;
}

struct BiasParams {
    uint d_rel;
    uint extent;
};

// The attention bias of each head at each distance back: its relative state through the distance profiles.
// The grid is extent x heads.
kernel void distance_bias(const device float* rel [[buffer(0)]],
                          const device float* profiles [[buffer(1)]],
                          device float* bias [[buffer(2)]],
                          constant BiasParams& p [[buffer(3)]],
                          uint2 id [[thread_position_in_grid]]) {
    if (id.x >= p.extent) return;
    float sum = 0.0f;
    for (uint r = 0; r < p.d_rel; r++) sum += rel[id.y * p.d_rel + r] * profiles[r * p.extent + id.x];
    bias[id.y * p.extent + id.x] = sum;
}

struct AttentionParams {
    uint heads;
    // Query heads to a key head.
    uint share;
    uint dim;
    // A sliding layer keeps its keys and values in a ring of `ring` tokens and sees the last `window`.
    uint local;
    uint window;
    uint ring;
    uint extent;
    float scaling;
    // Floats from one head's scores to the next.
    uint stride;
    // Groups of 32 threads to a head when turning scores into weights.
    uint width;
    // Groups of 32 threads to sixteen dimensions of a head when summing values.
    uint value_width;
};

static inline uint keys_seen(const device Token* token, constant AttentionParams& p) {
    return p.local != 0 ? min(token->position + 1, p.window) : token->position + 1;
}

// Which row of the table holds the oldest visible token.
static inline uint oldest_row(uint seen, const device Token* token, constant AttentionParams& p) {
    uint at = token->position + 1 - seen;
    return p.local != 0 ? at % p.ring : at;
}

// Where the `j`th visible token's keys and values start, oldest first.
static inline uint slot_of(uint j, uint oldest, constant AttentionParams& p) {
    uint row = oldest + j;
    if (p.local != 0 && row >= p.ring) row -= p.ring;
    return row * (p.heads / p.share) * p.dim;
}

// The score of every visible key for every head. The grid is keys x key heads: a thread reads one key and
// scores it for each of the query heads that share it.
kernel void attention_scores(const device float* queries [[buffer(0)]],
                             const device float* keys [[buffer(1)]],
                             const device float* bias [[buffer(2)]],
                             device float* scores [[buffer(3)]],
                             const device Token* token [[buffer(4)]],
                             constant AttentionParams& p [[buffer(5)]],
                             uint2 id [[thread_position_in_grid]]) {
    uint seen = keys_seen(token, p);
    if (id.x >= seen) return;
    const device packed_float4* key =
        (const device packed_float4*)(keys + slot_of(id.x, oldest_row(seen, token, p), p) + id.y * p.dim);
    const device packed_float4* query = (const device packed_float4*)(queries + id.y * p.share * p.dim);
    uint quarter = p.dim / 4;
    float4 acc[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
    for (uint i = 0; i < quarter; i++) {
        float4 k = float4(key[i]);
        for (uint g = 0; g < p.share; g++) acc[g] += float4(query[g * quarter + i]) * k;
    }
    uint back = seen - 1 - id.x;
    for (uint g = 0; g < p.share; g++) {
        uint head = id.y * p.share + g;
        float biased = back < p.extent ? bias[head * p.extent + back] : 0.0f;
        scores[head * p.stride + id.x] = (acc[g].x + acc[g].y + acc[g].z + acc[g].w) * p.scaling + biased;
    }
}

// Scores to weights, in place: a softmax over the keys and the head's sink, which takes weight and gives
// nothing back. One group of `32 * width` threads per head.
kernel void attention_weights(device float* scores [[buffer(0)]],
                              const device float* sinks [[buffer(1)]],
                              const device Token* token [[buffer(2)]],
                              constant AttentionParams& p [[buffer(3)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_threadgroup]],
                              ushort simd_lane [[thread_index_in_simdgroup]],
                              ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float highs[32];
    threadgroup float sums[32];
    uint seen = keys_seen(token, p);
    uint threads = 32 * p.width;
    device float* mine = scores + group.x * p.stride;
    float sink = sinks[group.x];
    float highest = sink;
    for (uint j = lane; j < seen; j += threads) highest = max(highest, mine[j]);
    highest = simd_max(highest);
    if (simd_lane == 0) highs[simd_group] = highest;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 0; g < p.width; g++) highest = max(highest, highs[g]);
    float sum = 0.0f;
    for (uint j = lane; j < seen; j += threads) {
        float e = exp(mine[j] - highest);
        mine[j] = e;
        sum += e;
    }
    sum = simd_sum(sum);
    if (simd_lane == 0) sums[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = exp(sink - highest);
    for (uint g = 0; g < p.width; g++) total += sums[g];
    for (uint j = lane; j < seen; j += threads) mine[j] /= total;
}

// Each head's weighted sum of values, through its output gate. One group of `32 * value_width` threads per
// (head, sixteen dimensions), sharing the keys between them: many threads with a few keys each finish sooner
// than a few threads with many.
kernel void attention_values(const device float* weights [[buffer(0)]],
                             const device float* values [[buffer(1)]],
                             const device float* gate [[buffer(2)]],
                             device float* out [[buffer(3)]],
                             const device Token* token [[buffer(4)]],
                             constant AttentionParams& p [[buffer(5)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_threadgroup]],
                             ushort simd_lane [[thread_index_in_simdgroup]],
                             ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[32 * 16];
    uint seen = keys_seen(token, p);
    uint oldest = oldest_row(seen, token, p);
    uint sixteenths = p.dim / 16;
    uint head = group.x / sixteenths, part = group.x % sixteenths;
    uint threads = 32 * p.value_width;
    const device float* mine = weights + head * p.stride;
    uint at = (head / p.share) * p.dim + part * 16;
    float4 acc[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
    for (uint j = lane; j < seen; j += threads) {
        const device packed_float4* value = (const device packed_float4*)(values + slot_of(j, oldest, p) + at);
        float w = mine[j];
        for (uint i = 0; i < 4; i++) acc[i] += w * float4(value[i]);
    }
    for (uint i = 0; i < 4; i++) {
        float4 sum = float4(simd_sum(acc[i].x), simd_sum(acc[i].y), simd_sum(acc[i].z), simd_sum(acc[i].w));
        if (simd_lane == 0) {
            for (uint c = 0; c < 4; c++) partial[simd_group * 16 + 4 * i + c] = sum[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 16) {
        float total = 0.0f;
        for (uint g = 0; g < p.value_width; g++) total += partial[g * 16 + lane];
        uint to = head * p.dim + part * 16 + lane;
        out[to] = total / (1.0f + exp(-gate[to]));
    }
}

struct StoreParams {
    uint n;
    uint local;
    uint ring;
};

// Files this token's keys or values in the layer's table.
kernel void store(const device float* from [[buffer(0)]],
                  device float* table [[buffer(1)]],
                  const device Token* token [[buffer(2)]],
                  constant StoreParams& p [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint slot = p.local != 0 ? token->position % p.ring : token->position;
    table[slot * p.n + i] = from[i];
}

struct RouteParams {
    uint experts;
    uint chosen;
    float scale;
};

// Picks the experts: the highest sigmoid score plus bias, weighted by their scores' share. One group with a
// thread per expert (padded to whole groups of 32), which finds its own place in the ranking.
kernel void route(const device float* logits [[buffer(0)]],
                  const device float* bias [[buffer(1)]],
                  device uint* order [[buffer(2)]],
                  device float* weight [[buffer(3)]],
                  constant RouteParams& p [[buffer(4)]],
                  uint lane [[thread_index_in_threadgroup]],
                  ushort simd_lane [[thread_index_in_simdgroup]],
                  ushort simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float ranked[256];
    threadgroup float partial[8];
    bool real = lane < p.experts;
    float score = real ? 1.0f / (1.0f + exp(-logits[lane])) : 0.0f;
    if (real) ranked[lane] = score + bias[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Ties go to the lower index, as a stable sort leaves them.
    uint ahead = 0;
    if (real) {
        float mine = ranked[lane];
        for (uint e = 0; e < p.experts; e++) {
            float other = ranked[e];
            ahead += (other > mine || (other == mine && e < lane)) ? 1 : 0;
        }
    }
    bool chosen = real && ahead < p.chosen;
    float sum = simd_sum(chosen ? score : 0.0f);
    if (simd_lane == 0) partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 1e-20f;
    for (uint g = 0; g < (p.experts + 31) / 32; g++) total += partial[g];
    if (chosen) {
        order[ahead] = lane;
        weight[ahead] = score / total * p.scale;
    }
}

struct ActivationParams {
    uint units;
    float gate_cap;
    float up_cap;
};

// The experts' activation: softcap(gate) * sigmoid(gate) * softcap(up). Each expert's products hold its gate
// units and then its up units. The grid is units x experts.
kernel void activation(const device float* products [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant ActivationParams& p [[buffer(2)]],
                       uint2 id [[thread_position_in_grid]]) {
    if (id.x >= p.units) return;
    float gate = products[id.y * 2 * p.units + id.x];
    float up = products[id.y * 2 * p.units + p.units + id.x];
    out[id.y * p.units + id.x] =
        tanh(gate / p.gate_cap) * p.gate_cap / (1.0f + exp(-gate)) * tanh(up / p.up_cap) * p.up_cap;
}

struct CombineParams {
    uint n;
    uint count;
};

// The chosen experts' outputs summed with their routing weights.
kernel void combine(const device float* outputs [[buffer(0)]],
                    const device float* weight [[buffer(1)]],
                    device float* out [[buffer(2)]],
                    constant CombineParams& p [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    float sum = 0.0f;
    for (uint e = 0; e < p.count; e++) sum += weight[e] * outputs[e * p.n + i];
    out[i] = sum;
}

struct CapParams {
    uint n;
    float cap;
};

// Logits squashed into (-cap, cap).
kernel void soft_cap(device float* values [[buffer(0)]],
                     constant CapParams& p [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    values[i] = tanh(values[i] / p.cap) * p.cap;
}
