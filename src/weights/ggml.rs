//! The block formats GGUF files store weights in, as defined by ggml (`ggml-quants.c`): how to decode a row and
//! how to take its dot product with a vector.
//!
//! A row is a whole number of blocks. `Q8_0` and `Q5_0` blocks hold 32 values with one scale; the `K` formats
//! hold 256 values in sub-blocks that each have their own small integer scale.

use half::f16;

use crate::weights::quant;

/// How a row of weights is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    F32,
    Bf16,
    /// 4-bit values, eight sub-blocks of 32 with 6-bit scales and offsets: 4.5 bits per weight.
    Q4K,
    /// 5-bit values in blocks of 32: 5.5 bits per weight.
    Q5_0,
    /// 6-bit values, sixteen sub-blocks of 16 with 8-bit scales: 6.5625 bits per weight.
    Q6K,
    /// 8-bit values in blocks of 32: 8.5 bits per weight.
    Q8_0,
}

impl Format {
    /// Values and bytes in one block.
    pub fn block(self) -> (usize, usize) {
        match self {
            Format::F32 => (1, 4),
            Format::Bf16 => (1, 2),
            Format::Q4K => (256, 144),
            Format::Q5_0 => (32, 22),
            Format::Q6K => (256, 210),
            Format::Q8_0 => (32, 34),
        }
    }

    /// The bytes a row of `cols` values takes, or `None` when a row is not a whole number of blocks.
    pub fn row_bytes(self, cols: usize) -> Option<usize> {
        let (values, bytes) = self.block();
        cols.is_multiple_of(values).then_some(cols / values * bytes)
    }

    /// Whether the weights are quantised, as opposed to plain floats.
    pub fn is_quantised(self) -> bool {
        !matches!(self, Format::F32 | Format::Bf16)
    }

    /// Decodes one stored row into `out`.
    pub fn decode_row(self, row: &[u8], out: &mut [f32]) {
        assert_eq!(Some(row.len()), self.row_bytes(out.len()), "{self:?} row of {} values", out.len());
        let (values, bytes) = self.block();
        match self {
            Format::F32 => {
                out.iter_mut().zip(row.as_chunks::<4>().0).for_each(|(o, b)| *o = f32::from_le_bytes(*b));
            }
            Format::Bf16 => quant::decode_bf16_row(row, out),
            _ => {
                for (block, out) in row.chunks_exact(bytes).zip(out.chunks_exact_mut(values)) {
                    match self {
                        Format::Q4K => decode_q4k(block, out),
                        Format::Q5_0 => decode_q5_0(block, out),
                        Format::Q6K => decode_q6k(block, out),
                        Format::Q8_0 => decode_q8_0(block, out),
                        Format::F32 | Format::Bf16 => unreachable!(),
                    }
                }
            }
        }
    }

    /// `row . x`, decoding the row into `scratch` first unless it is stored as floats.
    pub fn dot_row(self, row: &[u8], x: &[f32], scratch: &mut Vec<f32>) -> f32 {
        match self {
            Format::Bf16 => quant::dot_bf16_row(row, x),
            // SAFETY: any four bytes are a valid f32; only an aligned middle part is used.
            Format::F32 if unsafe { row.align_to::<f32>() }.0.is_empty() => {
                quant::dot_f32_row(unsafe { row.align_to::<f32>() }.1, x)
            }
            _ => {
                scratch.resize(x.len(), 0.0);
                self.decode_row(row, scratch);
                quant::dot_f32_row(scratch, x)
            }
        }
    }
}

fn half(bytes: &[u8]) -> f32 {
    f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()
}

/// `d: f16`, then 32 signed bytes.
fn decode_q8_0(block: &[u8], out: &mut [f32]) {
    let d = half(block);
    out.iter_mut().zip(&block[2..]).for_each(|(o, q)| *o = d * (*q as i8) as f32);
}

/// `d: f16`, 32 high bits, then 16 bytes of low nibbles: value `i` in the low half of byte `i`, value `i + 16` in
/// the high half.
fn decode_q5_0(block: &[u8], out: &mut [f32]) {
    let d = half(block);
    let high = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let low = &block[6..22];
    for j in 0..16 {
        let first = (low[j] & 0x0F) as i32 | ((high >> j) & 1) as i32 * 16;
        let second = (low[j] >> 4) as i32 | ((high >> (j + 16)) & 1) as i32 * 16;
        out[j] = d * (first - 16) as f32;
        out[j + 16] = d * (second - 16) as f32;
    }
}

/// The 6-bit scale and offset of sub-block `j` from the 12 packed bytes of a `Q4_K` block.
pub(crate) fn q4k_scale_min(j: usize, packed: &[u8]) -> (u8, u8) {
    if j < 4 {
        (packed[j] & 63, packed[j + 4] & 63)
    } else {
        ((packed[j + 4] & 0x0F) | ((packed[j - 4] >> 6) << 4), (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4))
    }
}

/// `d: f16`, `dmin: f16`, 12 bytes of scales and offsets, then 128 bytes of nibbles. Each 32 bytes hold two
/// sub-blocks: the low nibbles, then the high ones. A value is `d * scale * q - dmin * offset`.
fn decode_q4k(block: &[u8], out: &mut [f32]) {
    let (d, dmin) = (half(block), half(&block[2..]));
    let (packed, nibbles) = (&block[4..16], &block[16..144]);
    for (pair, (q, out)) in nibbles.chunks_exact(32).zip(out.chunks_exact_mut(64)).enumerate() {
        let (low, high) = out.split_at_mut(32);
        let (scale, min) = q4k_scale_min(2 * pair, packed);
        let (scale, min) = (d * scale as f32, dmin * min as f32);
        low.iter_mut().zip(q).for_each(|(o, q)| *o = scale * (q & 0x0F) as f32 - min);
        let (scale, min) = q4k_scale_min(2 * pair + 1, packed);
        let (scale, min) = (d * scale as f32, dmin * min as f32);
        high.iter_mut().zip(q).for_each(|(o, q)| *o = scale * (q >> 4) as f32 - min);
    }
}

/// 128 bytes of low nibbles, 64 bytes of high bit pairs, 16 signed scales, then `d: f16`. A value is
/// `d * scale * (q - 32)`.
fn decode_q6k(block: &[u8], out: &mut [f32]) {
    let d = half(&block[208..]);
    let scales = &block[192..208];
    for half_block in 0..2 {
        let low = &block[half_block * 64..][..64];
        let high = &block[128 + half_block * 32..][..32];
        let scales = &scales[half_block * 8..][..8];
        let out = &mut out[half_block * 128..][..128];
        for l in 0..32 {
            let scale = |k: usize| d * (scales[l / 16 + k] as i8) as f32;
            let q = [
                (low[l] & 0x0F) | ((high[l] & 3) << 4),
                (low[l + 32] & 0x0F) | (((high[l] >> 2) & 3) << 4),
                (low[l] >> 4) | (((high[l] >> 4) & 3) << 4),
                (low[l + 32] >> 4) | (((high[l] >> 6) & 3) << 4),
            ];
            for (k, q) in q.into_iter().enumerate() {
                out[l + 32 * k] = scale(2 * k) * (q as i32 - 32) as f32;
            }
        }
    }
}

/// The length of the blocks an [`Int8Activation`] is scaled in.
pub const INT8_BLOCK: usize = 32;

/// How many activations [`dot_row_int8_many`] takes at once.
pub const INT8_MAX_BATCH: usize = 8;

/// A vector rounded to signed bytes, each block of 32 with its own scale, so that its dot product with a
/// quantised row can be taken in integer arithmetic. This is how ggml multiplies quantised weights on a CPU; the
/// rounding costs about one part in 250 of each block's largest value.
#[derive(Debug, Clone)]
pub struct Int8Activation {
    values: Vec<i8>,
    /// What one step of a block's values is worth.
    scales: Vec<f32>,
    /// Each block's values summed and scaled: what the block contributes per unit of a constant weight offset.
    sums: Vec<f32>,
}

impl Int8Activation {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The values the rounded vector stands for.
    pub fn dequantize(&self) -> Vec<f32> {
        let blocks = self.values.chunks_exact(INT8_BLOCK).zip(&self.scales);
        blocks.flat_map(|(block, scale)| block.iter().map(move |q| *q as f32 * scale)).collect()
    }
}

/// Packs `values`, whose length must be a multiple of 32, as a `Q8_0` row: per block, a half-float scale and 32
/// signed bytes.
pub fn encode_q8_0(values: &[f32]) -> Vec<u8> {
    assert!(values.len().is_multiple_of(32), "{} values do not make whole blocks", values.len());
    let mut row = Vec::with_capacity(values.len() / 32 * 34);
    for block in values.as_chunks::<32>().0 {
        let largest = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = largest / 127.0;
        let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        row.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        row.extend(block.iter().map(|v| (v * inverse).round_ties_even() as i8 as u8));
    }
    row
}

/// Rounds `x`, whose length must be a multiple of 32, to signed bytes.
pub fn quantize_int8(x: &[f32]) -> Int8Activation {
    assert!(x.len().is_multiple_of(INT8_BLOCK), "{} values do not make whole blocks", x.len());
    let mut values = vec![0i8; x.len()];
    let blocks = x.len() / INT8_BLOCK;
    let (mut scales, mut sums) = (Vec::with_capacity(blocks), Vec::with_capacity(blocks));
    for (block, out) in x.as_chunks::<INT8_BLOCK>().0.iter().zip(values.as_chunks_mut::<INT8_BLOCK>().0) {
        let largest = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if largest == 0.0 || !largest.is_finite() {
            scales.push(0.0);
            sums.push(0.0);
            continue;
        }
        let inverse = 127.0 / largest;
        let mut sum = 0i32;
        for (q, v) in out.iter_mut().zip(block) {
            *q = (v * inverse).round_ties_even() as i8;
            sum += *q as i32;
        }
        let scale = largest / 127.0;
        scales.push(scale);
        sums.push(scale * sum as f32);
    }
    Int8Activation { values, scales, sums }
}

/// `row . x` for a quantised row and each of up to [`INT8_MAX_BATCH`] rounded activations, the row unpacked once.
/// Every activation gets exactly the arithmetic it would get alone, so results do not depend on batching.
pub fn dot_row_int8_many(format: Format, row: &[u8], xs: &[&Int8Activation], out: &mut [f32]) {
    assert!(format.is_quantised(), "{format:?} rows take float activations");
    assert!(xs.len() == out.len() && xs.len() <= INT8_MAX_BATCH);
    for x in xs {
        assert_eq!(Some(row.len()), format.row_bytes(x.len()), "{format:?} row against {} values", x.len());
    }
    #[cfg(target_arch = "aarch64")]
    return neon::dot_row_many(format, row, xs, out);
    #[cfg(not(target_arch = "aarch64"))]
    for (x, out) in xs.iter().zip(out) {
        *out = dot_row_int8_portable(format, row, x);
    }
}

/// The plain form of [`dot_row_int8_many`]: the same sum, in a different order.
pub fn dot_row_int8_portable(format: Format, row: &[u8], x: &Int8Activation) -> f32 {
    let mut weights = vec![0.0f32; x.len()];
    format.decode_row(row, &mut weights);
    let blocks = weights.chunks_exact(INT8_BLOCK).zip(x.values.chunks_exact(INT8_BLOCK)).zip(&x.scales);
    blocks.map(|((w, q), scale)| w.iter().zip(q).map(|(w, q)| w * *q as f32).sum::<f32>() * scale).sum()
}

/// NEON kernels. Each unpacks sixteen weights at a time into signed bytes and multiplies them with sixteen
/// activation bytes in one signed dot-product instruction; the integer sums of a sub-block are then scaled by the
/// weights' and the activation's scales together.
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    use super::{Format, Int8Activation, half, q4k_scale_min};

    /// Splits the activations into groups of 8, 4, 2 and 1 so each runs with a fixed-size inner loop.
    pub fn dot_row_many(format: Format, row: &[u8], mut xs: &[&Int8Activation], mut out: &mut [f32]) {
        fn take<const N: usize>(format: Format, row: &[u8], xs: &mut &[&Int8Activation], out: &mut &mut [f32]) {
            while let Some((group, rest)) = xs.split_first_chunk::<N>() {
                let (head, tail) = std::mem::take(out).split_at_mut(N);
                // SAFETY: the caller checked that the row is whole blocks covering every activation's length.
                head.copy_from_slice(&unsafe {
                    match format {
                        Format::Q4K => q4k(row, group),
                        Format::Q5_0 => q5_0(row, group),
                        Format::Q6K => q6k(row, group),
                        Format::Q8_0 => q8_0(row, group),
                        Format::F32 | Format::Bf16 => unreachable!("checked by the caller"),
                    }
                });
                (*xs, *out) = (rest, tail);
            }
        }
        take::<8>(format, row, &mut xs, &mut out);
        take::<4>(format, row, &mut xs, &mut out);
        take::<2>(format, row, &mut xs, &mut out);
        take::<1>(format, row, &mut xs, &mut out);
    }

    #[inline(always)]
    unsafe fn finish<const N: usize>(sums: [float32x4_t; N]) -> [f32; N] {
        unsafe { sums.map(|sum| vaddvq_f32(sum)) }
    }

    /// # Safety
    /// `row` must be whole 34-byte blocks, one for each 32 values of every activation.
    #[inline(always)]
    unsafe fn q8_0<const N: usize>(row: &[u8], xs: &[&Int8Activation; N]) -> [f32; N] {
        unsafe {
            let mut sums = [vdupq_n_f32(0.0); N];
            for (b, block) in row.chunks_exact(34).enumerate() {
                let d = half(block);
                let weights = [vld1q_s8(block.as_ptr().add(2).cast()), vld1q_s8(block.as_ptr().add(18).cast())];
                for (x, sum) in xs.iter().zip(&mut sums) {
                    let values = x.values.as_ptr().add(32 * b);
                    let dots = vdotq_s32(vdupq_n_s32(0), weights[0], vld1q_s8(values));
                    let dots = vdotq_s32(dots, weights[1], vld1q_s8(values.add(16)));
                    *sum = vfmaq_n_f32(*sum, vcvtq_f32_s32(dots), d * *x.scales.get_unchecked(b));
                }
            }
            finish(sums)
        }
    }

    /// # Safety
    /// `row` must be whole 22-byte blocks, one for each 32 values of every activation.
    #[inline(always)]
    unsafe fn q5_0<const N: usize>(row: &[u8], xs: &[&Int8Activation; N]) -> [f32; N] {
        const BYTE_OF: [u8; 32] =
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3];
        const BIT_OF: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        unsafe {
            let (first_bytes, second_bytes) = (vld1q_u8(BYTE_OF.as_ptr()), vld1q_u8(BYTE_OF.as_ptr().add(16)));
            let (bit, nibble, fifth, offset) = (vld1q_u8(BIT_OF.as_ptr()), vdupq_n_u8(15), vdupq_n_u8(16), vdupq_n_s8(16));
            let mut sums = [vdupq_n_f32(0.0); N];
            for (b, block) in row.chunks_exact(22).enumerate() {
                let d = half(block);
                // One bit per value: spread the four bytes so each lane can test its own bit.
                let high = vreinterpretq_u8_u32(vdupq_n_u32(u32::from_le_bytes([block[2], block[3], block[4], block[5]])));
                let high = |bytes| vandq_u8(vtstq_u8(vqtbl1q_u8(high, bytes), bit), fifth);
                let low = vld1q_u8(block.as_ptr().add(6));
                let weights = [
                    vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(low, nibble), high(first_bytes))), offset),
                    vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(low), high(second_bytes))), offset),
                ];
                for (x, sum) in xs.iter().zip(&mut sums) {
                    let values = x.values.as_ptr().add(32 * b);
                    let dots = vdotq_s32(vdupq_n_s32(0), weights[0], vld1q_s8(values));
                    let dots = vdotq_s32(dots, weights[1], vld1q_s8(values.add(16)));
                    *sum = vfmaq_n_f32(*sum, vcvtq_f32_s32(dots), d * *x.scales.get_unchecked(b));
                }
            }
            finish(sums)
        }
    }

    /// # Safety
    /// `row` must be whole 144-byte blocks, one for each 256 values of every activation.
    #[inline(always)]
    unsafe fn q4k<const N: usize>(row: &[u8], xs: &[&Int8Activation; N]) -> [f32; N] {
        unsafe {
            let nibble = vdupq_n_u8(15);
            let mut sums = [vdupq_n_f32(0.0); N];
            let mut offsets = [vdupq_n_f32(0.0); N];
            for (b, block) in row.chunks_exact(144).enumerate() {
                let (d, dmin) = (half(block), half(&block[2..]));
                let (mut scales, mut mins) = ([0.0f32; 8], [0.0f32; 8]);
                for j in 0..8 {
                    let (scale, min) = q4k_scale_min(j, &block[4..16]);
                    (scales[j], mins[j]) = (d * scale as f32, dmin * min as f32);
                }
                // Every weight of a sub-block carries the same offset, which meets the sum of the activations.
                let mins = [vld1q_f32(mins.as_ptr()), vld1q_f32(mins.as_ptr().add(4))];
                for (x, offset) in xs.iter().zip(&mut offsets) {
                    let block_sums = x.sums.as_ptr().add(8 * b);
                    *offset = vfmaq_f32(*offset, mins[0], vld1q_f32(block_sums));
                    *offset = vfmaq_f32(*offset, mins[1], vld1q_f32(block_sums.add(4)));
                }
                // 32 bytes hold two sub-blocks: the low nibbles, then the high ones.
                for pair in 0..4 {
                    let packed = block.as_ptr().add(16 + 32 * pair);
                    let (first, second) = (vld1q_u8(packed), vld1q_u8(packed.add(16)));
                    let low = [vreinterpretq_s8_u8(vandq_u8(first, nibble)), vreinterpretq_s8_u8(vandq_u8(second, nibble))];
                    let high = [vreinterpretq_s8_u8(vshrq_n_u8::<4>(first)), vreinterpretq_s8_u8(vshrq_n_u8::<4>(second))];
                    for (x, sum) in xs.iter().zip(&mut sums) {
                        let values = x.values.as_ptr().add(256 * b + 64 * pair);
                        let activation = x.scales.as_ptr().add(8 * b + 2 * pair);
                        let dots = vdotq_s32(vdupq_n_s32(0), low[0], vld1q_s8(values));
                        let dots = vdotq_s32(dots, low[1], vld1q_s8(values.add(16)));
                        *sum = vfmaq_n_f32(*sum, vcvtq_f32_s32(dots), scales[2 * pair] * *activation);
                        let dots = vdotq_s32(vdupq_n_s32(0), high[0], vld1q_s8(values.add(32)));
                        let dots = vdotq_s32(dots, high[1], vld1q_s8(values.add(48)));
                        *sum = vfmaq_n_f32(*sum, vcvtq_f32_s32(dots), scales[2 * pair + 1] * *activation.add(1));
                    }
                }
            }
            let (sums, offsets) = (finish(sums), finish(offsets));
            std::array::from_fn(|i| sums[i] - offsets[i])
        }
    }

    /// # Safety
    /// `row` must be whole 210-byte blocks, one for each 256 values of every activation.
    #[inline(always)]
    unsafe fn q6k<const N: usize>(row: &[u8], xs: &[&Int8Activation; N]) -> [f32; N] {
        unsafe {
            let (nibble, pair, offset) = (vdupq_n_u8(15), vdupq_n_u8(3), vdupq_n_s8(32));
            let mut sums = [vdupq_n_f32(0.0); N];
            for (b, block) in row.chunks_exact(210).enumerate() {
                let d = half(&block[208..]);
                for half_block in 0..2 {
                    let low = block.as_ptr().add(64 * half_block);
                    let high = block.as_ptr().add(128 + 32 * half_block);
                    let low = [vld1q_u8(low), vld1q_u8(low.add(16)), vld1q_u8(low.add(32)), vld1q_u8(low.add(48))];
                    let high = [vld1q_u8(high), vld1q_u8(high.add(16))];
                    // Value `16g + i` takes its low four bits from one nibble and its top two from a bit pair.
                    let join = |low: uint8x16_t, top: uint8x16_t| {
                        let joined = vorrq_u8(low, vshlq_n_u8::<4>(vandq_u8(top, pair)));
                        vsubq_s8(vreinterpretq_s8_u8(joined), offset)
                    };
                    let weights = [
                        join(vandq_u8(low[0], nibble), high[0]),
                        join(vandq_u8(low[1], nibble), high[1]),
                        join(vandq_u8(low[2], nibble), vshrq_n_u8::<2>(high[0])),
                        join(vandq_u8(low[3], nibble), vshrq_n_u8::<2>(high[1])),
                        join(vshrq_n_u8::<4>(low[0]), vshrq_n_u8::<4>(high[0])),
                        join(vshrq_n_u8::<4>(low[1]), vshrq_n_u8::<4>(high[1])),
                        join(vshrq_n_u8::<4>(low[2]), vshrq_n_u8::<6>(high[0])),
                        join(vshrq_n_u8::<4>(low[3]), vshrq_n_u8::<6>(high[1])),
                    ];
                    let scales: [f32; 8] = std::array::from_fn(|g| d * (block[192 + 8 * half_block + g] as i8) as f32);
                    for (x, sum) in xs.iter().zip(&mut sums) {
                        let values = x.values.as_ptr().add(256 * b + 128 * half_block);
                        let activation = x.scales.as_ptr().add(8 * b + 4 * half_block);
                        for g in 0..8 {
                            let dots = vdotq_s32(vdupq_n_s32(0), weights[g], vld1q_s8(values.add(16 * g)));
                            *sum = vfmaq_n_f32(*sum, vcvtq_f32_s32(dots), scales[g] * *activation.add(g / 2));
                        }
                    }
                }
            }
            finish(sums)
        }
    }
}

/// A row of pseudo-random bytes, shaped so every block is valid: the half-float scales are kept small.
#[cfg(test)]
pub(crate) fn random_row(format: Format, cols: usize, seed: u32) -> Vec<u8> {
    let (_, block_bytes) = format.block();
    let mut state = seed;
    let mut row: Vec<u8> = (0..format.row_bytes(cols).unwrap())
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect();
    let scale_at: &[usize] = match format {
        Format::Q4K => &[0, 2],
        Format::Q6K => &[208],
        _ => &[0],
    };
    for block in row.chunks_exact_mut(block_bytes) {
        for &at in scale_at {
            let scale = f16::from_f32(0.01 + block[at] as f32 * 1e-4).to_le_bytes();
            block[at..at + 2].copy_from_slice(&scale);
        }
    }
    row
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct Row {
        format: String,
        bytes: String,
        values: Vec<f32>,
    }

    /// Rows taken from a published GGUF file must decode to exactly what the reference decoder gives.
    #[test]
    fn a_row_packed_as_q8_0_decodes_to_nearly_the_same_values() {
        let values: Vec<f32> = (0..96).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013).collect();
        let row = encode_q8_0(&values);
        assert_eq!(row.len(), Format::Q8_0.row_bytes(96).unwrap());
        let mut decoded = vec![0.0f32; 96];
        Format::Q8_0.decode_row(&row, &mut decoded);
        // Half a step of the largest value's 127 levels, plus the scale's own rounding to a half float.
        let step = 0.65 / 127.0;
        assert!(values.iter().zip(&decoded).all(|(v, d)| (v - d).abs() <= step * 0.51));
        assert!(encode_q8_0(&[0.0; 32]).iter().all(|b| *b == 0));
    }

    #[test]
    fn rows_decode_as_the_reference_does() {
        let rows: Vec<Row> = serde_json::from_str(include_str!("../../tests/golden/ggml_rows.json")).unwrap();
        assert_eq!(rows.len(), 4);
        for row in rows {
            let format = match row.format.as_str() {
                "Q4_K" => Format::Q4K,
                "Q5_0" => Format::Q5_0,
                "Q6_K" => Format::Q6K,
                "Q8_0" => Format::Q8_0,
                other => panic!("unexpected format {other}"),
            };
            let bytes: Vec<u8> =
                (0..row.bytes.len() / 2).map(|i| u8::from_str_radix(&row.bytes[2 * i..2 * i + 2], 16).unwrap()).collect();
            let mut decoded = vec![0.0f32; row.values.len()];
            format.decode_row(&bytes, &mut decoded);
            assert_eq!(decoded, row.values, "{format:?}");
            assert!(decoded.iter().any(|v| *v != 0.0));
        }
    }

    #[test]
    fn rounding_to_bytes_stays_within_half_a_step() {
        let x: Vec<f32> = (0..96).map(|i| ((i * 37 % 53) as f32 - 26.0) * if i < 32 { 0.0 } else { 0.013 }).collect();
        let rounded = quantize_int8(&x).dequantize();
        for (block, rounded) in x.chunks(32).zip(rounded.chunks(32)) {
            let step = block.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
            assert!(block.iter().zip(rounded).all(|(a, b)| (a - b).abs() <= step * 0.5001));
        }
    }

    /// The integer kernels must give the dot product of the decoded row with the rounded activation, for every
    /// format and however many activations share the row.
    #[test]
    fn integer_dot_products_match_the_decoded_rows() {
        for format in [Format::Q4K, Format::Q5_0, Format::Q6K, Format::Q8_0] {
            let cols = 768;
            let row = random_row(format, cols, 7);
            let xs: Vec<Int8Activation> = (0..15)
                .map(|v| quantize_int8(&(0..cols).map(|i| ((i * 31 + v * 17) % 101) as f32 * 0.02 - 1.0).collect::<Vec<_>>()))
                .collect();
            let mut decoded = vec![0.0f32; cols];
            format.decode_row(&row, &mut decoded);
            let alone: Vec<f32> = xs
                .iter()
                .map(|x| {
                    let mut out = [0.0f32];
                    dot_row_int8_many(format, &row, &[x], &mut out);
                    let exact: f64 = decoded.iter().zip(x.dequantize()).map(|(w, x)| *w as f64 * x as f64).sum();
                    let size: f64 = decoded.iter().zip(x.dequantize()).map(|(w, x)| (*w as f64 * x as f64).abs()).sum();
                    assert!((out[0] as f64 - exact).abs() < 1e-5 * size, "{format:?}: {} against {exact}", out[0]);
                    assert!((dot_row_int8_portable(format, &row, x) as f64 - exact).abs() < 1e-5 * size);
                    out[0]
                })
                .collect();
            // Batched in every group size, each result is bit for bit the one the activation gets alone.
            for batch in [2, 3, 5, 8] {
                for (xs, alone) in xs.chunks(batch).zip(alone.chunks(batch)) {
                    let refs: Vec<&Int8Activation> = xs.iter().collect();
                    let mut out = vec![0.0f32; refs.len()];
                    dot_row_int8_many(format, &row, &refs, &mut out);
                    assert_eq!(out, alone, "{format:?} in batches of {batch}");
                }
            }
        }
    }
}
