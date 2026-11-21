//! The two block floating-point formats the expert weights are stored in, and kernels that work on them directly.
//!
//! - **NVFP4**: 4-bit E2M1 values, two per byte (low nibble first), in blocks of 16 along a row. Each block has an
//!   8-bit E4M3 scale, and each expert has one more f32 scale, `amax / outer_scale_denominator`.
//! - **MXFP8**: 8-bit E4M3 values in blocks of 32 along a row, each block scaled by `2^(scale - 127)`.
//!
//! Each format has a row decoder and a dot product that reads the packed row without materialising it. Decoding
//! matches llama.cpp's converter (`conversion/inkling2.py`) value for value. On aarch64 both run on NEON; the
//! portable versions below are the specification they are tested against.
//!
//! The model was trained with the expert *inputs* quantised too, each in the format of the weights it meets.
//! [`quantize_nvfp4`] and [`requantize_mxfp8`] reproduce that rounding (as SGLang's and vLLM's kernels do it), and
//! with both sides in NVFP4 a row product becomes sums of small integer products: [`dot_nvfp4_row_quantized`].
//!
//! The NEON MXFP8 path does not preserve the two E4M3 NaN codes (`0x7F`, `0xFF`), which a quantiser never emits.

use std::sync::LazyLock;

pub const NVFP4_BLOCK: usize = 16;
pub const MXFP8_BLOCK: usize = 32;

/// E2M1 code -> value: one sign bit, two exponent bits, one mantissa bit.
pub const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// E4M3FN code -> value. Codes `0x7F` and `0xFF` are NaN; there are no infinities.
pub static E4M3: LazyLock<[f32; 256]> = LazyLock::new(|| {
    let mut table = [0.0f32; 256];
    for (code, value) in table.iter_mut().enumerate() {
        let exponent = ((code & 0x7F) >> 3) as i32;
        let mantissa = (code & 7) as f32;
        let magnitude = if code & 0x7F == 0x7F {
            f32::NAN
        } else if exponent == 0 {
            mantissa * 2f32.powi(-9)
        } else {
            (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
        };
        *value = if code & 0x80 != 0 { -magnitude } else { magnitude };
    }
    table
});

/// `2^(scale - 127)`. For every scale but 0 that is the float whose biased exponent field is `scale`.
fn mxfp8_block_scale(scale: u8) -> f32 {
    match scale {
        0 => 2f32.powi(-127),
        scale => f32::from_bits((scale as u32) << 23),
    }
}

/// Decodes one NVFP4 row. `packed` holds `out.len() / 2` bytes and `scales` one E4M3 byte per 16 values.
pub fn decode_nvfp4_row(packed: &[u8], scales: &[u8], outer_scale: f32, out: &mut [f32]) {
    assert_eq!(out.len(), packed.len() * 2);
    assert_eq!(out.len(), scales.len() * NVFP4_BLOCK);
    #[cfg(target_arch = "aarch64")]
    neon::decode_nvfp4_row(packed, scales, outer_scale, out);
    #[cfg(not(target_arch = "aarch64"))]
    portable::decode_nvfp4_row(packed, scales, outer_scale, out);
}

/// `row . x` for one NVFP4 row, read straight from its packed form.
pub fn dot_nvfp4_row(packed: &[u8], scales: &[u8], outer_scale: f32, x: &[f32]) -> f32 {
    assert_eq!(x.len(), packed.len() * 2);
    assert_eq!(x.len(), scales.len() * NVFP4_BLOCK);
    #[cfg(target_arch = "aarch64")]
    return neon::dot_nvfp4_row(packed, scales, outer_scale, x);
    #[cfg(not(target_arch = "aarch64"))]
    portable::dot_nvfp4_row(packed, scales, outer_scale, x)
}

/// Decodes one MXFP8 row. `values` holds one E4M3 byte per value and `scales` one exponent byte per 32 values.
pub fn decode_mxfp8_row(values: &[u8], scales: &[u8], out: &mut [f32]) {
    assert_eq!(out.len(), values.len());
    assert_eq!(out.len(), scales.len() * MXFP8_BLOCK);
    #[cfg(target_arch = "aarch64")]
    neon::decode_mxfp8_row(values, scales, out);
    #[cfg(not(target_arch = "aarch64"))]
    portable::decode_mxfp8_row(values, scales, out);
}

/// `row . x` for one MXFP8 row, read straight from its stored form.
pub fn dot_mxfp8_row(values: &[u8], scales: &[u8], x: &[f32]) -> f32 {
    assert_eq!(x.len(), values.len());
    assert_eq!(x.len(), scales.len() * MXFP8_BLOCK);
    #[cfg(target_arch = "aarch64")]
    return neon::dot_mxfp8_row(values, scales, x);
    #[cfg(not(target_arch = "aarch64"))]
    portable::dot_mxfp8_row(values, scales, x)
}

/// Rounds to the nearest E4M3 code, ties to the even code, saturating at the largest finite value (448).
pub fn encode_e4m3(value: f32) -> u8 {
    let table = &*E4M3;
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs();
    if magnitude.is_nan() {
        return sign | 0x7F;
    }
    // Codes 0x00..=0x7E are the finite magnitudes in increasing order.
    let above = table[..0x7F].partition_point(|v| *v <= magnitude);
    let low = above.saturating_sub(1);
    let high = above.min(0x7E);
    let (below, over) = (magnitude - table[low], table[high] - magnitude);
    let code = if over < below || (over == below && high % 2 == 0) { high } else { low };
    sign | code as u8
}

/// Twice the E2M1 values, so they fit in a byte.
const TWICE_E2M1: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Rounds to the nearest E2M1 code, ties to the even code, saturating at 6.
fn encode_e2m1(value: f32) -> u8 {
    let a = value.abs();
    let code = match a {
        _ if a <= 0.25 => 0,
        _ if a < 0.75 => 1,
        _ if a <= 1.25 => 2,
        _ if a < 1.75 => 3,
        _ if a <= 2.5 => 4,
        _ if a < 3.5 => 5,
        _ if a <= 5.0 => 6,
        _ => 7,
    };
    code | ((value.is_sign_negative() as u8) << 3)
}

/// One vector quantised to NVFP4, laid out for [`dot_nvfp4_row_quantized`].
///
/// Element `j` of block `b` is `twice[j] / 2 * block_scales[b] * outer_scale`. The doubled values are split by
/// position parity, because a packed weight byte holds an even-position value in its low nibble and the following
/// odd-position value in its high nibble.
#[derive(Clone)]
pub struct Nvfp4Activation {
    /// Doubled values at even positions, eight per block.
    even: Vec<i8>,
    /// Doubled values at odd positions, eight per block.
    odd: Vec<i8>,
    block_scales: Vec<f32>,
    outer_scale: f32,
}

impl Nvfp4Activation {
    pub fn len(&self) -> usize {
        self.even.len() * 2
    }

    pub fn is_empty(&self) -> bool {
        self.even.is_empty()
    }

    /// The doubled even-position values, the doubled odd-position values, the block scales and the outer scale.
    pub fn parts(&self) -> (&[i8], &[i8], &[f32], f32) {
        (&self.even, &self.odd, &self.block_scales, self.outer_scale)
    }

    /// The values the quantised vector stands for.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len());
        for ((even, odd), scale) in self.even.chunks_exact(8).zip(self.odd.chunks_exact(8)).zip(&self.block_scales) {
            for (e, o) in even.iter().zip(odd) {
                out.push(*e as f32 * 0.5 * scale * self.outer_scale);
                out.push(*o as f32 * 0.5 * scale * self.outer_scale);
            }
        }
        out
    }
}

/// Quantises one activation vector to NVFP4 the way the model's kernels do ("fouroversix_mae").
///
/// The vector gets one outer scale, `max|x| / 1536`. Each block of 16 then tries two E4M3 block scales, one that
/// maps the block's largest value to 6 and one that maps it to 4, and keeps whichever reconstructs the block with
/// the smaller total absolute error (6 on a tie).
pub fn quantize_nvfp4(x: &[f32]) -> Nvfp4Activation {
    assert_eq!(x.len() % NVFP4_BLOCK, 0);
    let e4m3 = &*E4M3;
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let encoder = (1536.0 / amax).min(f32::MAX);
    let decoder = 1.0 / encoder;
    let outer_scale = amax * (1.0 / 1536.0);

    let mut out = Nvfp4Activation {
        even: Vec::with_capacity(x.len() / 2),
        odd: Vec::with_capacity(x.len() / 2),
        block_scales: Vec::with_capacity(x.len() / NVFP4_BLOCK),
        outer_scale,
    };
    for block in x.as_chunks::<NVFP4_BLOCK>().0 {
        let block_max = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let candidate = |inverse_target: f32| {
            let scale = e4m3[encode_e4m3((block_max * (encoder * inverse_target)).min(448.0)) as usize];
            let inverse = (1.0 / (scale * decoder)).min(f32::MAX);
            let codes = block.map(|v| encode_e2m1(v * inverse));
            let mut errors: [f32; NVFP4_BLOCK] =
                std::array::from_fn(|i| ((E2M1[codes[i] as usize] * scale) * outer_scale - block[i]).abs());
            // Summed pairwise, as the reference does, so the comparison below rounds the same way.
            let mut width = NVFP4_BLOCK;
            while width > 1 {
                width /= 2;
                (0..width).for_each(|i| errors[i] = errors[2 * i] + errors[2 * i + 1]);
            }
            (codes, scale, errors[0])
        };
        let (codes6, scale6, error6) = candidate(1.0 / 6.0);
        let (codes4, scale4, error4) = candidate(0.25);
        let (codes, scale) = if error4 < error6 { (codes4, scale4) } else { (codes6, scale6) };
        for pair in codes.as_chunks::<2>().0 {
            out.even.push(TWICE_E2M1[pair[0] as usize]);
            out.odd.push(TWICE_E2M1[pair[1] as usize]);
        }
        out.block_scales.push(scale);
    }
    out
}

/// The most activations [`dot_nvfp4_row_quantized_many`] takes at once.
pub const NVFP4_MAX_BATCH: usize = 16;

/// `row . x` for one NVFP4 row and an NVFP4 activation: within a block, a sum of integer products.
pub fn dot_nvfp4_row_quantized(packed: &[u8], scales: &[u8], outer_scale: f32, x: &Nvfp4Activation) -> f32 {
    let mut out = [0.0f32];
    dot_nvfp4_row_quantized_many(packed, scales, outer_scale, &[x], &mut out);
    out[0]
}

/// `row . x` for one NVFP4 row and each of several activations, unpacking the row once. Every activation gets
/// exactly the arithmetic it would get alone, so results do not depend on how activations are batched.
pub fn dot_nvfp4_row_quantized_many(
    packed: &[u8],
    scales: &[u8],
    outer_scale: f32,
    xs: &[&Nvfp4Activation],
    out: &mut [f32],
) {
    assert!(xs.len() <= NVFP4_MAX_BATCH && xs.len() == out.len());
    assert_eq!(packed.len(), scales.len() * NVFP4_BLOCK / 2);
    for x in xs {
        assert_eq!(x.even.len(), packed.len());
        assert_eq!(x.odd.len(), packed.len());
        assert_eq!(x.block_scales.len(), scales.len());
    }
    #[cfg(target_arch = "aarch64")]
    if scales.len() % 4 == 0 {
        return neon::dot_nvfp4_row_quantized_many(packed, scales, outer_scale, xs, out);
    }
    for (x, out) in xs.iter().zip(out) {
        *out = portable::dot_nvfp4_row_quantized(packed, scales, outer_scale, x);
    }
}

/// Rounds an activation vector to MXFP8 and back, in place, the way the model's kernels do: each block of 32
/// gets the smallest power-of-two scale that brings its largest value within 448, then every value is rounded
/// to E4M3.
pub fn requantize_mxfp8(x: &mut [f32]) {
    assert_eq!(x.len() % MXFP8_BLOCK, 0);
    let e4m3 = &*E4M3;
    let min_scale = 2f32.powi(-127);
    for block in x.as_chunks_mut::<MXFP8_BLOCK>().0 {
        let raw = block.iter().fold(0.0f32, |m, v| m.max(v.abs())) * (1.0 / 448.0);
        // Round the scale up to a power of two by carrying any mantissa bits into the exponent field.
        let exponent = match raw <= min_scale {
            true => 0,
            false => (((raw.to_bits() + 0x007F_FFFF) & 0x7F80_0000) >> 23).min(254),
        };
        let scale = f32::from_bits(exponent << 23).max(min_scale);
        let inverse = if exponent == 0 { 2f32.powi(127) } else { 1.0 / scale };
        for v in block {
            *v = e4m3[encode_e4m3((*v * inverse).clamp(-448.0, 448.0)) as usize] * mxfp8_block_scale(exponent as u8);
        }
    }
}

/// Widens a row of little-endian bfloat16 values to f32.
pub fn decode_bf16_row(row: &[u8], out: &mut [f32]) {
    assert_eq!(row.len(), out.len() * 2);
    for (out, bytes) in out.iter_mut().zip(row.as_chunks::<2>().0) {
        *out = f32::from_bits((u16::from_le_bytes(*bytes) as u32) << 16);
    }
}

/// `row . x` for f32 values, summed in the same order as [`dot_bf16_row`]: a bfloat16 row gives the same result
/// bit for bit whether it is widened first and passed here or passed to `dot_bf16_row` directly.
pub fn dot_f32_row(row: &[f32], x: &[f32]) -> f32 {
    assert_eq!(row.len(), x.len());
    #[cfg(target_arch = "aarch64")]
    return neon::dot_f32_row(row, x);
    #[cfg(not(target_arch = "aarch64"))]
    portable::dot_f32_row(row, x)
}

/// `row . x` for each of several vectors. Four at a time share the loads of `row` and keep sixteen independent
/// sums going, which is several times faster than one vector at a time; each result is still bit for bit what
/// [`dot_f32_row`] gives for that vector alone.
pub fn dot_f32_row_many(row: &[f32], xs: &[&[f32]], out: &mut [f32]) {
    assert_eq!(xs.len(), out.len());
    assert!(xs.iter().all(|x| x.len() == row.len()));
    let mut done = 0;
    #[cfg(target_arch = "aarch64")]
    for (xs, out) in xs.as_chunks::<4>().0.iter().zip(out.as_chunks_mut::<4>().0) {
        *out = neon::dot_f32_row_x4(row, xs);
        done += 4;
    }
    for (x, out) in xs[done..].iter().zip(&mut out[done..]) {
        *out = dot_f32_row(row, x);
    }
}

/// `row . x` for a row of little-endian bfloat16 values, widened on the fly. A bfloat16 is the top half of an f32.
pub fn dot_bf16_row(row: &[u8], x: &[f32]) -> f32 {
    assert_eq!(row.len(), x.len() * 2);
    #[cfg(target_arch = "aarch64")]
    return neon::dot_bf16_row(row, x);
    #[cfg(not(target_arch = "aarch64"))]
    portable::dot_bf16_row(row, x)
}

/// Straightforward table-driven versions: the fallback off aarch64 and the reference the NEON code is tested against.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
mod portable {
    use super::*;

    pub fn decode_nvfp4_row(packed: &[u8], scales: &[u8], outer_scale: f32, out: &mut [f32]) {
        let e4m3 = &*E4M3;
        let blocks = out.chunks_exact_mut(NVFP4_BLOCK).zip(packed.chunks_exact(NVFP4_BLOCK / 2));
        for ((out, packed), &scale) in blocks.zip(scales) {
            let block_scale = e4m3[scale as usize];
            for (pair, &byte) in out.chunks_exact_mut(2).zip(packed) {
                pair[0] = E2M1[(byte & 15) as usize] * block_scale * outer_scale;
                pair[1] = E2M1[(byte >> 4) as usize] * block_scale * outer_scale;
            }
        }
    }

    pub fn dot_nvfp4_row(packed: &[u8], scales: &[u8], outer_scale: f32, x: &[f32]) -> f32 {
        let e4m3 = &*E4M3;
        let blocks = x.chunks_exact(NVFP4_BLOCK).zip(packed.chunks_exact(NVFP4_BLOCK / 2));
        let mut sum = 0.0f32;
        for ((x, packed), &scale) in blocks.zip(scales) {
            let mut block = 0.0f32;
            for (pair, &byte) in x.chunks_exact(2).zip(packed) {
                block += E2M1[(byte & 15) as usize] * pair[0] + E2M1[(byte >> 4) as usize] * pair[1];
            }
            sum += block * e4m3[scale as usize];
        }
        sum * outer_scale
    }

    pub fn dot_nvfp4_row_quantized(packed: &[u8], scales: &[u8], outer_scale: f32, x: &Nvfp4Activation) -> f32 {
        let e4m3 = &*E4M3;
        let blocks = packed.chunks_exact(NVFP4_BLOCK / 2).zip(x.even.chunks_exact(8).zip(x.odd.chunks_exact(8)));
        let mut sum = 0.0f32;
        for ((packed, (even, odd)), (&scale, x_scale)) in blocks.zip(scales.iter().zip(&x.block_scales)) {
            let mut block = 0i32;
            for (&byte, (&even, &odd)) in packed.iter().zip(even.iter().zip(odd)) {
                block += TWICE_E2M1[(byte & 15) as usize] as i32 * even as i32;
                block += TWICE_E2M1[(byte >> 4) as usize] as i32 * odd as i32;
            }
            sum += block as f32 * (e4m3[scale as usize] * x_scale);
        }
        sum * (0.25 * outer_scale * x.outer_scale)
    }

    pub fn dot_f32_row(row: &[f32], x: &[f32]) -> f32 {
        const LANES: usize = 16;
        let (row_chunks, row_rest) = row.as_chunks::<LANES>();
        let (x_chunks, x_rest) = x.as_chunks::<LANES>();
        let mut acc = [0.0f32; LANES];
        for (row, x) in row_chunks.iter().zip(x_chunks) {
            for i in 0..LANES {
                acc[i] += row[i] * x[i];
            }
        }
        acc.iter().sum::<f32>() + row_rest.iter().zip(x_rest).map(|(w, x)| w * x).sum::<f32>()
    }

    pub fn dot_bf16_row(row: &[u8], x: &[f32]) -> f32 {
        const LANES: usize = 16;
        let widen = |b: &[u8; 2]| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16);
        let (row_chunks, row_rest) = row.as_chunks::<2>().0.as_chunks::<LANES>();
        let (x_chunks, x_rest) = x.as_chunks::<LANES>();
        let mut acc = [0.0f32; LANES];
        for (row, x) in row_chunks.iter().zip(x_chunks) {
            for i in 0..LANES {
                acc[i] += widen(&row[i]) * x[i];
            }
        }
        acc.iter().sum::<f32>() + row_rest.iter().zip(x_rest).map(|(w, x)| widen(w) * x).sum::<f32>()
    }

    pub fn decode_mxfp8_row(values: &[u8], scales: &[u8], out: &mut [f32]) {
        let e4m3 = &*E4M3;
        let blocks = out.chunks_exact_mut(MXFP8_BLOCK).zip(values.chunks_exact(MXFP8_BLOCK));
        for ((out, values), &scale) in blocks.zip(scales) {
            let block_scale = mxfp8_block_scale(scale);
            for (out, &value) in out.iter_mut().zip(values) {
                *out = e4m3[value as usize] * block_scale;
            }
        }
    }

    pub fn dot_mxfp8_row(values: &[u8], scales: &[u8], x: &[f32]) -> f32 {
        let e4m3 = &*E4M3;
        let blocks = x.chunks_exact(MXFP8_BLOCK).zip(values.chunks_exact(MXFP8_BLOCK));
        let mut sum = 0.0f32;
        for ((x, values), &scale) in blocks.zip(scales) {
            let block: f32 = x.iter().zip(values).map(|(x, &value)| e4m3[value as usize] * x).sum();
            sum += block * mxfp8_block_scale(scale);
        }
        sum
    }
}

/// NEON kernels. Both formats are expanded 16 values at a time into four f32 vectors without touching memory:
/// NVFP4 through a 16-entry byte table, MXFP8 by assembling each float from its sign, exponent and mantissa.
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    use super::{E4M3, MXFP8_BLOCK, NVFP4_BLOCK, Nvfp4Activation, TWICE_E2M1, mxfp8_block_scale};

    /// Sixteen NVFP4 values (eight packed bytes) as f32, times two.
    ///
    /// # Safety
    /// `packed` must be readable for 8 bytes.
    #[inline(always)]
    unsafe fn nvfp4_block(packed: *const u8, table: int8x16_t) -> [float32x4_t; 4] {
        unsafe {
            let bytes = vld1_u8(packed);
            let low = vand_u8(bytes, vdup_n_u8(15));
            let high = vshr_n_u8::<4>(bytes);
            // Low nibble first: interleave back into stream order.
            let codes = vcombine_u8(vzip1_u8(low, high), vzip2_u8(low, high));
            let values = vqtbl1q_s8(table, codes);
            let (a, b) = (vmovl_s8(vget_low_s8(values)), vmovl_high_s8(values));
            [
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(a))),
                vcvtq_f32_s32(vmovl_high_s16(a)),
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(b))),
                vcvtq_f32_s32(vmovl_high_s16(b)),
            ]
        }
    }

    /// Sixteen E4M3 bytes as f32: `sign * mantissa * 2^(max(exponent, 1) - 10)`, where the mantissa gains its
    /// implicit leading bit (8) unless the exponent is zero.
    ///
    /// # Safety
    /// `values` must be readable for 16 bytes.
    #[inline(always)]
    unsafe fn mxfp8_sixteen(values: *const u8) -> [float32x4_t; 4] {
        unsafe {
            let bytes = vld1q_u8(values);
            let exponent = vandq_u8(vshrq_n_u8::<3>(bytes), vdupq_n_u8(15));
            let normal = vtstq_u8(exponent, exponent);
            let mantissa = vorrq_u8(vandq_u8(bytes, vdupq_n_u8(7)), vandq_u8(normal, vdupq_n_u8(8)));
            let exponent = vmaxq_u8(exponent, vdupq_n_u8(1));
            let sign = vandq_u8(bytes, vdupq_n_u8(0x80));

            let quarters = |v: uint8x16_t| -> [uint32x4_t; 4] {
                let (a, b) = (vmovl_u8(vget_low_u8(v)), vmovl_high_u8(v));
                [vmovl_u16(vget_low_u16(a)), vmovl_high_u16(a), vmovl_u16(vget_low_u16(b)), vmovl_high_u16(b)]
            };
            let (mantissa, exponent, sign) = (quarters(mantissa), quarters(exponent), quarters(sign));
            let mut out = [vdupq_n_f32(0.0); 4];
            for i in 0..4 {
                // 2^(exponent - 10) as raw float bits: biased exponent `exponent - 10 + 127`.
                let power = vreinterpretq_f32_u32(vshlq_n_u32::<23>(vaddq_u32(exponent[i], vdupq_n_u32(117))));
                let magnitude = vmulq_f32(vcvtq_f32_u32(mantissa[i]), power);
                let sign_bit = vshlq_n_u32::<24>(sign[i]);
                out[i] = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(magnitude), sign_bit));
            }
            out
        }
    }

    /// `sum(values[i] * x[i])` over sixteen values, as four partial sums.
    ///
    /// # Safety
    /// `x` must be readable for 16 floats.
    #[inline(always)]
    unsafe fn dot_sixteen(values: [float32x4_t; 4], x: *const f32) -> float32x4_t {
        unsafe {
            let sum = vmulq_f32(values[0], vld1q_f32(x));
            let sum = vfmaq_f32(sum, values[1], vld1q_f32(x.add(4)));
            let sum = vfmaq_f32(sum, values[2], vld1q_f32(x.add(8)));
            vfmaq_f32(sum, values[3], vld1q_f32(x.add(12)))
        }
    }

    // The public wrappers in the parent module check that the slice lengths agree before calling these.

    pub fn decode_nvfp4_row(packed: &[u8], scales: &[u8], outer_scale: f32, out: &mut [f32]) {
        let e4m3 = &*E4M3;
        let blocks = out.chunks_exact_mut(NVFP4_BLOCK).zip(packed.chunks_exact(NVFP4_BLOCK / 2));
        // SAFETY: every chunk is exactly 8 packed bytes and 16 floats.
        unsafe {
            let table = vld1q_s8(TWICE_E2M1.as_ptr());
            for ((out, packed), &scale) in blocks.zip(scales) {
                // Halving is exact, so this is `value * block_scale * outer_scale` bit for bit.
                let block_scale = vdupq_n_f32(e4m3[scale as usize] * 0.5);
                let outer = vdupq_n_f32(outer_scale);
                for (i, values) in nvfp4_block(packed.as_ptr(), table).into_iter().enumerate() {
                    vst1q_f32(out.as_mut_ptr().add(4 * i), vmulq_f32(vmulq_f32(values, block_scale), outer));
                }
            }
        }
    }

    pub fn dot_nvfp4_row(packed: &[u8], scales: &[u8], outer_scale: f32, x: &[f32]) -> f32 {
        let e4m3 = &*E4M3;
        let blocks = x.chunks_exact(NVFP4_BLOCK).zip(packed.chunks_exact(NVFP4_BLOCK / 2));
        // SAFETY: every chunk is exactly 8 packed bytes and 16 floats.
        unsafe {
            let table = vld1q_s8(TWICE_E2M1.as_ptr());
            let mut sum = vdupq_n_f32(0.0);
            for ((x, packed), &scale) in blocks.zip(scales) {
                let block = dot_sixteen(nvfp4_block(packed.as_ptr(), table), x.as_ptr());
                sum = vfmaq_n_f32(sum, block, e4m3[scale as usize]);
            }
            vaddvq_f32(sum) * (0.5 * outer_scale)
        }
    }

    /// `dot_f32_row` for four vectors at once: the same sums per vector, interleaved.
    pub fn dot_f32_row_x4(row: &[f32], xs: &[&[f32]; 4]) -> [f32; 4] {
        let (row_chunks, row_rest) = row.as_chunks::<16>();
        let tail = row.len() - row_rest.len();
        let mut out = [0.0f32; 4];
        // SAFETY: the caller checked each vector is as long as `row`, so offset `16 * i + 4 * quarter` with
        // `i < row_chunks.len()` reads four floats in bounds.
        unsafe {
            let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
            for (i, row) in row_chunks.iter().enumerate() {
                for quarter in 0..4 {
                    let w = vld1q_f32(row.as_ptr().add(4 * quarter));
                    for (acc, x) in acc.iter_mut().zip(xs) {
                        let x = vld1q_f32(x.as_ptr().add(16 * i + 4 * quarter));
                        acc[quarter] = vfmaq_f32(acc[quarter], w, x);
                    }
                }
            }
            for ((out, acc), x) in out.iter_mut().zip(acc).zip(xs) {
                let sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3])));
                *out = sum + row_rest.iter().zip(&x[tail..]).map(|(w, x)| w * x).sum::<f32>();
            }
        }
        out
    }

    /// Mirrors `dot_bf16_row` below operation for operation, minus the widening.
    pub fn dot_f32_row(row: &[f32], x: &[f32]) -> f32 {
        let (x_chunks, x_rest) = x.as_chunks::<16>();
        let (row_chunks, row_rest) = row.as_chunks::<16>();
        // SAFETY: every chunk is exactly 16 floats.
        let sum = unsafe {
            let mut acc = [vdupq_n_f32(0.0); 4];
            for (row, x) in row_chunks.iter().zip(x_chunks) {
                for quarter in 0..4 {
                    let (w, x) = (vld1q_f32(row.as_ptr().add(4 * quarter)), vld1q_f32(x.as_ptr().add(4 * quarter)));
                    acc[quarter] = vfmaq_f32(acc[quarter], w, x);
                }
            }
            vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3])))
        };
        sum + row_rest.iter().zip(x_rest).map(|(w, x)| w * x).sum::<f32>()
    }

    pub fn dot_bf16_row(row: &[u8], x: &[f32]) -> f32 {
        let (x_chunks, x_rest) = x.as_chunks::<16>();
        let (row_chunks, row_rest) = row.as_chunks::<32>();
        // SAFETY: every chunk is exactly 32 bytes and 16 floats; the loads have no alignment requirement.
        let sum = unsafe {
            let mut acc = [vdupq_n_f32(0.0); 4];
            for (row, x) in row_chunks.iter().zip(x_chunks) {
                for half in 0..2 {
                    let bits = vreinterpretq_u16_u8(vld1q_u8(row.as_ptr().add(16 * half)));
                    let low = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bits)));
                    let high = vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bits));
                    acc[2 * half] = vfmaq_f32(acc[2 * half], low, vld1q_f32(x.as_ptr().add(8 * half)));
                    acc[2 * half + 1] = vfmaq_f32(acc[2 * half + 1], high, vld1q_f32(x.as_ptr().add(8 * half + 4)));
                }
            }
            vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3])))
        };
        let widen = |b: &[u8; 2]| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16);
        sum + row_rest.as_chunks::<2>().0.iter().zip(x_rest).map(|(w, x)| widen(w) * x).sum::<f32>()
    }

    /// Splits the activations into groups of 16, 8, 4, 2 and 1 so each group runs with a fixed-size, fully
    /// unrolled inner loop. Grouping does not change any result.
    pub fn dot_nvfp4_row_quantized_many(
        packed: &[u8],
        scales: &[u8],
        outer_scale: f32,
        mut xs: &[&Nvfp4Activation],
        mut out: &mut [f32],
    ) {
        fn take<'a, 'b, const N: usize>(
            packed: &[u8],
            scales: &[u8],
            outer_scale: f32,
            xs: &mut &'a [&'b Nvfp4Activation],
            out: &mut &mut [f32],
        ) {
            while let Some((group, rest)) = xs.split_first_chunk::<N>() {
                let (head, tail) = std::mem::take(out).split_at_mut(N);
                head.copy_from_slice(&dot_group(packed, scales, outer_scale, group));
                (*xs, *out) = (rest, tail);
            }
        }
        take::<16>(packed, scales, outer_scale, &mut xs, &mut out);
        take::<8>(packed, scales, outer_scale, &mut xs, &mut out);
        take::<4>(packed, scales, outer_scale, &mut xs, &mut out);
        take::<2>(packed, scales, outer_scale, &mut xs, &mut out);
        take::<1>(packed, scales, outer_scale, &mut xs, &mut out);
    }

    /// Four blocks (64 weights) per step. Sixteen packed bytes cover two blocks; their low nibbles are the
    /// even-position weights and their high nibbles the odd-position ones, matching the activation's two arrays,
    /// so no interleaving is needed. Each signed dot-product instruction sums sixteen byte products into four
    /// lanes, two per block. The row is unpacked once per step and reused for every activation.
    #[inline(always)]
    fn dot_group<const N: usize>(
        packed: &[u8],
        scales: &[u8],
        outer_scale: f32,
        xs: &[&Nvfp4Activation; N],
    ) -> [f32; N] {
        let e4m3 = &*E4M3;
        let groups = packed.as_chunks::<32>().0.iter().zip(scales.as_chunks::<4>().0);
        // SAFETY: the caller checked that each activation has as many even and odd values as the row has packed
        // bytes and one scale per block, and that the block count is a multiple of four. So group `g` reads 32
        // bytes and 4 scales at offsets 32g and 4g, all in bounds.
        unsafe {
            let table = vld1q_s8(TWICE_E2M1.as_ptr());
            let nibble = vdupq_n_u8(15);
            let mut sums = [vdupq_n_f32(0.0); N];
            for (g, (packed, scales)) in groups.enumerate() {
                let (first, second) = (vld1q_u8(packed.as_ptr()), vld1q_u8(packed.as_ptr().add(16)));
                let low = [vqtbl1q_s8(table, vandq_u8(first, nibble)), vqtbl1q_s8(table, vandq_u8(second, nibble))];
                let high = [vqtbl1q_s8(table, vshrq_n_u8::<4>(first)), vqtbl1q_s8(table, vshrq_n_u8::<4>(second))];
                let weight_scales = scales.map(|s| e4m3[s as usize]);
                let weight_scales = vld1q_f32(weight_scales.as_ptr());
                for (x, sum) in xs.iter().zip(&mut sums) {
                    let (even, odd) = (x.even.as_ptr().add(32 * g), x.odd.as_ptr().add(32 * g));
                    let pair = |half: usize| {
                        let acc = vdotq_s32(vdupq_n_s32(0), low[half], vld1q_s8(even.add(16 * half)));
                        vdotq_s32(acc, high[half], vld1q_s8(odd.add(16 * half)))
                    };
                    // Each `pair` holds two half-sums per block; pairwise addition leaves one sum per block.
                    let block_sums = vcvtq_f32_s32(vpaddq_s32(pair(0), pair(1)));
                    let block_scales = vmulq_f32(weight_scales, vld1q_f32(x.block_scales.as_ptr().add(4 * g)));
                    *sum = vfmaq_f32(*sum, block_sums, block_scales);
                }
            }
            std::array::from_fn(|i| vaddvq_f32(sums[i]) * (0.25 * outer_scale * xs[i].outer_scale))
        }
    }

    pub fn decode_mxfp8_row(values: &[u8], scales: &[u8], out: &mut [f32]) {
        let blocks = out.chunks_exact_mut(MXFP8_BLOCK).zip(values.chunks_exact(MXFP8_BLOCK));
        // SAFETY: every chunk is exactly 32 bytes and 32 floats, read and written 16 at a time.
        unsafe {
            for ((out, values), &scale) in blocks.zip(scales) {
                let block_scale = vdupq_n_f32(mxfp8_block_scale(scale));
                for half in 0..2 {
                    let decoded = mxfp8_sixteen(values.as_ptr().add(16 * half));
                    for (i, decoded) in decoded.into_iter().enumerate() {
                        vst1q_f32(out.as_mut_ptr().add(16 * half + 4 * i), vmulq_f32(decoded, block_scale));
                    }
                }
            }
        }
    }

    pub fn dot_mxfp8_row(values: &[u8], scales: &[u8], x: &[f32]) -> f32 {
        let blocks = x.chunks_exact(MXFP8_BLOCK).zip(values.chunks_exact(MXFP8_BLOCK));
        // SAFETY: every chunk is exactly 32 bytes and 32 floats, read 16 at a time.
        unsafe {
            let mut sum = vdupq_n_f32(0.0);
            for ((x, values), &scale) in blocks.zip(scales) {
                let first = dot_sixteen(mxfp8_sixteen(values.as_ptr()), x.as_ptr());
                let second = dot_sixteen(mxfp8_sixteen(values.as_ptr().add(16)), x.as_ptr().add(16));
                sum = vfmaq_n_f32(sum, vaddq_f32(first, second), mxfp8_block_scale(scale));
            }
            vaddvq_f32(sum)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic filler bytes (a small LCG), so the tests need no extra dependency.
    fn bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 56) as u8
            })
            .collect()
    }

    fn floats(len: usize, seed: u64) -> Vec<f32> {
        bytes(len, seed).iter().map(|b| (*b as f32 - 127.5) / 64.0).collect()
    }

    /// E4M3 bytes without the NaN codes, as a quantiser would emit.
    fn e4m3_bytes(len: usize, seed: u64) -> Vec<u8> {
        bytes(len, seed).into_iter().map(|b| if b & 0x7F == 0x7F { b & 0xF0 } else { b }).collect()
    }

    #[test]
    fn e4m3_known_values() {
        let t = &*E4M3;
        assert_eq!(t[0x00], 0.0);
        assert_eq!(t[0x01], 2f32.powi(-9)); // smallest subnormal
        assert_eq!(t[0x08], 2f32.powi(-6)); // smallest normal
        assert_eq!(t[0x38], 1.0);
        assert_eq!(t[0x39], 1.125);
        assert_eq!(t[0x7E], 448.0); // largest finite
        assert_eq!(t[0xB8], -1.0);
        assert!(t[0x7F].is_nan() && t[0xFF].is_nan());
        assert_eq!(t.iter().filter(|v| v.is_nan()).count(), 2);
    }

    #[test]
    fn nvfp4_low_nibble_first_and_scales() {
        // One block of 16 values: codes 0..=15 in order, so byte i packs code 2i (low) and 2i + 1 (high).
        let packed: Vec<u8> = (0..8u8).map(|i| (2 * i) | ((2 * i + 1) << 4)).collect();
        let mut out = [0.0f32; 16];
        decode_nvfp4_row(&packed, &[0x40], 0.5, &mut out); // 0x40 = 2.0
        for (code, value) in out.iter().enumerate() {
            assert_eq!(*value, E2M1[code] * 2.0 * 0.5);
        }
    }

    #[test]
    fn mxfp8_block_scale_is_a_power_of_two() {
        for scale in 0..=254u8 {
            assert_eq!(mxfp8_block_scale(scale), 2f32.powi(scale as i32 - 127), "scale {scale}");
        }
    }

    #[test]
    fn mxfp8_block_exponent() {
        let values = [0x38u8; 64]; // all 1.0
        let mut out = [0.0f32; 64];
        decode_mxfp8_row(&values, &[127, 124], &mut out);
        assert!(out[..32].iter().all(|v| *v == 1.0));
        assert!(out[32..].iter().all(|v| *v == 0.125));
    }

    #[test]
    fn mxfp8_decodes_every_finite_code() {
        // All 256 codes, NaNs replaced, against the table.
        let values: Vec<u8> = (0..=255u8).map(|b| if b & 0x7F == 0x7F { 0 } else { b }).collect();
        let mut out = [0.0f32; 256];
        decode_mxfp8_row(&values, &[127; 8], &mut out);
        for (value, decoded) in values.iter().zip(out) {
            assert_eq!(decoded, E4M3[*value as usize], "code {value:#04x}");
        }
    }

    #[test]
    fn decoders_match_the_portable_reference_exactly() {
        let cols = 2304;
        let (packed, scales) = (bytes(cols / 2, 1), e4m3_bytes(cols / NVFP4_BLOCK, 2));
        let (mut fast, mut reference) = (vec![0.0f32; cols], vec![0.0f32; cols]);
        decode_nvfp4_row(&packed, &scales, 0.0123, &mut fast);
        portable::decode_nvfp4_row(&packed, &scales, 0.0123, &mut reference);
        assert_eq!(fast, reference);

        // Block exponents around 127 keep the products finite.
        let values = e4m3_bytes(cols, 3);
        let scales: Vec<u8> = bytes(cols / MXFP8_BLOCK, 4).iter().map(|b| 110 + b % 24).collect();
        decode_mxfp8_row(&values, &scales, &mut fast);
        portable::decode_mxfp8_row(&values, &scales, &mut reference);
        assert_eq!(fast, reference);
    }

    #[test]
    fn e4m3_encoding_rounds_to_nearest_even_and_saturates() {
        let t = &*E4M3;
        for code in (0..=0x7Eu8).chain(0x80..=0xFE) {
            assert_eq!(encode_e4m3(t[code as usize]), code, "code {code:#04x}");
        }
        for code in 0..0x7Eu8 {
            let (low, high) = (t[code as usize], t[code as usize + 1]);
            let even = if code % 2 == 0 { code } else { code + 1 };
            assert_eq!(encode_e4m3((low + high) / 2.0), even, "midpoint above {code:#04x}");
            assert_eq!(encode_e4m3(low + (high - low) * 0.25), code);
            assert_eq!(encode_e4m3(low + (high - low) * 0.75), code + 1);
        }
        assert_eq!(encode_e4m3(1e9), 0x7E);
        assert_eq!(encode_e4m3(-1e9), 0xFE);
    }

    #[test]
    fn e2m1_encoding_rounds_ties_to_even() {
        // Midpoints between neighbouring grid values go to the even code.
        for (value, code) in [(0.25, 0), (0.75, 2), (1.25, 2), (1.75, 4), (2.5, 4), (3.5, 6), (5.0, 6), (100.0, 7)] {
            assert_eq!(encode_e2m1(value), code, "{value}");
            assert_eq!(encode_e2m1(-value), code | 8, "-{value}");
        }
        for code in 0..8u8 {
            assert_eq!(encode_e2m1(E2M1[code as usize]), code);
        }
    }

    #[test]
    fn nvfp4_activation_is_exact_on_grid_values() {
        // With the largest value at 6 the scales are powers of two and every grid value survives unchanged.
        let x: Vec<f32> = (0..32).map(|i| E2M1[(i * 5 + 7) % 16]).chain(E2M1).collect();
        assert_eq!(quantize_nvfp4(&x).dequantize(), x);
        // Scaling the whole vector by a power of two only changes the outer scale.
        let scaled: Vec<f32> = x.iter().map(|v| v * 0.03125).collect();
        assert_eq!(quantize_nvfp4(&scaled).dequantize(), scaled);
        // An all-zero vector stays zero.
        assert_eq!(quantize_nvfp4(&[0.0; 16]).dequantize(), [0.0; 16]);
    }

    #[test]
    fn nvfp4_activation_error_is_bounded() {
        let x = floats(2304, 12);
        let back = quantize_nvfp4(&x).dequantize();
        for (block, back) in x.chunks_exact(16).zip(back.chunks_exact(16)) {
            let block_max = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            // The coarsest grid step is 2 out of a full scale of 6, so no value is off by more than a sixth of
            // the block maximum, plus a little for the 8-bit block scale.
            for (v, b) in block.iter().zip(back) {
                assert!((v - b).abs() <= block_max * 0.19, "{v} became {b} (block max {block_max})");
            }
        }
    }

    #[test]
    fn nvfp4_activation_picks_the_better_block_scale() {
        // Mapping the maximum to 4 puts every value on the grid (4, 2, 1, 0.5); mapping it to 6 does not.
        let block: Vec<f32> = [4.0, 2.0, 1.0, 0.5].repeat(4);
        let x: Vec<f32> = block.iter().copied().chain([6.0; 16]).collect();
        assert_eq!(quantize_nvfp4(&x).dequantize(), x);
    }

    #[test]
    fn quantized_dot_matches_dot_with_the_dequantized_activation() {
        let cols = 2304;
        let activation = quantize_nvfp4(&floats(cols, 13));
        let x = activation.dequantize();
        let (packed, scales) = (bytes(cols / 2, 14), e4m3_bytes(cols / NVFP4_BLOCK, 15));
        let mut row = vec![0.0f32; cols];
        portable::decode_nvfp4_row(&packed, &scales, 0.0123, &mut row);
        let exact: f64 = row.iter().zip(&x).map(|(w, x)| (*w as f64) * (*x as f64)).sum();
        let scale: f64 = row.iter().zip(&x).map(|(w, x)| ((*w as f64) * (*x as f64)).abs()).sum();

        let fast = dot_nvfp4_row_quantized(&packed, &scales, 0.0123, &activation) as f64;
        let reference = portable::dot_nvfp4_row_quantized(&packed, &scales, 0.0123, &activation) as f64;
        assert!((fast - exact).abs() <= 1e-5 * scale, "{fast} vs {exact}");
        assert!((reference - exact).abs() <= 1e-5 * scale, "{reference} vs {exact}");
    }

    /// Single-thread throughput of the kernels generation leans on:
    /// `cargo test --release --lib quant::tests::bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench() {
        use std::time::Instant;
        let (rows, cols) = (2304, 2304);
        let packed = bytes(rows * cols / 2, 70);
        let scales = e4m3_bytes(rows * cols / NVFP4_BLOCK, 71);
        let activation = quantize_nvfp4(&floats(cols, 72));
        let run = || -> f32 {
            (0..rows)
                .map(|r| {
                    dot_nvfp4_row_quantized(
                        &packed[r * cols / 2..][..cols / 2],
                        &scales[r * cols / 16..][..cols / 16],
                        0.0123,
                        &activation,
                    )
                })
                .sum()
        };
        let mut best = f64::INFINITY;
        let mut check = 0.0;
        for _ in 0..20 {
            let started = Instant::now();
            check = run();
            best = best.min(started.elapsed().as_secs_f64());
        }
        eprintln!("nvfp4 x nvfp4, one activation: {:.1} G products/s on one thread (checksum {check})", (rows * cols) as f64 / best / 1e9);

        let weights = bytes(rows * cols * 2, 73);
        let x = floats(cols, 74);
        let mut best = f64::INFINITY;
        for _ in 0..20 {
            let started = Instant::now();
            let sum: f32 = (0..rows).map(|r| dot_bf16_row(&weights[r * cols * 2..][..cols * 2], &x)).sum();
            std::hint::black_box(sum);
            best = best.min(started.elapsed().as_secs_f64());
        }
        eprintln!("bf16 x f32: {:.1} G products/s on one thread", (rows * cols) as f64 / best / 1e9);
    }

    #[test]
    fn batched_quantized_dot_equals_one_at_a_time_exactly() {
        let cols = 2304;
        let (packed, scales) = (bytes(cols / 2, 20), e4m3_bytes(cols / NVFP4_BLOCK, 21));
        let activations: Vec<Nvfp4Activation> =
            (0..NVFP4_MAX_BATCH as u64).map(|seed| quantize_nvfp4(&floats(cols, 30 + seed))).collect();
        let single: Vec<f32> =
            activations.iter().map(|x| dot_nvfp4_row_quantized(&packed, &scales, 0.0123, x)).collect();
        for batch in [1, 3, NVFP4_MAX_BATCH] {
            let xs: Vec<&Nvfp4Activation> = activations[..batch].iter().collect();
            let mut out = vec![0.0f32; batch];
            dot_nvfp4_row_quantized_many(&packed, &scales, 0.0123, &xs, &mut out);
            assert_eq!(out, single[..batch], "batch of {batch}");
        }
    }

    #[test]
    fn f32_dot_of_a_widened_row_equals_the_bf16_dot_exactly() {
        for len in [2304, 3072, 37] {
            // Small bfloat16 values: sign, a modest exponent, random mantissa.
            let row: Vec<u8> = bytes(len * 2, 40).as_chunks::<2>().0.iter().flat_map(|b| [b[0], 0x3C | (b[1] & 0x83)]).collect();
            let x = floats(len, 41);
            let mut widened = vec![0.0f32; len];
            decode_bf16_row(&row, &mut widened);
            assert!(widened.iter().all(|w| w.is_finite()));
            assert_eq!(dot_f32_row(&widened, &x), dot_bf16_row(&row, &x), "len {len}");
            assert_eq!(portable::dot_f32_row(&widened, &x), portable::dot_bf16_row(&row, &x), "len {len}");
        }
    }

    #[test]
    fn many_f32_dots_equal_one_at_a_time_exactly() {
        for len in [2304, 37] {
            let row = floats(len, 50);
            let vectors: Vec<Vec<f32>> = (0..7).map(|seed| floats(len, 60 + seed)).collect();
            let xs: Vec<&[f32]> = vectors.iter().map(|x| x.as_slice()).collect();
            let single: Vec<f32> = xs.iter().map(|x| dot_f32_row(&row, x)).collect();
            let mut many = vec![0.0f32; xs.len()];
            dot_f32_row_many(&row, &xs, &mut many);
            assert_eq!(many, single, "len {len}");
        }
    }

    #[test]
    fn mxfp8_activation_rounds_within_a_sixteenth_and_is_idempotent() {
        let x: Vec<f32> = floats(2304, 16).iter().map(|v| v * 37.0).collect();
        let mut once = x.clone();
        requantize_mxfp8(&mut once);
        for (v, q) in x.iter().zip(&once) {
            // E4M3 has three mantissa bits: at most one part in sixteen off, except near the bottom of the range.
            assert!((v - q).abs() <= v.abs() / 16.0 + 1e-3, "{v} became {q}");
        }
        let mut twice = once.clone();
        requantize_mxfp8(&mut twice);
        assert_eq!(once, twice);
        // A block whose maximum is exactly 448 * 2^k needs no rounding up of its scale.
        let mut block = [0.0f32; 32];
        block[3] = 448.0 * 4.0;
        block[9] = -3.0 * 4.0;
        let expected = block;
        requantize_mxfp8(&mut block);
        assert_eq!(block, expected);
    }

    #[test]
    fn bf16_dot_matches_widening_each_value() {
        // 2304 is a multiple of 16; 37 exercises the tail.
        for len in [2304, 37] {
            let (row, x) = (bytes(len * 2, 10), floats(len, 11));
            let widened: Vec<f32> = row.as_chunks::<2>().0.iter().map(|b| half::bf16::from_le_bytes(*b).to_f32()).collect();
            let finite: Vec<u8> = row
                .as_chunks::<2>()
                .0
                .iter()
                .zip(&widened)
                .flat_map(|(b, w)| if w.is_finite() && w.abs() < 1e6 { *b } else { [0, 0] })
                .collect();
            let widened: Vec<f32> = widened.iter().map(|w| if w.is_finite() && w.abs() < 1e6 { *w } else { 0.0 }).collect();
            let exact: f64 = widened.iter().zip(&x).map(|(w, x)| (*w as f64) * (*x as f64)).sum();
            let scale: f64 = widened.iter().zip(&x).map(|(w, x)| ((*w as f64) * (*x as f64)).abs()).sum();
            let fast = dot_bf16_row(&finite, &x) as f64;
            assert!((fast - exact).abs() <= 1e-5 * scale, "len {len}: {fast} vs {exact}");
            assert!((portable::dot_bf16_row(&finite, &x) as f64 - exact).abs() <= 1e-5 * scale);
        }
    }

    #[test]
    fn dots_match_decode_then_dot() {
        let cols = 2304;
        let x = floats(cols, 5);
        let dot = |row: &[f32]| row.iter().zip(&x).map(|(w, x)| (*w as f64) * (*x as f64)).sum::<f64>();
        let scale_of = |row: &[f32]| row.iter().zip(&x).map(|(w, x)| ((*w as f64) * (*x as f64)).abs()).sum::<f64>();
        let mut row = vec![0.0f32; cols];

        let (packed, scales) = (bytes(cols / 2, 6), e4m3_bytes(cols / NVFP4_BLOCK, 7));
        portable::decode_nvfp4_row(&packed, &scales, 0.0123, &mut row);
        let fast = dot_nvfp4_row(&packed, &scales, 0.0123, &x) as f64;
        assert!((fast - dot(&row)).abs() <= 1e-5 * scale_of(&row), "nvfp4: {fast} vs {}", dot(&row));

        let values = e4m3_bytes(cols, 8);
        let scales: Vec<u8> = bytes(cols / MXFP8_BLOCK, 9).iter().map(|b| 110 + b % 24).collect();
        portable::decode_mxfp8_row(&values, &scales, &mut row);
        let fast = dot_mxfp8_row(&values, &scales, &x) as f64;
        assert!((fast - dot(&row)).abs() <= 1e-5 * scale_of(&row), "mxfp8: {fast} vs {}", dot(&row));
    }
}
