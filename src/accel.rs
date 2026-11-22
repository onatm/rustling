//! Vector math for the hot loops that are plain float arithmetic: elementwise `exp` and `tanh`. On macOS these
//! are Apple's Accelerate routines, several times faster than scalar code; elsewhere, straightforward loops. The two agree only to the last bit or so, so results differ slightly between
//! platforms, but never between two runs or two ways of batching on the same one.

/// `x[i] = exp(x[i])`.
pub(crate) fn exp_in_place(x: &mut [f32]) {
    #[cfg(target_os = "macos")]
    // SAFETY: reads and writes `x.len()` floats of `x`; in-place use is allowed.
    unsafe {
        ffi::vvexpf(x.as_mut_ptr(), x.as_ptr(), &(x.len() as i32));
    }
    #[cfg(not(target_os = "macos"))]
    x.iter_mut().for_each(|v| *v = v.exp());
}

/// `x[i] = tanh(x[i])`.
pub(crate) fn tanh_in_place(x: &mut [f32]) {
    #[cfg(target_os = "macos")]
    // SAFETY: reads and writes `x.len()` floats of `x`; in-place use is allowed.
    unsafe {
        ffi::vvtanhf(x.as_mut_ptr(), x.as_ptr(), &(x.len() as i32));
    }
    #[cfg(not(target_os = "macos"))]
    x.iter_mut().for_each(|v| *v = v.tanh());
}

/// A matrix laid out row by row inside a longer slice: row `r` is `values[r * stride..][..cols]`. A stride wider
/// than the row picks a band of columns out of a larger table without copying it.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(crate) struct Rows<'a> {
    pub values: &'a [f32],
    pub rows: usize,
    pub cols: usize,
    pub stride: usize,
}

/// `out = scale * a . b` (or `a . b^T` with `transposed`), by Apple's tuned matrix product. `out` is row by row,
/// `a.rows` rows. The sums come out in that routine's order, so results match a plain loop only to rounding.
#[cfg(target_os = "macos")]
pub(crate) fn multiply(a: Rows, b: Rows, transposed: bool, scale: f32, out: &mut [f32]) {
    let (inner, width) = if transposed { (b.cols, b.rows) } else { (b.rows, b.cols) };
    let holds = |m: Rows| m.rows == 0 || m.values.len() >= (m.rows - 1) * m.stride + m.cols;
    assert!(a.cols == inner && out.len() == a.rows * width && holds(a) && holds(b) && a.stride >= a.cols && b.stride >= b.cols);
    if a.rows == 0 || width == 0 || inner == 0 {
        return out.fill(0.0);
    }
    const ROW_MAJOR: i32 = 101;
    let transpose = if transposed { 112 } else { 111 };
    // SAFETY: the assertion above bounds every element the routine reads through the strides, and `out` holds
    // `a.rows * width` floats, all of which it writes.
    unsafe {
        ffi::cblas_sgemm(
            ROW_MAJOR, 111, transpose, a.rows as i32, width as i32, inner as i32, scale,
            a.values.as_ptr(), a.stride as i32, b.values.as_ptr(), b.stride as i32, 0.0,
            out.as_mut_ptr(), width as i32,
        );
    }
}

#[cfg(target_os = "macos")]
mod ffi {
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
        pub fn vvtanhf(y: *mut f32, x: *const f32, n: *const i32);
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_sgemm(
            order: i32, trans_a: i32, trans_b: i32, m: i32, n: i32, k: i32, alpha: f32,
            a: *const f32, lda: i32, b: *const f32, ldb: i32, beta: f32, c: *mut f32, ldc: i32,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn the_tuned_product_matches_a_plain_one() {
        // a is 3 x 4; b is 5 x 4 taken out of a table 6 wide, two columns in.
        let a: Vec<f32> = (0..12).map(|i| (i as f32 * 0.37).sin()).collect();
        let table: Vec<f32> = (0..30).map(|i| (i as f32 * 0.91).cos()).collect();
        let b = Rows { values: &table[2..], rows: 5, cols: 4, stride: 6 };
        let mut out = vec![0.0f32; 15];
        multiply(Rows { values: &a, rows: 3, cols: 4, stride: 4 }, b, true, 0.5, &mut out);
        for (r, c) in (0..3).flat_map(|r| (0..5).map(move |c| (r, c))) {
            let expected: f32 = (0..4).map(|j| a[r * 4 + j] * table[2 + c * 6 + j]).sum::<f32>() * 0.5;
            assert!((out[r * 5 + c] - expected).abs() < 1e-5);
        }
        // Not transposed: a is 3 x 4, b is 4 x 3 out of the same table.
        let b = Rows { values: &table[2..], rows: 4, cols: 3, stride: 6 };
        let mut out = vec![0.0f32; 9];
        multiply(Rows { values: &a, rows: 3, cols: 4, stride: 4 }, b, false, 1.0, &mut out);
        for (r, c) in (0..3).flat_map(|r| (0..3).map(move |c| (r, c))) {
            let expected: f32 = (0..4).map(|j| a[r * 4 + j] * table[2 + j * 6 + c]).sum();
            assert!((out[r * 3 + c] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn elementwise_functions_match_the_scalar_ones() {
        let x: Vec<f32> = (0..97).map(|i| (i as f32 - 48.0) / 7.0).collect();
        let (mut exp, mut tanh) = (x.clone(), x.clone());
        exp_in_place(&mut exp);
        tanh_in_place(&mut tanh);
        for ((x, e), t) in x.iter().zip(exp).zip(tanh) {
            assert!((e - x.exp()).abs() <= 1e-6 * x.exp(), "exp({x})");
            assert!((t - x.tanh()).abs() <= 1e-6, "tanh({x})");
        }
    }
}
