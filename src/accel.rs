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

#[cfg(target_os = "macos")]
mod ffi {
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
        pub fn vvtanhf(y: *mut f32, x: *const f32, n: *const i32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
