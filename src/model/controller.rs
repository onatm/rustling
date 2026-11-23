//! The controllers of the looped layers, which merge a pass's state with messages from the pass before.

use super::*;

impl Controller<'_> {
    /// `(1 - forget) * hidden + gain * message`, with both gates read from the normed pair through a low rank.
    pub(super) fn merge(&self, gpu: Option<&Gpu>, hidden: &Seq, message: &Seq, h: usize, eps: f32, bf16: bool) -> Seq {
        // A long batch on the fast path: the three products on the GPU, which is several times quicker at them
        // than every core together.
        #[cfg(target_os = "macos")]
        if let Some(gpu) = gpu.filter(|gpu| gpu.fast() && hidden.len() / h >= FAST_MIN_TOKENS && !bf16)
            && [&self.down, &self.forget, &self.gain].iter().all(|matrix| matrix.format == Format::F32)
        {
            let mut pairs = vec![0.0f32; 2 * hidden.len()];
            pairs.par_chunks_exact_mut(2 * h).zip(hidden.par_chunks_exact(h)).zip(message.par_chunks_exact(h)).for_each(
                |((pair, hidden), message)| {
                    let (a, b) = pair.split_at_mut(h);
                    a.copy_from_slice(hidden);
                    b.copy_from_slice(message);
                    rms_norm(a, None, eps);
                    rms_norm(b, None, eps);
                },
            );
            match gpu.gates_fast(&pairs, &self.down.bytes, &self.forget.bytes, &self.gain.bytes, (self.down.rows, h)) {
                Ok((forget, gain)) if !forget.par_iter().chain(&gain).all(|value| value.is_finite()) => {
                    GPU_PRODUCTS_REDONE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Ok((mut forget, mut gain)) => {
                    let mut merged = vec![0.0f32; hidden.len()];
                    (merged.par_chunks_mut(h), forget.par_chunks_mut(h), gain.par_chunks_mut(h), hidden.par_chunks(h), message.par_chunks(h))
                        .into_par_iter()
                        .for_each(|(merged, forget, gain, hidden, message)| {
                            accel::tanh_in_place(forget);
                            accel::tanh_in_place(gain);
                            for i in 0..h {
                                merged[i] = (1.0 - forget[i]) * hidden[i] + gain[i] * message[i];
                            }
                        });
                    return merged;
                }
                Err(error) => gpu_failed(&error),
            }
        }
        #[cfg(not(target_os = "macos"))]
        let _ = gpu;
        let merged = per_token(hidden.len() / h, |t| {
            let (hidden, message) = (&hidden[t * h..][..h], &message[t * h..][..h]);
            let mut pair = [hidden, message].concat();
            let (a, b) = pair.split_at_mut(h);
            rms_norm(a, None, eps);
            rms_norm(b, None, eps);
            round_all(bf16, &mut pair);
            let mut low = self.down.apply_one(&pair);
            round_all(bf16, &mut low);
            let (mut forget, mut gain) = (self.forget.apply_one(&low), self.gain.apply_one(&low));
            accel::tanh_in_place(&mut forget);
            accel::tanh_in_place(&mut gain);
            (0..h).map(|i| (1.0 - forget[i]) * hidden[i] + gain[i] * message[i]).collect::<Vec<f32>>()
        });
        merged.concat()
    }
}
