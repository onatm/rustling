//! A weight matrix in whatever format it is stored in, and its products with batches of vectors.

use super::*;

/// Row-major dense matrix in its stored form (bfloat16 in the checkpoint, a block format in a GGUF file),
/// borrowed from the weights and applied as `y = W x`.
pub(crate) struct Mat<'a> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) format: Format,
    pub(super) row_bytes: usize,
    pub(crate) bytes: std::borrow::Cow<'a, [u8]>,
}

impl<'a> Mat<'a> {
    pub(crate) fn open(weights: &'a Checkpoint, name: &str, rows: usize, cols: usize) -> Result<Self> {
        let (format, shape, bytes) = weights.rows(name)?;
        let row_bytes = format.row_bytes(cols).with_context(|| format!("{name}: rows of {cols} in {format:?}"))?;
        ensure!(bytes.len() == rows * row_bytes, "{name}: expected {rows}x{cols}, found shape {shape:?}");
        ensure!(format != Format::F32 || weights.is_gguf(), "{name}: expected bfloat16");
        Ok(Mat { rows, cols, format, row_bytes, bytes: bytes.into() })
    }

    /// A copy with every row packed as `Q8_0`, so that it is multiplied in integer arithmetic: several times
    /// faster than bfloat16, and off by a little. The matrix itself when it is quantised already, or its rows do
    /// not make whole blocks.
    pub(crate) fn rounded_to_bytes(self) -> Mat<'a> {
        self.packed().unwrap_or(self)
    }

    /// The copy [`Mat::rounded_to_bytes`] makes, or nothing where that returns the matrix itself.
    pub(crate) fn packed(&self) -> Option<Mat<'static>> {
        if self.format.is_quantised() || !self.cols.is_multiple_of(crate::weights::ggml::INT8_BLOCK) {
            return None;
        }
        let format = Format::Q8_0;
        let row_bytes = format.row_bytes(self.cols).expect("whole blocks");
        let mut bytes = vec![0u8; self.rows * row_bytes];
        bytes.par_chunks_exact_mut(row_bytes).zip(self.bytes.par_chunks_exact(self.row_bytes)).for_each_init(
            || vec![0.0f32; self.cols],
            |widened, (packed, row)| {
                self.format.decode_row(row, widened);
                packed.copy_from_slice(&crate::weights::ggml::encode_q8_0(widened));
            },
        );
        Some(Mat { rows: self.rows, cols: self.cols, format, row_bytes, bytes: bytes.into() })
    }

    pub(crate) fn row(&self, row: usize) -> &[u8] {
        &self.bytes[row * self.row_bytes..][..self.row_bytes]
    }

    /// Rows `first..first + count` as floats, one after another.
    pub(crate) fn widened(&self, first: usize, count: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; count * self.cols];
        for (r, out) in out.chunks_exact_mut(self.cols).enumerate() {
            self.format.decode_row(self.row(first + r), out);
        }
        out
    }

    /// Whether the matrix is multiplied in integer arithmetic, against activations rounded to bytes.
    pub(super) fn int8(&self) -> bool {
        self.format.is_quantised() && self.cols.is_multiple_of(crate::weights::ggml::INT8_BLOCK) && !float_reference()
    }

    /// `W x` for each of the rounded activations `xs`, rows `first..first + outs[i].len()`, into `outs`.
    pub(super) fn apply_int8(&self, first: usize, xs: &[&Int8Activation], outs: &mut [&mut [f32]]) {
        let mut dots = [0.0f32; INT8_MAX_BATCH];
        for (xs, outs) in xs.chunks(INT8_MAX_BATCH).zip(outs.chunks_mut(INT8_MAX_BATCH)) {
            for r in 0..outs[0].len() {
                dot_row_int8_many(self.format, self.row(first + r), xs, &mut dots[..xs.len()]);
                outs.iter_mut().zip(&dots).for_each(|(out, dot)| out[r] = *dot);
            }
        }
    }

    /// Applies the matrix to every vector of a sequence.
    pub(crate) fn apply(&self, xs: &[f32]) -> Seq {
        apply_many(None, &[(self, xs)]).remove(0)
    }

    /// Applies the matrix to one vector on the calling thread.
    pub(crate) fn apply_one(&self, x: &[f32]) -> Vec<f32> {
        if self.int8() {
            let mut out = vec![0.0f32; self.rows];
            self.apply_int8(0, &[&quantize_int8(x)], &mut [&mut out[..]]);
            return out;
        }
        let mut scratch = Vec::new();
        self.bytes.chunks_exact(self.row_bytes).map(|row| self.format.dot_row(row, x, &mut scratch)).collect()
    }
}

/// Applies each matrix to its sequence. The products share one parallel region, split into row ranges, because
/// waking the thread pool separately for every matrix costs more than the smaller products themselves.
///
/// Tokens go through in tiles: each weight row is widened once and applied to every token of the tile, so a long
/// sequence reads the weights once per tile instead of once per token. A lone token skips the widening; the two
/// routes give the same result bit for bit.
///
/// With a GPU and enough tokens the whole batch goes there instead; the results are identical.
pub(crate) fn apply_many(gpu: Option<&Gpu>, jobs: &[(&Mat, &[f32])]) -> Vec<Seq> {
    #[cfg(target_os = "macos")]
    if let Some(gpu) = gpu
        && jobs.iter().any(|(m, xs)| xs.len() / m.cols >= GPU_MIN_TOKENS)
    {
        let fast = gpu.fast() && jobs.iter().any(|(m, xs)| xs.len() / m.cols >= FAST_MIN_TOKENS);
        // The exact kernel reads bfloat16; the fast path also unpacks a GGUF file's block formats.
        let suits = |m: &Mat| match m.format {
            Format::Bf16 => m.cols % 16 == 0,
            format => fast && format.is_quantised() && m.cols % 32 == 0,
        };
        if jobs.iter().all(|(m, _)| suits(m)) {
        let dense: Vec<DenseJob> = jobs
            .iter()
            .map(|(m, xs)| DenseJob { format: m.format, weights: &m.bytes, rows: m.rows, cols: m.cols, xs })
            .collect();
        // The fast product now and then returns NaN (see `Gpu::dense_fast`), which nothing downstream would
        // catch: it would sit in the state and spoil every token after it. A batch that comes back so is redone
        // in short runs, still on the GPU, and only then on the CPU.
        let sound = |outs: &Vec<Seq>| outs.par_iter().all(|out| out.iter().all(|value| value.is_finite()));
        for in_runs in [false, true] {
            match if fast { gpu.dense_fast(&dense, in_runs) } else { gpu.dense(&dense) } {
                Ok(outs) if !fast || sound(&outs) => return outs,
                Ok(_) => products_redone(in_runs),
                Err(error) => {
                    gpu_failed(&error);
                    break;
                }
            }
        }
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = gpu;
    let mut outs: Vec<Seq> = jobs.iter().map(|(m, xs)| vec![0.0f32; xs.len() / m.cols * m.rows]).collect();
    let work: usize = jobs.iter().map(|(m, xs)| xs.len() * m.rows).sum();

    // One task per (matrix, tile of tokens, range of rows); it owns that range of each token's output.
    // Quantised matrices meet their inputs rounded to bytes, once for all the rows.
    let rounded: Vec<Vec<Int8Activation>> = jobs
        .iter()
        .map(|(m, xs)| match m.int8() {
            true => xs.chunks_exact(m.cols).map(quantize_int8).collect(),
            false => Vec::new(),
        })
        .collect();
    let mut tasks = Vec::new();
    for (((m, xs), out), rounded) in jobs.iter().zip(&mut outs).zip(&rounded) {
        let mut ranges: Vec<_> = out.chunks_exact_mut(m.rows).map(|out| out.chunks_mut(TASK_ROWS)).collect();
        let tiles = xs.chunks(DENSE_TILE * m.cols).zip(ranges.chunks_mut(DENSE_TILE)).enumerate();
        for (t, (tile, ranges)) in tiles {
            let rounded = rounded.get(t * DENSE_TILE..).unwrap_or_default();
            for range in 0..m.rows.div_ceil(TASK_ROWS) {
                let outs: Vec<&mut [f32]> = ranges.iter_mut().map(|r| r.next().expect("a range per token")).collect();
                tasks.push((*m, tile, rounded, range * TASK_ROWS, outs));
            }
        }
    }
    let run = |(m, xs, rounded, first, mut outs): (&Mat, &[f32], &[Int8Activation], usize, Vec<&mut [f32]>)| {
        if m.int8() {
            let xs: Vec<&Int8Activation> = rounded.iter().take(outs.len()).collect();
            return m.apply_int8(first, &xs, &mut outs);
        }
        let rows = m.bytes[first * m.row_bytes..].chunks_exact(m.row_bytes).take(outs[0].len());
        if let [out] = &mut outs[..] {
            let mut scratch = Vec::new();
            out.iter_mut().zip(rows).for_each(|(o, row)| *o = m.format.dot_row(row, xs, &mut scratch));
            return;
        }
        let xs: Vec<&[f32]> = xs.chunks_exact(m.cols).collect();
        let mut widened = vec![0.0f32; m.cols];
        let mut dots = vec![0.0f32; xs.len()];
        for (r, row) in rows.enumerate() {
            m.format.decode_row(row, &mut widened);
            dot_f32_row_many(&widened, &xs, &mut dots);
            outs.iter_mut().zip(&dots).for_each(|(out, dot)| out[r] = *dot);
        }
    };
    match work < PARALLEL_MIN_WORK {
        true => tasks.into_iter().for_each(run),
        false => tasks.into_par_iter().for_each(run),
    }
    outs
}
