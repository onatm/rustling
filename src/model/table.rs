//! Where a layer visit keeps the keys or the values of the tokens it has seen: a row per token for each key
//! head, the heads one after another, so that attention reads a head's rows as one run of memory.
//!
//! The rows live in ordinary memory, or in a buffer shared with the GPU once the GPU decoder has used them. The
//! decoder then reads and adds to the same rows the CPU paths do, and nothing is kept twice.

use serde::{Deserialize, Serialize};

#[cfg(target_os = "macos")]
use crate::gpu::{Gpu, Shared};

/// Rows of room a table is left with when it is cut down after rows are dropped.
const SPARE_ROWS: usize = 256;

pub(crate) struct Table {
    heads: usize,
    dim: usize,
    /// Rows in use, of the `capacity` each head has room for.
    rows: usize,
    capacity: usize,
    store: Store,
}

enum Store {
    Plain(Vec<f32>),
    #[cfg(target_os = "macos")]
    Shared(Shared),
}

impl Store {
    fn floats(&self) -> &[f32] {
        match self {
            Store::Plain(floats) => floats,
            #[cfg(target_os = "macos")]
            Store::Shared(shared) => shared.floats(),
        }
    }

    fn floats_mut(&mut self) -> &mut [f32] {
        match self {
            Store::Plain(floats) => floats,
            #[cfg(target_os = "macos")]
            Store::Shared(shared) => shared.floats_mut(),
        }
    }

    /// Room for `floats` more of the same kind; ordinary memory if the GPU has none to give.
    fn another(&self, floats: usize) -> Store {
        match self {
            #[cfg(target_os = "macos")]
            Store::Shared(shared) => match shared.sibling(floats * 4) {
                Ok(sibling) => Store::Shared(sibling),
                Err(_) => Store::Plain(vec![0.0; floats]),
            },
            _ => Store::Plain(vec![0.0; floats]),
        }
    }
}

impl Default for Table {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl Table {
    pub(crate) fn new(heads: usize, dim: usize) -> Self {
        Self { heads, dim, rows: 0, capacity: 0, store: Store::Plain(Vec::new()) }
    }

    /// A table holding the given rows, a head's worth in each vector.
    pub(crate) fn from_heads(heads: Vec<Vec<f32>>, dim: usize) -> Self {
        let rows = heads.first().map_or(0, |head| head.len() / dim.max(1));
        assert!(heads.iter().all(|head| head.len() == rows * dim), "heads of different lengths");
        Self { heads: heads.len(), dim, rows, capacity: rows, store: Store::Plain(heads.concat()) }
    }

    pub(crate) fn heads(&self) -> usize {
        self.heads
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// How much memory the rows in use take.
    pub(crate) fn bytes(&self) -> usize {
        self.rows * self.heads * self.dim * size_of::<f32>()
    }

    /// One head's rows, oldest first.
    pub(crate) fn head(&self, head: usize) -> &[f32] {
        &self.store.floats()[head * self.capacity * self.dim..][..self.rows * self.dim]
    }

    /// Makes room for `more` rows.
    pub(crate) fn reserve(&mut self, more: usize) {
        let capacity = self.capacity_for(more);
        if capacity != self.capacity {
            self.resize(capacity);
        }
    }

    fn capacity_for(&self, more: usize) -> usize {
        let needed = self.rows + more;
        if needed <= self.capacity { self.capacity } else { needed.max(2 * self.capacity).max(16) }
    }

    fn resize(&mut self, capacity: usize) {
        let store = self.store.another(self.heads * capacity * self.dim);
        self.move_into(store, capacity);
    }

    fn move_into(&mut self, mut store: Store, capacity: usize) {
        let used = self.rows * self.dim;
        for head in 0..self.heads {
            store.floats_mut()[head * capacity * self.dim..][..used].copy_from_slice(self.head(head));
        }
        (self.store, self.capacity) = (store, capacity);
    }

    /// Adds rows given token by token, each token's heads one after another.
    pub(crate) fn push(&mut self, tokens: &[f32]) {
        let (heads, dim) = (self.heads, self.dim);
        let count = tokens.len() / (heads * dim);
        self.reserve(count);
        let (capacity, rows) = (self.capacity, self.rows);
        let floats = self.store.floats_mut();
        for (t, token) in tokens.chunks_exact(heads * dim).enumerate() {
            for (head, new) in token.chunks_exact(dim).enumerate() {
                floats[(head * capacity + rows + t) * dim..][..dim].copy_from_slice(new);
            }
        }
        self.rows += count;
    }

    /// Counts `more` rows that were written in place, into room made by [`Table::reserve`].
    #[cfg(target_os = "macos")]
    pub(crate) fn added(&mut self, more: usize) {
        assert!(self.rows + more <= self.capacity, "rows beyond the table's room");
        self.rows += more;
    }

    /// Forgets the newest rows, keeping `rows`.
    pub(crate) fn truncate(&mut self, rows: usize) {
        self.rows = self.rows.min(rows);
    }

    /// Drops the `count` oldest rows.
    pub(crate) fn drop_front(&mut self, count: usize) {
        let (capacity, dim, rows) = (self.capacity, self.dim, self.rows);
        let floats = self.store.floats_mut();
        for head in 0..self.heads {
            let start = head * capacity * dim;
            floats.copy_within(start + count * dim..start + rows * dim, start);
        }
        self.rows -= count;
        // A long pass leaves a sliding layer's table several times the size it settles at.
        if self.capacity > 2 * self.rows + SPARE_ROWS {
            self.resize(self.rows + SPARE_ROWS);
        }
    }

    /// Ensures GPU-visible rows and room for the next batch in a single allocation and copy.
    #[cfg(target_os = "macos")]
    pub(crate) fn reserve_shared(&mut self, gpu: &Gpu, more: usize) -> anyhow::Result<()> {
        let capacity = self.capacity_for(more);
        if capacity == self.capacity && matches!(self.store, Store::Shared(_)) {
            return Ok(());
        }
        // Only the rows in use are copied: the rest of the room has never been touched and takes no memory yet.
        let shared = gpu.shared(self.heads * capacity * self.dim * 4)?;
        self.move_into(Store::Shared(shared), capacity);
        Ok(())
    }

    /// The shared buffer, once [`Table::reserve_shared`] has made one.
    #[cfg(target_os = "macos")]
    pub(crate) fn shared_mut(&mut self) -> Option<&mut Shared> {
        match &mut self.store {
            Store::Shared(shared) => Some(shared),
            Store::Plain(_) => None,
        }
    }

    /// Copy the retained suffix of each head directly into a compact table.
    pub(super) fn clone_skipping(&self, skip: usize) -> Self {
        let rows = self.rows - skip;
        self.copy_into(self.store.another(self.heads * rows * self.dim), rows, skip)
    }

    /// Fork the cached rows into an independent GPU buffer with space for the imminent batch.
    #[cfg(target_os = "macos")]
    pub(super) fn clone_shared(&self, gpu: &Gpu, more: usize) -> anyhow::Result<Self> {
        let capacity = self.capacity_for(more);
        let shared = gpu.shared(self.heads * capacity * self.dim * 4)?;
        Ok(self.copy_into(Store::Shared(shared), capacity, 0))
    }

    fn copy_into(&self, mut store: Store, capacity: usize, skip: usize) -> Self {
        let rows = self.rows - skip;
        for head in 0..self.heads {
            store.floats_mut()[head * capacity * self.dim..][..rows * self.dim].copy_from_slice(&self.head(head)[skip * self.dim..]);
        }
        Self { heads: self.heads, dim: self.dim, rows, capacity, store }
    }
}

impl Clone for Table {
    fn clone(&self) -> Self {
        self.clone_skipping(0)
    }
}

/// A table as it is written to disk: each head's rows.
#[derive(Serialize, Deserialize)]
struct Saved {
    dim: usize,
    heads: Vec<Vec<f32>>,
}

impl Serialize for Table {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Saved { dim: self.dim, heads: (0..self.heads).map(|head| self.head(head).to_vec()).collect() }.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Table {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let saved = Saved::deserialize(deserializer)?;
        let rows = saved.heads.first().map_or(0, |head| head.len() / saved.dim.max(1));
        if saved.heads.iter().any(|head| head.len() != rows * saved.dim) {
            return Err(serde::de::Error::custom("a table's heads hold different numbers of rows"));
        }
        Ok(Table::from_heads(saved.heads, saved.dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn sharing_full_tables_keeps_each_heads_stride_when_they_grow() {
        let gpu = Gpu::new(&[], true).unwrap();
        let mut table = Table::from_heads(vec![vec![1., 2., 3., 4.], vec![5., 6., 7., 8.]], 2);
        table.reserve_shared(&gpu, 1).unwrap();
        table.push(&[9., 10., 11., 12.]);
        let capacity = table.capacity();
        table.reserve_shared(&gpu, capacity).unwrap();
        assert!(table.capacity() >= capacity + 3);
        assert!(table.shared_mut().is_some());
        assert_eq!(table.head(0), &[1., 2., 3., 4., 9., 10.]);
        assert_eq!(table.head(1), &[5., 6., 7., 8., 11., 12.]);
        let mut suffix = table.clone_skipping(1);
        assert!(suffix.shared_mut().is_some());
        assert_eq!(suffix.head(0), &[3., 4., 9., 10.]);
        assert_eq!(suffix.head(1), &[7., 8., 11., 12.]);
    }

    #[test]
    fn suffix_copy_serializes_like_dropping_rows_from_a_full_copy() {
        let mut table = Table::new(3, 2);
        for t in 0..800 {
            table.push(&std::array::from_fn::<_, 6, _>(|i| (6 * t + i) as f32));
        }
        table.drop_front(17);
        for skip in [0, 1, 640, table.rows()] {
            let mut reference = table.clone();
            reference.drop_front(skip);
            let copy = table.clone_skipping(skip);
            assert_eq!(copy.capacity, table.rows() - skip);
            assert_eq!(postcard::to_allocvec(&copy).unwrap(), postcard::to_allocvec(&reference).unwrap());
        }
    }

    #[test]
    fn rows_survive_growing_dropping_and_copying() {
        // Two heads of three values; token `t` holds `t` in head 0 and `100 + t` in head 1.
        let token = |t: usize| -> Vec<f32> { [t as f32; 3].into_iter().chain([100.0 + t as f32; 3]).collect() };
        let mut table = Table::new(2, 3);
        for t in 0..40 {
            table.push(&token(t));
        }
        assert_eq!(table.rows(), 40);
        table.drop_front(5);
        table.truncate(30);
        let check = |table: &Table| {
            assert_eq!(table.rows(), 30);
            for (r, row) in table.head(0).chunks_exact(3).enumerate() {
                assert_eq!(row, [(5 + r) as f32; 3]);
            }
            for (r, row) in table.head(1).chunks_exact(3).enumerate() {
                assert_eq!(row, [(105 + r) as f32; 3]);
            }
        };
        check(&table);
        let copy = table.clone();
        table.push(&token(99));
        check(&copy);
        let saved = postcard::to_allocvec(&copy).unwrap();
        check(&postcard::from_bytes(&saved).unwrap());
    }
}
