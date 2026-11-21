//! Named intermediate tensors of the first decoder layer, captured on request so a mismatch with the reference can
//! be pinned to one step. Off unless [`start`] has been called; then the first layer of the next forward pass
//! records into it.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);
static TENSORS: Mutex<BTreeMap<String, Vec<f32>>> = Mutex::new(BTreeMap::new());

pub fn start() {
    TENSORS.lock().expect("trace lock").clear();
    ENABLED.store(true, Ordering::Relaxed);
}

/// Stops capturing and returns what was recorded.
pub fn take() -> BTreeMap<String, Vec<f32>> {
    ENABLED.store(false, Ordering::Relaxed);
    std::mem::take(&mut TENSORS.lock().expect("trace lock"))
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn record(name: &str, data: &[f32]) {
    if enabled() {
        TENSORS.lock().expect("trace lock").insert(name.to_owned(), data.to_vec());
    }
}

/// Records `data` at `offset` of a tensor of `len` values, for tensors filled in pieces from several threads.
pub fn record_at(name: &str, len: usize, offset: usize, data: &[f32]) {
    if enabled() {
        let mut tensors = TENSORS.lock().expect("trace lock");
        let tensor = tensors.entry(name.to_owned()).or_insert_with(|| vec![0.0; len]);
        tensor[offset..offset + data.len()].copy_from_slice(data);
    }
}
