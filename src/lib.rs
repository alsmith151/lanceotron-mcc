//! Python bindings for mcc-peaks, exposed as `mcc_peaks._core`.
//!
//! Algorithms live in the Python-free [`kernels`] module; `bindings` only
//! borrows NumPy arrays as slices, validates them, releases the GIL and runs
//! parallel work on rayon pools.

#[cfg(feature = "python")]
mod bindings;
pub mod kernels;

#[cfg(feature = "python")]
#[pyo3::pymodule]
fn _core(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    bindings::register(m)
}
