//! Peak-calling and footprinting algorithms over plain slices.
//!
//! Nothing here touches pyo3, so the algorithms are unit-tested with plain
//! `cargo test`; the `bindings` module exposes them to Python. Parallel kernels only
//! write disjoint output slices, so they are free of data races by
//! construction.
//!
//! - [`consensus`]: multi-scale enrichment votes for distance-aware peak calling.
//! - [`signal`]: BigWig value storage and replicate-mean prefix sums.
//! - [`peaks`]: local maxima and distance/prominence filtering (valley splits).
//! - [`footprint`]: junction-end profiles and capture x reporter contacts.

pub mod consensus;
pub mod footprint;
pub mod peaks;
pub mod signal;
