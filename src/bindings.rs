//! Python bindings: borrow NumPy arrays as slices, validate, release the GIL.
//!
//! Input arrays (including memory maps) are borrowed without copying; outputs
//! are written into caller-provided arrays or returned as new NumPy arrays.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use numpy::{
    IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::ThreadPool;

use crate::kernels::consensus::{chromosome_votes, Grid, InvalidSignal};
use crate::kernels::footprint::{accumulate, End, Layout, Regions};
use crate::kernels::peaks;
use crate::kernels::signal::{store, Track};

fn invalid(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn slice<'a, T: numpy::Element>(
    array: &'a PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_| invalid(format!("{name} must be a contiguous array")))
}

fn slice_mut<'a, T: numpy::Element>(
    array: &'a mut PyReadwriteArray1<'_, T>,
    name: &str,
) -> PyResult<&'a mut [T]> {
    array
        .as_slice_mut()
        .map_err(|_| invalid(format!("{name} must be a contiguous array")))
}

/// Shared rayon pools, one per requested thread count.
fn pool(threads: usize) -> PyResult<Arc<ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();
    let threads = threads.max(1);
    let mut pools = POOLS.get_or_init(Default::default).lock().unwrap();
    if let Some(existing) = pools.get(&threads) {
        return Ok(existing.clone());
    }
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("mcc-peaks-{i}"))
        .build()
        .map_err(|e| invalid(format!("Could not start {threads} threads: {e}")))?;
    let built = Arc::new(built);
    pools.insert(threads, built.clone());
    Ok(built)
}

/// Borrowed replicate tracks: float32 or float64 arrays with optional scales.
enum Borrowed<'py> {
    F32(PyReadonlyArray1<'py, f32>),
    F64(PyReadonlyArray1<'py, f64>),
}

fn borrow_tracks<'py>(
    replicates: &[(Bound<'py, PyAny>, Option<f64>)],
) -> PyResult<Vec<Borrowed<'py>>> {
    replicates
        .iter()
        .map(|(array, _)| {
            if let Ok(a) = array.extract::<PyReadonlyArray1<'py, f32>>() {
                Ok(Borrowed::F32(a))
            } else {
                Ok(Borrowed::F64(
                    array.extract::<PyReadonlyArray1<'py, f64>>().map_err(|_| {
                        invalid("replicate signal must be a float32 or float64 array")
                    })?,
                ))
            }
        })
        .collect()
}

fn tracks<'a>(
    borrowed: &'a [Borrowed<'_>],
    replicates: &[(Bound<'_, PyAny>, Option<f64>)],
) -> PyResult<Vec<(Track<'a>, Option<f64>)>> {
    borrowed
        .iter()
        .zip(replicates)
        .map(|(b, (_, scale))| {
            let track = match b {
                Borrowed::F32(a) => Track::F32(slice(a, "replicate")?),
                Borrowed::F64(a) => Track::F64(slice(a, "replicate")?),
            };
            Ok((track, *scale))
        })
        .collect()
}

/// Store BigWig values as float32 (missing as zero); returns (non-finite, negative).
#[pyfunction]
fn store_signal(
    py: Python<'_>,
    values: PyReadonlyArray1<'_, f64>,
    mut out: PyReadwriteArray1<'_, f32>,
) -> PyResult<(bool, bool)> {
    let values = slice(&values, "values")?;
    let out = slice_mut(&mut out, "out")?;
    if values.len() != out.len() {
        return Err(invalid("values and out lengths differ"));
    }
    Ok(py.detach(|| store(values, out)))
}

/// Consensus votes for a whole chromosome, written into `out` (one per unit).
///
/// `replicates` are (signal, scale) pairs; `excluded` are sorted, merged
/// half-open unit intervals. See `consensus::chromosome_votes`.
#[pyfunction]
#[pyo3(signature = (replicates, excluded, eligible, profile_positions, profile_values, unit, chrom_length, windows, window_index, backgrounds, thresholds, chunk, halo, out, threads=1))]
#[allow(clippy::too_many_arguments)]
fn consensus_votes(
    py: Python<'_>,
    replicates: Vec<(Bound<'_, PyAny>, Option<f64>)>,
    excluded: Vec<(usize, usize)>,
    eligible: Option<PyReadonlyArray1<'_, bool>>,
    profile_positions: PyReadonlyArray1<'_, f64>,
    profile_values: PyReadonlyArray1<'_, f64>,
    unit: i64,
    chrom_length: i64,
    windows: Vec<i64>,
    window_index: Vec<i64>,
    backgrounds: Vec<i64>,
    thresholds: Vec<f64>,
    chunk: usize,
    halo: usize,
    mut out: PyReadwriteArray1<'_, i32>,
    threads: usize,
) -> PyResult<()> {
    let borrowed = borrow_tracks(&replicates)?;
    let tracks = tracks(&borrowed, &replicates)?;
    let eligible = match &eligible {
        Some(e) => Some(slice(e, "eligible")?),
        None => None,
    };
    let grid = Grid {
        profile_positions: slice(&profile_positions, "profile_positions")?,
        profile_values: slice(&profile_values, "profile_values")?,
        unit,
        chrom_length,
        windows: &windows,
        window_index: &window_index,
        backgrounds: &backgrounds,
        thresholds: &thresholds,
    };
    let out = slice_mut(&mut out, "out")?;
    let n = out.len();
    if tracks.is_empty()
        || tracks.iter().any(|(t, _)| t.len() != n)
        || eligible.is_some_and(|e| e.len() != n)
        || grid.profile_positions.is_empty()
        || grid.profile_positions.len() != grid.profile_values.len()
        || unit < 1
        || windows.iter().chain(&backgrounds).any(|&w| w < 1)
        || window_index
            .iter()
            .any(|&w| w < 0 || w as usize >= windows.len())
        || excluded.windows(2).any(|w| w[0].1 > w[1].0)
        || excluded.iter().any(|&(lo, hi)| lo > hi || hi > n)
    {
        return Err(invalid("inconsistent consensus vote inputs"));
    }
    let pool = pool(threads)?;
    py.detach(|| {
        pool.install(|| chromosome_votes(&tracks, &excluded, eligible, &grid, chunk, halo, out))
    })
    .map_err(|InvalidSignal| invalid("Replicate signal must contain finite nonnegative values"))
}

/// Peak indices of `x` (scipy.signal.find_peaks with distance and prominence).
#[pyfunction]
fn find_peaks<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    distance: f64,
    prominence: f64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let x = slice(&x, "x")?;
    let found: Vec<i64> = py.detach(|| {
        peaks::find_peaks(x, distance, prominence)
            .into_iter()
            .map(|p| p as i64)
            .collect()
    });
    Ok(found.into_pyarray(py))
}

type Table<'py> = (
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, i64>,
);

fn regions<'a>(table: &'a Table<'_>) -> PyResult<Regions<'a>> {
    let regions = Regions {
        offsets: slice(&table.0, "offsets")?,
        starts: slice(&table.1, "starts")?,
        ends: slice(&table.2, "ends")?,
        max_ends: slice(&table.3, "max_ends")?,
        ids: slice(&table.4, "ids")?,
    };
    let n = regions.starts.len();
    if [
        regions.ends.len(),
        regions.max_ends.len(),
        regions.ids.len(),
    ]
    .iter()
    .any(|&l| l != n)
        || regions.offsets.iter().any(|&o| o < 0 || o as usize > n)
    {
        return Err(invalid("inconsistent region table"));
    }
    Ok(regions)
}

/// Accumulate selected contacts; see `footprint::accumulate`.
///
/// Returns (contacts written, junction ends counted, contacts with a capture end).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn footprint_rows(
    py: Python<'_>,
    chrom0: PyReadonlyArray1<'_, i64>,
    pos0: PyReadonlyArray1<'_, i64>,
    dir0: PyReadonlyArray1<'_, i64>,
    chrom1: PyReadonlyArray1<'_, i64>,
    pos1: PyReadonlyArray1<'_, i64>,
    dir1: PyReadonlyArray1<'_, i64>,
    reporter: Table<'_>,
    capture: Table<'_>,
    region_start: PyReadonlyArray1<'_, i64>,
    base: PyReadonlyArray1<'_, i64>,
    store: PyReadonlyArray1<'_, i64>,
    mut profile: PyReadwriteArray1<'_, i64>,
    mut contacts: PyReadwriteArray2<'_, i64>,
) -> PyResult<(usize, i64, i64)> {
    let columns = [
        slice(&chrom0, "chrom0")?,
        slice(&pos0, "pos0")?,
        slice(&dir0, "dir0")?,
        slice(&chrom1, "chrom1")?,
        slice(&pos1, "pos1")?,
        slice(&dir1, "dir1")?,
    ];
    let n = columns[0].len();
    if columns.iter().any(|c| c.len() != n) {
        return Err(invalid("row arrays must have equal lengths"));
    }
    let rows: Vec<[End; 2]> = (0..n)
        .map(|i| {
            [
                (columns[0][i], columns[1][i], columns[2][i]),
                (columns[3][i], columns[4][i], columns[5][i]),
            ]
        })
        .collect();
    let reporter = regions(&reporter)?;
    let capture = regions(&capture)?;
    let layout = Layout {
        base: slice(&base, "base")?,
        region_start: slice(&region_start, "region_start")?,
        column: slice(&store, "store")?,
    };
    if contacts.shape()[1] != 8 {
        return Err(invalid("contacts must have 8 columns"));
    }
    let capacity = contacts.shape()[0];
    let contacts = contacts
        .as_slice_mut()
        .map_err(|_| invalid("contacts must be a contiguous array"))?;
    let profile = slice_mut(&mut profile, "profile")?;
    let tally = py.detach(|| accumulate(&rows, &reporter, &capture, &layout, profile, contacts));
    if tally.contacts > capacity {
        return Err(invalid("contact buffer too small"));
    }
    Ok((tally.contacts, tally.junction_ends, tally.mapped))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(store_signal, m)?)?;
    m.add_function(wrap_pyfunction!(consensus_votes, m)?)?;
    m.add_function(wrap_pyfunction!(find_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(footprint_rows, m)?)?;
    Ok(())
}
