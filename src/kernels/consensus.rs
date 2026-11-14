//! Consensus votes over the background x window x threshold grid.

use rayon::prelude::*;

use super::signal::{replicate_prefix, Track};

/// Centred moving mean at index `j` from prefix sums over `n` bases.
///
/// Edge windows are truncated. With `counts` (prefix counts of valid bases)
/// the denominator is the number of valid bases, and windows without any are
/// zero. A window of at least `n` is the mean of everything.
#[inline(always)]
pub fn centered_mean(sums: &[f64], counts: Option<&[i64]>, n: usize, window: i64, j: usize) -> f64 {
    // Invalid bases contribute zero to `sums`, so a window without any valid
    // base has a zero mean.
    let mean = |lo: usize, hi: usize| {
        let total = sums[hi] - sums[lo];
        match counts {
            Some(c) if c[hi] > c[lo] => total / (c[hi] - c[lo]) as f64,
            Some(_) => 0.0,
            None => total / (hi - lo) as f64,
        }
    };
    if n == 0 {
        return 0.0;
    }
    if window >= n as i64 {
        return mean(0, n);
    }
    let window = window as usize;
    let left = window / 2;
    let right = window - left - 1;
    mean(j.saturating_sub(left), (j + right + 1).min(n))
}

/// Linear interpolation of `fp` at `x`, clamped to the end values, given
/// `bracket`, the last index with `xp[bracket] <= x` (or 0).
#[inline(always)]
pub fn interpolate(xp: &[f64], fp: &[f64], bracket: usize, x: f64) -> f64 {
    let last = xp.len() - 1;
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[last] {
        return fp[last];
    }
    let (x0, x1) = (xp[bracket], xp[bracket + 1]);
    if x == x0 {
        return fp[bracket];
    }
    fp[bracket] + (fp[bracket + 1] - fp[bracket]) * ((x - x0) / (x1 - x0))
}

/// Chunk inputs shared by every position of a vote computation.
pub struct VoteGrid<'a> {
    /// Prefix sums of the replicate-mean signal over the chunk with halo.
    pub sums: &'a [f64],
    /// Prefix counts of valid bases, when some bases are excluded.
    pub counts: Option<&'a [i64]>,
    pub valid: &'a [bool],
    /// Coarse distance-background profile: bin centres (bp) and values.
    pub profile_positions: &'a [f64],
    pub profile_values: &'a [f64],
    /// Unit index of the first core position, unit width (bp) and chromosome
    /// length (bp); the last unit may be truncated.
    pub first_unit: i64,
    pub unit: i64,
    pub chrom_length: i64,
    /// Chunk index of the first core position.
    pub core_start: usize,
    /// Distinct smoothing widths (units); each grid window maps to one.
    pub windows: &'a [i64],
    pub window_index: &'a [i64],
    pub backgrounds: &'a [i64],
    pub thresholds: &'a [f64],
    /// Widths of at least `whole_length` use `whole_mean`.
    pub whole_length: i64,
    pub whole_mean: f64,
}

impl VoteGrid<'_> {
    fn smooth(&self, width: i64, j: usize) -> f64 {
        if width >= self.whole_length {
            self.whole_mean
        } else {
            centered_mean(self.sums, self.counts, self.sums.len() - 1, width, j)
        }
    }

    /// Fill `out` with votes for core positions `offset..offset + out.len()`.
    ///
    /// A cell votes when the window mean is positive and at least its
    /// threshold times the larger of the background mean and the distance
    /// background.
    pub fn votes(&self, offset: usize, out: &mut [i32]) {
        let (xp, fp) = (self.profile_positions, self.profile_values);
        let mut observed = vec![0.0f64; self.windows.len()];
        // Unit centres increase along the block, so the bracket only moves right.
        let first_start = ((self.first_unit + offset as i64) * self.unit) as f64;
        let mut bracket = xp.partition_point(|&p| p <= first_start).saturating_sub(1);
        for (i, slot) in out.iter_mut().enumerate() {
            let k = offset + i;
            let j = self.core_start + k;
            if !self.valid[j] {
                *slot = 0;
                continue;
            }
            let mut any_signal = false;
            for (value, &width) in observed.iter_mut().zip(self.windows) {
                *value = self.smooth(width, j);
                any_signal |= *value > 0.0;
            }
            if !any_signal {
                *slot = 0;
                continue;
            }
            let start = (self.first_unit + k as i64) * self.unit;
            let centre = start as f64 + self.unit.min(self.chrom_length - start) as f64 / 2.0;
            while bracket + 1 < xp.len() && xp[bracket + 1] <= centre {
                bracket += 1;
            }
            let distance = interpolate(xp, fp, bracket, centre);
            let mut votes = 0i32;
            for &width in self.backgrounds {
                let baseline = distance.max(self.smooth(width, j));
                for &w in self.window_index {
                    let value = observed[w as usize];
                    if value > 0.0 {
                        votes += self
                            .thresholds
                            .iter()
                            .filter(|&&t| value >= t * baseline)
                            .count() as i32;
                    }
                }
            }
            *slot = votes;
        }
    }
}

/// Grid settings for [`chromosome_votes`].
pub struct Grid<'a> {
    pub profile_positions: &'a [f64],
    pub profile_values: &'a [f64],
    pub unit: i64,
    pub chrom_length: i64,
    pub windows: &'a [i64],
    pub window_index: &'a [i64],
    pub backgrounds: &'a [i64],
    pub thresholds: &'a [f64],
}

/// Error raised when replicate signal cannot be called.
#[derive(Debug, PartialEq)]
pub struct InvalidSignal;

/// Validity of `lo..hi` given sorted, merged excluded intervals.
fn validity(lo: usize, hi: usize, excluded: &[(usize, usize)]) -> Vec<bool> {
    let mut valid = vec![true; hi - lo];
    let first = excluded.partition_point(|&(_, end)| end <= lo);
    for &(start, end) in &excluded[first..] {
        if start >= hi {
            break;
        }
        valid[start.max(lo) - lo..end.min(hi) - lo].fill(false);
    }
    valid
}

/// Mean replicate signal over valid bases; rejects negative or non-finite values.
pub fn global_mean(
    tracks: &[(Track<'_>, Option<f64>)],
    excluded: &[(usize, usize)],
    n: usize,
) -> Result<f64, InvalidSignal> {
    const BLOCK: usize = 1 << 20;
    let parts: Vec<Option<(f64, usize)>> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| {
            let (lo, hi) = (b * BLOCK, ((b + 1) * BLOCK).min(n));
            let valid = validity(lo, hi, excluded);
            let mut total = 0.0;
            for (track, scale) in tracks {
                for (i, &ok) in valid.iter().enumerate() {
                    let value = scale.map_or(track.at(lo + i), |s| track.at(lo + i) * s);
                    if !value.is_finite() || value < 0.0 {
                        return None;
                    }
                    if ok {
                        total += value;
                    }
                }
            }
            Some((total, valid.iter().filter(|&&ok| ok).count()))
        })
        .collect();
    let mut total = 0.0;
    let mut count = 0usize;
    for part in parts {
        let (t, c) = part.ok_or(InvalidSignal)?;
        total += t;
        count += c;
    }
    Ok(total / (count.max(1) * tracks.len()) as f64)
}

/// Consensus votes for every unit of a chromosome.
///
/// Work is split into chunks of `chunk` units processed in parallel; each
/// smooths over its chunk plus `halo` units either side, so results do not
/// depend on chunking beyond floating-point rounding. Chunks with no signal
/// within the widest window cannot vote and are skipped. `eligible` (count
/// mode) zeroes votes in ineligible bins.
pub fn chromosome_votes(
    tracks: &[(Track<'_>, Option<f64>)],
    excluded: &[(usize, usize)],
    eligible: Option<&[bool]>,
    grid: &Grid<'_>,
    chunk: usize,
    halo: usize,
    out: &mut [i32],
) -> Result<(), InvalidSignal> {
    let n = out.len();
    let whole_mean = global_mean(tracks, excluded, n)?;
    let widest = *grid.windows.iter().max().unwrap_or(&1) as usize;
    out.par_chunks_mut(chunk.max(1))
        .enumerate()
        .for_each(|(task, core)| {
            let lo = task * chunk.max(1);
            let hi = lo + core.len();
            let has_signal = if widest >= n {
                whole_mean > 0.0
            } else {
                let (probe_lo, probe_hi) = (
                    lo.saturating_sub(widest / 2 + 1),
                    (hi + widest / 2 + 1).min(n),
                );
                tracks
                    .iter()
                    .any(|(track, _)| (probe_lo..probe_hi).any(|i| track.at(i) > 0.0))
            };
            if !has_signal {
                core.fill(0);
                return;
            }
            let (a, b) = (lo.saturating_sub(halo), (hi + halo).min(n));
            let valid = validity(a, b, excluded);
            let all_valid = valid.iter().all(|&ok| ok);
            let mut sums = vec![0.0f64; b - a + 1];
            let mut counts = if all_valid {
                None
            } else {
                Some(vec![0i64; b - a + 1])
            };
            replicate_prefix(
                tracks,
                a,
                if all_valid { None } else { Some(&valid) },
                &mut sums,
                counts.as_deref_mut(),
            );
            VoteGrid {
                sums: &sums,
                counts: counts.as_deref(),
                valid: &valid,
                profile_positions: grid.profile_positions,
                profile_values: grid.profile_values,
                first_unit: lo as i64,
                unit: grid.unit,
                chrom_length: grid.chrom_length,
                core_start: lo - a,
                windows: grid.windows,
                window_index: grid.window_index,
                backgrounds: grid.backgrounds,
                thresholds: grid.thresholds,
                whole_length: n as i64,
                whole_mean,
            }
            .votes(0, core);
            if let Some(e) = eligible {
                for (slot, &ok) in core.iter_mut().zip(&e[lo..hi]) {
                    if !ok {
                        *slot = 0;
                    }
                }
            }
        });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefix(values: &[f64]) -> Vec<f64> {
        let mut sums = vec![0.0];
        for v in values {
            sums.push(sums.last().unwrap() + v);
        }
        sums
    }

    #[test]
    fn centred_mean_matches_truncated_window_average() {
        let values: Vec<f64> = (0..23).map(|i| ((i * 7) % 5) as f64).collect();
        let sums = prefix(&values);
        for window in 1..30i64 {
            for j in 0..values.len() {
                let w = window as usize;
                let (lo, hi) = if w >= values.len() {
                    (0, values.len())
                } else {
                    (j.saturating_sub(w / 2), (j + w - w / 2).min(values.len()))
                };
                let expected = values[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
                let got = centered_mean(&sums, None, values.len(), window, j);
                assert!((got - expected).abs() < 1e-12, "window {window} j {j}");
            }
        }
    }

    #[test]
    fn centred_mean_uses_valid_bases_only() {
        let values = [4.0, 0.0, 0.0, 2.0];
        let valid = [true, false, false, true];
        let masked: Vec<f64> = values
            .iter()
            .zip(valid)
            .map(|(v, ok)| if ok { *v } else { 0.0 })
            .collect();
        let sums = prefix(&masked);
        let mut counts = vec![0i64];
        for ok in valid {
            counts.push(counts.last().unwrap() + ok as i64);
        }
        assert_eq!(centered_mean(&sums, Some(&counts), 4, 3, 0), 4.0);
        assert_eq!(centered_mean(&sums, Some(&counts), 4, 3, 1), 4.0);
        assert_eq!(centered_mean(&sums, Some(&counts), 4, 10, 2), 3.0);
    }

    #[test]
    fn interpolation_clamps_and_is_linear() {
        let xp = [10.0, 20.0, 40.0];
        let fp = [1.0, 3.0, 7.0];
        assert_eq!(interpolate(&xp, &fp, 0, 5.0), 1.0);
        assert_eq!(interpolate(&xp, &fp, 0, 15.0), 2.0);
        assert_eq!(interpolate(&xp, &fp, 1, 30.0), 5.0);
        assert_eq!(interpolate(&xp, &fp, 2, 50.0), 7.0);
    }
}
