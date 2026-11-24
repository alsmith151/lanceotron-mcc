//! Signal preparation: BigWig value storage and replicate-mean prefix sums.

/// Store values as float32 with missing (NaN) bases as zero.
///
/// Returns whether any stored value is non-finite and whether any is negative.
pub fn store(values: &[f64], out: &mut [f32]) -> (bool, bool) {
    let (mut nonfinite, mut negative) = (false, false);
    for (slot, &value) in out.iter_mut().zip(values) {
        let stored = value as f32;
        let stored = if stored.is_nan() { 0.0 } else { stored };
        nonfinite |= !stored.is_finite();
        negative |= stored < 0.0;
        *slot = stored;
    }
    (nonfinite, negative)
}

/// A replicate track, stored as float32 (BigWig) or float64 (counts).
#[derive(Clone, Copy)]
pub enum Track<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}

impl Track<'_> {
    pub fn len(&self) -> usize {
        match self {
            Track::F32(s) => s.len(),
            Track::F64(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    pub fn at(&self, i: usize) -> f64 {
        match self {
            Track::F32(s) => s[i] as f64,
            Track::F64(s) => s[i],
        }
    }
}

/// Prefix sums of the scaled replicate mean over `start..start + sums.len() - 1`.
///
/// Invalid bases count as zero; `counts` (when given, requires `valid`)
/// receives prefix counts of valid bases.
pub fn replicate_prefix(
    tracks: &[(Track<'_>, Option<f64>)],
    start: usize,
    valid: Option<&[bool]>,
    sums: &mut [f64],
    counts: Option<&mut [i64]>,
) {
    let replicates = tracks.len() as f64;
    sums[0] = 0.0;
    let mut running = 0.0f64;
    for i in 0..sums.len() - 1 {
        let mut total = 0.0f64;
        for (track, scale) in tracks {
            let value = track.at(start + i);
            total += scale.map_or(value, |s| value * s);
        }
        if valid.is_none_or(|v| v[i]) {
            running += total / replicates;
        }
        sums[i + 1] = running;
    }
    if let (Some(c), Some(v)) = (counts, valid) {
        c[0] = 0;
        for (i, &ok) in v.iter().enumerate() {
            c[i + 1] = c[i] + ok as i64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_zeroes_missing_and_flags_bad_values() {
        let mut out = [0.0f32; 3];
        assert_eq!(store(&[1.5, f64::NAN, 2.0], &mut out), (false, false));
        assert_eq!(out, [1.5, 0.0, 2.0]);
        assert_eq!(store(&[f64::INFINITY, -1.0, 0.0], &mut out), (true, true));
    }

    #[test]
    fn prefix_averages_scaled_replicates_and_masks_invalid_bases() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [3.0f64, 2.0, 1.0];
        let tracks = [(Track::F32(&a), Some(2.0)), (Track::F64(&b), None)];
        let mut sums = [0.0; 3];
        let mut counts = [0i64; 3];
        replicate_prefix(
            &tracks,
            1,
            Some(&[true, false]),
            &mut sums,
            Some(&mut counts),
        );
        assert_eq!(sums, [0.0, 3.0, 3.0]);
        assert_eq!(counts, [0, 1, 1]);
    }
}
