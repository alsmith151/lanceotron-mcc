//! Peak finding used to split broad regions at internal signal valleys.
//!
//! Follows `scipy.signal.find_peaks` with `distance` and `prominence`.

/// Midpoints of local maxima; a plateau counts once, at its (lower) middle.
pub fn local_maxima(x: &[f64]) -> Vec<usize> {
    let mut peaks = vec![];
    if x.len() < 3 {
        return peaks;
    }
    let last = x.len() - 1;
    let mut i = 1;
    while i < last {
        if x[i - 1] < x[i] {
            let mut ahead = i + 1;
            while ahead < last && x[ahead] == x[i] {
                ahead += 1;
            }
            if x[ahead] < x[i] {
                peaks.push((i + ahead - 1) / 2);
                i = ahead;
            }
        }
        i += 1;
    }
    peaks
}

/// Drop peaks closer than `distance` to a higher-priority peak.
///
/// `order` ranks peaks by ascending height (ties as the caller's sort left
/// them); the last entries win.
pub fn thin_by_distance(peaks: &[usize], order: &[usize], distance: f64) -> Vec<bool> {
    let distance = distance.ceil() as usize;
    let mut keep = vec![true; peaks.len()];
    for &j in order.iter().rev() {
        if !keep[j] {
            continue;
        }
        let mut k = j;
        while k > 0 && peaks[j] - peaks[k - 1] < distance {
            keep[k - 1] = false;
            k -= 1;
        }
        let mut k = j + 1;
        while k < peaks.len() && peaks[k] - peaks[j] < distance {
            keep[k] = false;
            k += 1;
        }
    }
    keep
}

/// Height of a peak above the higher of its lowest points on either side,
/// searching outwards until the signal exceeds the peak.
pub fn prominence(x: &[f64], peak: usize) -> f64 {
    let top = x[peak];
    let left = x[..=peak]
        .iter()
        .rev()
        .take_while(|&&v| v <= top)
        .fold(top, |m, &v| m.min(v));
    let right = x[peak..]
        .iter()
        .take_while(|&&v| v <= top)
        .fold(top, |m, &v| m.min(v));
    top - left.max(right)
}

/// Peaks of `x` at least `distance` apart (higher peaks win; ties keep the
/// later-ranked of a stable height sort) with at least `min_prominence`.
pub fn find_peaks(x: &[f64], distance: f64, min_prominence: f64) -> Vec<usize> {
    let peaks = local_maxima(x);
    let mut order: Vec<usize> = (0..peaks.len()).collect();
    order.sort_by(|&a, &b| x[peaks[a]].total_cmp(&x[peaks[b]]));
    let keep = thin_by_distance(&peaks, &order, distance);
    peaks
        .into_iter()
        .zip(keep)
        .filter(|&(peak, kept)| kept && min_prominence <= prominence(x, peak))
        .map(|(peak, _)| peak)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plateaus_and_edges() {
        assert_eq!(
            local_maxima(&[0.0, 1.0, 1.0, 0.0, 2.0, 0.0, 3.0]),
            vec![1, 4]
        );
        assert_eq!(local_maxima(&[1.0, 0.0]), Vec::<usize>::new());
    }

    #[test]
    fn distance_keeps_highest_and_prominence_measures_drop() {
        let x = [0.0, 3.0, 0.0, 2.0, 0.0, 5.0, 4.0, 4.5, 0.0];
        let peaks = local_maxima(&x);
        assert_eq!(peaks, vec![1, 3, 5, 7]);
        let order = [1, 0, 3, 2];
        assert_eq!(
            thin_by_distance(&peaks, &order, 3.0),
            vec![true, false, true, false]
        );
        assert_eq!(prominence(&x, 5), 5.0);
        assert_eq!(prominence(&x, 7), 0.5);
    }
}
