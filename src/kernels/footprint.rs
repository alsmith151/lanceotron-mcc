//! Junction-end profiles and capture x reporter contacts for footprinting.

/// Region lookup over per-chromosome entries sorted by start.
///
/// `max_ends` holds running maxima of ends within each chromosome, so all
/// regions containing a point are found by walking left from the last start
/// at or before it, even when regions overlap.
pub struct Regions<'a> {
    pub offsets: &'a [i64],
    pub starts: &'a [i64],
    pub ends: &'a [i64],
    pub max_ends: &'a [i64],
    pub ids: &'a [i64],
}

impl Regions<'_> {
    pub fn hits(&self, chrom: i64, pos: i64, out: &mut Vec<i64>) {
        out.clear();
        if chrom < 0 || chrom as usize + 1 >= self.offsets.len() {
            return;
        }
        let lo = self.offsets[chrom as usize] as usize;
        let hi = self.offsets[chrom as usize + 1] as usize;
        let mut i = lo + self.starts[lo..hi].partition_point(|&s| s <= pos);
        while i > lo {
            i -= 1;
            if self.max_ends[i] <= pos {
                break;
            }
            if pos < self.ends[i] {
                out.push(self.ids[i]);
            }
        }
    }
}

/// One contact endpoint: chromosome code, position and direction code.
pub type End = (i64, i64, i64);

/// Profile layout: region id -> first slot and region start; direction code
/// -> profile column. Slot of (region, pos, direction) is
/// `(base[region] + pos - start[region]) * 3 + column[direction]`.
pub struct Layout<'a> {
    pub base: &'a [i64],
    pub region_start: &'a [i64],
    pub column: &'a [i64],
}

impl Layout<'_> {
    #[inline(always)]
    fn slot(&self, region: i64, end: End) -> usize {
        let g = region as usize;
        ((self.base[g] + end.1 - self.region_start[g]) * 3 + self.column[end.2 as usize]) as usize
    }
}

/// Counts produced by [`accumulate`].
#[derive(Debug, Default, PartialEq)]
pub struct Tally {
    /// Contact rows emitted (may exceed the buffer; the caller checks).
    pub contacts: usize,
    /// Junction ends counted in reporter regions.
    pub junction_ends: i64,
    /// Selected contacts with an endpoint in a capture window.
    pub mapped: i64,
}

/// Add selected contacts to `profile` and emit capture x reporter rows.
///
/// Each endpoint inside a reporter region counts in that region's profile.
/// For each reporter region hit, a contact needs its other end in a capture
/// window; when both orientations qualify, the smaller (capture chrom, pos,
/// reporter chrom, pos, directions, end) tuple wins, and codes are ranked by
/// name so this matches comparing names. The chosen capture end counts once
/// per capture window. `contacts` rows are (row, region, capture chrom,
/// capture pos, reporter chrom, reporter pos, capture direction, reporter
/// direction), 8 values each.
pub fn accumulate(
    rows: &[[End; 2]],
    reporter: &Regions<'_>,
    capture: &Regions<'_>,
    layout: &Layout<'_>,
    profile: &mut [i64],
    contacts: &mut [i64],
) -> Tally {
    let capacity = contacts.len() / 8;
    let (mut hits0, mut hits1, mut caps0, mut caps1) = (vec![], vec![], vec![], vec![]);
    let mut union: Vec<i64> = vec![];
    let mut contributions: Vec<(i64, usize)> = vec![];
    let mut tally = Tally::default();
    for (row, ends) in rows.iter().enumerate() {
        reporter.hits(ends[0].0, ends[0].1, &mut hits0);
        reporter.hits(ends[1].0, ends[1].1, &mut hits1);
        for &g in &hits0 {
            profile[layout.slot(g, ends[0])] += 1;
        }
        for &g in &hits1 {
            profile[layout.slot(g, ends[1])] += 1;
        }
        tally.junction_ends += (hits0.len() + hits1.len()) as i64;
        capture.hits(ends[0].0, ends[0].1, &mut caps0);
        capture.hits(ends[1].0, ends[1].1, &mut caps1);
        union.clear();
        union.extend_from_slice(&hits0);
        union.extend(hits1.iter().filter(|g| !hits0.contains(g)));
        contributions.clear();
        let mut mapped = false;
        for &g in &union {
            let orient = |capture_end: usize| {
                let (c, r) = (ends[capture_end], ends[1 - capture_end]);
                [c.0, c.1, r.0, r.1, c.2, r.2, capture_end as i64]
            };
            let mut best: Option<[i64; 7]> = None;
            if !caps1.is_empty() && hits0.contains(&g) {
                best = Some(orient(1));
            }
            if !caps0.is_empty() && hits1.contains(&g) {
                let candidate = orient(0);
                if best.is_none_or(|b| candidate < b) {
                    best = Some(candidate);
                }
            }
            let Some(best) = best else { continue };
            if tally.contacts < capacity {
                let at = tally.contacts * 8;
                contacts[at] = row as i64;
                contacts[at + 1] = g;
                contacts[at + 2..at + 8].copy_from_slice(&best[..6]);
            }
            tally.contacts += 1;
            mapped = true;
            let end = best[6] as usize;
            for &cap in if end == 1 { &caps1 } else { &caps0 } {
                if !contributions.contains(&(cap, end)) {
                    contributions.push((cap, end));
                }
            }
        }
        for &(cap, end) in &contributions {
            profile[layout.slot(cap, ends[end])] += 1;
        }
        tally.mapped += mapped as i64;
    }
    tally
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_regions_are_all_found() {
        let regions = Regions {
            offsets: &[0, 3],
            starts: &[10, 15, 40],
            ends: &[30, 20, 50],
            max_ends: &[30, 30, 50],
            ids: &[0, 1, 2],
        };
        let mut out = vec![];
        regions.hits(0, 16, &mut out);
        assert_eq!(out, vec![1, 0]);
        regions.hits(0, 25, &mut out);
        assert_eq!(out, vec![0]);
        regions.hits(0, 35, &mut out);
        assert!(out.is_empty());
        regions.hits(1, 16, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn contact_maps_reporter_end_to_capture_window() {
        // Region 0: capture window 100..110; region 1: reporter 500..520.
        let capture = Regions {
            offsets: &[0, 1],
            starts: &[100],
            ends: &[110],
            max_ends: &[110],
            ids: &[0],
        };
        let reporter = Regions {
            offsets: &[0, 1],
            starts: &[500],
            ends: &[520],
            max_ends: &[520],
            ids: &[1],
        };
        let layout = Layout {
            base: &[0, 10],
            region_start: &[100, 500],
            column: &[1, 2, 0],
        };
        let mut profile = vec![0i64; 30 * 3];
        let mut contacts = vec![0i64; 8];
        let tally = accumulate(
            &[[(0, 505, 0), (0, 103, 2)]],
            &reporter,
            &capture,
            &layout,
            &mut profile,
            &mut contacts,
        );
        assert_eq!(
            tally,
            Tally {
                contacts: 1,
                junction_ends: 1,
                mapped: 1
            }
        );
        assert_eq!(contacts, vec![0, 1, 0, 103, 0, 505, 2, 0]);
        assert_eq!(profile[(10 + 5) * 3 + 1], 1);
        assert_eq!(profile[3 * 3], 1);
    }
}
