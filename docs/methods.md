# Calling methods and interpretation

For commands and input formats, see [CLI usage](cli.md). These methods report
heuristic enrichment and replicate support; raw-count significance is experimental.

## BigWig background and replicate support

Background profiles use 200 bp bins by default (`--background-bin-size`). On each
viewpoint arm, log-distance bands pool observations to fit a non-increasing mean
curve with iterative strong-enrichment masking. Profile bins touching exclusions
are omitted. The larger of this curve and each local background estimate is used
for candidate generation. This is an enrichment heuristic, not a probability model
for BigWig values. Negative/difference tracks are unsupported.

Replicates contribute equally after usable cis-integral normalization to the
median depth, excluding bait and blacklist intervals. Peak scores are evaluated
separately in each replicate. BigWig coverage is always treated as signal;
unnormalized coverage is not converted into independent read counts.

## Consensus and region refinement

The defaults require support from 9 of 27 combinations. Support remains a fraction
when the grid changes. Region refinement raises the support threshold to separate
broad runs. Distance-aware methods also split substantial internal signal valleys
in regions up to 16,384 bp; longer or flat regions that cannot be resolved are
reported separately. These decisions do not depend on chunk size or memory budget.

## Experimental count statistics

For each contact with exactly one end assigned to the selected capture oligos,
the other end contributes one reporter event. Endpoint order does not assign
biological roles. Standard contacts use BED overlap; recognized MCCNado read IDs
also provide capture-oligo and junction-side context, including oligo-end boundary
handling. The provided pairs positions remain the counting coordinates. Contacts with neither or both ends in the selected viewpoint
and unmapped contacts do not contribute; trans contacts are reported in QC only.

Counts are aggregated into 50 bp bins by default. Background-profile widths are
rounded up to whole count bins; BigWig profile widths use the requested bp value.
Expected counts depend on
viewpoint distance, sample depth and partial-bin exposure. NB2 dispersion uses
moments after strong-enrichment masking; inadequate information falls back to
Poisson, which is not guaranteed conservative. Fisher's method combines separate
biological-replicate upper-tail tests. BH adjustment covers every eligible cis bin
for the viewpoint before candidate selection. Bins touching exclusions are
ineligible. Do not treat technical replicates as independent biological samples.

**Count statistics remain experimental.** Background/dispersion estimates,
within-molecule dependence and replicate assumptions need real-data calibration.
`--alpha 0.05` annotates significant bins; it does not filter peak calls. Peak
selection uses enrichment, consensus, majority replicate support and minimum
pooled **raw** reporter events. `MinimumBinQ` summarizes a peak's bins and is not
a peak-level FDR. Direct BAM and Cooler adapters are not included.

## Junction footprints

MNase digests DNA that is not protected by bound protein, so the ligation
junctions of an MCC fragment sit at the edges of the protected region. Counting
junction ends at single-base resolution, instead of read coverage, gives a
profile similar to DNase I footprinting: a protected binding site shows a
depletion of junctions flanked by peaks of junctions.

Each junction end is assigned the direction in which its ligated fragment
extends. For MCCNado reporters this follows from the reporter side in the read
name (`start` is a segment to the right of the viewpoint in the read, `end` to the
left) and the alignment strand; the capture end takes the complementary
orientation. `Downstream` junctions are the left boundary of a protected fragment
and `Upstream` junctions its right boundary, so plotting them separately places
the protein between the two peaks rather than over the whole fragment.

The contact map pairs each reporter junction with the capture junction from the
same read. Arrows show the fragment direction at both ends, which identifies
the protein at the capture site and the one at the reporter site that gave rise
to each ligation (for example the two interacting CTCF sites around a protected
central motif).

Profiles and maps are descriptive. No footprint calling or significance test is
applied; counts are junction events, not unique molecules, and depend on MNase
digestion level. Compare samples at matched depth and digestion.

## Validation limits

There is no independent gold standard in this repository. Evaluate held-out
biological replicates, downsampling, localization and call-count stability before
treating significance as calibrated. Replicate agreement alone is not accuracy.
Synthetic regression tests and local runtime measurements do not establish
biological sensitivity, specificity or Ceph performance.
