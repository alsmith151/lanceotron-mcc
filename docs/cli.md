# CLI and pipeline usage

Run these examples from the repository root after `uv sync`. Use
`uv run mcc-peaks --help` or append `--help` to any command to inspect its
options. Each command writes a BED plus score/provenance sidecars.

## BigWig calling

```bash
uv run mcc-peaks call-bigwig-peaks \
  --bigwigs replicate1.bigWig replicate2.bigWig replicate3.bigWig \
  --viewpoint-bed viewpoints.bed --viewpoint BRD4 \
  --exclude-bed exclusion_regions.bed --bait-padding 100 \
  --output peaks.bed --memory-budget-mb 1024 --threads 2
```

For a single sample, supply one path to `--bigwigs`. Multiple paths must represent
separate biological replicates. Input files cannot alias the same physical file.
A strict majority must meet the minimum enrichment threshold; two replicates
therefore both need support. A single track has no reproducibility claim.

Viewpoints use zero-based, half-open BED intervals. Multiple oligos may share a
BED4 group name. Names ending in `-chr…-start-end` can be selected by their common
prefix. `--viewpoint` is required when the BED contains multiple named groups;
each call handles one group on one chromosome. Bait oligos and optional padding
are excluded, together with `--exclude-bed`. Empty exclusion BEDs are accepted.
Set padding according to the assay's bait-proximal artifact region; it defaults
to zero.

See [calling methods](methods.md) for normalization, background fitting and
replicate support. Each replicate must have positive usable cis signal; empty
replicates are rejected rather than silently removed.

## Shared detection controls

| Option | Default | Meaning |
|---|---|---|
| `--window-sizes` | `200 400 600` | Foreground smoothing widths in bp |
| `--background-windows` | `10000 100000 1000000` | Local background widths in bp |
| `--thresholds` | `2 4 6` | Enrichment multipliers |
| `--support-fraction` | `1/3` | Fraction of parameter combinations passing at each position |
| `--min-peak-size` | `50` | Minimum retained width in bp |
| `--max-peak-size` | `500` | Maximum resolved width in bp |

The default support fraction is one third (9 of the default 27 combinations).
Set fractions as decimals, for example `--support-fraction 0.5`.

## Reporter-count input

Create a tab-separated manifest. Paths are relative to the manifest directory:

```text
sample	viewpoint	pairs
rep1	BRD4	mcc/replicates/rep1/ligation_junctions/BRD4.pairs.gz
rep2	BRD4	mcc/replicates/rep2/ligation_junctions/BRD4.pairs.gz
rep3	BRD4	mcc/replicates/rep3/ligation_junctions/BRD4.pairs.gz
```

```bash
uv run mcc-peaks call-count-peaks \
  --manifest samples.tsv --viewpoint BRD4 --viewpoint-bed viewpoints.bed \
  --chrom-sizes hg38.chrom.sizes --exclude-bed exclusion_regions.bed \
  --bin-size 50 --min-read-count 5 --memory-budget-mb 2048 \
  --output count_peaks.bed
```

Input uses the standard tab-separated `.pairs` columns:

```text
readID chr1 pos1 chr2 pos2 strand1 strand2
```

These are the first seven columns, not an MCCNado-specific format. Coordinates
are one-based. Standard `#` headers and additional columns (for example
`pair_type`) are supported; extra annotations are not used for scoring.
Headerless files produced by MCCNado are also accepted. See the
[4DN pairs specification](https://github.com/4dn-dcic/pairix/blob/master/pairs_format_specification.md)
and [pairtools format guide](https://pairtools.readthedocs.io/en/latest/formats.html).

For ordinary pairs, the adapter identifies the capture by overlap with the
selected viewpoint BED, regardless of endpoint order. The opposite end is the
reporter, so upper-triangular/flipped pairs need no conversion. Contacts touching
neither selected viewpoint interval, contacts with both ends in the selected
viewpoint, and unmapped pairs are skipped and recorded in QC. Trans contacts are
recorded in QC and do not enter cis calling.

### MCCNado reporter-name annotations

MCCNado uses the standard pairs container and retains reporter metadata in
`readID`, with this structure:

```text
parent_id__viewpoint-chrom-start-end__start_or_end__read_number__flashed_status
```

The name identifies the parent read, capture oligo and which side of the original
read contains the viewpoint. The embedded genomic interval describes the capture
oligo; it is not the reporter's genomic junction coordinate. MCCNado combines
this side information with alignments and strands upstream, then writes the
calculated junction positions into `pos1`/`pos2`.

The adapter bins the supplied junction positions without recalculating them from
read names or replacing them with generic alignment starts. Recognized MCCNado
annotations are used to validate the selected capture oligo and resolve either
endpoint order. A junction at the encoded oligo's end boundary is accepted for
these annotations; ordinary mapped-point overlap remains half-open. Contacts
whose name identifies another capture oligo are skipped and recorded in QC.
Parent IDs are not collapsed into molecule counts.

Upstream mapping-quality filtering and deduplication remain the pipeline's
responsibility; optional `pair_type` values are not an implicit filter. Counts
represent reporter events, not unique molecules; several reporters may originate
from one molecule.

See [the experimental count model](methods.md#experimental-count-statistics)
for score interpretation. `--alpha` annotates bins and does not filter peaks.

## Junction footprints

`footprint` reports the exact base-pair positions of ligation junctions instead of
read pile-ups. It reads the same manifest and MCCNado `.pairs` files as
`call-count-peaks`:

```bash
uv sync --extra plot
uv run --extra plot mcc-peaks footprint \
  --manifest manifest.tsv --viewpoint BRD4 \
  --viewpoint-bed viewpoints.bed --chrom-sizes genome.chrom.sizes \
  --peaks brd4_peaks.bed --output footprints/BRD4
```

`--peaks` takes BED3/BED4 reporter regions, such as the output of
`call-count-peaks`; names come from column 4, otherwise `peak_1`, `peak_2`, ….
Without it the capture-oligo tiles are profiled. Each reporter region is widened
by `--reporter-flank` (default 100 bp). A capture junction must fall within
`--capture-flank` (default 50 bp) of an oligo to enter the contact map.

Inputs must come from a current MCCNado release, where `pos1` is the reporter
ligation junction and `pos2` the capture-side junction, and the read name carries
the reporter side (`start`/`end`). Older releases wrote the far end of the reporter
segment into `pos2`; with such files the contact map is empty and the command
warns. Regenerate the ligation junctions in that case. Flipped (`#shape`) pairs
lose MCC endpoint order and are rejected unless `--orientation-source unknown`.
Contacts without side or strand information stop the run unless
`--allow-unoriented` is given, in which case they are counted as `Unknown`.

| Output | Content |
|---|---|
| `PREFIX.profiles.tsv.gz` | One row per sample, region and base: `Upstream`, `Downstream`, `Unknown`, `Total` junction ends |
| `PREFIX.contacts.tsv` | Sparse single-base map: `CapturePosition`, `ReporterPosition`, the fragment direction at each end, and `Count` |
| `PREFIX.regions.tsv` | Profiled capture (`capture:oligo_N`) and reporter (`reporter:NAME`) windows after flanking |
| `PREFIX.metrics.json` | Parameters, input identities, versions and per-sample QC counters and warnings |
| `PREFIX.pdf` | One page per reporter region (up to `--max-panels`); omitted with `--no-plot` |

Positions are zero-based. `Upstream` or `Downstream` gives the side of the
junction on which the ligated fragment, and therefore the protecting protein,
lies. Exact repeated read-ID/endpoint rows are collapsed by default
(`--no-deduplicate` keeps them); parent reads are not collapsed. See
[junction footprints](methods.md#junction-footprints) for interpretation.
`--max-profile-bases` and `--max-contacts` bound memory use.

## SeqNado integration

Use existing replicate BigWigs for signal calling. For count mode, SeqNado already
produces `mcc/replicates/{sample}/ligation_junctions/{viewpoint}.pairs.gz`; supply
these through the manifest above with the pipeline's viewpoint BED, exclusion
BED and chromosome sizes.

The BigWig invocation in a Snakemake shell block becomes:

```text
mcc-peaks call-bigwig-peaks \
  --bigwigs {input.bigwigs:q} --viewpoint-bed {input.viewpoints:q} \
  --viewpoint {wildcards.viewpoint_group:q} --exclude-bed {input.exclusions:q} \
  --threads {threads} --output {output.peaks:q}
```

Declare all replicate files and metadata as inputs, and the required artifacts as
outputs. Use `call-count-peaks` for the manifest-based workflow. All commands are
CPU-only; `--threads` parallelises the compiled vote kernel. SeqNado itself is
not modified here.

The shell example invokes the binary directly in a prepared workflow environment
or container. Prepare that environment with uv before submitting jobs; see
[HPC and Ceph](hpc.md). Do not sync a new environment for each peak-calling job.

## Outputs and diagnostics

For `peaks.bed`:

| File | Contents |
|---|---|
| `peaks.bed` | Resolved calls, BED3 |
| `peaks.bed.unresolved.bed` | Broad unresolved regions, BED3 |
| `peaks.bed.tsv` | All retained regions, statistics and support |
| `peaks.bed.json` | Parameters, original input identities, versions, artifact paths and timings |
| `peaks.bed.bins.tsv.gz` | Count mode: all bins, eligibility, raw counts, expected values and tests |

Coordinates are int64, zero-based and half-open. Count-mode summits identify the
start of the strongest bin. `Signal`/`Sum` use normalized units across replicates;
`Count` is the pooled raw reporter-event count. Process peak memory is a lifetime
high-water mark, and read counters measure logical requests, not physical Ceph
traffic.

Output paths are checked against inputs, including symlink/hardlink aliases and
sidecars. Artifacts are staged before replacement; renames are atomic per file,
not across the whole bundle, and provenance is published last. Reusing a count
output stem for signal calling removes its old count-bin sidecar. Original input
identities are checked again before publication. Errors are concise by default;
`--debug` includes diagnostic logs and runtime tracebacks, and `--quiet` suppresses
informational messages.

## CLI migration

The old `call-mcc-peaks` command, the torch-based `call-legacy-peaks` command
and compatibility aliases are removed. Use `call-bigwig-peaks --bigwigs ...
--output ...` or `call-count-peaks`. Shared flags are `--thresholds`,
`--background-windows`, `--support-fraction` and `--threads`; the absolute vote
threshold and BigWig read-estimation flags are no longer exposed.
