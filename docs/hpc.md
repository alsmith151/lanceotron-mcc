# HPC use

Prepare the environment with `uv sync` on a cluster login/build node that has
package-index access, a Rust toolchain (for source builds), and matches the
compute nodes' OS, architecture and Python runtime. Alternatively use a release
wheel or the container, which need no toolchain. Install at its final path; virtual environments are not portable between
workstations and cluster nodes.

In each job, activate that environment or call its installed executable directly.
From the project checkout (use an absolute executable path from other directories):

```bash
.venv/bin/mcc-peaks call-bigwig-peaks \
  --bigwigs rep1.bigWig rep2.bigWig --viewpoint-bed viewpoints.bed \
  --viewpoint BRD4 --scratch-dir "$TMPDIR" --output peaks.bed --threads 2
```

Compute nodes without network access should not run `uv sync`; use the prepared
environment as-is. If the cluster has persistent local staging for uv's package
cache, set `UV_CACHE_DIR` to that provisioned path while preparing or updating
the environment. Do not point it automatically at job-ephemeral scratch.

## Storage and resources

- `--scratch-dir "$TMPDIR"` stages BigWigs and holds temporary signal/vote arrays.
  Reserve space for compressed inputs and decoded data. Temporary arrays are
  removed after the run.
- `--cache-dir` enables reusable array caches; it is distinct from uv's package
  cache (`UV_CACHE_DIR`). Array caches persist until removed by the user or
  scheduler and do not have built-in quota or eviction.
- `--threads` sets the compiled vote kernel's worker pool (chunks run in
  parallel). `.pairs` decompression runs on a background thread and parsing is
  multithreaded by polars; BigWig decoding remains serialized.
- `--chunk-size` (default 8 Mb) caps bases per vote chunk; the memory budget
  lowers it further when needed, accounting for one chunk plus halo per thread.
- `--memory-budget-mb` guides array cache, chunk and worker sizing. It is not a
  hard RSS limit; mapped arrays, Python and libraries add overhead. Methods can
  reject budgets too small for required halos, profile fitting or count tables.
- Keep inputs immutable during a run. Input identity checks use paths and file
  metadata rather than hashing large remote files.

See [CLI reference](cli.md) for all options and [methods](methods.md) for the
statistical behavior and limitations.

## I/O benchmark

Run a comparison on the target HPC filesystem:

```bash
uv run python -m benchmarks.benchmark_pipeline --bigwig sample.bigWig \
  --viewpoint-bed viewpoints.bed --viewpoint BRD4 \
  --scratch-dir "$TMPDIR" --output benchmark.json
```

On offline compute nodes, run `.venv/bin/python -m benchmarks.benchmark_pipeline`
with the same arguments from the prepared checkout instead of allowing uv to
synchronize the environment.

The harness compares direct, staged and cached runs and verifies call
equivalence. It does not clear filesystem or OS caches. Omit the input and
viewpoint arguments for a synthetic smoke benchmark; use job or Ceph monitoring
for physical traffic metrics.

## Pairs and footprint benchmark

Time `.pairs` parsing, count binning and footprinting on real data:

```bash
uv run python -m benchmarks.benchmark_pairs --manifest manifest.tsv \
  --viewpoint BRD4 --viewpoint-bed viewpoints.bed --chrom-sizes genome.sizes \
  --output pairs-benchmark.json
```

Without inputs it generates synthetic MCCNado-style pairs (`--rows`, default
2,000,000). On a laptop-class CPU, 5 million gzipped rows parse at roughly
3 million rows per second.
