# mcc-peaks

Distance-aware MCC peak calling from BigWig signal or SeqNado reporter events,
with replicate support and reusable local caches, plus base-pair ligation-junction
footprinting at capture and reporter sites.

## Install and run with uv

Use Python 3.10 or newer and [install uv](https://docs.astral.sh/uv/getting-started/installation/).
From this repository's root:

```bash
uv sync
uv run mcc-peaks --help
```

`uv sync` creates the project's `.venv` and installs this package in editable mode.
`uv run` selects that environment without requiring shell activation. The committed
`uv.lock` pins the resolution; see [development](docs/development.md) for locking and
optional extras.

## Call BigWig peaks

```bash
uv run mcc-peaks call-bigwig-peaks \
  --bigwigs replicate1.bigWig replicate2.bigWig replicate3.bigWig \
  --viewpoint-bed viewpoints.bed --viewpoint BRD4 \
  --output peaks.bed
```

Supply one track for single-sample calling, or separate biological replicate tracks
for majority support. Viewpoint BEDs and optional exclusions use zero-based,
half-open coordinates. Outputs include BED3, a score TSV, unresolved regions and
provenance JSON.

| Command | Input and purpose |
|---|---|
| `call-bigwig-peaks` | BigWig signal; distance-aware enrichment and replicate support |
| `call-count-peaks` | Standard `.pairs` contacts; experimental bin statistics |
| `footprint` | MCCNado ligation junctions; base-pair junction footprints and capture × reporter maps (PDF needs the `plot` extra) |
| `call-legacy-peaks` | Original multi-window heuristic; requires the `legacy` extra |

For the original heuristic, include its extra when syncing and running:

```bash
uv sync --extra legacy
uv run --extra legacy mcc-peaks call-legacy-peaks --help
```

Count statistics remain experimental; adjusted bin values are not peak-level FDR.
See [calling methods](docs/methods.md) before interpreting scores.

## Documentation

- [CLI, input formats, outputs and SeqNado integration](docs/cli.md)
- [Calling methods and statistical interpretation](docs/methods.md)
- [HPC, Ceph, memory and caching](docs/hpc.md)
- [Development, tests, builds and dependency locking](docs/development.md)

## Development quick start

```bash
uv sync --extra dev
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv build
```

The `dev` extra includes the optional legacy dependencies needed for the complete
test suite. Keep `--extra dev` on development run commands. For offline compute
nodes, provision the environment before submission and run its installed binary;
see [the HPC guide](docs/hpc.md).
