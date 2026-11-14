# mcc-peaks

Distance-aware MCC peak calling from BigWig signal or SeqNado reporter events,
with replicate support and reusable local caches, plus base-pair ligation-junction
footprinting at capture and reporter sites.

Experimental. Formerly `lanceotron-mcc`. Hot loops run in compiled Rust
(parallel with `--threads`) and `.pairs` files are parsed as columns with polars.

A container with the plotting extra is published on release:

```bash
docker run --rm -v "$PWD:/data" ghcr.io/alsmith151/mcc-peaks:latest mcc-peaks --help
```

## Install and run with uv

Use Python 3.10 or newer and [install uv](https://docs.astral.sh/uv/getting-started/installation/).
Building from source also needs a [Rust toolchain](https://rustup.rs) (1.85+);
release wheels and the container need neither. From this repository's root:

```bash
uv sync
uv run mcc-peaks --help
```

`uv sync` creates the project's `.venv`, compiles the Rust extension with maturin
and installs this package in editable mode; editing Rust sources triggers a rebuild
on the next `uv sync`/`uv run`.
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

The original torch-based heuristic (`call-legacy-peaks`) has been removed; use
`call-bigwig-peaks`.

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
cargo test
uv build
```

Keep `--extra dev` on development run commands. For offline compute
nodes, provision the environment before submission and run its installed binary;
see [the HPC guide](docs/hpc.md).
