# Development

Install the package and its runtime dependencies with uv. Building from source
compiles the Rust extension with [maturin](https://www.maturin.rs), so a Rust
toolchain (1.85+, e.g. via [rustup](https://rustup.rs)) is required:

```bash
uv sync
uv run mcc-peaks --help
```

For development, sync the optional `dev` extra (pytest, Ruff, maturin and
matplotlib):

```bash
uv sync --extra dev
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --release
uv build
```

## Code layout

| Path | Contents |
|---|---|
| `mcc_peaks/` | Python package: CLI, I/O, orchestration, statistics |
| `mcc_peaks/api/pairs.py` | Columnar `.pairs` reader (polars), exact per-line validation |
| `src/lib.rs`, `src/bindings.rs` | `mcc_peaks._core`: pyo3 bindings over the kernels |
| `src/kernels/` | Python-free Rust algorithms with unit tests |

`src/kernels/` modules:

| Module | Algorithm |
|---|---|
| `consensus` | Background x window x threshold enrichment votes, parallel over chunks |
| `signal` | BigWig value storage and replicate-mean prefix sums |
| `peaks` | Local maxima with distance and prominence filters (valley splitting) |
| `footprint` | Junction-end profiles and capture x reporter contacts |

The bindings borrow NumPy arrays (including memory maps) without copying,
release the GIL and run parallel work on rayon pools sized by `--threads`.
Parallel kernels write disjoint output slices only. After editing Rust code,
the next `uv sync` or `uv run` rebuilds the extension (`[tool.uv] cache-keys`).

`footprint` PDFs need the `plot` extra (matplotlib):
`uv sync --extra plot` and `uv run --extra plot mcc-peaks footprint ...`, or
pass `--no-plot` for tables only. The `dev` extra includes matplotlib. See [CLI reference](cli.md) for command options and
[methods](methods.md) for detection behavior and interpretation.

## Dependency locking

`uv.lock` is committed, so every checkout shares one pinned resolution, including
all optional extras. Use `uv sync --locked` (and `uv lock --check`) in CI to reject
a lockfile that no longer matches `pyproject.toml`. Runtime dependencies are
declared in `pyproject.toml`; Rust dependencies are pinned in `Cargo.lock`.

After changing dependency declarations, run `uv lock` and sync the required extra
again. Use `uv lock --upgrade` only for an intentional dependency refresh, then
rerun the checks above. The `dev` and `plot` sets are optional extras, so use
`--extra dev` and `--extra plot`, not dependency-group flags. See the
[uv syncing guide](https://docs.astral.sh/uv/concepts/projects/sync/) for details.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on pushes to `main`:

| Job | Checks |
|---|---|
| `lint` | `uv lock --check`, `ruff check`, `ruff format --check`, `cargo fmt`, `cargo clippy` |
| `rust` | `cargo test` for the Rust kernels |
| `test` | full pytest suite with the `dev` extra on Python 3.10–3.13 (Linux) and 3.12 (macOS) |
| `core-install` | CLI runs without optional extras |
| `wheels` | abi3 wheels for Linux x86_64/aarch64 (manylinux) and macOS arm64/x86_64 |
| `sdist` | source distribution and `twine check` |
| `docker` | image builds and smoke-tests (not pushed) |

## Releasing

Releases are cut from `main` by pushing a version tag:

1. Bump `version` in `pyproject.toml`, run `uv lock`, and merge to `main`.
2. Tag and push: `git tag v0.2.0 && git push origin v0.2.0`.

`.github/workflows/release.yml` then verifies the tag matches the
`pyproject.toml` version, reruns CI, and publishes:

- the sdist and per-platform wheels to PyPI (trusted publishing, no stored token);
- a GitHub release with generated notes and the built distributions;
- `ghcr.io/alsmith151/mcc-peaks:<version>`, `:<major>.<minor>` and `:latest`
  (pre-release tags such as `v0.2.0-rc1` skip `latest` and are marked pre-release).

One-time setup: on PyPI, add a trusted publisher for this repository with
workflow `release.yml` and environment `pypi`, and create the `pypi` environment
under the repository's GitHub settings. After the first image push, make the ghcr
package public if it should be pullable without authentication.

## Container image

The image includes the `plot` extra; the Rust extension is compiled in the
builder stage, so the runtime image carries no toolchain:

```bash
docker run --rm -v "$PWD:/data" ghcr.io/alsmith151/mcc-peaks:latest \
  mcc-peaks call-bigwig-peaks --help
apptainer exec docker://ghcr.io/alsmith151/mcc-peaks:0.1.0 \
  mcc-peaks footprint --help
```

The working directory is `/data`, and there is no entrypoint, so workflow managers
pass the full command. Build without matplotlib with
`docker build --build-arg EXTRAS="" .`.

## Validation limits

The historical `test_peak_calling.ipynb` notebook preserves exploratory results;
it is excluded from maintained examples and automated checks. There is no
independent gold standard in this repository. Count scores need calibration on
held-out biological replicates, downsampling, localization and call-count
stability. Replicate agreement alone does not establish accuracy.
