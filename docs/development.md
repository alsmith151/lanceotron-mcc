# Development

Install the core package and its runtime dependencies with uv:

```bash
uv sync
uv run lanceotron-mcc --help
```

For development, sync the optional `dev` extra. It includes pytest, Ruff and the
optional legacy-method dependencies:

```bash
uv sync --extra dev
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv build
```

To use the original heuristic without the full development extra, sync its
dependencies and pass the extra when running it:

```bash
uv sync --extra legacy
uv run --extra legacy lanceotron-mcc call-legacy-peaks --help
```

Extras are selected per `uv run` invocation, so include `--extra legacy` on each
legacy command. See [CLI reference](cli.md) for command options and
[methods](methods.md) for detection behavior and interpretation.

## Dependency locking

`uv sync` and `uv run` resolve dependencies and create `uv.lock` when it is absent.
No lockfile is currently checked into this repository, so a fresh checkout does
not yet share a pinned resolution. Review and commit a generated lockfile if the
project adopts one; only then use `--locked` in CI to reject accidental updates.

After changing dependency declarations, run `uv lock` and sync the required extra
again. Use `uv lock --upgrade` only for an intentional dependency refresh, then
rerun the checks above. The `dev` and `legacy` sets are optional extras, so use
`--extra dev` and `--extra legacy`, not dependency-group flags. See the
[uv syncing guide](https://docs.astral.sh/uv/concepts/projects/sync/) for details.

## Validation limits

The historical `test_peak_calling.ipynb` notebook preserves exploratory results;
it is excluded from maintained examples and automated checks. There is no
independent gold standard in this repository. Count scores need calibration on
held-out biological replicates, downsampling, localization and call-count
stability. Replicate agreement alone does not establish accuracy.
