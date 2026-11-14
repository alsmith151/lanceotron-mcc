"""Validate and publish peak artifacts without truncating input files."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import source_identity


@dataclass(frozen=True)
class OutputPaths:
    bed: Path
    table: Path
    unresolved: Path
    provenance: Path
    bins: Path

    @classmethod
    def from_bed(cls, outfile: str | Path) -> OutputPaths:
        bed = Path(outfile).expanduser().absolute()
        return cls(
            bed,
            Path(f"{bed}.tsv"),
            Path(f"{bed}.unresolved.bed"),
            Path(f"{bed}.json"),
            Path(f"{bed}.bins.tsv"),
        )

    def artifacts(self, include_bins: bool = False) -> tuple[Path, ...]:
        paths = (self.bed, self.table, self.unresolved)
        return (*paths, self.bins, self.provenance) if include_bins else (*paths, self.provenance)


def preflight_outputs(outfile: str | Path, inputs: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Snapshot input identities and reject output/input aliases before computing."""
    identities = [source_identity(path) for path in inputs]
    source_paths = {Path(item["path"]) for item in identities}
    inode_ids = {(path.stat().st_dev, path.stat().st_ino) for path in source_paths}
    # Reserve the whole output namespace, including a possible old count table.
    for target in OutputPaths.from_bed(outfile).artifacts(include_bins=True):
        if target.is_dir():
            raise ValueError(f"Output path is a directory: {target}")
        if target.resolve() in source_paths:
            raise ValueError(f"Output would overwrite an input file: {target}")
        if target.exists():
            stat = target.stat()
            if (stat.st_dev, stat.st_ino) in inode_ids:
                raise ValueError(f"Output aliases an input file: {target}")
    return identities


def _versions() -> dict[str, str]:
    result = {}
    for name in ("mcc_peaks", "pybigtools", "numpy", "scipy", "pandas", "torch"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "uninstalled"
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported provenance value: {type(value).__name__}")


def write_outputs(
    peaks: pd.DataFrame,
    outfile: str | Path,
    *,
    method: str,
    parameters: dict,
    metrics: dict,
    input_identities: list[dict],
    bins: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Stage every artifact before replacing outputs; publish provenance last.

    Renames are atomic per file, not a transaction over the complete bundle.
    """
    source_paths = [item["path"] for item in input_identities]
    current = preflight_outputs(outfile, source_paths)
    if current != input_identities:
        raise ValueError(
            "An input changed while peak calling was running; outputs were not published"
        )
    paths = OutputPaths.from_bed(outfile)
    frame = peaks.copy()
    required = ["Chromosome", "Start", "End"]
    if frame.empty:
        frame = frame.reindex(columns=list(dict.fromkeys([*required, *frame.columns])))
    elif not set(required).issubset(frame):
        raise ValueError("Peak table must contain Chromosome, Start and End")
    if frame.Chromosome.isna().any() or frame.Chromosome.astype(str).str.strip().eq("").any():
        raise ValueError("Peak chromosome names must be nonempty")
    for name in ("Start", "End"):
        values = pd.to_numeric(frame[name], errors="raise")
        if (
            not np.all(np.isfinite(values))
            or (values < 0).any()
            or (values % 1 != 0).any()
            or (values > np.iinfo(np.int64).max).any()
        ):
            raise ValueError(f"Peak {name} coordinates must be nonnegative int64 values")
        frame[name] = values.astype(np.int64)
    if (frame.End <= frame.Start).any():
        raise ValueError("Peak intervals must have positive width")
    if "Unresolved" not in frame:
        frame["Unresolved"] = False
    if not frame.Unresolved.dropna().map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("Unresolved flags must be boolean")
    unresolved = frame.Unresolved.fillna(False).astype(bool)
    provenance = {
        "method": method,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters,
        "inputs": input_identities,
        "versions": _versions(),
        "metrics": metrics,
        "resolved_peaks": int((~unresolved).sum()),
        "unresolved_regions": int(unresolved.sum()),
        "artifacts": {path.name: str(path) for path in paths.artifacts(bins is not None)},
        "statistical_status": (
            "experimental; bin adjusted values are not peak-level FDR"
            if method == "counts"
            else "heuristic enrichment"
        ),
    }
    # Validate serialization before creating or replacing any final artifact.
    metadata = json.dumps(provenance, indent=2, default=_json_default, allow_nan=False) + "\n"
    # A previous count run may have used this output stem. Remove only its
    # known bin sidecar after the replacement bundle has been prepared.
    previous_count_output = False
    try:
        previous_count_output = json.loads(paths.provenance.read_text()).get("method") == "counts"
    except (OSError, ValueError, AttributeError):
        pass
    paths.bed.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=paths.bed.parent, prefix=".mcc-peaks-") as staging:
        staged = {
            target: Path(staging) / target.name for target in paths.artifacts(bins is not None)
        }
        frame.loc[~unresolved, required].to_csv(
            staged[paths.bed], sep="\t", header=False, index=False
        )
        frame.loc[unresolved, required].to_csv(
            staged[paths.unresolved], sep="\t", header=False, index=False
        )
        frame.to_csv(staged[paths.table], sep="\t", index=False)
        if bins is not None:
            bins.to_csv(staged[paths.bins], sep="\t", index=False)
        staged[paths.provenance].write_text(metadata)
        if preflight_outputs(outfile, source_paths) != input_identities:
            raise ValueError("An input changed while staging outputs; outputs were not published")
        for target, temporary in staged.items():
            if target == paths.provenance and bins is None and previous_count_output:
                paths.bins.unlink(missing_ok=True)
            os.replace(temporary, target)
    return provenance
