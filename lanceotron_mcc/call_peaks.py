"""Focused entry points for signal, raw counts and the original MCC heuristic."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from cyclopts import App, Group, Parameter
from loguru import logger
from pydantic import ValidationError

from .api.adaptive import AdaptiveParams, call_adaptive_bigwigs, call_count_manifest, read_bed
from .api.counts import load_manifest
from .api.io import peak_memory_mb
from .api.results import preflight_outputs, write_outputs
from .cli_options import (
    BACKGROUND,
    INPUTS,
    LEGACY,
    BackgroundOptions,
    ChunkOptions,
    DetectionOptions,
    RuntimeOptions,
)


def _version() -> str:
    try:
        return version("lanceotron_mcc")
    except PackageNotFoundError:
        return "0.1.0 (source checkout)"


app = App(
    name="lanceotron-mcc",
    version=_version,
    help="Call MCC peaks from BigWig signal or SeqNado reporter counts.",
    help_epilogue="All commands write BED3, a score TSV, unresolved regions and provenance JSON.",
)

PEAK_COMMANDS = Group("Peak calling", sort_key=0)
LEGACY_COMMANDS = Group("Original heuristic", sort_key=1)

Detection = Annotated[DetectionOptions, Parameter(name="*")]
Background = Annotated[BackgroundOptions, Parameter(name="*")]
Runtime = Annotated[RuntimeOptions, Parameter(name="*")]
Chunks = Annotated[ChunkOptions, Parameter(name="*")]
Output = Annotated[
    Path, Parameter(group=INPUTS, help="Output BED path; sidecars are appended to this filename.")
]
ViewpointBed = Annotated[
    Path, Parameter(group=INPUTS, help="BED3/4 capture oligos for one selected viewpoint group.")
]
ViewpointName = Annotated[
    str | None,
    Parameter(
        group=BACKGROUND, help="BED4 group name; required when the BED contains multiple groups."
    ),
]


def _configure_runtime(runtime: RuntimeOptions) -> None:
    if runtime.debug and runtime.quiet:
        raise ValueError("--debug and --quiet are mutually exclusive")
    if runtime.threads < 1:
        raise ValueError("--threads must be a positive integer")
    if not math.isfinite(runtime.memory_budget_mb) or runtime.memory_budget_mb < 16:
        raise ValueError("--memory-budget-mb must be at least 16")
    for name, directory in (("scratch", runtime.scratch_dir), ("cache", runtime.cache_dir)):
        if directory is not None and directory.exists() and not directory.is_dir():
            raise ValueError(f"The {name} directory is a file: {directory}")
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if runtime.debug else "WARNING" if runtime.quiet else "INFO",
        format="<level>{level}</level> | {message}",
    )


def _check_detection(options: DetectionOptions) -> int:
    if not math.isfinite(options.support_fraction) or not 0 < options.support_fraction <= 1:
        raise ValueError("--support-fraction must be in (0, 1]")
    grid_size = (
        len(options.window_sizes) * len(options.background_windows) * len(options.thresholds)
    )
    if not grid_size:
        raise ValueError("Window, background and threshold lists must be nonempty")
    return grid_size


def _adaptive_params(
    detection: DetectionOptions,
    background: BackgroundOptions,
    runtime: RuntimeOptions,
    chunks: ChunkOptions,
    *,
    bin_size: int = 50,
    min_read_count: int = 5,
    alpha: float = 0.05,
) -> AdaptiveParams:
    _check_detection(detection)
    return AdaptiveParams(
        window_sizes=list(detection.window_sizes),
        backgrounds=list(detection.background_windows),
        thresholds=list(detection.thresholds),
        support_fraction=detection.support_fraction,
        min_peak_size=detection.min_peak_size,
        max_peak_size=detection.max_peak_size,
        memory_budget_mb=runtime.memory_budget_mb,
        chunk_size=chunks.chunk_size,
        n_jobs=runtime.threads,
        bait_padding=background.bait_padding,
        background_bin_size=background.background_bin_size,
        bin_size=bin_size,
        min_read_count=min_read_count,
        alpha=alpha,
    )


def _read_regions(
    viewpoint_bed: Path, viewpoint: str | None, background: BackgroundOptions
) -> tuple[dict, dict]:
    viewpoints = read_bed(viewpoint_bed, viewpoint, require_single_group=True)
    exclusions = (
        read_bed(background.exclude_bed, allow_empty=True) if background.exclude_bed else {}
    )
    return viewpoints, exclusions


def _region_files(viewpoint_bed: Path, background: BackgroundOptions) -> list[Path]:
    return [viewpoint_bed, *([background.exclude_bed] if background.exclude_bed else [])]


def _finish(
    peaks: pd.DataFrame,
    output: Path,
    *,
    method: str,
    parameters: dict,
    metrics: dict,
    identities: list[dict],
    runtime: RuntimeOptions,
    bins: pd.DataFrame | None = None,
) -> None:
    parameters = parameters | {"execution": asdict(runtime)}
    metadata = write_outputs(
        peaks,
        output,
        method=method,
        parameters=parameters,
        metrics=metrics,
        input_identities=identities,
        bins=bins,
    )
    logger.info(
        "Wrote {} resolved peaks and {} unresolved regions to {}",
        metadata["resolved_peaks"],
        metadata["unresolved_regions"],
        output,
    )


@app.command(group=PEAK_COMMANDS)
def call_bigwig_peaks(
    *,
    bigwigs: Annotated[
        list[Path],
        Parameter(
            group=INPUTS,
            consume_multiple=True,
            negative=[],
            help="One signal track, or separate biological replicate tracks.",
        ),
    ],
    viewpoint_bed: ViewpointBed,
    output: Output,
    viewpoint: ViewpointName = None,
    detection: Detection = DetectionOptions(),
    background: Background = BackgroundOptions(),
    runtime: Runtime = RuntimeOptions(),
    chunks: Chunks = ChunkOptions(),
) -> None:
    """Call distance-aware peaks from BigWig signal, with optional replicate support.

    BigWig coverage is signal, not independent read counts. Multiple inputs must
    be biological replicates; calls require enrichment in a strict majority.
    """
    _configure_runtime(runtime)
    params = _adaptive_params(detection, background, runtime, chunks)
    identities = preflight_outputs(output, [*bigwigs, *_region_files(viewpoint_bed, background)])
    viewpoints, exclusions = _read_regions(viewpoint_bed, viewpoint, background)
    peaks, metrics = call_adaptive_bigwigs(
        paths=bigwigs,
        viewpoints=viewpoints,
        exclusions=exclusions,
        params=params,
        scratch_dir=runtime.scratch_dir,
        cache_dir=runtime.cache_dir,
    )
    parameters = params.model_dump(exclude={"bin_size", "min_read_count", "alpha"}) | {
        "viewpoint": viewpoint,
        "viewpoints": viewpoints,
        "exclusions": exclusions,
    }
    _finish(
        peaks,
        output,
        method="adaptive",
        parameters=parameters,
        metrics=metrics,
        identities=identities,
        runtime=runtime,
    )


def _read_chrom_sizes(path: Path) -> dict[str, int]:
    try:
        frame = pd.read_csv(
            path, sep=r"\s+", header=None, comment="#", dtype=str, keep_default_na=False
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Chromosome-size file is empty: {path}") from exc
    if frame.shape[1] != 2 or frame[0].duplicated().any():
        raise ValueError("Chromosome sizes require unique names and positive integer lengths")
    if not frame[1].str.fullmatch(r"[1-9][0-9]*").all():
        raise ValueError("Chromosome lengths must be positive integers")
    sizes = {name: int(size) for name, size in frame.itertuples(index=False, name=None)}
    if any(size > sys.maxsize for size in sizes.values()):
        raise ValueError("Chromosome lengths exceed the supported integer range")
    return sizes


@app.command(group=PEAK_COMMANDS)
def call_count_peaks(
    *,
    manifest: Annotated[
        Path,
        Parameter(
            group=INPUTS,
            help="TSV with sample, viewpoint and pairs columns; paths are relative to the manifest.",
        ),
    ],
    viewpoint: Annotated[
        str, Parameter(group=INPUTS, help="Viewpoint group to select from the manifest and BED.")
    ],
    viewpoint_bed: ViewpointBed,
    chrom_sizes: Annotated[
        Path, Parameter(group=INPUTS, help="Two-column chromosome names and lengths.")
    ],
    output: Output,
    bin_size: Annotated[
        int, Parameter(group="Count model", help="Reporter-event bin width in bp.")
    ] = 50,
    min_read_count: Annotated[
        int,
        Parameter(group="Count model", help="Minimum pooled raw reporter-event count per peak."),
    ] = 5,
    alpha: Annotated[
        float,
        Parameter(
            group="Count model",
            help="Bin significance annotation threshold; does not filter peak calls.",
        ),
    ] = 0.05,
    detection: Detection = DetectionOptions(),
    background: Background = BackgroundOptions(),
    runtime: Runtime = RuntimeOptions(),
    chunks: Chunks = ChunkOptions(),
) -> None:
    """Call peaks and experimental bin statistics from standard .pairs contacts.

    Counts represent reporter events, not independent molecules. Adjusted bin
    values are experimental and are not peak-level FDR estimates.
    """
    _configure_runtime(runtime)
    params = _adaptive_params(
        detection,
        background,
        runtime,
        chunks,
        bin_size=bin_size,
        min_read_count=min_read_count,
        alpha=alpha,
    )
    base_inputs = [manifest, chrom_sizes, *_region_files(viewpoint_bed, background)]
    base_identities = preflight_outputs(output, base_inputs)
    sizes = _read_chrom_sizes(chrom_sizes)
    viewpoints, exclusions = _read_regions(viewpoint_bed, viewpoint, background)
    samples = load_manifest(manifest, viewpoint)
    identities = preflight_outputs(output, [*base_inputs, *samples["pairs"]])
    if identities[: len(base_identities)] != base_identities:
        raise ValueError("An input changed while reading count inputs")
    logger.warning("Count statistics are experimental; adjusted bin values are not peak-level FDR")
    peaks, bins, metrics = call_count_manifest(
        manifest=samples,
        viewpoint=viewpoint,
        viewpoints=viewpoints,
        chrom_sizes=sizes,
        exclusions=exclusions,
        params=params,
        scratch_dir=runtime.scratch_dir,
        cache_dir=runtime.cache_dir,
    )
    parameters = params.model_dump() | {
        "viewpoint": viewpoint,
        "viewpoints": viewpoints,
        "exclusions": exclusions,
    }
    _finish(
        peaks,
        output,
        method="counts",
        parameters=parameters,
        metrics=metrics,
        identities=identities,
        runtime=runtime,
        bins=bins,
    )


@app.command(group=LEGACY_COMMANDS)
def call_legacy_peaks(
    *,
    bigwig: Annotated[Path, Parameter(group=INPUTS, help="Single BigWig signal track.")],
    output: Output,
    cis_chromosome: Annotated[
        str, Parameter(group=LEGACY, help="Cis chromosome, or estimate from total signal.")
    ] = "estimate",
    background_chromosomes: Annotated[
        int, Parameter(group=LEGACY, help="Number of trans chromosomes to sample.")
    ] = 1,
    filter_background: Annotated[
        bool, Parameter(group=LEGACY, help="Filter using the 95th percentile of trans peak signal.")
    ] = True,
    filter_outliers: Annotated[
        bool, Parameter(group=LEGACY, help="Keep extreme log-signal outliers above Q3 + 3 IQR.")
    ] = True,
    global_background_floor: Annotated[
        float | Literal["mean"],
        Parameter(group=LEGACY, help="Minimum signal threshold, or chromosome mean."),
    ] = "mean",
    device: Annotated[
        Literal["auto", "cpu", "cuda", "mps"],
        Parameter(group=LEGACY, help="Compute backend; auto chooses an available accelerator."),
    ] = "auto",
    seed: Annotated[
        int, Parameter(group=LEGACY, help="Random seed for trans chromosome sampling.")
    ] = 0,
    detection: Detection = DetectionOptions(),
    runtime: Runtime = RuntimeOptions(),
) -> None:
    """Run the original multi-window heuristic and optional intensity filters.

    This method processes whole chromosomes. Its working memory can exceed the
    cache budget; prefer call-bigwig-peaks for chunked distance-aware calling.
    """
    try:
        import torch

        from .api.candidates import CandidatePeakCallerMCC, CandidatePeakCallParams
        from .api.utils import get_device
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "pyranges"}:
            raise ValueError(
                "Legacy calling requires the legacy extra. From the project checkout, run 'uv sync --extra legacy' and retry using 'uv run --extra legacy'."
            ) from exc
        raise

    _configure_runtime(runtime)
    grid_size = _check_detection(detection)
    params = CandidatePeakCallParams(
        window_sizes=list(detection.window_sizes),
        threshold_lst=list(detection.thresholds),
        threshold_background_lst=list(detection.background_windows),
        threshold_initial=detection.support_fraction * grid_size,
        min_peak_size=detection.min_peak_size,
        max_peak_size=detection.max_peak_size,
        background_global_min=global_background_floor,
    )
    selected_device = get_device(True) if device == "auto" else device
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use --device cpu or auto")
    if selected_device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available; use --device cpu or auto")
    identities = preflight_outputs(output, [bigwig])
    started = time.perf_counter()
    with CandidatePeakCallerMCC(
        bigwig=bigwig,
        params=params,
        cis_chromosome=cis_chromosome,
        n_background=background_chromosomes,
        apply_background_filter=filter_background,
        apply_signal_filter=filter_outliers,
        device=selected_device,
        memory_budget_mb=runtime.memory_budget_mb,
        scratch_dir=runtime.scratch_dir,
        cache_dir=runtime.cache_dir,
        seed=seed,
    ) as caller:
        peaks = caller.call_candidate_peaks(n_jobs=runtime.threads)
        metrics = {
            **caller.reader.metrics,
            "cache": dict(caller.reader.cache.metrics),
            "cis_chromosome": caller.cis_chromosome,
            "device": selected_device,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_mb": peak_memory_mb(),
        }
    parameters = params.model_dump() | {
        "seed": seed,
        "background_chromosomes": background_chromosomes,
        "filter_background": filter_background,
        "filter_outliers": filter_outliers,
        "support_fraction": detection.support_fraction,
    }
    _finish(
        peaks,
        output,
        method="legacy",
        parameters=parameters,
        metrics=metrics,
        identities=identities,
        runtime=runtime,
    )


def main() -> None:
    try:
        app()
    except (ValueError, OSError) as exc:
        if "--debug" in sys.argv:
            raise
        if isinstance(exc, ValidationError):
            message = "; ".join(
                f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
                for item in exc.errors(include_input=False, include_url=False)
            )
        else:
            message = str(exc)
        print(f"Error: {message}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
