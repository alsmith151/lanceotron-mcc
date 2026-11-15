"""Command-line entry points for signal and count peak calling and junction footprints."""

from __future__ import annotations

import json
import math
import sys
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
from .api.io import source_identity
from .api.results import _json_default, _versions, preflight_outputs, write_outputs
from .cli_options import (
    BACKGROUND,
    INPUTS,
    LOGGING,
    BackgroundOptions,
    ChunkOptions,
    DetectionOptions,
    RuntimeOptions,
)


def _version() -> str:
    try:
        return version("mcc_peaks")
    except PackageNotFoundError:
        return "0.1.0 (source checkout)"


app = App(
    name="mcc-peaks",
    version=_version,
    help="Call MCC peaks and profile base-pair ligation-junction footprints.",
    help_epilogue=(
        "Peak commands write BED3, a score TSV, unresolved regions and provenance JSON; "
        "footprint writes junction profile tables, a sparse contact map and an optional PDF."
    ),
)

PEAK_COMMANDS = Group("Peak calling", sort_key=0)
JUNCTION_COMMANDS = Group("Junction footprints", sort_key=1)
JUNCTIONS = Group("Junction profiling", sort_key=1)
PLOTTING = Group("Plotting", sort_key=2)

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
    # Two-column chrom.sizes, or a samtools faidx index (.fai): name and length lead.
    if frame.shape[1] < 2 or frame.shape[1] not in (2, 5, 6):
        raise ValueError(
            "Chromosome sizes require two columns (name, length) or a samtools .fai index"
        )
    frame = frame.iloc[:, :2]
    frame.columns = [0, 1]
    if frame[0].duplicated().any():
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


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    """Write an integer/string table as TSV (gzip for ``.gz``), using polars for speed.

    Float columns are not supported; their text would differ from pandas.
    """
    import gzip

    import polars as pl

    table = pl.DataFrame(
        {
            name: (
                column.to_numpy()
                if column.dtype.kind in "iub"
                else pl.Series(column.astype(str).tolist(), dtype=pl.String)
            )
            for name, column in frame.items()
        }
    )
    if path.suffix == ".gz":
        with gzip.open(path, "wb", compresslevel=6) as handle:
            table.write_csv(handle, separator="\t")
    else:
        table.write_csv(path, separator="\t")


def _footprint_outputs(prefix: Path, plot: bool) -> dict[str, Path]:
    prefix = prefix.expanduser().absolute()
    outputs = {
        "profiles": Path(f"{prefix}.profiles.tsv.gz"),
        "contacts": Path(f"{prefix}.contacts.tsv"),
        "regions": Path(f"{prefix}.regions.tsv"),
        "metrics": Path(f"{prefix}.metrics.json"),
    }
    if plot:
        outputs["plot"] = Path(f"{prefix}.pdf")
    return outputs


def _preflight_footprint(outputs: dict[str, Path], inputs: list[Path]) -> list[dict]:
    identities = [source_identity(path) for path in inputs]
    sources = {Path(item["path"]).resolve() for item in identities}
    inodes = {(path.stat().st_dev, path.stat().st_ino) for path in sources}
    for target in outputs.values():
        if target.is_dir():
            raise ValueError(f"Output path is a directory: {target}")
        if target.resolve() in sources or (
            target.exists() and (target.stat().st_dev, target.stat().st_ino) in inodes
        ):
            raise ValueError(f"Output would overwrite an input file: {target}")
    return identities


@app.command(group=JUNCTION_COMMANDS)
def footprint(
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
    output: Annotated[
        Path,
        Parameter(
            group=INPUTS,
            help="Output prefix; .profiles.tsv.gz, .contacts.tsv, .regions.tsv, "
            ".metrics.json and .pdf are appended.",
        ),
    ],
    peaks: Annotated[
        Path | None,
        Parameter(
            group=INPUTS,
            help="BED3/4 reporter regions, e.g. called peaks. Default: the capture oligo tiles.",
        ),
    ] = None,
    exclude_bed: Annotated[
        Path | None,
        Parameter(group=INPUTS, help="BED regions whose contacts are dropped; may be empty."),
    ] = None,
    reporter_flank: Annotated[
        int, Parameter(group=JUNCTIONS, help="Bases profiled either side of each reporter region.")
    ] = 100,
    capture_flank: Annotated[
        int,
        Parameter(
            group=JUNCTIONS,
            help="Bases either side of each oligo in which a capture junction is accepted.",
        ),
    ] = 50,
    allow_unoriented: Annotated[
        bool,
        Parameter(
            group=JUNCTIONS,
            negative=[],
            help="Keep contacts without MCCNado side/strand information as 'unknown'.",
        ),
    ] = False,
    deduplicate: Annotated[
        bool,
        Parameter(group=JUNCTIONS, help="Collapse exact repeated read-ID/endpoint rows."),
    ] = True,
    orientation_source: Annotated[
        Literal["mccnado", "unknown"],
        Parameter(
            group=JUNCTIONS,
            help="Derive fragment direction from MCCNado read names, or report all as unknown.",
        ),
    ] = "mccnado",
    max_profile_bases: Annotated[
        int, Parameter(group=JUNCTIONS, help="Upper bound on profiled bases across all samples.")
    ] = 2_000_000,
    max_contacts: Annotated[
        int, Parameter(group=JUNCTIONS, help="Upper bound on distinct contacts held in memory.")
    ] = 2_000_000,
    plot: Annotated[
        bool, Parameter(group=PLOTTING, help="Write a PDF; requires the 'plot' extra.")
    ] = True,
    max_panels: Annotated[
        int, Parameter(group=PLOTTING, help="Maximum reporter regions plotted, in input order.")
    ] = 12,
    arrow_length: Annotated[
        float, Parameter(group=PLOTTING, help="Direction arrow length on the contact map, in bp.")
    ] = 20,
    debug: Annotated[
        bool,
        Parameter(
            group=LOGGING,
            negative=[],
            show_default=False,
            help="Show diagnostic logs and tracebacks for runtime errors.",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        Parameter(
            group=LOGGING, negative=[], show_default=False, help="Only show warnings and errors."
        ),
    ] = False,
) -> None:
    """Profile base-pair ligation junctions at capture and reporter sites.

    Junction ends are counted per base and split by whether the ligated fragment
    extends upstream or downstream of the junction, so protected sites appear as
    footprints. Contacts with one end in a capture window and one in a reporter
    region also form a sparse single-base capture x reporter map. Profiles are
    descriptive; no footprint significance is estimated.
    """
    _configure_runtime(RuntimeOptions(debug=debug, quiet=quiet))
    if plot:
        try:
            import matplotlib  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ValueError(
                "Plotting requires matplotlib. Run 'uv sync --extra plot' and retry with "
                "'uv run --extra plot', or pass --no-plot."
            ) from exc
    if isinstance(max_panels, bool) or max_panels < 1:
        raise ValueError("--max-panels must be a positive integer")
    if not math.isfinite(arrow_length) or arrow_length <= 0:
        raise ValueError("--arrow-length must be a finite positive number")
    from .api.junctions import profile_junctions, read_peak_regions

    outputs = _footprint_outputs(output, plot)
    region_files = [viewpoint_bed, *(path for path in (peaks, exclude_bed) if path)]
    base_inputs = [manifest, chrom_sizes, *region_files]
    base_identities = _preflight_footprint(outputs, base_inputs)
    sizes = _read_chrom_sizes(chrom_sizes)
    viewpoints = read_bed(viewpoint_bed, viewpoint, require_single_group=True)
    exclusions = read_bed(exclude_bed, allow_empty=True) if exclude_bed else {}
    reporter_regions = read_peak_regions(peaks, sizes) if peaks else None
    samples = load_manifest(manifest, viewpoint)
    identities = _preflight_footprint(outputs, [*base_inputs, *samples["pairs"]])
    if identities[: len(base_identities)] != base_identities:
        raise ValueError("An input changed while reading junction inputs")
    profiles, contacts, regions, metrics = profile_junctions(
        samples,
        viewpoint,
        viewpoints,
        sizes,
        reporter_regions,
        reporter_flank=reporter_flank,
        capture_flank=capture_flank,
        exclusions=exclusions,
        allow_unoriented=allow_unoriented,
        deduplicate=deduplicate,
        orientation_source=orientation_source,
        max_profile_bases=max_profile_bases,
        max_contacts=max_contacts,
    )
    outputs["metrics"].parent.mkdir(parents=True, exist_ok=True)
    for name, frame in (("profiles", profiles), ("contacts", contacts), ("regions", regions)):
        _write_tsv(frame, outputs[name])
    if plot:
        from .api.junction_plotting import plot_junctions

        plot_junctions(
            profiles, contacts, outputs["plot"], max_panels=max_panels, arrow_length=arrow_length
        )
    provenance = {
        "method": "junction-footprint",
        "versions": _versions(),
        "parameters": {
            "viewpoint": viewpoint,
            "viewpoints": viewpoints,
            "exclusions": exclusions,
            "reporter_flank": reporter_flank,
            "capture_flank": capture_flank,
            "allow_unoriented": allow_unoriented,
            "deduplicate": deduplicate,
            "orientation_source": orientation_source,
            "max_profile_bases": max_profile_bases,
            "max_contacts": max_contacts,
            "plot": plot,
            "max_panels": max_panels,
            "arrow_length": arrow_length,
        },
        "inputs": identities,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "metrics": metrics,
    }
    outputs["metrics"].write_text(json.dumps(provenance, indent=2, default=_json_default) + "\n")
    logger.info(
        "Wrote {} profile rows, {} sparse contacts and {} regions to {}.*",
        len(profiles),
        len(contacts),
        len(regions),
        output,
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
