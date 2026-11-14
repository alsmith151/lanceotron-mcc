"""Shared command-line options; numerical validation belongs to the API models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import Group, Parameter

INPUTS = Group("Inputs", sort_key=0)
DETECTION = Group("Peak detection", sort_key=1)
BACKGROUND = Group("Viewpoint and background", sort_key=2)
RESOURCES = Group("Resources and caching", sort_key=4)
LOGGING = Group("Logging", sort_key=5)


@dataclass(frozen=True)
class DetectionOptions:
    window_sizes: Annotated[
        tuple[int, ...],
        Parameter(
            group=DETECTION,
            consume_multiple=True,
            negative=[],
            help="Foreground smoothing windows in bp.",
        ),
    ] = (200, 400, 600)
    thresholds: Annotated[
        tuple[float, ...],
        Parameter(
            group=DETECTION,
            consume_multiple=True,
            negative=[],
            help="Enrichment multipliers above background.",
        ),
    ] = (2.0, 4.0, 6.0)
    background_windows: Annotated[
        tuple[int, ...],
        Parameter(
            group=DETECTION,
            consume_multiple=True,
            negative=[],
            help="Local background windows in bp.",
        ),
    ] = (10_000, 100_000, 1_000_000)
    support_fraction: Annotated[
        float,
        Parameter(
            group=DETECTION,
            show_default=False,
            help="Required fraction of grid combinations, in (0, 1]. Default: 1/3.",
        ),
    ] = 1 / 3
    min_peak_size: Annotated[int, Parameter(group=DETECTION, help="Minimum peak width in bp.")] = 50
    max_peak_size: Annotated[
        int,
        Parameter(
            group=DETECTION,
            help="Maximum resolved width in bp; wider regions are reported separately.",
        ),
    ] = 500


@dataclass(frozen=True)
class BackgroundOptions:
    exclude_bed: Annotated[
        Path | None,
        Parameter(group=BACKGROUND, help="BED regions to exclude; an empty file is allowed."),
    ] = None
    bait_padding: Annotated[
        int,
        Parameter(
            group=BACKGROUND,
            help="Additional exclusion on either side of each capture oligo, in bp.",
        ),
    ] = 0
    background_bin_size: Annotated[
        int,
        Parameter(
            group=BACKGROUND,
            help="Distance-profile resolution in bp, independent of count-bin size.",
        ),
    ] = 200


@dataclass(frozen=True)
class RuntimeOptions:
    threads: Annotated[
        int, Parameter(group=RESOURCES, help="Compute workers; input reads remain serialized.")
    ] = 1
    memory_budget_mb: Annotated[
        float,
        Parameter(
            group=RESOURCES,
            help="Array cache and working-size budget in MiB; not a hard RSS limit.",
        ),
    ] = 512
    scratch_dir: Annotated[
        Path | None,
        Parameter(
            group=RESOURCES, help="Node-local directory for staged BigWigs and temporary arrays."
        ),
    ] = None
    cache_dir: Annotated[
        Path | None,
        Parameter(
            group=RESOURCES, help="Optional persistent array cache; prefer node-local scratch."
        ),
    ] = None
    debug: Annotated[
        bool,
        Parameter(
            group=LOGGING,
            negative=[],
            show_default=False,
            help="Show diagnostic logs and tracebacks for runtime errors.",
        ),
    ] = False
    quiet: Annotated[
        bool,
        Parameter(
            group=LOGGING, negative=[], show_default=False, help="Only show warnings and errors."
        ),
    ] = False


@dataclass(frozen=True)
class ChunkOptions:
    chunk_size: Annotated[
        int,
        Parameter(
            group=RESOURCES,
            help="Upper bound on bases processed per chunk; the memory budget may lower it.",
        ),
    ] = 8_000_000
