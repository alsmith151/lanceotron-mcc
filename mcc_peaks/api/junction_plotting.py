"""Render base-pair junction profiles and sparse capture-reporter maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROFILE_COLUMNS = {
    "Sample",
    "Region",
    "Chromosome",
    "Position",
    "RegionStart",
    "RegionEnd",
    "Upstream",
    "Downstream",
    "Unknown",
    "Total",
}
CONTACT_COLUMNS = {
    "Sample",
    "Region",
    "CaptureChromosome",
    "CapturePosition",
    "ReporterChromosome",
    "ReporterPosition",
    "CaptureDirection",
    "ReporterDirection",
    "Count",
}
DIRECTIONS = {"upstream", "downstream", "unknown"}


def _validate(profiles: pd.DataFrame, contacts: pd.DataFrame) -> None:
    missing_profiles = PROFILE_COLUMNS.difference(profiles.columns)
    missing_contacts = CONTACT_COLUMNS.difference(contacts.columns)
    if missing_profiles:
        raise ValueError(f"Profiles are missing columns: {', '.join(sorted(missing_profiles))}")
    if missing_contacts:
        raise ValueError(f"Contacts are missing columns: {', '.join(sorted(missing_contacts))}")

    for frame, column in (
        (profiles, "Position"),
        (profiles, "RegionStart"),
        (profiles, "RegionEnd"),
        (profiles, "Upstream"),
        (profiles, "Downstream"),
        (profiles, "Unknown"),
        (profiles, "Total"),
        (contacts, "CapturePosition"),
        (contacts, "ReporterPosition"),
        (contacts, "Count"),
    ):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} values must be finite numbers")
        if column in {"Upstream", "Downstream", "Unknown", "Total", "Count"} and (values < 0).any():
            raise ValueError(f"{column} values must be nonnegative")
    for frame, columns in (
        (profiles, ("Upstream", "Downstream", "Unknown", "Total")),
        (contacts, ("Count",)),
    ):
        for column in columns:
            values = pd.to_numeric(frame[column], errors="raise")
            if column == "Count" and (values <= 0).any():
                raise ValueError("Count values in sparse contacts must be positive")
    for column in ("CaptureDirection", "ReporterDirection"):
        directions = contacts[column].fillna("unknown").astype(str).str.lower()
        if not directions.isin(DIRECTIONS).all():
            raise ValueError(f"{column} must contain upstream, downstream or unknown")


def _draw_profile(ax: Any, data: pd.DataFrame, *, title: str) -> None:
    """Plot a mirrored upstream/downstream trace for each sample."""
    ax.set_title(title, loc="left", fontsize=10)
    if data.empty:
        ax.text(0.5, 0.5, "No profile data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return

    samples = list(pd.unique(data["Sample"].fillna("(unknown sample)").astype(str)))
    colors = {sample: f"C{index % 10}" for index, sample in enumerate(samples)}
    for sample in samples:
        sample_values = data.loc[data["Sample"].fillna("(unknown sample)").astype(str) == sample]
        grouped = sample_values.groupby("Position", sort=True)[
            ["Upstream", "Downstream", "Unknown"]
        ].sum()
        x = grouped.index.to_numpy(dtype=float)
        color = colors[sample]
        ax.plot(x, grouped["Upstream"], color=color, linewidth=1.2, label=f"{sample} upstream")
        ax.plot(
            x,
            -grouped["Downstream"],
            color=color,
            linewidth=1.0,
            linestyle="--",
            label=f"{sample} downstream",
        )
        unknown = grouped["Unknown"].to_numpy(dtype=float)
        if np.any(unknown):
            ax.plot(x, unknown, color=color, linewidth=0.8, linestyle=":", alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("1 bp contacts\nupstream + / downstream −")
    ax.set_xlabel("Position (0-based bp)")
    ax.grid(axis="y", color="0.88", linewidth=0.5)
    if len(samples) <= 5:
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    ax.margins(x=0.02)


def _direction_sign(value: Any) -> int | None:
    direction = "unknown" if pd.isna(value) else str(value).lower()
    if direction == "unknown":
        return None
    return -1 if direction == "upstream" else 1


def _draw_contacts(ax: Any, contacts: pd.DataFrame, *, arrow_length: float) -> None:
    ax.set_title("Sparse capture × reporter 1 bp map", loc="left", fontsize=10)
    ax.set_xlabel("Capture position (0-based bp)")
    ax.set_ylabel("Reporter position (0-based bp)")
    if contacts.empty:
        ax.text(
            0.5,
            0.5,
            "No capture–reporter contacts in this region",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return

    x = contacts["CapturePosition"].to_numpy(dtype=float)
    y = contacts["ReporterPosition"].to_numpy(dtype=float)
    count = contacts["Count"].to_numpy(dtype=float)
    sizes = 18 + 18 * np.log1p(count)
    from matplotlib.colors import LogNorm

    points = ax.scatter(
        x,
        y,
        c=count,
        s=sizes,
        cmap="viridis",
        norm=LogNorm(vmin=float(count.min()), vmax=float(count.max()) * 1.000001),
        marker="s",
        edgecolors="black",
        linewidths=0.25,
        zorder=2,
    )
    colorbar = ax.figure.colorbar(points, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Contact count (log scale)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    for row in contacts.itertuples(index=False):
        dx = _direction_sign(row.CaptureDirection)
        dy = _direction_sign(row.ReporterDirection)
        if dx is None or dy is None:
            ax.scatter(
                [row.CapturePosition],
                [row.ReporterPosition],
                s=70,
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                zorder=4,
            )
        else:
            ax.annotate(
                "",
                xy=(
                    row.CapturePosition + dx * arrow_length,
                    row.ReporterPosition + dy * arrow_length,
                ),
                xytext=(row.CapturePosition, row.ReporterPosition),
                arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 0.8, "mutation_scale": 7},
                zorder=3,
            )
    ax.text(
        0.01,
        0.99,
        f"Arrow vectors: fixed {arrow_length:g} bp; open circle: direction unknown",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2},
    )
    ax.margins(0.08)
    ax.grid(color="0.9", linewidth=0.4)


def _page(regions: pd.DataFrame, contacts: pd.DataFrame, region: str, arrow_length: float):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(11.7, 9.0), constrained_layout=True)
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(3, 1, height_ratios=(1, 1, 1.6))
    capture = regions.loc[regions["Region"].astype(str).str.startswith("capture:")]
    reporter = regions.loc[regions["Region"].astype(str) == region]
    region_contacts = contacts.loc[contacts["Region"].astype(str) == region]
    chrom = (
        ", ".join(pd.unique(reporter["Chromosome"].astype(str)))
        if not reporter.empty
        else "unknown"
    )
    starts = reporter["RegionStart"].min() if not reporter.empty else "?"
    ends = reporter["RegionEnd"].max() if not reporter.empty else "?"
    fig.suptitle(
        f"{region} — {chrom}:{starts}-{ends}\n"
        "Base-pair junction profile and sparse capture–reporter contact map",
        fontsize=13,
    )
    _draw_profile(fig.add_subplot(grid[0]), capture, title="Capture profile")
    _draw_profile(fig.add_subplot(grid[1]), reporter, title=f"Reporter profile: {region}")
    _draw_contacts(fig.add_subplot(grid[2]), region_contacts, arrow_length=arrow_length)
    return fig


def plot_junctions(
    profiles: pd.DataFrame,
    contacts: pd.DataFrame,
    output: str | Path,
    *,
    max_panels: int = 12,
    arrow_length: float = 20,
) -> Path:
    """Write a PDF of mirrored junction profiles and sparse direction maps.

    Each reporter region gets one page, preserving input region order. Upstream
    profile counts are positive and downstream counts negative. Contact maps
    are drawn directly from sparse rows, so no dense genomic matrix is made.
    ``arrow_length`` is only a fixed visual vector length in base pairs; counts
    are represented by square marker size and color.
    """
    if isinstance(max_panels, bool) or not isinstance(max_panels, int) or max_panels < 1:
        raise ValueError("max_panels must be a positive integer")
    if not np.isfinite(arrow_length) or arrow_length <= 0:
        raise ValueError("arrow_length must be a finite positive number of base pairs")
    _validate(profiles, contacts)
    profiles = profiles.copy()
    contacts = contacts.copy()
    for column in (
        "Position",
        "RegionStart",
        "RegionEnd",
        "Upstream",
        "Downstream",
        "Unknown",
        "Total",
    ):
        profiles[column] = pd.to_numeric(profiles[column], errors="raise")
    for column in ("CapturePosition", "ReporterPosition", "Count"):
        contacts[column] = pd.to_numeric(contacts[column], errors="raise")
    for column in ("CaptureDirection", "ReporterDirection"):
        contacts[column] = contacts[column].fillna("unknown").astype(str).str.lower()

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reporter_regions = [
        region
        for region in pd.unique(profiles["Region"].dropna().astype(str))
        if region.startswith("reporter:")
    ]
    selected = reporter_regions[:max_panels]
    omitted = reporter_regions[max_panels:]
    with PdfPages(output_path) as pdf:
        if not selected:
            fig = Figure(figsize=(11.7, 8.3))
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            ax.axis("off")
            ax.text(
                0.5,
                0.58,
                "No reporter junction profiles to plot",
                ha="center",
                va="center",
                fontsize=16,
            )
            ax.text(
                0.5,
                0.45,
                f"Input profiles: {len(profiles):,} rows; contacts: {len(contacts):,} rows",
                ha="center",
                va="center",
                fontsize=10,
            )
            pdf.savefig(fig)
        else:
            for region in selected:
                fig = _page(profiles, contacts, region, float(arrow_length))
                pdf.savefig(fig)
                fig.clear()
        if omitted:
            fig = Figure(figsize=(11.7, 8.3), constrained_layout=True)
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            ax.axis("off")
            ax.set_title("Reporter regions omitted by max_panels", loc="left")
            ax.text(
                0.02,
                0.92,
                f"Plotted {len(selected)} of {len(reporter_regions)} reporter regions. "
                f"max_panels={max_panels}; omitted {len(omitted)} region(s) in input order.",
                transform=ax.transAxes,
                va="top",
            )
            ax.text(
                0.02,
                0.84,
                "\n".join(omitted),
                transform=ax.transAxes,
                va="top",
                family="monospace",
                fontsize=9,
            )
            pdf.savefig(fig)
            fig.clear()
    return output_path
