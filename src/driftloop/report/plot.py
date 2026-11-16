"""The headline chart: quality per window for every policy, retrain events marked.

The chart is a static PNG written by the CLI and embedded in the README later, so it
is drawn with matplotlib's non-interactive Agg backend and needs no display.

Identity is never carried by colour alone: every policy also has its own line style
and is labelled at the end of its line, and the legend says what the markers mean.
The colours are the first slots of a colour-blind-validated categorical palette, in
a fixed order, so a policy keeps its colour across charts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Imported after use("Agg") so pyplot never tries to open a GUI backend.
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.ticker import MaxNLocator

#: Categorical slots in fixed order (validated for colour-vision deficiency on adjacent
#: pairs). Assigned by the policy's position in the results, never cycled.
PALETTE: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
#: One dash pattern per palette slot, so lines stay distinguishable in greyscale.
LINESTYLES: tuple[str | tuple[int, tuple[int, ...]], ...] = (
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 3)),
    (0, (8, 2, 1, 2)),
)
_TEXT = "#0b0b0b"
_MUTED = "#52514e"
_GRID = "#e4e3df"
_SURFACE = "#fcfcfb"
_SHADE = "#efeeea"

METRIC_LABELS: dict[str, str] = {
    "recall": "Recall@{k}",
    "ndcg": "NDCG@{k}",
    "hit_rate": "Hit rate@{k}",
    "coverage": "Coverage",
}


def _spread(values: list[float], min_gap: float) -> list[float]:
    """Nudge sorted label positions apart so no two are closer than ``min_gap``.

    Lines that end at (nearly) the same value would otherwise get overlapping labels.
    Labels keep their order and only colliding ones move (upwards), so a label that
    has room stays level with the end of its line.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [values[i] for i in order]
    for j in range(1, len(placed)):
        placed[j] = max(placed[j], placed[j - 1] + min_gap)
    out = [0.0] * len(values)
    for j, i in enumerate(order):
        out[i] = placed[j]
    return out


def _window_days(rows: pl.DataFrame) -> float | None:
    if not {"start", "end"} <= set(rows.columns):
        return None
    length = rows.get_column("end")[0] - rows.get_column("start")[0]
    return float(length) / 86_400


def plot_quality(rows: pl.DataFrame, out_path: Path, *, metric: str = "recall") -> Path:
    """Draw ``metric`` over window index, one line per policy, and save it as PNG.

    Filled markers on a line are windows where that policy retrained. Coincident
    curves hide each other's markers, so a strip under the chart repeats the retrain
    events with one row per policy. Tuning windows (not part of the summary) are
    shaded so a reader does not compare policies on them. Returns ``out_path``.
    """
    if metric not in METRIC_LABELS:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(METRIC_LABELS)}")
    missing = {"policy", "window", "retrained", metric} - set(rows.columns)
    if missing:
        raise ValueError(f"results are missing columns {sorted(missing)}")
    if rows.height == 0:
        raise ValueError("no result rows to plot")
    policies = rows.get_column("policy").unique(maintain_order=True).to_list()
    if len(policies) > len(PALETTE):
        raise ValueError(f"at most {len(PALETTE)} policies per chart, got {len(policies)}")

    k = int(rows.get_column("k")[0]) if "k" in rows.columns else None
    ylabel = METRIC_LABELS[metric].format(k=k if k is not None else "k")
    days = _window_days(rows)
    xlabel = "Replay window" + (f" ({days:g}-day steps)" if days is not None else "")
    last_x = rows.get_column("window").max()
    assert isinstance(last_x, int)

    fig, (ax, strip) = plt.subplots(
        2,
        1,
        figsize=(10, 6.4),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [5, 0.35 * len(policies) + 0.4]},
        layout="constrained",
    )
    fig.patch.set_facecolor(_SURFACE)

    unscored: list[int] = []
    if "scored" in rows.columns:
        unscored = rows.filter(~pl.col("scored")).get_column("window").unique().to_list()
    for axis, label in ((ax, "tuning windows (not in summary)"), (strip, None)):
        axis.set_facecolor(_SURFACE)
        if unscored:
            axis.axvspan(
                min(unscored) - 0.5, max(unscored) + 0.5, color=_SHADE, zorder=0, label=label
            )

    ends: list[tuple[str, float, float]] = []
    for i, policy in enumerate(policies):
        part = rows.filter(pl.col("policy") == policy).sort("window")
        x = part.get_column("window").to_numpy()
        y = part.get_column(metric).cast(pl.Float64).to_numpy()
        retrained = part.get_column("retrained").fill_null(False).to_numpy().astype(bool)
        color = PALETTE[i]
        ax.plot(x, y, color=color, linestyle=LINESTYLES[i], linewidth=2, label=policy)
        ax.plot(
            x[retrained],
            y[retrained],
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=3,
        )
        row_y = len(policies) - 1 - i
        strip.plot(
            x[retrained],
            [row_y] * int(retrained.sum()),
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="white",
        )
        if y.size and y[-1] == y[-1]:  # NaN != NaN: no label for an unscorable last window
            ends.append((policy, float(x[-1]), float(y[-1])))

    # End-of-line labels, spread to at least ~1/25 of the y range apart.
    ax.relim()
    ax.autoscale_view()
    lo, hi = ax.get_ylim()
    label_y = _spread([e[2] for e in ends], (hi - lo) / 25)
    for (policy, x_end, _), y_label in zip(ends, label_y, strict=True):
        ax.annotate(
            policy,
            xy=(x_end, y_label),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=_TEXT,
        )

    # A proxy artist so the legend explains the markers once, in neutral ink.
    ax.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=7,
        markerfacecolor=_MUTED,
        markeredgecolor="white",
        label="retrained before this window",
    )
    ax.set_xlim(-0.5, last_x + 1.6)
    ax.set_ylabel(ylabel, color=_TEXT)
    ax.set_title(f"{ylabel} per window by retrain policy", color=_TEXT, loc="left")
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, labelcolor=_TEXT, loc="upper right")

    strip.set_yticks(range(len(policies)), list(reversed(policies)), fontsize=8, color=_TEXT)
    strip.set_ylim(-0.6, len(policies) - 0.4)
    strip.set_ylabel("retrains", color=_TEXT, fontsize=9)
    strip.set_xlabel(xlabel, color=_TEXT)
    strip.xaxis.set_major_locator(MaxNLocator(integer=True))
    strip.grid(axis="x", color=_GRID, linewidth=0.8)
    strip.set_axisbelow(True)

    for axis in (ax, strip):
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(_MUTED)
        axis.tick_params(colors=_MUTED)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.align_ylabels([ax, strip])
    fig.savefig(out_path, facecolor=_SURFACE)
    plt.close(fig)
    return out_path
