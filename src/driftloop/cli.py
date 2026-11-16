"""The ``driftloop`` command: thin I/O wrappers around the library.

Every command reads files, calls one library function and writes files. Nothing here
computes a result, so anything the CLI prints can be reproduced from Python.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any

import polars as pl
import typer

from driftloop.config import load_config, load_interactions
from driftloop.data import yambda
from driftloop.loop.simulate import RESULTS_FILE, run_simulation, write_run
from driftloop.report.plot import METRIC_LABELS, plot_quality

app = typer.Typer(
    help="Replay interactions in time order and compare retrain policies.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config", exists=True, dir_okay=False, readable=True, help="Experiment YAML file."
    ),
]


def _fmt(value: Any, spec: str) -> str:
    """Format a summary value; None or NaN (undefined ratio, empty mean) prints as a dash."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return format("-", ">" + spec.split(".")[0])
    return format(value, spec)


def format_summary(summary: dict[str, Any]) -> str:
    """The per-policy table printed after a run, one line per policy."""
    metric = summary["headline_metric"]
    header = (
        f"{'policy':<10} {'mean_' + metric:>12} {'mean_ndcg':>10} {'retrains':>9} "
        f"{'fit_s':>8} {'quality_kept':>13} {'cost_share':>11}"
    )
    lines = [
        f"{summary['name']}: {summary['n_scored_windows']} scored windows "
        f"(of {summary['n_windows']}; first {summary['tuning_windows']} used for tuning), "
        f"{summary['n_interactions']} interactions",
        header,
    ]
    for policy, s in summary["policies"].items():
        lines.append(
            f"{policy:<10} {_fmt(s[f'mean_{metric}'], '12.4f')} {_fmt(s['mean_ndcg'], '10.4f')} "
            f"{s['retrains']:>9d} {s['retrain_seconds']:>8.3f} "
            f"{_fmt(s['quality_kept'], '13.3f')} {_fmt(s['cost_share'], '11.3f')}"
        )
    return "\n".join(lines)


@app.command()
def simulate(
    config: ConfigOption,
    out: Annotated[Path, typer.Option("--out", file_okay=False, help="Run directory to write.")],
) -> None:
    """Run the replay and write results.csv, summary.json and config.yaml to OUT."""
    cfg = load_config(config)
    interactions = load_interactions(cfg)
    result = run_simulation(interactions, cfg)
    write_run(result, cfg, out)
    typer.echo(format_summary(result.summary))
    typer.echo(f"wrote {out}")


@app.command()
def plot(
    run_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="A directory written by simulate.")
    ],
    metric: Annotated[str, typer.Option(help=f"One of: {', '.join(METRIC_LABELS)}.")] = "recall",
) -> None:
    """Draw quality per window from RUN_DIR/results.csv into RUN_DIR/quality.png."""
    results = run_dir / RESULTS_FILE
    if not results.is_file():
        raise typer.BadParameter(f"{results} not found; run `driftloop simulate` first")
    rows = pl.read_csv(results)
    path = plot_quality(rows, run_dir / "quality.png", metric=metric)
    typer.echo(f"wrote {path}")


@app.command()
def synth(
    config: ConfigOption,
    out: Annotated[Path, typer.Option("--out", dir_okay=False, help="Parquet file to write.")],
) -> None:
    """Write the synthetic interactions of a synthetic-source config as Parquet."""
    cfg = load_config(config)
    if cfg.data.source != "synthetic":
        raise typer.BadParameter(f"{config} uses data.source={cfg.data.source!r}, not synthetic")
    interactions = load_interactions(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    interactions.write_parquet(out)
    typer.echo(f"wrote {interactions.height} interactions to {out}")


@app.command("yambda-download")
def yambda_download(
    out: Annotated[Path, typer.Option("--out", file_okay=False, help="Data directory.")],
    size: Annotated[str, typer.Option(help="Dataset size: 50m, 500m or 5b.")] = "50m",
) -> None:
    """Download Yambda listens into OUT (needs the optional `data` extra)."""
    try:
        paths = yambda.download(size, out)
    except (ImportError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for path in paths:
        typer.echo(f"wrote {path}")
