"""Generate report-ready comparison plots for the trajectory study."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "trajectory_study"
OUTPUT = RESULTS / "plots"
PATHS = ("circle", "figure_eight", "helix")
LABELS = {"circle": "Circle", "figure_eight": "Figure eight", "helix": "3D helix"}
COLORS = {"circle": "#2563EB", "figure_eight": "#EA580C", "helix": "#16A34A"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.7,
    })


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate_comparison(summary: list[dict[str, str]]) -> None:
    ordered = [next(row for row in summary if row["path"] == path) for path in PATHS]
    x = np.arange(len(PATHS))
    labels = [LABELS[path] for path in PATHS]
    colors = [COLORS[path] for path in PATHS]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    fig.suptitle("Trajectory tracking performance across complex paths")
    metrics = (
        ("position_rmse_m", "Position RMSE (mm)", 1000.0),
        ("position_max_error_m", "Maximum position error (mm)", 1000.0),
        ("orientation_max_error_rad", "Maximum orientation error (deg)", 180.0 / np.pi),
    )
    for axis, (metric, title, scale) in zip(axes, metrics):
        mean = np.asarray([number(row, f"{metric}_mean") for row in ordered]) * scale
        std = np.asarray([number(row, f"{metric}_std") for row in ordered]) * scale
        bars = axis.bar(x, mean, color=colors, width=0.62, alpha=0.9)
        axis.errorbar(x, mean, yerr=std, fmt="none", color="black", capsize=4)
        axis.set(title=title, xticks=x, xticklabels=labels)
        axis.tick_params(axis="x", rotation=15)
        for bar, value, error in zip(bars, mean, std):
            axis.text(bar.get_x() + bar.get_width()/2, value + error,
                      f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "trajectory_aggregate_comparison")


def run_comparison(runs: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    fig.suptitle("Run-to-run variation across trajectory types")
    metrics = (
        ("position_rmse_m", "Position RMSE (mm)", 1000.0),
        ("position_max_error_m", "Maximum position error (mm)", 1000.0),
        ("orientation_max_error_rad", "Maximum orientation error (deg)", 180.0 / np.pi),
    )
    for axis, (metric, title, scale) in zip(axes, metrics):
        for path in PATHS:
            selected = sorted(
                [row for row in runs if row["path"] == path],
                key=lambda row: int(row["run"]),
            )
            axis.plot(
                [int(row["run"]) for row in selected],
                [number(row, metric) * scale for row in selected],
                marker="o", linewidth=1.6, color=COLORS[path], label=LABELS[path],
            )
        axis.set(title=title, xlabel="Run", xticks=range(1, 6))
    axes[0].legend(frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "trajectory_run_comparison")


def path_shape_comparison() -> None:
    fig = plt.figure(figsize=(11.0, 3.8))
    fig.suptitle("Desired and measured paths for representative runs")
    for index, path in enumerate(PATHS, start=1):
        data = np.load(RESULTS / f"{path}_run_01.npz")
        desired = data["desired_positions"]
        actual = data["actual_positions"]
        axis = fig.add_subplot(1, 3, index, projection="3d")
        axis.plot(*desired.T, linestyle="--", color="#111827", linewidth=1.5,
                  label="Desired")
        axis.plot(*actual.T, color=COLORS[path], linewidth=1.5, label="Measured")
        axis.scatter(*desired[0], color="#16A34A", s=18, label="Start")
        axis.set_title(LABELS[path])
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_zlabel("z (m)")
        combined = np.vstack((desired, actual))
        span = np.ptp(combined, axis=0)
        axis.set_box_aspect(np.maximum(span, 1e-3))
        axis.view_init(elev=24, azim=-55)
        if index == 1:
            axis.legend(frameon=False, loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "trajectory_path_comparison")


def main() -> None:
    style()
    summary = load_csv(RESULTS / "summary.csv")
    runs = load_csv(RESULTS / "runs.csv")
    aggregate_comparison(summary)
    run_comparison(runs)
    path_shape_comparison()
    for path in sorted(OUTPUT.glob("*")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
