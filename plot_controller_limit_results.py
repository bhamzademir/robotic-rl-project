"""Generate report-ready plots for the controller-limit experiments."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "controller_limits"
OUTPUT = RESULTS / "plots"

BLUE = "#2563EB"
ORANGE = "#EA580C"
GREEN = "#16A34A"
RED = "#DC2626"
GRAY = "#64748B"
LIGHT_GRAY = "#E2E8F0"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def values(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=float)


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
        "grid.color": LIGHT_GRAY,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.8,
    })


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def step_response_plots() -> None:
    rows = load_csv(RESULTS / "step_response" / "summary.csv")
    position = [row for row in rows if row["test"] == "position"]
    orientation = [row for row in rows if row["test"] == "orientation"]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    fig.suptitle("Task-space controller step-response performance")

    pos_mm = 1000.0 * values(position, "magnitude")
    axes[0, 0].errorbar(
        pos_mm, values(position, "rise_time_s_mean"),
        yerr=values(position, "rise_time_s_std"), marker="o", color=BLUE,
        capsize=4, linewidth=1.8, label="Rise time",
    )
    axes[0, 0].errorbar(
        pos_mm, values(position, "settling_time_s_mean"),
        yerr=values(position, "settling_time_s_std"), marker="s", color=ORANGE,
        capsize=4, linewidth=1.8, label="Settling time",
    )
    axes[0, 0].set(title="Position response timing", xlabel="Position step (mm)", ylabel="Time (s)")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].errorbar(
        pos_mm, values(position, "overshoot_percent_mean"),
        yerr=values(position, "overshoot_percent_std"), marker="o", color=RED,
        capsize=4, linewidth=1.8,
    )
    axes[0, 1].set(title="Position overshoot", xlabel="Position step (mm)", ylabel="Overshoot (%)")

    ori_deg = values(orientation, "magnitude")
    axes[1, 0].errorbar(
        ori_deg, values(orientation, "rise_time_s_mean"),
        yerr=values(orientation, "rise_time_s_std"), marker="o", color=BLUE,
        capsize=4, linewidth=1.8, label="Rise time",
    )
    axes[1, 0].errorbar(
        ori_deg, values(orientation, "settling_time_s_mean"),
        yerr=values(orientation, "settling_time_s_std"), marker="s", color=ORANGE,
        capsize=4, linewidth=1.8, label="Settling time",
    )
    axes[1, 0].set(title="Orientation response timing", xlabel="Orientation step (deg)", ylabel="Time (s)")
    axes[1, 0].legend(frameon=False)

    ori_error_deg = np.rad2deg(values(orientation, "steady_state_error_si_mean"))
    ori_error_std_deg = np.rad2deg(values(orientation, "steady_state_error_si_std"))
    bars = axes[1, 1].bar(ori_deg, ori_error_deg, width=3.5, color=GREEN, alpha=0.9)
    axes[1, 1].errorbar(ori_deg, ori_error_deg, yerr=ori_error_std_deg,
                        fmt="none", color="black", capsize=4)
    axes[1, 1].set(title="Orientation steady-state error", xlabel="Orientation step (deg)", ylabel="Error (deg)")
    for bar, value, error in zip(bars, ori_error_deg, ori_error_std_deg):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, value + error + 0.001,
                        f"{value:.3f}°",
                        ha="center", va="bottom", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "step_response_summary")


def speed_limit_plots() -> None:
    rows = load_csv(RESULTS / "speed_limit" / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.1))
    fig.suptitle("Trajectory tracking degradation with commanded speed")

    for path, label, color, marker in (
        ("circle", "Circle", BLUE, "o"),
        ("figure_eight", "Figure eight", ORANGE, "s"),
    ):
        selected = sorted(
            [row for row in rows if row["path"] == path],
            key=lambda row: float(row["speed_scale"]),
        )
        speed = values(selected, "speed_scale")
        maximum = 1000.0 * values(selected, "position_max_error_m_mean")
        maximum_std = 1000.0 * values(selected, "position_max_error_m_std")
        rmse = 1000.0 * values(selected, "position_rmse_m_mean")
        rmse_std = 1000.0 * values(selected, "position_rmse_m_std")
        axes[0].errorbar(speed, maximum, yerr=maximum_std, marker=marker,
                         color=color, capsize=4, linewidth=1.8, label=label)
        axes[1].errorbar(speed, rmse, yerr=rmse_std, marker=marker,
                         color=color, capsize=4, linewidth=1.8, label=label)

    axes[0].axhline(50.0, color=RED, linestyle="--", linewidth=1.5,
                    label="50 mm failure threshold")
    axes[0].set(title="Maximum tracking error", xlabel="Commanded speed scale", ylabel="Maximum error (mm)")
    axes[0].legend(frameon=False)
    axes[1].set(title="Root-mean-square tracking error", xlabel="Commanded speed scale", ylabel="Position RMSE (mm)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(sorted(set(values(rows, "speed_scale"))))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "speed_limit_tracking")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for path, label, color, marker in (
        ("circle", "Circle", BLUE, "o"),
        ("figure_eight", "Figure eight", ORANGE, "s"),
    ):
        selected = sorted([row for row in rows if row["path"] == path],
                          key=lambda row: float(row["speed_scale"]))
        ax.errorbar(values(selected, "speed_scale"),
                    100.0 * values(selected, "max_joint_velocity_ratio_mean"),
                    yerr=100.0 * values(selected, "max_joint_velocity_ratio_std"),
                    marker=marker, color=color, capsize=4, linewidth=1.8, label=label)
    ax.axhline(100.0, color=RED, linestyle="--", linewidth=1.5,
               label="Joint-velocity limit")
    ax.set(title="Joint-velocity utilization during speed testing",
           xlabel="Commanded speed scale", ylabel="Maximum velocity-limit utilization (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "speed_limit_joint_velocity")


def robustness_plots() -> None:
    rows = load_csv(RESULTS / "robustness" / "summary.csv")
    payload = sorted(
        [row for row in rows if float(row["disturbance_force_n"]) == 0.0],
        key=lambda row: float(row["payload_kg"]),
    )
    disturbance = sorted(
        [row for row in rows if float(row["disturbance_force_n"]) > 0.0],
        key=lambda row: float(row["disturbance_force_n"]),
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    mass = values(payload, "payload_kg")
    peak = 1000.0 * values(payload, "peak_displacement_m_mean")
    peak_std = 1000.0 * values(payload, "peak_displacement_m_std")
    residual = 1000.0 * values(payload, "residual_error_m_mean")
    residual_std = 1000.0 * values(payload, "residual_error_m_std")
    ax.errorbar(mass, peak, yerr=peak_std, marker="o", color=ORANGE,
                capsize=4, linewidth=1.8, label="Peak displacement")
    ax.errorbar(mass, residual, yerr=residual_std, marker="s", color=BLUE,
                capsize=4, linewidth=1.8, label="Residual error")
    ax.axhline(10.0, color=RED, linestyle="--", linewidth=1.5,
               label="10 mm hold-error threshold")
    ax.set(title="Effect of unmodeled payload on hold accuracy",
           xlabel="Additional payload (kg)", ylabel="Position error (mm)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "robustness_payload")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1))
    force = values(disturbance, "disturbance_force_n")
    axes[0].errorbar(force, 1000.0 * values(disturbance, "peak_displacement_m_mean"),
                     yerr=1000.0 * values(disturbance, "peak_displacement_m_std"),
                     marker="o", color=RED, capsize=4, linewidth=1.8,
                     label="Peak displacement")
    axes[0].errorbar(force, 1000.0 * values(disturbance, "residual_error_m_mean"),
                     yerr=1000.0 * values(disturbance, "residual_error_m_std"),
                     marker="s", color=GREEN, capsize=4, linewidth=1.8,
                     label="Residual error")
    axes[0].set(title="Displacement under lateral force pulses",
                xlabel="Disturbance force (N)", ylabel="Position error (mm)")
    axes[0].legend(frameon=False)

    axes[1].errorbar(force, values(disturbance, "recovery_time_s_mean"),
                     yerr=values(disturbance, "recovery_time_s_std"),
                     marker="o", color=BLUE, capsize=4, linewidth=1.8)
    axes[1].set(title="Recovery after disturbance removal",
                xlabel="Disturbance force (N)", ylabel="Recovery time (s)")
    fig.tight_layout()
    save(fig, "robustness_disturbance")


def main() -> None:
    style()
    step_response_plots()
    speed_limit_plots()
    robustness_plots()
    for path in sorted(OUTPUT.glob("*")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
