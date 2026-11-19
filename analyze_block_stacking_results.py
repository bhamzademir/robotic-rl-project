"""Analyze randomized five-block stacking performance for the final report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results" / "block_stacking" / "episodes.csv"
OUTPUT = ROOT / "results" / "block_stacking" / "analysis"
SAFE_TRANSIT_XY = np.array([0.42, 0.0])

BLUE = "#2563EB"
ORANGE = "#EA580C"
GREEN = "#16A34A"
RED = "#DC2626"
PURPLE = "#7C3AED"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else np.nan


def expand_blocks(episodes: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for episode in episodes:
        initial = json.loads(episode["initial_positions_json"])
        errors = json.loads(episode["placement_errors_json"])
        towers = {
            "tower_1": episode["tower_1_order"].split("-"),
            "tower_2": episode["tower_2_order"].split("-"),
        }
        membership = {
            block: (tower, level, len(order))
            for tower, order in towers.items()
            for level, block in enumerate(order)
        }
        # placement_errors_json preserves target insertion order, which is the
        # actual controller execution/pick order for the episode.
        pick_order = {block: index for index, block in enumerate(errors, start=1)}
        for block, error_vector in errors.items():
            tower, level, tower_size = membership[block]
            position = np.asarray(initial[block], dtype=float)
            error = np.asarray(error_vector, dtype=float)
            rows.append({
                "episode": int(episode["episode"]),
                "seed": int(episode["seed"]),
                "block": block,
                "success": episode["success"],
                "tower": tower,
                "tower_size": tower_size,
                "tower_level": level + 1,
                "pick_sequence": pick_order[block],
                "initial_x_m": position[0],
                "initial_y_m": position[1],
                "initial_abs_y_m": abs(position[1]),
                "pickup_distance_m": float(np.linalg.norm(position[:2] - SAFE_TRANSIT_XY)),
                "error_x_m": error[0],
                "error_y_m": error[1],
                "error_z_m": error[2],
                "placement_error_m": float(np.linalg.norm(error)),
                "horizontal_error_m": float(np.linalg.norm(error[:2])),
                "vertical_error_m": float(abs(error[2])),
            })
    return rows


def group_summary(
    rows: list[dict[str, object]], fields: tuple[str, ...]
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        groups.setdefault(key, []).append(row)
    output = []
    for key in sorted(groups):
        selected = groups[key]
        placement = [float(row["placement_error_m"]) for row in selected]
        horizontal = [float(row["horizontal_error_m"]) for row in selected]
        vertical = [float(row["vertical_error_m"]) for row in selected]
        p_mean, p_std = mean_std(placement)
        h_mean, h_std = mean_std(horizontal)
        v_mean, v_std = mean_std(vertical)
        output.append({
            **dict(zip(fields, key)),
            "placements": len(selected),
            "placement_error_mean_m": p_mean,
            "placement_error_std_m": p_std,
            "horizontal_error_mean_m": h_mean,
            "horizontal_error_std_m": h_std,
            "vertical_error_mean_m": v_mean,
            "vertical_error_std_m": v_std,
        })
    return output


def correlations(rows: list[dict[str, object]]) -> dict[str, float | None]:
    error = np.asarray([float(row["placement_error_m"]) for row in rows])
    level = np.asarray([int(row["tower_level"]) for row in rows])
    output: dict[str, float | None] = {}
    for field in ("initial_x_m", "initial_abs_y_m", "pickup_distance_m", "pick_sequence"):
        predictor = np.asarray([float(row[field]) for row in rows])
        output[f"raw_{field}_pearson_r"] = float(np.corrcoef(predictor, error)[0, 1])
        # Remove the dominant tower-level mean before estimating the remaining
        # relation. This avoids attributing level-dependent placement bias to
        # the randomized initial position or chronological pick order.
        adjusted_error = error.copy()
        adjusted_predictor = predictor.copy()
        for value in np.unique(level):
            mask = level == value
            adjusted_error[mask] -= np.mean(adjusted_error[mask])
            adjusted_predictor[mask] -= np.mean(adjusted_predictor[mask])
        denominator = np.std(adjusted_predictor) * np.std(adjusted_error)
        output[f"level_adjusted_{field}_pearson_r"] = (
            float(np.corrcoef(adjusted_predictor, adjusted_error)[0, 1])
            if denominator > 0.0 else None
        )
    return output


def report_table(
    level_summary: list[dict[str, object]],
    tower_level_summary: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in level_summary:
        rows.append({
            "comparison": "Tower level",
            "condition": f"Level {row['tower_level']}",
            "placements": row["placements"],
            "placement_error_mean_mm": 1000 * float(row["placement_error_mean_m"]),
            "placement_error_std_mm": 1000 * float(row["placement_error_std_m"]),
        })
    for row in tower_level_summary:
        rows.append({
            "comparison": "Tower size at fixed level",
            "condition": f"{row['tower_size']}-block tower, level {row['tower_level']}",
            "placements": row["placements"],
            "placement_error_mean_mm": 1000 * float(row["placement_error_mean_m"]),
            "placement_error_std_mm": 1000 * float(row["placement_error_std_m"]),
        })
    return rows


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


def factor_plot(
    blocks: list[dict[str, object]],
    level_summary: list[dict[str, object]],
    pick_summary: list[dict[str, object]],
    tower_level_summary: list[dict[str, object]],
    episodes: list[dict[str, str]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    fig.suptitle("Randomized five-block stacking performance factors")

    levels = np.asarray([int(row["tower_level"]) for row in level_summary])
    means = 1000 * np.asarray([float(row["placement_error_mean_m"]) for row in level_summary])
    stds = 1000 * np.asarray([float(row["placement_error_std_m"]) for row in level_summary])
    axes[0, 0].errorbar(levels, means, yerr=stds, marker="o", color=RED,
                        linewidth=1.8, capsize=4)
    for row in blocks:
        axes[0, 0].scatter(int(row["tower_level"]), 1000 * float(row["placement_error_m"]),
                           color=RED, alpha=0.18, s=18)
    axes[0, 0].set(title="Placement error increases with tower level",
                    xlabel="Tower level", ylabel="Placement error (mm)", xticks=[1, 2, 3])

    pick = np.asarray([int(row["pick_sequence"]) for row in pick_summary])
    p_mean = 1000 * np.asarray([float(row["placement_error_mean_m"]) for row in pick_summary])
    p_std = 1000 * np.asarray([float(row["placement_error_std_m"]) for row in pick_summary])
    axes[0, 1].errorbar(pick, p_mean, yerr=p_std, marker="s", color=PURPLE,
                        linewidth=1.8, capsize=4)
    axes[0, 1].set(title="Error by chronological pick sequence",
                    xlabel="Pick sequence", ylabel="Placement error (mm)", xticks=range(1, 6))

    width = 0.34
    common_levels = (1, 2)
    for offset, tower_size, color in ((-width/2, 2, BLUE), (width/2, 3, ORANGE)):
        selected = {
            int(row["tower_level"]): row for row in tower_level_summary
            if int(row["tower_size"]) == tower_size
        }
        mean = [1000 * float(selected[level]["placement_error_mean_m"]) for level in common_levels]
        std = [1000 * float(selected[level]["placement_error_std_m"]) for level in common_levels]
        axes[1, 0].bar(np.asarray(common_levels) + offset, mean, width=width,
                       yerr=std, capsize=4, color=color, alpha=0.9,
                       label=f"{tower_size}-block tower")
    axes[1, 0].set(title="Tower-size comparison at shared levels",
                    xlabel="Tower level", ylabel="Placement error (mm)", xticks=common_levels)
    axes[1, 0].legend(frameon=False)

    episode_ids = [int(row["episode"]) for row in episodes]
    episode_mean = [1000 * float(row["mean_placement_error_m"]) for row in episodes]
    episode_max = [1000 * float(row["max_placement_error_m"]) for row in episodes]
    axes[1, 1].plot(episode_ids, episode_mean, marker="o", color=BLUE,
                    linewidth=1.6, label="Episode mean")
    axes[1, 1].plot(episode_ids, episode_max, marker="s", color=ORANGE,
                    linewidth=1.6, label="Episode maximum")
    axes[1, 1].set(title="Episode-to-episode placement error",
                    xlabel="Episode", ylabel="Placement error (mm)", xticks=episode_ids)
    axes[1, 1].legend(frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "block_stacking_factor_analysis")


def position_plot(blocks: list[dict[str, object]]) -> None:
    x = np.asarray([float(row["initial_x_m"]) for row in blocks])
    y = np.asarray([float(row["initial_y_m"]) for row in blocks])
    error = 1000 * np.asarray([float(row["placement_error_m"]) for row in blocks])
    distance = np.asarray([float(row["pickup_distance_m"]) for row in blocks])
    level = np.asarray([int(row["tower_level"]) for row in blocks])

    fig, axes = plt.subplots(1, 2, figsize=(9.7, 4.1))
    fig.suptitle("Initial block position and final placement accuracy")
    scatter = axes[0].scatter(1000*x, 1000*y, c=error, cmap="viridis",
                              s=58, edgecolor="white", linewidth=0.5)
    axes[0].set(title="Randomized pickup positions", xlabel="Initial x (mm)",
                ylabel="Initial y (mm)")
    colorbar = fig.colorbar(scatter, ax=axes[0])
    colorbar.set_label("Placement error (mm)")

    markers = {1: "o", 2: "s", 3: "^"}
    colors = {1: BLUE, 2: ORANGE, 3: GREEN}
    for tower_level in (1, 2, 3):
        mask = level == tower_level
        axes[1].scatter(1000*distance[mask], error[mask], marker=markers[tower_level],
                        color=colors[tower_level], alpha=0.8,
                        label=f"Level {tower_level}")
    axes[1].set(title="Pickup distance versus placement error",
                xlabel="Distance from safe transit (mm)", ylabel="Placement error (mm)")
    axes[1].legend(frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "block_stacking_initial_position")


def main() -> None:
    episodes = read_csv(INPUT)
    blocks = expand_blocks(episodes)
    level_summary = group_summary(blocks, ("tower_level",))
    pick_summary = group_summary(blocks, ("pick_sequence",))
    tower_level_summary = group_summary(blocks, ("tower_size", "tower_level"))
    episode_summary = group_summary(blocks, ("episode",))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "per_block_results.csv", blocks)
    write_csv(OUTPUT / "summary_by_tower_level.csv", level_summary)
    write_csv(OUTPUT / "summary_by_pick_sequence.csv", pick_summary)
    write_csv(OUTPUT / "summary_by_tower_size_and_level.csv", tower_level_summary)
    write_csv(OUTPUT / "summary_by_episode.csv", episode_summary)
    write_csv(OUTPUT / "report_summary_table.csv", report_table(level_summary, tower_level_summary))

    stats = {
        "episodes": len(episodes),
        "placements": len(blocks),
        "successes": sum(row["success"] == "True" for row in episodes),
        "success_rate": sum(row["success"] == "True" for row in episodes) / len(episodes),
        "success_effect_of_initial_position_estimable": False,
        "reason_success_effect_not_estimable": "All 10 episodes succeeded; there is no failure variation.",
        "correlations": correlations(blocks),
    }
    (OUTPUT / "analysis_summary.json").write_text(json.dumps(stats, indent=2) + "\n")

    style()
    factor_plot(blocks, level_summary, pick_summary, tower_level_summary, episodes)
    position_plot(blocks)

    print(json.dumps(stats, indent=2))
    for path in sorted(OUTPUT.glob("*")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
