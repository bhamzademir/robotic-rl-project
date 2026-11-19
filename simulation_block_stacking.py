"""Randomized five-block, two-tower stacking experiment.

Each episode samples collision-free block positions and a new ordered task:
three labeled blocks form tower 1 and the remaining two form tower 2. Results
include success, simulated completion time, placement errors, and random seeds.

Examples:
    python simulation_block_stacking.py --headless --episodes 10 --seed 42
    mjpython simulation_block_stacking.py --episodes 1 --seed 42
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from environment import FrankaPickAndPlaceEnv, TaskSpacePDConfig
from trajectory_generator import interpolate_pose


HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
GRASP_HAND_OFFSET_Z = 0.090
TRANSIT_CLEARANCE_Z = 0.190
PLACE_CLEARANCE_Z = 0.006
# The task-space controller consistently places higher levels short in X.
# Compensate the measured bias while retaining the nominal tower centres for
# final validation: roughly 8 mm, 16 mm, and 25 mm for levels 0, 1, and 2.
PLACE_X_COMPENSATION = np.array([0.008, 0.016, 0.025])
MOVE_DURATION = 2.5
GRIPPER_DURATION = 1.0
SETTLE_DURATION = 0.5
SAFE_MOVE_DURATION = 1.5
DISPLAY_EVERY = 10
STACK_XY_TOLERANCE = 0.035
STACK_Z_TOLERANCE = 0.025
DEFAULT_HORIZON = 120.0
# Keep source blocks away from the demonstrated workspace boundaries.  The
# controller can reach the old limits, but tracking degrades there and leaves
# too little clearance for a centered grasp.
BLOCK_SAMPLE_LOW = np.array([0.470, -0.13])
BLOCK_SAMPLE_HIGH = np.array([0.565, 0.13])
# Blocks are 40 mm wide, so 60 mm centre separation still leaves a 20 mm gap
# while allowing five blocks to fit inside the safer, inset pickup region.
MIN_BLOCK_SEPARATION = 0.060
MAX_LAYOUT_ATTEMPTS = 200
MAX_BLOCK_SAMPLE_ATTEMPTS = 1000
SAFE_TRANSIT_POSITION = np.array([0.42, 0.0, 0.68])
SAFE_TRANSIT_Z = float(SAFE_TRANSIT_POSITION[2])
MIN_PICKUP_CORRIDOR_CLEARANCE = 0.140
TOWERS = ("tower_1", "tower_2")
MAX_GRASP_ATTEMPTS = 2
RETRY_GRASP_OFFSET = 0.012
MAX_PREGRASP_POSITION_ERROR = 0.025
MIN_GRASP_LIFT = 0.080
MAX_CARRIED_XY_OFFSET = 0.050
MIN_CARRIED_Z_OFFSET = -0.115
MAX_CARRIED_Z_OFFSET = -0.080
OUTPUT_DIR = Path("results/block_stacking")


@dataclass(frozen=True)
class EpisodeTask:
    seed: int
    initial_positions: dict[str, np.ndarray]
    towers: dict[str, tuple[str, ...]]


def point_to_segment_distance(
    point: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray
) -> float:
    """Return the planar distance from a point to a closed line segment."""
    segment = segment_end - segment_start
    length_squared = float(segment @ segment)
    if length_squared == 0.0:
        return float(np.linalg.norm(point - segment_start))
    fraction = float(np.clip((point - segment_start) @ segment / length_squared, 0.0, 1.0))
    closest = segment_start + fraction * segment
    return float(np.linalg.norm(point - closest))


def sample_episode_task(env: FrankaPickAndPlaceEnv, seed: int) -> EpisodeTask:
    """Sample reproducible starts and an ordered 3+2 tower assignment."""
    rng = np.random.default_rng(seed)
    positions: dict[str, np.ndarray] = {}
    # Restart the complete layout if sequential rejection sampling paints
    # itself into a corner.  Five separated blocks in the deliberately tighter
    # spawn region are feasible, but a poor early draw can consume the space.
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        candidate_positions: dict[str, np.ndarray] = {}
        for name in env.block_names:
            for _ in range(MAX_BLOCK_SAMPLE_ATTEMPTS):
                xy = rng.uniform(BLOCK_SAMPLE_LOW, BLOCK_SAMPLE_HIGH)
                separated = all(
                    np.linalg.norm(xy - existing[:2]) >= MIN_BLOCK_SEPARATION
                    for existing in candidate_positions.values()
                )
                # The pickup motion starts at the common safe-transit point and
                # approaches the source block from above.  Reserve a capsule-like
                # planar corridor for that motion so the hand/forearm cannot pass
                # through a tower that has already been built.
                pickup_corridor_clear = all(
                    point_to_segment_distance(
                        tower_xy, SAFE_TRANSIT_POSITION[:2], xy
                    ) >= MIN_PICKUP_CORRIDOR_CLEARANCE
                    for tower_xy in env.tower_positions.values()
                )
                if separated and pickup_corridor_clear:
                    candidate_positions[name] = np.array([xy[0], xy[1], 0.42])
                    break
            else:
                break
        if len(candidate_positions) == len(env.block_names):
            positions = candidate_positions
            break
    else:
        raise RuntimeError("Could not sample a collision-free pickup layout.")

    order = list(rng.permutation(env.block_names))
    return EpisodeTask(
        seed=seed,
        initial_positions=positions,
        towers={"tower_1": tuple(order[:3]), "tower_2": tuple(order[3:])},
    )


def controller_config() -> TaskSpacePDConfig:
    return TaskSpacePDConfig(
        kp_position=np.array([180.0, 180.0, 220.0]),
        kd_position=np.array([28.0, 28.0, 34.0]),
        kp_orientation=np.array([45.0, 45.0, 35.0]),
        kd_orientation=np.array([8.0, 8.0, 7.0]),
        nullspace_kp=18.0,
        nullspace_kd=5.0,
        damping=0.05,
    )


def tower_execution_order(
    task: EpisodeTask, tower_positions: dict[str, np.ndarray]
) -> tuple[str, str]:
    """Build first the tower that leaves the safest later pickup paths."""
    scores: dict[str, float] = {}
    for first_tower in TOWERS:
        second_tower = next(tower for tower in TOWERS if tower != first_tower)
        first_xy = tower_positions[first_tower]
        scores[first_tower] = min(
            point_to_segment_distance(
                first_xy,
                SAFE_TRANSIT_POSITION[:2],
                task.initial_positions[block_name][:2],
            )
            for block_name in task.towers[second_tower]
        )
    first = max(TOWERS, key=lambda tower: scores[tower])
    second = next(tower for tower in TOWERS if tower != first)
    return first, second


def retry_grasp_offset(
    block_xy: np.ndarray,
    placed_targets: dict[str, np.ndarray],
) -> np.ndarray:
    """Return a small offset away from the nearest constructed tower."""
    if not placed_targets:
        return np.zeros(2)
    placed_xy = min(
        (target[:2] for target in placed_targets.values()),
        key=lambda xy: float(np.linalg.norm(block_xy - xy)),
    )
    away = block_xy - placed_xy
    norm = float(np.linalg.norm(away))
    if norm < 1e-9:
        away = SAFE_TRANSIT_POSITION[:2] - placed_xy
        norm = float(np.linalg.norm(away))
    return RETRY_GRASP_OFFSET * away / norm


def run_episode(
    episode_index: int,
    seed: int,
    horizon: float = DEFAULT_HORIZON,
    render: bool = False,
    realtime: bool = False,
    save_step_log: bool = False,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    env = FrankaPickAndPlaceEnv(render_mode="human" if render else None)
    _, _ = env.reset(seed=seed)
    env.data.qpos[env.arm_qpos_indices] = HOME_Q
    env.data.qvel[:] = 0.0
    env.open_gripper()

    task = sample_episode_task(env, seed)
    for name, position in task.initial_positions.items():
        env.set_block_position(name, position)
    mujoco.mj_forward(env.model, env.data)

    labels = env.block_labels
    tower_labels = {
        tower: [labels[name] for name in order]
        for tower, order in task.towers.items()
    }
    execution_order = tower_execution_order(task, env.tower_positions)
    print(
        f"Episode {episode_index}: seed={seed}, "
        f"tower_1={tower_labels['tower_1']}, tower_2={tower_labels['tower_2']}, "
        f"execution_order={execution_order}"
    )

    preferred_q = HOME_Q.copy()
    target_rotation = env.get_end_effector_pose().rotation.copy()
    current_target = env.get_end_effector_pose().position.copy()
    dt = float(env.model.opt.timestep)
    elapsed = 0.0
    timed_out = False
    step_logs: list[dict[str, object]] = []
    config = controller_config()
    targets: dict[str, np.ndarray] = {}
    placed_targets: dict[str, np.ndarray] = {}
    placed_order: list[str] = []
    first_disturbance: dict[str, object] | None = None
    failure_stage = ""
    failure_block = ""
    failure_tower = ""
    failure_level: int | str = ""
    failure_details: dict[str, object] = {}

    def phase(name: str, goal: np.ndarray, duration: float, opening: float) -> None:
        nonlocal current_target, elapsed, timed_out, first_disturbance
        if timed_out:
            return
        start = current_target.copy()
        steps = max(1, int(np.ceil(duration / dt)))
        for step in range(steps + 1):
            if elapsed >= horizon:
                timed_out = True
                return
            local_t = min(step * dt, duration)
            sample = interpolate_pose(
                start, goal, target_rotation, target_rotation, local_t, duration
            )
            env.set_gripper_opening(opening)
            diagnostics = env.step_task_space_pd(
                target_position=sample.position,
                target_rotation=sample.rotation,
                target_linear_velocity=sample.velocity,
                target_angular_velocity=np.zeros(3),
                preferred_q=preferred_q,
                config=config,
            )
            if first_disturbance is None:
                for placed_block, placed_target in placed_targets.items():
                    placed_position = env.get_block_position(placed_block)
                    placed_error = placed_position - placed_target
                    xy_error = float(np.linalg.norm(placed_error[:2]))
                    z_error = float(abs(placed_error[2]))
                    if (
                        xy_error > STACK_XY_TOLERANCE
                        or z_error > STACK_Z_TOLERANCE
                    ):
                        first_disturbance = {
                            "phase": name,
                            "time_s": elapsed,
                            "block": labels[placed_block],
                            "xy_error_m": xy_error,
                            "z_error_m": z_error,
                        }
            if save_step_log:
                pose = env.get_end_effector_pose()
                step_logs.append({
                    "episode": episode_index,
                    "seed": seed,
                    "time_s": elapsed,
                    "phase": name,
                    "desired_x": sample.position[0],
                    "desired_y": sample.position[1],
                    "desired_z": sample.position[2],
                    "actual_x": pose.position[0],
                    "actual_y": pose.position[1],
                    "actual_z": pose.position[2],
                    "position_error_m": diagnostics["position_error"],
                    "orientation_error_rad": diagnostics["orientation_error"],
                    "torque_saturated": int(diagnostics["torque_saturated"]),
                    **{
                        f"{labels[block]}_{axis}": position[axis_index]
                        for block, position in env.get_all_block_positions().items()
                        for axis_index, axis in enumerate(("x", "y", "z"))
                    },
                })
            elapsed += dt
            if render and step % DISPLAY_EVERY == 0:
                env.render()
                if realtime:
                    time.sleep(DISPLAY_EVERY * dt)
        current_target = np.asarray(goal, dtype=float).copy()

    failure_reason = ""
    abort_episode = False
    try:
        # Choose the first tower from the sampled geometry so its completed
        # blocks leave the largest clearance for later source pickups.
        for tower in execution_order:
            order = task.towers[tower]
            for level, block_name in enumerate(order):
                if timed_out or abort_episode:
                    break
                target = env.get_stack_target_position(level, tower=tower)
                targets[block_name] = target
                compensated_target = target.copy()
                compensated_target[0] += PLACE_X_COMPENSATION[level]
                nominal_place = compensated_target + np.array(
                    [0.0, 0.0, GRASP_HAND_OFFSET_Z + PLACE_CLEARANCE_Z]
                )
                place_above = nominal_place + np.array(
                    [0.0, 0.0, TRANSIT_CLEARANCE_Z - GRASP_HAND_OFFSET_Z]
                )
                prefix = f"{tower}_{labels[block_name]}_level_{level}"

                grasp_confirmed = False
                for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
                    block = env.get_block_position(block_name)
                    block_start_z = float(block[2])
                    pick = block + np.array([0.0, 0.0, GRASP_HAND_OFFSET_Z])
                    pick_above = block + np.array([0.0, 0.0, TRANSIT_CLEARANCE_Z])
                    if attempt > 1:
                        xy_offset = retry_grasp_offset(block[:2], placed_targets)
                        pick[:2] += xy_offset
                        pick_above[:2] += xy_offset
                    attempt_prefix = f"{prefix}_attempt_{attempt}"
                    phase(f"{attempt_prefix}_approach", pick_above, MOVE_DURATION, 1.0)
                    phase(f"{attempt_prefix}_descend", pick, MOVE_DURATION, 1.0)

                    pregrasp_error = float(
                        np.linalg.norm(env.get_end_effector_pose().position - pick)
                    )
                    if pregrasp_error <= MAX_PREGRASP_POSITION_ERROR:
                        phase(f"{attempt_prefix}_grasp", pick, GRIPPER_DURATION, 0.0)
                        phase(f"{attempt_prefix}_lift", pick_above, MOVE_DURATION, 0.0)
                    else:
                        print(
                            f"  {labels[block_name]} pre-grasp pose blocked on attempt "
                            f"{attempt} (error={1000 * pregrasp_error:.1f} mm)"
                        )

                    lifted_block = env.get_block_position(block_name)
                    hand_position = env.get_end_effector_pose().position
                    carried_offset = lifted_block - hand_position
                    lift_distance = float(lifted_block[2] - block_start_z)
                    carried_xy_offset = float(np.linalg.norm(carried_offset[:2]))
                    lifted_enough = lift_distance >= MIN_GRASP_LIFT
                    offset_valid = (
                        carried_xy_offset <= MAX_CARRIED_XY_OFFSET
                        and MIN_CARRIED_Z_OFFSET
                        <= carried_offset[2]
                        <= MAX_CARRIED_Z_OFFSET
                    )
                    grasp_confirmed = bool(lifted_enough and offset_valid)
                    if grasp_confirmed:
                        print(
                            f"  {labels[block_name]} grasp confirmed "
                            f"on attempt {attempt} "
                            f"(lift={1000 * lift_distance:.1f} mm, "
                            f"hand_offset_z={1000 * carried_offset[2]:.1f} mm)"
                        )
                        break

                    print(
                        f"  {labels[block_name]} grasp validation failed on attempt {attempt} "
                        f"(lift={1000 * lift_distance:.1f} mm, "
                        f"hand_offset_xy={1000 * carried_xy_offset:.1f} mm, "
                        f"hand_offset_z={1000 * carried_offset[2]:.1f} mm); "
                        f"{'retrying' if attempt < MAX_GRASP_ATTEMPTS else 'aborting episode'}"
                    )
                    failure_details = {
                        "attempt": attempt,
                        "pregrasp_error_m": pregrasp_error,
                        "retry_xy_offset_m": (
                            [float(value) for value in xy_offset]
                            if attempt > 1
                            else [0.0, 0.0]
                        ),
                        "lift_m": lift_distance,
                        "carried_xy_offset_m": carried_xy_offset,
                        "carried_z_offset_m": float(carried_offset[2]),
                        "lifted_enough": bool(lifted_enough),
                        "offset_valid": bool(offset_valid),
                    }
                    phase(
                        f"{attempt_prefix}_failed_open",
                        env.get_end_effector_pose().position.copy(),
                        GRIPPER_DURATION,
                        1.0,
                    )
                    failed_retreat = env.get_end_effector_pose().position.copy()
                    failed_retreat[2] = SAFE_TRANSIT_Z
                    phase(
                        f"{attempt_prefix}_failed_vertical_retreat",
                        failed_retreat,
                        SAFE_MOVE_DURATION,
                        1.0,
                    )
                    phase(
                        f"{attempt_prefix}_failed_safe_transit",
                        SAFE_TRANSIT_POSITION,
                        SAFE_MOVE_DURATION,
                        1.0,
                    )

                if not grasp_confirmed:
                    failure_reason = f"grasp_failed_{labels[block_name]}"
                    failure_stage = "grasp_validation"
                    failure_block = labels[block_name]
                    failure_tower = tower
                    failure_level = level
                    abort_episode = True
                    break

                # Do not move diagonally from the pickup to the destination.
                # First raise the carried block above every completed tower,
                # move through a point away from the towers, and only then
                # approach the destination tower from above.
                carried_vertical_clearance = current_target.copy()
                carried_vertical_clearance[2] = SAFE_TRANSIT_Z
                phase(
                    f"{prefix}_carried_vertical_clearance",
                    carried_vertical_clearance,
                    SAFE_MOVE_DURATION,
                    0.0,
                )
                phase(
                    f"{prefix}_carried_safe_transit",
                    SAFE_TRANSIT_POSITION,
                    SAFE_MOVE_DURATION,
                    0.0,
                )
                carried_above_tower = np.array(
                    [target[0], target[1], SAFE_TRANSIT_Z], dtype=float
                )
                phase(
                    f"{prefix}_carried_above_tower",
                    carried_above_tower,
                    SAFE_MOVE_DURATION,
                    0.0,
                )
                phase(f"{prefix}_transfer", place_above, MOVE_DURATION, 0.0)
                # A grasp is rarely centered exactly. Measure the carried
                # block-to-hand offset and compensate before the final descent
                # instead of assuming a fixed vertical offset.
                hand_position = env.get_end_effector_pose().position
                carried_position = env.get_block_position(block_name)
                carried_offset = carried_position - hand_position
                place = compensated_target - carried_offset
                place[2] += PLACE_CLEARANCE_Z
                phase(f"{prefix}_place", place, MOVE_DURATION, 0.0)
                phase(f"{prefix}_release", place, GRIPPER_DURATION, 1.0)
                # From release onward, continuously monitor this block and all
                # previously completed levels.  If a later robot motion knocks
                # a tower over, the first offending phase and time are retained.
                placed_targets[block_name] = target.copy()
                # Leave the completed tower vertically before any lateral
                # motion, then move to a transit point away from both towers.
                vertical_retreat = place.copy()
                vertical_retreat[2] = SAFE_TRANSIT_Z
                phase(
                    f"{prefix}_vertical_retreat",
                    vertical_retreat,
                    SAFE_MOVE_DURATION,
                    1.0,
                )
                phase(
                    f"{prefix}_safe_transit",
                    SAFE_TRANSIT_POSITION,
                    SAFE_MOVE_DURATION,
                    1.0,
                )
                phase(
                    f"{prefix}_settle",
                    SAFE_TRANSIT_POSITION,
                    SETTLE_DURATION,
                    1.0,
                )
                if not timed_out:
                    placed_order.append(block_name)

            if timed_out or abort_episode:
                break

        final_positions = env.get_all_block_positions()
        errors = {
            name: final_positions[name] - target
            for name, target in targets.items()
        }
        expected_order = [
            name
            for tower in execution_order
            for name in task.towers[tower]
        ]
        placements_ok = len(errors) == len(env.block_names) and all(
            np.linalg.norm(error[:2]) <= STACK_XY_TOLERANCE
            and abs(error[2]) <= STACK_Z_TOLERANCE
            for error in errors.values()
        )
        order_ok = placed_order == expected_order
        success = bool(not timed_out and not abort_episode and placements_ok and order_ok)
        if not success and not timed_out and not failure_stage:
            if first_disturbance is not None:
                failure_stage = "tower_disturbance"
                failure_block = str(first_disturbance["block"])
                failure_details = first_disturbance.copy()
                failure_reason = f"tower_disturbed_during_{first_disturbance['phase']}"
            elif not placements_ok:
                failure_stage = "final_placement_check"
                failure_reason = "placement_tolerance"
            elif not order_ok:
                failure_stage = "order_check"
                failure_reason = "placement_order"
    finally:
        env.close()

    if save_step_log and step_logs:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"episode_{episode_index:02d}_steps.csv"
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=step_logs[0].keys())
            writer.writeheader()
            writer.writerows(step_logs)

    error_norms = [float(np.linalg.norm(error)) for error in errors.values()]
    return {
        "episode": episode_index,
        "seed": seed,
        "success": success,
        "timed_out": timed_out,
        "completion_time_s": elapsed,
        "tower_1_order": "-".join(tower_labels["tower_1"]),
        "tower_2_order": "-".join(tower_labels["tower_2"]),
        "initial_positions_json": json.dumps({
            labels[name]: np.round(position, 6).tolist()
            for name, position in task.initial_positions.items()
        }),
        "mean_placement_error_m": float(np.mean(error_norms)) if error_norms else np.nan,
        "max_placement_error_m": float(np.max(error_norms)) if error_norms else np.nan,
        "placement_errors_json": json.dumps({
            labels[name]: np.round(error, 6).tolist()
            for name, error in errors.items()
        }),
        "failure_stage": failure_stage,
        "failure_block": failure_block,
        "failure_tower": failure_tower,
        "failure_level": failure_level,
        "failure_details_json": json.dumps(failure_details),
        "failure_reason": "" if success else (
            "horizon" if timed_out else failure_reason or "placement_or_order"
        ),
    }


def write_results(results: list[dict[str, object]], output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "episodes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    successes = np.asarray([bool(row["success"]) for row in results])
    completion = np.asarray([
        float(row["completion_time_s"]) for row in results if bool(row["success"])
    ])
    placement = np.asarray([float(row["mean_placement_error_m"]) for row in results])
    summary = {
        "episodes": len(results),
        "successes": int(successes.sum()),
        "success_rate": float(successes.mean()),
        "completion_time_mean_s": float(np.mean(completion)) if completion.size else None,
        "completion_time_std_s": float(np.std(completion, ddof=1)) if completion.size > 1 else None,
        "placement_error_mean_m": float(np.mean(placement)),
        "placement_error_std_m": float(np.std(placement, ddof=1)) if placement.size > 1 else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--episode", type=int, help="run only this episode from the seeded sequence (1-based)"
    )
    parser.add_argument("--seed", type=int, default=42, help="base experiment seed")
    parser.add_argument("--horizon", type=float, default=DEFAULT_HORIZON)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--save-step-logs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1.")
    if args.episode is not None and not 1 <= args.episode <= args.episodes:
        raise ValueError("--episode must be between 1 and --episodes.")
    seed_sequence = np.random.SeedSequence(args.seed)
    episode_seeds = [
        int(child.generate_state(1)[0]) for child in seed_sequence.spawn(args.episodes)
    ]
    output_dir = OUTPUT_DIR
    if args.episode is not None or args.episodes == 1:
        output_dir = OUTPUT_DIR / "replays" / f"seed_{args.seed}_episode_{args.episode or 1:02d}"
    experiment_results = [
        run_episode(
            episode_index=index,
            seed=episode_seed,
            horizon=args.horizon,
            render=not args.headless,
            realtime=not args.no_realtime,
            save_step_log=args.save_step_logs,
            output_dir=output_dir,
        )
        for index, episode_seed in enumerate(episode_seeds, start=1)
        if args.episode is None or index == args.episode
    ]
    write_results(experiment_results, output_dir)
    print(f"Results saved to {output_dir}")
