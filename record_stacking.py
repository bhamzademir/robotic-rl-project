"""Open report stacking Episode 2 for manual screen recording."""

import subprocess
from pathlib import Path
from tempfile import mkdtemp


def main() -> int:
    project = Path(__file__).resolve().parent
    viewer_python = project / "venv" / "bin" / "mjpython"
    if not viewer_python.is_file():
        raise SystemExit(f"MuJoCo's Python launcher was not found at {viewer_python}")

    capture_dir = Path(mkdtemp(prefix="panda-stacking-capture-"))
    print(f"Recording run logs will be saved in {capture_dir / 'results' / 'block_stacking' / 'replays' / 'seed_42_episode_02'}.", flush=True)
    print("Episode 2: D–C–E and B–A (bottom to top).", flush=True)

    return subprocess.call(
        [
            str(viewer_python),
            str(project / "simulation_block_stacking.py"),
            "--episodes", "2",
            "--episode", "2",
            "--seed", "42",
            "--horizon", "120",
            "--save-step-logs",
        ],
        cwd=capture_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
