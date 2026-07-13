"""Export collected PushT episodes to a LeRobot dataset (like lerobot/pusht).

Reads server/episodes/*.json, renders each frame to an image (goal T in green,
block T in gray, agent in blue — matching the gym-pusht look), and writes a
LeRobot v3 dataset with encoded videos via LeRobotDataset.create/add_frame.

Who collected each episode is stored in <out>/meta/collectors.json
(episode_index -> name), keeping the dataset itself fully standard.

Usage:
    .venv/bin/python scripts/export_lerobot.py \
        --episodes server/episodes --out datasets/pusht_mobile
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pusht_mobile_env.render import IMG, render_frame

TASK = "Push the T-shaped block onto the T-shaped target."

FEATURES = {
    "observation.image": {"dtype": "video", "shape": (IMG, IMG, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": ["agent_x", "agent_y"]},
    "action": {"dtype": "float32", "shape": (2,), "names": ["target_x", "target_y"]},
}


def ep_frame(agent: list[float], block: list[float], ep: dict) -> np.ndarray:
    g = ep["goalPose"]
    return render_frame(tuple(agent), tuple(block), (g["x"], g["y"], g["angle"]), ep["scale"], ep["agentRadius"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, default=Path("server/episodes"))
    p.add_argument("--out", type=Path, default=Path("datasets/pusht_mobile"))
    p.add_argument("--repo-id", default="zaringleb/pusht_mobile")
    p.add_argument("--force", action="store_true", help="overwrite an existing output dir")
    p.add_argument("--push", action="store_true", help="push to the Hugging Face Hub as <repo-id> after export")
    p.add_argument("--private", action="store_true", help="make the Hub dataset private (with --push)")
    args = p.parse_args()

    files = sorted(args.episodes.glob("*.json"))
    if not files:
        raise SystemExit(f"no episodes found in {args.episodes}")
    if args.out.exists():
        if not args.force:
            raise SystemExit(f"{args.out} exists — pass --force to overwrite")
        shutil.rmtree(args.out)

    first = json.loads(files[0].read_text())
    fps = first["controlHz"]

    ds = LeRobotDataset.create(args.repo_id, fps=fps, features=FEATURES, root=args.out)

    collectors = []
    for i, f in enumerate(files):
        ep = json.loads(f.read_text())
        assert ep["controlHz"] == fps, f"{f.name}: controlHz {ep['controlHz']} != {fps}"
        for fr in ep["frames"]:
            ds.add_frame(
                {
                    "observation.image": ep_frame(fr["agent"], fr["block"], ep),
                    "observation.state": np.asarray(fr["agent"], dtype=np.float32),
                    "action": np.asarray(fr["action"], dtype=np.float32),
                    "task": TASK,
                }
            )
        ds.save_episode()
        collectors.append({"episode_index": i, "name": ep["name"], "file": f.name, "n_frames": len(ep["frames"])})
        print(f"episode {i}: {ep['name']} ({len(ep['frames'])} frames) <- {f.name}")

    ds.finalize()
    (args.out / "meta" / "collectors.json").write_text(json.dumps(collectors, indent=2))
    print(f"\ndone: {len(files)} episodes -> {args.out}")

    if args.push:
        ds.push_to_hub(tags=["pusht", "mobile-teleop"], private=args.private)
        print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
