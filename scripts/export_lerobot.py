"""Export version 2 recordings, preserving exact state, timing and raw JSON.

Use --state-only to skip images. Old recordings must be exported separately:
they do not have the action alignment required by this format.
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.recording import Episode

STATE_NAMES = ["agent_x", "agent_y", "block_x", "block_y", "block_angle"]

TASK = "Push the T-shaped block onto the T-shaped target."

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (5,), "names": STATE_NAMES},
    "next.observation.state": {"dtype": "float32", "shape": (5,), "names": STATE_NAMES},
    "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    "action": {"dtype": "float32", "shape": (2,), "names": ["target_x", "target_y"]},
}


def ep_frame(agent: list[float], block: list[float], ep: dict) -> np.ndarray:
    from pusht_mobile_env.render import render_frame

    g = ep["goalPose"]
    return render_frame(tuple(agent), tuple(block), (g["x"], g["y"], g["angle"]), ep["scale"], ep["agentRadius"])


def state(frame: dict) -> np.ndarray:
    values = np.asarray([*frame["agent"], *frame["block"]], dtype=np.float64)
    values[4] = (values[4] + np.pi) % (2 * np.pi) - np.pi
    return values.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, default=Path("server/episodes"))
    p.add_argument("--out", type=Path, default=Path("datasets/pusht_mobile_state_v2"))
    p.add_argument("--repo-id", default="zaringleb/pusht_mobile_state_v2")
    p.add_argument("--state-only", action="store_true", help="skip video rendering and encoding")
    p.add_argument("--force", action="store_true", help="overwrite an existing output dir")
    p.add_argument("--push", action="store_true", help="push to the Hugging Face Hub as <repo-id> after export")
    p.add_argument("--private", action="store_true", help="make the Hub dataset private (with --push)")
    args = p.parse_args()

    files = sorted(args.episodes.glob("*.json"))
    if not files:
        raise SystemExit(f"no episodes found in {args.episodes}")
    # Validate every input before creating or replacing an export.
    episodes = [Episode.model_validate_json(f.read_text()).model_dump(mode="json") for f in files]
    if args.out.resolve() == args.episodes.resolve() or args.out.resolve() in args.episodes.resolve().parents:
        raise SystemExit("output must not contain the source recordings")
    if args.out.exists():
        if not args.force:
            raise SystemExit(f"{args.out} exists — pass --force to overwrite")
        shutil.rmtree(args.out)

    first = episodes[0]
    fps = first["controlHz"]

    # Encode video as H.264 instead of LeRobot's libsvtav1 default. AV1 software
    # encoding of these tiny 96x96 frames is ~30x slower (~10 min vs ~20 s for
    # 100 episodes) and yields larger files; H.264 is standard and fully
    # training-compatible.
    #
    # sched_getaffinity, not os.cpu_count(): on a shared host (e.g. a RunPod pod
    # pinned to 2 cores) cpu_count() reports the whole machine (e.g. 192), and
    # asking libx264 for that many threads oversubscribes and is ~15x SLOWER.
    n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    features = dict(FEATURES)
    if not args.state_only:
        from pusht_mobile_env.render import IMG
        features["observation.image"] = {"dtype": "video", "shape": (IMG, IMG, 3), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(
        args.repo_id,
        fps=fps,
        features=features,
        root=args.out,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="veryfast", g=2),
        encoder_threads=n_cpu,
    )

    collectors = []
    metadata = []
    raw_dir = args.out / "raw"
    raw_dir.mkdir()
    for i, (f, ep) in enumerate(zip(files, episodes)):
        for j, fr in enumerate(ep["frames"]):
            next_state = ep["frames"][j + 1] if j + 1 < ep["nFrames"] else ep["finalState"]
            row = {
                "observation.state": state(fr),
                "next.observation.state": state(next_state),
                "next.done": np.asarray([j == ep["nFrames"] - 1], dtype=bool),
                "action": np.asarray(fr["action"], dtype=np.float32),
                "task": TASK,
            }
            if not args.state_only:
                row["observation.image"] = ep_frame(fr["agent"], fr["block"], ep)
            ds.add_frame(row)
        shutil.copy2(f, raw_dir / f.name)
        metadata.append({"episode_index": i, "file": f.name, **{k: v for k, v in ep.items() if k != "frames"}})
        ds.save_episode()
        # slug matches the server's per-name stats key, so "Gleb"/"gleb" merge
        slug = re.sub(r"[^a-z0-9]+", "-", ep["name"].lower()).strip("-") or "anon"
        collectors.append(
            {"episode_index": i, "name": ep["name"], "slug": slug, "file": f.name, "n_frames": len(ep["frames"])}
        )
        print(f"episode {i}: {ep['name']} ({len(ep['frames'])} frames) <- {f.name}")

    ds.finalize()
    (args.out / "meta" / "collectors.json").write_text(json.dumps(collectors, indent=2))
    (args.out / "meta" / "recordings.json").write_text(json.dumps(metadata, indent=2))
    print(f"\ndone: {len(files)} episodes -> {args.out}")

    if args.push:
        ds.push_to_hub(tags=["pusht", "mobile-teleop"], private=args.private)
        print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
