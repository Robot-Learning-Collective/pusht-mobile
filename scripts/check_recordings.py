"""Check saved version 2 demos by replaying every action in the Python simulator."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pusht_core as pc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server.recording import Episode


def replay(episode: dict) -> float:
    sim = pc.PushTSim("product")
    initial = episode["init"]
    sim.reset(tuple(initial["agent"]), tuple(initial["blockPos"]), initial["blockAngle"])
    worst = 0.0
    for i, recorded in enumerate([*episode["frames"], episode["finalState"]]):
        actual = np.array([*sim.agent_pos(), *sim.block_pose()])
        expected = np.array([*recorded["agent"], *recorded["block"]])
        error = actual - expected
        error[4] = (error[4] + np.pi) % (2 * np.pi) - np.pi
        worst = max(worst, float(np.max(np.abs(error))))
        # Native and WASM f32 contact arithmetic can differ by sub-millipixels.
        if np.max(np.abs(error[:4])) > 1e-3 or abs(error[4]) > 1e-4:
            raise ValueError(f"step {i}: replay differs from recording by {error.tolist()}")
        if i < len(episode["frames"]):
            sim.step(tuple(recorded["action"]))
    if episode["success"] and sim.coverage() < episode["successThreshold"]:
        raise ValueError("replay did not reach success")
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path)
    args = parser.parse_args()
    files = sorted(args.episodes.glob("*.json"))
    if not files:
        raise SystemExit("no recordings found")
    for path in files:
        ep = json.loads(path.read_text())
        Episode.model_validate(ep)
        error = replay(ep)
        print(f"{path.name}: {len(ep['frames'])} steps, replay error {error:g}")
    print(f"Passed: {len(files)} episodes")


if __name__ == "__main__":
    main()
