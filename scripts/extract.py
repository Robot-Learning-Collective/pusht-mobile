"""Extract PushT episodes from the lerobot/pusht_keypoints parquet into JSON.

observation.state           : [agent_x, agent_y]            (2,)
observation.environment_state: 8 T keypoints, flat [x0,y0,...] (16,)
action                      : [target_x, target_y]         (2,)
All coordinates are in PushT world pixels [0, 512].
"""
import json
import sys
import numpy as np
import pyarrow.parquet as pq

PARQUET = "data/pusht_keypoints.parquet"
OUT = "data/episodes.json"
N_EPISODES = 5
SEED = 0


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
    n = int(sys.argv[2]) if len(sys.argv) > 2 else N_EPISODES
    out = sys.argv[3] if len(sys.argv) > 3 else OUT

    t = pq.read_table(PARQUET)

    def col2d(name, dim):
        # list<float> column -> (N, dim) array
        flat = t[name].combine_chunks().flatten().to_numpy(zero_copy_only=False)
        return flat.reshape(-1, dim)

    def scalar(name):
        return np.asarray(t[name].combine_chunks().to_pylist())

    ep_idx = scalar("episode_index").astype(np.int64)
    frame_idx = scalar("frame_index").astype(np.int64)
    state_all = col2d("observation.state", 2)
    env_all = col2d("observation.environment_state", 16)
    action_all = col2d("action", 2)
    success_all = scalar("next.success").astype(bool)

    ep_ids = np.unique(ep_idx)
    rng = np.random.RandomState(seed)
    chosen = sorted(rng.choice(ep_ids, size=n, replace=False).tolist())
    print(f"total episodes={len(ep_ids)}  chosen={chosen}", file=sys.stderr)

    episodes = []
    for ep in chosen:
        mask = ep_idx == ep
        order = np.argsort(frame_idx[mask])
        state = state_all[mask][order]
        env = env_all[mask][order]
        action = action_all[mask][order]
        success = success_all[mask][order]
        episodes.append({
            "episode_index": int(ep),
            "n_frames": int(len(state)),
            "agent": state.astype(float).tolist(),
            "keypoints": env.astype(float).tolist(),   # flat 16 per frame
            "action": action.astype(float).tolist(),
            "success": success.tolist(),
        })

    with open(out, "w") as f:
        json.dump({"seed": seed, "episodes": episodes}, f)
    print(f"wrote {out} with {len(episodes)} episodes", file=sys.stderr)

    # quick sanity: print frame-0 keypoints of first chosen episode
    kp = np.array(episodes[0]["keypoints"][0]).reshape(8, 2)
    print("ep", episodes[0]["episode_index"], "frame0 keypoints:", file=sys.stderr)
    for i, (x, y) in enumerate(kp):
        print(f"  kp{i}: ({x:.2f}, {y:.2f})", file=sys.stderr)
    # pairwise distances kp0..kp3 (box1 bar) to verify geometry
    def d(a, b):
        return float(np.hypot(*(kp[a] - kp[b])))
    print(f"  |kp0-kp1|={d(0,1):.2f} (expect bar length ~120)", file=sys.stderr)
    print(f"  |kp1-kp2|={d(1,2):.2f} (expect bar height ~30)", file=sys.stderr)
    print(f"  |kp4-kp5|={d(4,5):.2f} (expect stem length ~90)", file=sys.stderr)
    print(f"  agent frame0: {episodes[0]['agent'][0]}", file=sys.stderr)


if __name__ == "__main__":
    main()
