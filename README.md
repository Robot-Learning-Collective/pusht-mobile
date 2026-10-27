# PushT — Rapier2D simulation, validated against `lerobot/pusht`

A browser-ready PushT physics core (Rapier2D) for mobile teleop data
collection, validated for fidelity against the real HuggingFace
`lerobot/pusht` demonstrations.

## What this is

- `rust/pusht-core` — the PushT simulation on **Rapier2D**, compiled to wasm
  for the browser UI and to a Python module for training evals (see “Rust
  core” below). No rendering or DOM dependencies.
- `rust/pusht-core/src/bin/replay.rs` — an offline harness that replays
  recorded demonstrations through the sim and measures the difference from
  ground truth.
- `scripts/extract.py` — pulls episodes out of the dataset parquet into JSON.

## How the simulation matches the reference

PushT was originally implemented in **pymunk** (Chipmunk2D). The exact reference
parameters (from `gym-pusht` and `diffusion_policy`) are reproduced:

| Parameter | Value |
|---|---|
| World | 512×512 px, gravity (0,0) |
| sim / control rate | 100 Hz / 10 Hz → 10 substeps per action, dt = 0.01 s |
| Agent | **kinematic** circle r=15, PD-driven: `v += (100·(target−p) + 20·(0−v))·dt` |
| Block | dynamic T, mass 1, two cuboids (bar 120×30, stem 30×90) |
| Block COM | centroid of the 8 keypoints (== pymunk centre of gravity) |
| Block angular inertia | **3000** in reference mode (reproduces a copy-paste bug in the original that uses the bar's moment twice); geometry-derived (~1932, COM (0,−4.29)) in our physical mode — see "Reference vs physical" below |
| Contact friction / restitution | 0 / 0 |
| Surface friction | none — `space.damping = 0` |

The single most important modeling decision: in pymunk, `damping = 0` means the
block's velocity is **fully reset every physics step** and only re-acquired from
contact impulses — the block never coasts, it stops the instant the agent stops
pushing. We replicate this by zeroing the block's linear/angular velocity before
each substep (`resetBlockVelEachStep`).

## Reference vs physical block dynamics

The original PushT env has a **copy-paste bug** in the T's moment of inertia:

```python
inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)   # bar
inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)   # BUG: should be vertices2 (stem)
body = pymunk.Body(mass, inertia1 + inertia2)                 # => 3000, not the intended 7875
```

It also treats the centre of gravity as the *un-weighted* mean of the two box
centroids (the body origin), rather than the true area-weighted centre of mass.
So the shipped block has inertia **3000** about the wrong axis; a uniform-density
T of mass 1 physically has COM **(0, −4.29)** and inertia **1932**.

The sim has exactly two modes (`new PushTSim(mode)`):

- **`"product"`** (the game) — Rapier derives mass/COM/inertia from the actual
  geometry. Correct rigid-body dynamics, plus the robustness features below and
  1.5x-size T/robot for phone playability.
- **`"reference"`** (the eval harness) — reproduces the pymunk env bug-for-bug
  (mass 1, COM (0,0), inertia 3000, kinematic agent, 1x geometry). This is what
  matches the dataset.

Consequence: the UI differs from `lerobot/pusht` by design (the block rotates
about the true COM with correct inertia — median ~21 px divergence from the
recorded, buggy trajectories over the full dataset). The velocity-reset /
no-coast behaviour (`damping=0`) is kept in both modes — that is a deliberate
PushT design choice, not a bug.

## Robustness: walls & agent (product vs reference)

The original kinematic agent has infinite force, so it can crush the T through a
wall (especially into a corner) or, when flicked fast, tunnel through the block.
Faithful to the reference, but bad for a live demo. The product default fixes
this without touching dataset fidelity:

- **Dynamic agent** — the agent is a finite-mass body driven by a PD *force*
  (`F = m·(kp·e − kv·v)`). Unobstructed it tracks the target identically to the
  kinematic agent (the mass cancels: `a = kp·e − kv·v`), but walls and the block
  genuinely stop it, so no crush-through and no tunnelling. CCD adds swept
  collision for fast motion.
- **Thick walls** (extend outward from a fixed inner face) and block CCD guard
  the block against tunnelling.
- **Bounds guard** (`keepInBounds`) — the definitive fix for wall escapes. Solver contacts are
  compliant springs, so a fast heavy agent can *ratchet* the block into a wall
  over repeated impacts (adversarial corner-whipping reached 74 px of
  penetration; CCD cannot help once bodies already overlap, and stiffening
  contacts globally degrades dataset parity). The guard enforces the invariant
  at position level: after every substep any body outside the walls' inner
  faces is projected back inside. It is a strict no-op in the interior.
  Verified: 0.0 px worst penetration over 48k adversarial substeps.
- **No-pass-through guard** (`noPassThrough`) — when the block is
  wall-pinned, the agent's unbounded PD force drives its centre past the T's
  15 px half-thickness and the solver ejects it out the far side; no
  stiffness/force-cap setting prevents this (measured). After every substep, an
  agent centre that ended inside the T is bisected back along its own motion to
  the surface it entered through. The guard's distance field must be the T's
  *outline polygon* (`T_OUTLINE`), not a min over the two boxes — the per-box
  minimum reads ~0 on the interior bar/stem seam, leaving a blind slab exactly
  where the rectangles connect (measured: agent endpoints 15 px inside at the
  junction; 0 with the outline SDF). Verified: 0 pass-throughs over 30k
  substeps of junction-drilling pursuit; sliding *around* the frictionless T is
  still possible, as in the reference env.
- The UI clamps the pointer target into the play area and uses **pointer
  capture**, so drags keep tracking when the finger leaves the canvas — without
  it the target freezes short of the borders and the robot feels slow there.

Reference mode keeps the kinematic agent (+ the inertia bug) and no guards, so
dataset validation is unchanged — parity stays at median 0.088 px.

## Validation method

`lerobot/pusht` only stores the 2-D agent position, so we use
**`lerobot/pusht_keypoints`**, which adds `observation.environment_state`: the 8
world-space T keypoints (16 numbers) per frame. From those we recover the
ground-truth block pose by 2-D Procrustes alignment against the canonical
body-frame template (`pose_from_keypoints` in `pusht-core`). The keypoint
template is constant across
all frames to 1e-4 px, confirming a clean rigid body.

For each episode we do an **open-loop rollout**: reset the sim to the recorded
frame-0 state, then apply the recorded action sequence for the whole episode
(no re-syncing), and compare the simulated trajectory to the recording.

Metrics (per frame, averaged over the trajectory):
- **agent RMSE** — validates the PD controller / timestep (deterministic).
- **block position / angle error** — validates the contact physics.
- **mean keypoint error (px)** — the primary objective; captures position + angle together.

## Results

Validated on the **entire dataset — all 206 episodes**
(`cargo run --release --bin replay`), open-loop rollout per episode:

```
agent RMSE:           max 4.95e-5 px        (kinematic + PD is deterministic)
block kpMeanErr (px): mean 0.95  median 0.088  p90 0.35  p95 1.16  p99 21.6  max 68.8
fraction of episodes: <=0.5px 92.7%   <=1px 93.7%   <=2px 96.1%   <=5px 97.6%
```

- **Agent trajectory matches to floating-point precision** across all 206
  episodes — confirms the timestep, substep count, and PD gains are exact.
- **Median block error is 0.088 px and p95 is 1.16 px** over open-loop rollouts
  of 56–200+ frames while the block travels up to ~450 px and rotates up to
  ~200°. That is essentially the float32 noise floor of the dataset. **96% of
  episodes track to ≤2 px.**
- **A ~2.4% tail diverges (p99 ≈ 22 px, max 69 px).** These are contact-rich
  events where Rapier and pymunk split a push's impulse between translation and
  rotation slightly differently; in an *open-loop* replay that small difference
  then compounds. The tail is heterogeneous (worst offenders mix 200°-rotation
  episodes like 107/192/78 with a few low-rotation ones like 50), i.e. it is
  chaotic contact sensitivity, **not** a systematic model bias. In the
  interactive UI it does not accumulate because the human closes the loop.

Worst 10 episodes: `ep107` 68.8px (204° rot), `ep192` 56px (180°), `ep78` 22px
(196°), `ep204` 7.3px, `ep50` 5.2px, then everything ≤3.2px.

## What the tuning sweeps showed

(The sweep scripts served their purpose and were removed; the findings are
recorded here.) The chosen parameters are near-optimal, and the sweep doubled
as a correctness check:

- Letting the block coast (`resetBlockVel=false`) → **159 px** error. The
  velocity-reset is the dominant modeling factor.
- `angularInertia=3000` (the reproduced bug) is optimal; the "physically
  correct" 7875 gives 88 px, 1500 gives 33 px.
- `mass=1`, `friction=0`, `restitution=0` are all confirmed optimal; any
  deviation blows up the error.
- Solver iterations barely matter (the contact solution is geometric/converged).

Iterating on the full 206-episode set confirmed the parameters are optimal and
that the divergence tail is irreducible by global tuning:

```
variant          median   p95    p99    max    <=2px
default          0.088   1.16   21.6   68.8   96.1%
natFreq=40       0.101   1.01   22.1   60.6   96.6%
natFreq=10       1.071   6.42   44.4   64.9   80.1%   <- "fixes" ep78 but ruins the median
angInertia=4000 14.766  75.69  124    164     5.8%   <- confirms 3000 is dialed in
```

Softening contacts (`natFreq=10`) looked promising on a 10-episode subset
(it lowered ep78) but is catastrophic at scale — a good argument for always
validating on the whole dataset. The default TGS solver at the reference
parameters is the best-balanced choice.

## Reproduce

```bash
# 1. data (Python venv with pyarrow)
python3 -m venv .venv && .venv/bin/pip install pyarrow numpy
curl -sSL https://huggingface.co/datasets/lerobot/pusht_keypoints/resolve/main/data/chunk-000/file-000.parquet -o data/pusht_keypoints.parquet
.venv/bin/python scripts/extract.py 0 5            # 5 random episodes -> data/episodes.json

.venv/bin/python scripts/extract.py 0 206 data/episodes_all.json   # full dataset

# 2. sim + eval (Rust)
cd rust && cargo run --release --bin replay -- ../data/episodes_all.json
```

## Rust core — the single source of truth

All PushT-specific logic (scene setup, PD servo, pymunk-compat hacks, guards,
coverage, seeded resets) lives in one Rust crate and is compiled for every
consumer, so data collection and training evals run **the same code**:

```
rust/pusht-core    the sim + game logic (rapier2d = 0.22, the exact engine
                   version wrapped by @dimforge/rapier2d-compat@0.14)
rust/pusht-wasm    wasm-bindgen bindings -> browser UI
rust/pusht-py      PyO3 bindings -> `pusht_core` Python module
python/            `pusht_mobile_env` gymnasium wrapper (PushTMobile-v0)
```

Validation (same protocol as the TS sim, against the original lerobot/pusht
data, all 206 episodes — `cargo run --release --bin replay`):
agent RMSE max 4.95e-5 px; block kpMeanErr mean 0.95, median 0.088, p95 1.16 —
identical to the TS numbers above, digit for digit. Native and wasm builds
produce bit-identical trajectories (`eval/wasmParity.mjs` vs
`cargo run --release --bin traj`). The phone-collected episodes replay to
success in the Python env (12/14; the two misses end at coverage 0.947/0.920,
an artifact of 10 Hz action resampling, not an env mismatch).

Note the version-mapping trap: npm `rapier2d-compat@0.14` wraps **Rust rapier
0.22**, not 0.14 (see the rapier.js changelog).

Build the bindings:

```bash
npm run build:wasm                                   # -> rust/pusht-wasm/pkg (needed before npm run dev)
.venv/bin/maturin build --release -m rust/pusht-py/Cargo.toml
.venv/bin/pip install rust/target/wheels/pusht_py-*.whl
.venv/bin/pip install -e python                      # gymnasium wrapper
```

Use in training code (e.g. ~/pusht-slim):

```python
import gymnasium as gym
import pusht_mobile_env  # registers PushTMobile-v0

env = gym.make("PushTMobile-v0")  # obs: {"pixels": 96x96x3 u8, "agent_pos": [x,y]}
obs, info = env.reset(seed=0)     # action: target position in [0,512]^2
obs, reward, terminated, truncated, info = env.step(action)
# terminated == success (coverage >= 0.95); info: coverage, is_success
```

Observations are rendered by `pusht_mobile_env/render.py` — the same code the
dataset export uses, so eval pixels match training pixels exactly.

For state-only training, request five numbers instead of images:

```python
env = gym.make("PushTMobile-v0", obs_type="state", render_mode=None)
obs, info = env.reset(seed=0)  # float32 [agent_x, agent_y, T_x, T_y, T_angle]
```

The angle is in radians, wrapped to `[-pi, pi)`. State observations do not load
the renderer or generate images. Physics, actions and rewards are the same;
the default observation mode remains `pixels_agent_pos`.

Run the Python environment checks after installing the bindings:

```bash
python -m unittest discover -s python -p 'test_*.py'
```

## Interactive UI (PixiJS + wasm core)

A mobile-first web app on top of `pusht-wasm`:

- `index.html` — HTML/CSS shell (name field, status line, result dialogs,
  Save / Restart).
- `src/main.ts` — PixiJS renderer + pointer/touch controls + real-time loop +
  10 Hz episode recording (starts when the user grabs the robot, stops at the
  first successful control boundary).
- `leaderboard.html` — projector page, auto-refreshing per-name counts.
- `server/main.py` — FastAPI collector; one JSON file per successful episode in
  `server/episodes/`, plus `/api/stats`.

Run it locally:

```bash
npm install && npm run build:wasm
npm run dev                                          # UI on http://localhost:5173
.venv/bin/python -m uvicorn server.main:app --port 8000   # episode collector
```

Production (single origin, e.g. on a RunPod CPU pod): `npm run build`, then
`scripts/serve.sh` — uvicorn serves `dist/` and `/api` on one port
(`PORT`, `PUSHT_EPISODES_DIR` env vars).

Touch the blue robot to start, drag to push the grey T onto the green target;
at ≥95% coverage the episode can be saved (name required). Pre-upload checks reject teleport-fast motion or >20 s episodes (including
pauses); accepted episodes are attributed to the entered name. All stationary
frames are retained.

### State recordings (schema version 2)

Each frame contains the state **before** its action: `agent: [x, y]`,
`block: [x, y, angle]`, `action: [target_x, target_y]`, zero-based `step`, and
`timestamp` in simulation seconds. The sampled action stays fixed for ten
100 Hz physics ticks, matching one Python `env.step()`. Rendering remains
smooth at the display rate. Success is checked after the full control step.
The episode stores `finalState` and `finalTimestamp` separately, so the last
transition has a real next state without an invented extra action.

The raw JSON also preserves the reset seed and initial poses, physics/control
rates, geometry, goal, success threshold, engine version and SHA-256 of the
compiled simulator. T angles are radians; exported angles are wrapped to
`[-pi, pi)` to match the state environment. Coordinates and action targets
use the 512×512 simulation world, independent of phone screen size.

Use a new persistent directory for these demos. The collector rejects old or
incomplete recordings; refresh an old browser tab before recording. The
exporter also rejects old recordings rather than silently mixing time formats.

```bash
PUSHT_EPISODES_DIR=/path/to/state-v2/episodes scripts/serve.sh
python scripts/check_recordings.py /path/to/state-v2/episodes
python scripts/export_lerobot.py --episodes /path/to/state-v2/episodes \
  --out datasets/pusht_mobile_state_v2 --state-only
```

The LeRobot export contains `observation.state` and `next.observation.state`
with `[agent_x, agent_y, block_x, block_y, block_angle]`, `action`, `next.done`,
and standard timestamps, frame and episode indices. Omit `--state-only` to
include rendered videos. Full episode metadata goes in `meta/recordings.json`,
collector names in `meta/collectors.json`, and byte-for-byte original JSON
backups in `raw/`. Keep the server directory and exported backups; the original
dataset is not overwritten. Hub upload is opt-in with `--push`; the new default
repository is `zaringleb/pusht_mobile_state_v2`.

Before a full collection, record five successful demos and run the replay
check above. It compares every state and terminal success against Python;
tolerances are 0.001 pixels and 0.0001 radians for native/WASM rounding.

End-to-end checks (requires the existing LeRobot export dependencies plus
FastAPI and httpx):

```bash
npm run build:wasm
npx tsc --noEmit
python -m unittest discover -s tests -v
```
