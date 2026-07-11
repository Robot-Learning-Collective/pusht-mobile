# PushT — Rapier2D simulation, validated against `lerobot/pusht`

A browser-ready PushT physics core (Rapier2D) for the mobile UI described in
`project_plan.md`, validated for fidelity against the real
HuggingFace `lerobot/pusht` demonstrations.

## What this is

- `src/sim/pushtSim.ts` — framework-agnostic TypeScript PushT simulation on
  **Rapier2D** (`@dimforge/rapier2d-compat`, runs in both Node and the browser).
  This is the physics engine that will plug into the PixiJS renderer + touch
  controls. No rendering or DOM dependencies.
- `eval/` — an offline harness that replays recorded demonstrations through the
  sim and measures the difference from ground truth.
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

The sim supports both via `blockMassMode`:

- **`"physical"` (`DEFAULT_CONFIG`, our product default)** — Rapier derives
  mass/COM/inertia from the actual geometry. Correct rigid-body dynamics.
- **`"reference"` (`REFERENCE_CONFIG`)** — reproduces the pymunk env bug-for-bug
  (mass 1, COM (0,0), inertia 3000). This is what matches the dataset, so the
  eval harness uses it.

Consequence: our UI now differs from `lerobot/pusht` by design (the block
rotates about the true COM with correct inertia — median ~21 px divergence from
the recorded, buggy trajectories over the full dataset). The velocity-reset /
no-coast behaviour (`damping=0`) is kept — that is a deliberate PushT design
choice, not a bug. Dataset parity is still fully reproducible via
`REFERENCE_CONFIG`.

## Robustness: walls & agent (product vs reference)

The original kinematic agent has infinite force, so it can crush the T through a
wall (especially into a corner) or, when flicked fast, tunnel through the block.
Faithful to the reference, but bad for a live demo. The product default fixes
this without touching dataset fidelity:

- **`agentMode: "dynamic"`** — the agent is a finite-mass body driven by a PD
  *force* (`F = m·(kp·e − kv·v)`). Unobstructed it tracks the target identically
  to the kinematic agent (the mass cancels: `a = kp·e − kv·v`), but walls and the
  block genuinely stop it, so no crush-through and no tunnelling. `agentCcd` adds
  swept collision for fast motion.
- **Thick walls** (`wallThickness`, extend outward from a fixed inner face) and
  **`blockCcd`** guard the block against tunnelling.
- **`boundaryGuard`** — the definitive fix for wall escapes. Solver contacts are
  compliant springs, so a fast heavy agent can *ratchet* the block into a wall
  over repeated impacts (adversarial corner-whipping reached 74 px of
  penetration; CCD cannot help once bodies already overlap, and stiffening
  contacts globally degrades dataset parity). The guard enforces the invariant
  at position level: after every substep any body outside the walls' inner
  faces is projected back inside. It is a strict no-op in the interior.
  Verified: 0.0 px worst penetration over 48k adversarial substeps.
- **No-pass-through guard** (part of `boundaryGuard`) — when the block is
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

`REFERENCE_CONFIG` keeps `agentMode: "kinematic"` (+ the inertia bug), so dataset
validation is unchanged — parity stays at median 0.088 px.

## Validation method

`lerobot/pusht` only stores the 2-D agent position, so we use
**`lerobot/pusht_keypoints`**, which adds `observation.environment_state`: the 8
world-space T keypoints (16 numbers) per frame. From those we recover the
ground-truth block pose by 2-D Procrustes alignment against the canonical
body-frame template (`eval/pose.ts`). The keypoint template is constant across
all frames to 1e-4 px, confirming a clean rigid body.

For each episode we do an **open-loop rollout**: reset the sim to the recorded
frame-0 state, then apply the recorded action sequence for the whole episode
(no re-syncing), and compare the simulated trajectory to the recording.

Metrics (per frame, averaged over the trajectory):
- **agent RMSE** — validates the PD controller / timestep (deterministic).
- **block position / angle error** — validates the contact physics.
- **mean keypoint error (px)** — the primary objective; captures position + angle together.

## Results

Validated on the **entire dataset — all 206 episodes** (`eval/fullEval.ts`),
open-loop rollout per episode:

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

## What the tuning sweep showed (`eval/sweep.ts`)

The defaults are near-optimal, and the sweep doubles as a correctness check:

- Letting the block coast (`resetBlockVel=false`) → **159 px** error. The
  velocity-reset is the dominant modeling factor.
- `angularInertia=3000` (the reproduced bug) is optimal; the "physically
  correct" 7875 gives 88 px, 1500 gives 33 px.
- `mass=1`, `friction=0`, `restitution=0` are all confirmed optimal; any
  deviation blows up the error.
- Solver iterations barely matter (the contact solution is geometric/converged).

Iterating on the full 206-episode set (`eval/compareFull.ts`) confirms the
defaults are optimal and that the divergence tail is irreducible by global
tuning:

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

# 2. sim + eval (Node)
npm install
npx tsx eval/runEval.ts          # main diff table
npx tsx eval/diagnose.ts         # per-episode motion vs max error
npx tsx eval/sweep.ts            # parameter sensitivity
```

## Interactive UI (PixiJS)

A mobile-first web app wraps the validated `PushTSim`:

- `index.html` — HTML/CSS shell (status line, solved counter, Restart / Save).
- `src/main.ts` — PixiJS renderer + pointer/touch controls + real-time loop.
- `src/game/pushtGame.ts` — random reset (reference ranges), goal pose, coverage
  (grid-sampled overlap), success at ≥95%, and JSON episode recording.

Run it locally:

```bash
npm install
npm run dev          # Vite dev server on http://localhost:5173
```

How it plays: touch/click the blue robot to start, then drag — your pointer
becomes the PD target the robot chases. Push the grey T onto the green target;
at ≥95% coverage the episode is marked solved and **Save episode** downloads the
recorded trajectory (engine, config, init state, per-frame state+action, success
flag — the reproducible format from `project_plan.md`).

Real-time loop: the renderer accumulates wall-clock time and advances the sim in
fixed dt = 0.01 s substeps (`sim.advance`), applying the live pointer target each
substep, so the physics stays identical to the validated config while feeling
smooth. The block velocity-reset (pymunk `damping=0`) means it stops the instant
you stop pushing, exactly like the reference.

Engine interface the UI relies on: `reset()`, `advance(action, n)` /
`step(action)`, `agentPos()`, `blockPose()`, `blockKeypoints()`, plus
`pointInT()` / `T_BOXES` for rendering and coverage.

## Not yet built

The FastAPI upload backend, dashboard, and the richer client-side validation
(too-short / too-jittery / etc.) from `project_plan.md`. Save currently
downloads the episode JSON client-side, ready to POST to that backend later.
