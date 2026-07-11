## Physics + UI design

### Priority

Use **client-side simulation** because server-side stepping would add latency and make the public demo fragile.

But since physics quality matters, do **not** use a toy hand-written simulator. Use a real browser physics engine.

PushT itself is simple in state/action size, but the physics is contact-rich: a circular agent pushes a compound T-shaped rigid body into a target. The official `gym-pusht` task uses 2D continuous target-position actions, 5D state, optional pixels, coverage reward, and 95% overlap success.

## Physics engine options

### Option A — Rapier2D + PixiJS

Recommended.

Rapier is a modern Rust physics engine with official WebAssembly/JavaScript support, 2D/3D rigid bodies, contact events, scene queries, and optional cross-platform determinism.

Use:

* Rapier2D for physics
* PixiJS for rendering
* plain HTML/CSS for name input, buttons, and dashboard overlay

Pros:

* good physics quality
* browser-native
* mobile-friendly
* better long-term choice than Matter.js
* supports fixed-timestep simulation
* clean separation: physics state → renderer

Cons:

* not identical to official `gym-pusht`
* needs tuning: mass, friction, damping, solver settings
* WASM adds small integration complexity

Best fit: **high-quality lecture demo and useful IL data collection.**

### Option B — Chipmunk/Munk2D in WASM

Best physics parity, highest risk.

Official Pymunk is built on Munk2D, a fork of Chipmunk2D. Chipmunk2D itself is a lightweight C 2D rigid-body physics library.

Pros:

* closest to official PushT physics family
* best chance of matching Python/Pymunk behavior
* conceptually clean: browser runs almost the same kind of engine

Cons:

* more custom WASM/bindings work
* weaker JS ecosystem
* higher integration/debugging risk
* harder to finish reliably before a public lecture

Best fit: **only if exact PushT compatibility becomes important.**

### Option C — Planck.js / Box2D-style

Good fallback.

Planck.js is a JavaScript/TypeScript rewrite of Box2D for cross-platform HTML5 game development. Box2D is a standard 2D rigid-body engine for games.

Pros:

* mature 2D physics model
* pure JS/TS, no WASM complexity
* good examples
* likely enough for PushT-style pushing

Cons:

* not official PushT physics
* older ecosystem feel
* less attractive than Rapier for new code

Best fit: **fallback if Rapier integration is annoying.**

### Option D — Matter.js

Demote.

Matter.js is easy and browser-native, but I would not choose it now if physics quality matters. It is good for quick demos, but less convincing for contact-rich data collection. Phaser also uses Matter.js for its advanced physics mode, so Phaser does not solve the “better physics” problem.

Best fit: **prototype only, not final MVP.**

## UI/rendering options

### Option 1 — PixiJS canvas + HTML overlay

Recommended.

PixiJS uses WebGL/WebGL2, with Canvas fallback, and is designed for high-performance 2D rendering on the web.

Use PixiJS for:

* robot circle
* T block
* target zone
* success animation
* live visual feedback

Use normal HTML/CSS for:

* name form
* restart button
* save button
* submitted episode count
* rejection messages

Pros:

* smooth mobile rendering
* clean architecture
* not locked into a game framework
* easy to pair with Rapier

Cons:

* imperative rendering code
* need to manage coordinate transforms manually

Recommended UI architecture.

### Option 2 — Phaser

Full game framework.

Phaser supports Canvas/WebGL and mobile web games, and includes Matter.js physics support.

Pros:

* many game features included
* input, scenes, assets, game loop are already structured
* useful if this becomes a richer game-like demo

Cons:

* pushes us toward Matter.js
* more framework than needed
* harder to keep product simple
* UI/dashboard less natural than normal web app

Not recommended for this MVP.

### Option 3 — Konva

Good canvas UI library, weaker simulation fit.

Konva supports pointer events that work across mobile and desktop, including `pointerdown`, `pointermove`, and `pointerup`.

Pros:

* easy shape interaction
* good for editable canvas-style apps
* nice event model

Cons:

* less suitable for a tight physics/rendering game loop
* not a physics/rendering-performance-first choice

Possible, but PixiJS is better here.

### Option 4 — Plain Canvas 2D

Simplest but lower ceiling.

Pros:

* no rendering dependency
* easy to debug
* small bundle

Cons:

* more manual work
* weaker performance headroom
* no scene graph
* less nice if we add animations/replay

Acceptable for prototype, but PixiJS is cleaner.

## Recommended stack

Use:

* **Rapier2D** for physics
* **PixiJS** for rendering
* **TypeScript + Vite**
* **plain HTML/CSS overlay**
* **FastAPI backend**
* **SQLite/Postgres**
* **JSON trajectory upload**
* **server-side Python validation/export later**

Do not use:

* React inside the simulation loop
* server-side live stepping
* video upload
* Matter.js as final physics engine

## Simulation design

Use a fixed physics timestep:

* physics runs at 60 Hz
* rendering follows screen refresh
* pointer input is sampled into target actions
* uploaded episode stores resampled state/action frames

Record:

* engine name/version
* physics parameters
* initial state
* timestep
* per-step state
* per-step action
* success flag
* validation metadata

This makes episodes reproducible enough for teaching and later dataset export.

## Mobile control design

Initial screen:

> Touch robot to start

Behavior:

* recording is paused on reset
* user must touch the robot circle to begin
* after start, finger position becomes the 2D target action
* action is smoothed/clipped
* if finger leaves screen, keep last action briefly, then stop episode or require restart

Buttons:

* Restart
* Save episode

Visible status:

* total successfully submitted episodes
* rejection reason
* saved confirmation

No reward display.
No coverage display.
No pause button.

## Client-side validation

Reject before upload if:

* too short
* robot barely moved
* block barely moved
* no success
* too jittery
* too many action jumps
* too many missing frames
* episode duration too long
* physics exploded / NaNs / out-of-bounds

Show one clear reason, then reset.

Example messages:

* “Too short — try again.”
* “The block barely moved — push the T shape.”
* “Too jittery — move more smoothly.”
* “Not solved yet — try again.”
* “Tracking issue — please restart.”

## Video decision

Do not upload video from phone.

For poor mobile internet, video upload is the wrong default. PushT data is low-dimensional, so upload JSON trajectory only. A 10–20 second episode at 30–60 Hz is small as state/action data, but much larger and less reliable as video.

If video is needed, render it later on the backend from saved states.

## Dashboard

MVP dashboard shows only:

* total successful submitted episodes
* sorted table by participant:

  * name
  * successful episode count

No replay UI in MVP.

## Final recommendation

Build the MVP as:

> Rapier2D physics + PixiJS rendering + TypeScript mobile web app + FastAPI backend.

This gives good physics, low latency, small uploads, and a robust public lecture demo.

Keep Python `gym-pusht` for later validation/export, not for live interaction.

