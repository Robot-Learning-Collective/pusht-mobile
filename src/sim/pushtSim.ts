import RAPIER from "@dimforge/rapier2d-compat";

/**
 * PushT physics on Rapier2D. Two modes:
 *
 * - "reference": bug-for-bug port of the original pymunk env (gym-pusht /
 *   diffusion_policy), used to validate against the lerobot/pusht dataset.
 *   Kinematic PD agent (unstoppable), block inertia 3000 — reproducing the
 *   original's moment_for_poly copy-paste bug — and dataset-size geometry.
 *   Matches the recorded trajectories to ~0.1 px median (see README).
 *
 * - "product": what the game ships. Physically-derived block mass properties,
 *   a force-driven dynamic agent that walls and the block can actually stop,
 *   CCD, position-level guards against wall escapes / pass-through, and
 *   1.5x geometry for phone playability.
 *
 * Shared behaviour (both modes, matching the reference env):
 * - 512x512 px world, gravity (0,0), frictionless contacts
 * - 100 Hz physics, 10 substeps per 10 Hz control step
 * - PD control: v += (kp*(target - pos) + kv*(0 - v)) * dt
 * - pymunk `space.damping = 0` fully resets the block's velocity every step;
 *   it only re-acquires motion from contact impulses and never coasts. We
 *   replicate that by zeroing the block velocity before every substep.
 */

export type SimMode = "product" | "reference";

export interface BlockPose {
  x: number;
  y: number;
  angle: number;
}

export const WORLD = 512;
export const CONTROL_HZ = 10;
export const WALL = { lo: 7, hi: WORLD - 8 }; // walls' inner faces
export const PRODUCT = { scale: 1.5, agentRadius: 22.5 };
const REFERENCE = { scale: 1, agentRadius: 15 };

const DT = 0.01; // 100 Hz
const SUBSTEPS = 10;
const KP = 100;
const KV = 20;
const AGENT_MASS = 60; // product agent: heavy enough to push like the reference one
const BLOCK_MASS = 1;
const REFERENCE_INERTIA = 3000; // the original env's buggy value (see README)
const WALL_THICKNESS = 40; // half-thickness; slabs extend OUTWARD from the inner face
const GUARD_MARGIN = 0.5;

// The T as two axis-aligned boxes in body frame at unit scale (origin = COM =
// keypoint centroid). Used for colliders, rendering, and coverage.
export const T_BOXES = [
  { cx: 0, cy: -30, hx: 60, hy: 15 }, // bar  (x[-60,60], y[-45,-15])
  { cx: 0, cy: 30, hx: 15, hy: 45 }, // stem (x[-15,15], y[-15,75])
] as const;

// The T's outline polygon (the real union boundary; the bar/stem seam is
// interior and NOT part of it — a per-box distance field reads ~0 depth on the
// seam, which is why the pass-through guard must use this outline).
export const T_OUTLINE: readonly [number, number][] = [
  [-60, -45],
  [60, -45],
  [60, -15],
  [15, -15],
  [15, 75],
  [-15, 75],
  [-15, -15],
  [-60, -15],
];

// The 8 T keypoints in body frame, in lerobot/pusht_keypoints index order.
export const KEYPOINTS_BODY: readonly [number, number][] = [
  [-60, -45],
  [60, -45],
  [60, -15],
  [-60, -15],
  [-15, -15],
  [15, -15],
  [15, 75],
  [-15, 75],
];

/** True if a world point lies inside the T placed at the given pose. */
export function pointInT(px: number, py: number, pose: BlockPose, scale = 1): boolean {
  const ca = Math.cos(-pose.angle);
  const sa = Math.sin(-pose.angle);
  const dx = px - pose.x;
  const dy = py - pose.y;
  const bx = (ca * dx - sa * dy) / scale; // into unit-scale body frame
  const by = (sa * dx + ca * dy) / scale;
  for (const b of T_BOXES) {
    if (Math.abs(bx - b.cx) <= b.hx && Math.abs(by - b.cy) <= b.hy) return true;
  }
  return false;
}

let rapierReady = false;
export async function initRapier(): Promise<void> {
  if (!rapierReady) {
    await RAPIER.init();
    rapierReady = true;
  }
}

export class PushTSim {
  readonly mode: SimMode;
  readonly scale: number;
  readonly agentRadius: number;
  readonly dt = DT;
  private world!: RAPIER.World;
  private agent!: RAPIER.RigidBody;
  private block!: RAPIER.RigidBody;
  private agentVel = { x: 0, y: 0 }; // reference mode: manually integrated velocity

  constructor(mode: SimMode = "product") {
    this.mode = mode;
    const g = mode === "product" ? PRODUCT : REFERENCE;
    this.scale = g.scale;
    this.agentRadius = g.agentRadius;
  }

  reset(agent: [number, number], blockPos: [number, number], blockAngle: number): void {
    if (this.world) this.world.free();
    this.world = new RAPIER.World({ x: 0, y: 0 });
    this.world.lengthUnit = 10; // keeps solver tolerances well-scaled for px units
    const ip = this.world.integrationParameters;
    ip.dt = DT;
    ip.numSolverIterations = 10; // match pymunk's default iteration count

    const zeroFriction = (cd: RAPIER.ColliderDesc) => cd.setFriction(0).setRestitution(0);

    // Walls: thick slabs extending outward from fixed inner faces, so a fast
    // block can't skip over them in one step and thickness never changes the
    // interior dynamics.
    const t = WALL_THICKNESS;
    const mid = WORLD / 2;
    const span = WORLD / 2 + t; // covers the side plus both corners
    const walls = this.world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
    for (const [x, y, hx, hy] of [
      [mid, WALL.lo - t, span, t],
      [mid, WALL.hi + t, span, t],
      [WALL.lo - t, mid, t, span],
      [WALL.hi + t, mid, t, span],
    ] as const) {
      this.world.createCollider(zeroFriction(RAPIER.ColliderDesc.cuboid(hx, hy).setTranslation(x, y)), walls);
    }

    // Agent circle. Reference: kinematic (velocity-imposed, unstoppable).
    // Product: dynamic with finite mass, driven by a PD force, so walls and
    // the block genuinely stop it; CCD against tunnelling when fast.
    this.agentVel = { x: 0, y: 0 };
    this.agent = this.world.createRigidBody(
      this.mode === "product"
        ? RAPIER.RigidBodyDesc.dynamic()
            .setTranslation(agent[0], agent[1])
            .setCcdEnabled(true)
            .setAdditionalMassProperties(AGENT_MASS, { x: 0, y: 0 }, AGENT_MASS * this.agentRadius ** 2)
        : RAPIER.RigidBodyDesc.kinematicVelocityBased().setTranslation(agent[0], agent[1])
    );
    this.world.createCollider(zeroFriction(RAPIER.ColliderDesc.ball(this.agentRadius).setDensity(0)), this.agent);

    // Block: dynamic T. Reference: mass 1 with COM at the keypoint centroid and
    // the original's buggy inertia. Product: Rapier derives mass/COM/inertia
    // from the actual geometry at uniform density.
    const k = this.scale;
    const blockDesc = RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(blockPos[0], blockPos[1])
      .setRotation(blockAngle)
      .setCcdEnabled(true);
    if (this.mode === "reference") {
      blockDesc.setAdditionalMassProperties(BLOCK_MASS, { x: 0, y: 0 }, REFERENCE_INERTIA);
    }
    this.block = this.world.createRigidBody(blockDesc);
    const area = T_BOXES.reduce((s, b) => s + 4 * b.hx * b.hy, 0) * k * k;
    const density = this.mode === "product" ? BLOCK_MASS / area : 0;
    for (const b of T_BOXES) {
      this.world.createCollider(
        zeroFriction(
          RAPIER.ColliderDesc.cuboid(b.hx * k, b.hy * k).setTranslation(b.cx * k, b.cy * k).setDensity(density)
        ),
        this.block
      );
    }
  }

  /** Advance one 10 Hz control step (the unit used for dataset replay). */
  step(action: [number, number]): void {
    for (let i = 0; i < SUBSTEPS; i++) this.substep(action);
  }

  /** Advance `n` physics substeps (for a real-time render loop). */
  advance(action: [number, number], n: number): void {
    for (let i = 0; i < n; i++) this.substep(action);
  }

  agentPos(): [number, number] {
    const p = this.agent.translation();
    return [p.x, p.y];
  }

  blockPose(): BlockPose {
    const p = this.block.translation();
    return { x: p.x, y: p.y, angle: this.block.rotation() };
  }

  /** World-space keypoints (flat [x0,y0,...]) in the dataset's layout. */
  blockKeypoints(): number[] {
    const p = this.block.translation();
    const a = this.block.rotation();
    const ca = Math.cos(a);
    const sa = Math.sin(a);
    const out: number[] = [];
    for (const [bx, by] of KEYPOINTS_BODY) {
      out.push(p.x + (ca * bx - sa * by) * this.scale, p.y + (sa * bx + ca * by) * this.scale);
    }
    return out;
  }

  private substep(action: [number, number]): void {
    const p = this.agent.translation();

    if (this.mode === "product") {
      // Force-driven servo, F = m*(kp*(target-pos) + kv*(0-v)). Unobstructed it
      // tracks exactly like the kinematic agent (mass cancels), but contacts
      // genuinely resist it. Read the real velocity — contacts change it.
      const v = this.agent.linvel();
      this.agent.resetForces(false);
      this.agent.addForce(
        {
          x: AGENT_MASS * (KP * (action[0] - p.x) - KV * v.x),
          y: AGENT_MASS * (KP * (action[1] - p.y) - KV * v.y),
        },
        true
      );
    } else {
      // Kinematic velocity-imposed control, matching the pymunk env exactly.
      this.agentVel.x += (KP * (action[0] - p.x) - KV * this.agentVel.x) * DT;
      this.agentVel.y += (KP * (action[1] - p.y) - KV * this.agentVel.y) * DT;
      this.agent.setLinvel(this.agentVel, true);
    }

    // pymunk damping=0: block velocity resets every step (never coasts).
    this.block.setLinvel({ x: 0, y: 0 }, true);
    this.block.setAngvel(0, true);

    const prev = { x: p.x, y: p.y };
    this.world.step();

    if (this.mode === "product") {
      this.keepInBounds();
      this.noPassThrough(prev);
    }
  }

  /**
   * Position-level invariant #1: nothing leaves the play area. Solver contacts
   * are compliant springs, so a fast heavy agent can ratchet the block INTO a
   * wall over repeated impacts (CCD can't help once bodies already overlap).
   * After every substep, project the block (and the agent) back inside the
   * walls' inner faces. A no-op unless something has already penetrated.
   */
  private keepInBounds(): void {
    const kp = this.blockKeypoints();
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (let i = 0; i < kp.length; i += 2) {
      minx = Math.min(minx, kp[i]);
      maxx = Math.max(maxx, kp[i]);
      miny = Math.min(miny, kp[i + 1]);
      maxy = Math.max(maxy, kp[i + 1]);
    }
    const dx = minx < WALL.lo ? WALL.lo - minx : maxx > WALL.hi ? WALL.hi - maxx : 0;
    const dy = miny < WALL.lo ? WALL.lo - miny : maxy > WALL.hi ? WALL.hi - maxy : 0;
    if (dx !== 0 || dy !== 0) {
      const t = this.block.translation();
      this.block.setTranslation({ x: t.x + dx, y: t.y + dy }, true);
    }

    const r = this.agentRadius;
    const p = this.agent.translation();
    const cx = Math.max(WALL.lo + r, Math.min(WALL.hi - r, p.x));
    const cy = Math.max(WALL.lo + r, Math.min(WALL.hi - r, p.y));
    if (cx !== p.x || cy !== p.y) this.agent.setTranslation({ x: cx, y: cy }, true);
  }

  /**
   * Position-level invariant #2: the agent's centre never ends a substep inside
   * the T. When the block is pinned (wall/corner + keepInBounds) the agent's
   * unbounded PD force drives its centre past the T's half-thickness and the
   * solver ejects it out the FAR side — a pass-through no stiffness or force
   * cap prevents (measured). Fix: walk the centre back along this substep's
   * motion to the surface it entered through.
   */
  private noPassThrough(prev: { x: number; y: number }): void {
    const cur = this.agent.translation();
    if (this.tDist(cur.x, cur.y) >= 0) return;

    if (this.tDist(prev.x, prev.y) > GUARD_MARGIN) {
      // Bisect along cur -> prev for the point GUARD_MARGIN outside the T.
      let lo = 0; // cur (inside)
      let hi = 1; // prev (outside)
      for (let i = 0; i < 16; i++) {
        const m = (lo + hi) / 2;
        if (this.tDist(cur.x + (prev.x - cur.x) * m, cur.y + (prev.y - cur.y) * m) < GUARD_MARGIN) lo = m;
        else hi = m;
      }
      this.agent.setTranslation({ x: cur.x + (prev.x - cur.x) * hi, y: cur.y + (prev.y - cur.y) * hi }, true);
    } else {
      // Block moved/rotated onto the agent: eject along the outward normal.
      const d0 = this.tDist(cur.x, cur.y);
      const e = 0.5;
      let nx = (this.tDist(cur.x + e, cur.y) - d0) / e;
      let ny = (this.tDist(cur.x, cur.y + e) - d0) / e;
      const n = Math.hypot(nx, ny) || 1;
      nx /= n;
      ny /= n;
      this.agent.setTranslation({ x: cur.x + nx * (GUARD_MARGIN - d0), y: cur.y + ny * (GUARD_MARGIN - d0) }, true);
    }
  }

  /** Signed distance from a world point to the T's outline (negative = inside). */
  private tDist(wx: number, wy: number): number {
    const bp = this.block.translation();
    const a = this.block.rotation();
    const ca = Math.cos(-a);
    const sa = Math.sin(-a);
    const dx = wx - bp.x;
    const dy = wy - bp.y;
    const px = (ca * dx - sa * dy) / this.scale;
    const py = (sa * dx + ca * dy) / this.scale;
    let inside = false;
    let d = Infinity;
    for (let i = 0, j = T_OUTLINE.length - 1; i < T_OUTLINE.length; j = i++) {
      const [xi, yi] = T_OUTLINE[i];
      const [xj, yj] = T_OUTLINE[j];
      if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
      const l2 = (xj - xi) ** 2 + (yj - yi) ** 2;
      const t = Math.max(0, Math.min(1, ((px - xi) * (xj - xi) + (py - yi) * (yj - yi)) / l2));
      d = Math.min(d, Math.hypot(px - (xi + t * (xj - xi)), py - (yi + t * (yj - yi))));
    }
    return (inside ? -d : d) * this.scale;
  }
}
