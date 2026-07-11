import RAPIER from "@dimforge/rapier2d-compat";

/**
 * PushT simulation ported to Rapier2D, matching the reference gym-pusht /
 * diffusion_policy pymunk environment as closely as practical.
 *
 * Reference physics (pymunk):
 *  - world 512x512 px, gravity (0,0), space.damping = 0
 *  - sim_hz = 100, control_hz = 10  => 10 physics substeps per control step
 *  - agent: KINEMATIC circle r=15, PD-driven: v += (kp*(target-pos) + kv*(0-v))*dt
 *  - block: dynamic T (mass 1), rotates about its centre of gravity which
 *    coincides with the centroid of the 8 keypoints; angular inertia 3000.
 *  - damping=0 in pymunk means the block velocity is fully reset every step and
 *    only re-acquired from contact impulses (the block never coasts). We
 *    replicate that by zeroing the block velocity before every substep.
 *
 * We work directly in the dataset's pixel coordinate frame (no axis flip):
 * gravity is zero so the y-orientation is irrelevant as long as we are
 * self-consistent, and ground-truth poses are derived in this same frame.
 */

export interface SimConfig {
  worldSize: number;
  simHz: number;
  controlHz: number;
  agentRadius: number;
  kp: number;
  kv: number;
  blockMass: number;
  blockAngularInertia: number; // only used in "reference" mass mode
  blockMassMode: "reference" | "physical";
  contactFriction: number;
  contactRestitution: number;
  linearDamping: number; // extra coasting damping (0 = pure pymunk reset behaviour)
  angularDamping: number;
  resetBlockVelEachStep: boolean;
  lengthUnit: number;
  numSolverIterations: number;
  numInternalPgsIterations: number;
  numAdditionalFrictionIterations: number;
  walls: boolean;
  wallThickness: number; // half-thickness; extends OUTWARD so the inner face is fixed
  blockCcd: boolean; // continuous collision detection on the block (anti-tunneling)
  agentMode: "kinematic" | "dynamic"; // kinematic = reference (unstoppable); dynamic = force-driven, stoppable
  agentCcd: boolean; // CCD on the agent (anti tunnel-through-block when fast)
  maxAgentSpeed: number; // px/s cap on agent velocity (0 = unlimited)
  agentMass: number; // used in dynamic mode
  agentMaxForce: number; // dynamic-mode force clamp (0 = unlimited)
  boundaryGuard: boolean; // position-level invariant: bodies projected back into the play area after each substep
  sizeScale: number; // scales the T geometry (and everything derived from it); 1 = dataset size
  contactNaturalFrequency: number | null; // null = Rapier default
  solver: "default" | "pgsSmall" | "pgsSmallNoWarmstart" | "standardPgs";
}

export const DEFAULT_CONFIG: SimConfig = {
  worldSize: 512,
  simHz: 100,
  controlHz: 10,
  // Product default: 1.5x-size T and robot (finger occlusion on phones vs
  // playability); dataset-faithful 1x geometry lives in REFERENCE_CONFIG.
  agentRadius: 22.5,
  kp: 100,
  kv: 20,
  blockMass: 1,
  blockAngularInertia: 3000,
  // Our product uses physically-correct block dynamics by default. The original
  // PushT env has a T-inertia copy-paste bug; switch to "reference" to reproduce
  // it (and match the lerobot/pusht dataset) via REFERENCE_CONFIG below.
  blockMassMode: "physical",
  contactFriction: 0, // pymunk shape default friction is 0
  contactRestitution: 0,
  linearDamping: 0,
  angularDamping: 0,
  resetBlockVelEachStep: true,
  lengthUnit: 10,
  numSolverIterations: 10,
  numInternalPgsIterations: 1,
  numAdditionalFrictionIterations: 0,
  walls: true,
  wallThickness: 40,
  blockCcd: true,
  agentMode: "dynamic",
  agentCcd: true,
  maxAgentSpeed: 0,
  agentMass: 60,
  agentMaxForce: 0,
  boundaryGuard: true,
  sizeScale: 1.5,
  contactNaturalFrequency: null,
  solver: "default",
};

/**
 * Config that reproduces the original pymunk PushT env, including its T-inertia
 * copy-paste bug. Use this to validate against the lerobot/pusht dataset — with
 * it the sim matches the recorded trajectories to the float32 noise floor.
 */
export const REFERENCE_CONFIG: Partial<SimConfig> = {
  blockMassMode: "reference",
  blockMass: 1,
  blockAngularInertia: 3000,
  agentMode: "kinematic", // the reference agent is an unstoppable kinematic body
  agentCcd: false,
  boundaryGuard: false, // pure reference physics for dataset validation
  agentRadius: 15,
  sizeScale: 1,
};

// The T as two axis-aligned boxes in body frame (origin = COM = keypoint
// centroid). Used for colliders, rendering, and coverage. {cx,cy,hx,hy}.
export const T_BOXES = [
  { cx: 0, cy: -30, hx: 60, hy: 15 }, // bar  (x[-60,60], y[-45,-15])
  { cx: 0, cy: 30, hx: 15, hy: 45 }, // stem (x[-15,15], y[-15,75])
] as const;

// The T's outline polygon in body frame (counter-clockwise walk of the real
// union boundary). Note the bar/stem seam is interior and is NOT part of it —
// distance fields built per-box are wrong there (they report ~0 depth on the
// seam), which is why the pass-through guard must use this outline instead.
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

/** True if a world point lies inside the T placed at the given pose. */
export function pointInT(px: number, py: number, pose: BlockPose, scale = 1): boolean {
  const ca = Math.cos(-pose.angle);
  const sa = Math.sin(-pose.angle);
  const dx = px - pose.x;
  const dy = py - pose.y;
  const bx = (ca * dx - sa * dy) / scale; // into (unit-scale) body frame
  const by = (sa * dx + ca * dy) / scale;
  for (const b of T_BOXES) {
    if (Math.abs(bx - b.cx) <= b.hx && Math.abs(by - b.cy) <= b.hy) return true;
  }
  return false;
}

// 8 keypoints in body frame (centroid = COM at origin), in dataset index order.
export const KEYPOINTS_BODY: [number, number][] = [
  [-60, -45], // 0
  [60, -45], // 1
  [60, -15], // 2
  [-60, -15], // 3
  [-15, -15], // 4
  [15, -15], // 5
  [15, 75], // 6
  [-15, 75], // 7
];

export interface BlockPose {
  x: number;
  y: number;
  angle: number;
}

let _rapierReady = false;
export async function initRapier(): Promise<void> {
  if (!_rapierReady) {
    await RAPIER.init();
    _rapierReady = true;
  }
}

export class PushTSim {
  cfg: SimConfig;
  private world!: RAPIER.World;
  private agent!: RAPIER.RigidBody;
  private block!: RAPIER.RigidBody;
  private agentVel = { x: 0, y: 0 };
  readonly dt: number;
  readonly substeps: number;

  constructor(cfg: Partial<SimConfig> = {}) {
    this.cfg = { ...DEFAULT_CONFIG, ...cfg };
    this.dt = 1 / this.cfg.simHz;
    this.substeps = Math.round(this.cfg.simHz / this.cfg.controlHz);
  }

  reset(agent: [number, number], blockPos: [number, number], blockAngle: number): void {
    if (this.world) this.world.free();
    const c = this.cfg;
    this.world = new RAPIER.World({ x: 0, y: 0 });
    this.world.lengthUnit = c.lengthUnit;
    const ip = this.world.integrationParameters;
    ip.dt = this.dt;
    ip.numSolverIterations = c.numSolverIterations;
    ip.numInternalPgsIterations = c.numInternalPgsIterations;
    ip.numAdditionalFrictionIterations = c.numAdditionalFrictionIterations;
    if (c.contactNaturalFrequency !== null) ip.contact_natural_frequency = c.contactNaturalFrequency;
    if (c.solver === "pgsSmall") ip.switchToSmallStepsPgsSolver();
    else if (c.solver === "pgsSmallNoWarmstart") ip.switchToSmallStepsPgsSolverWithoutWarmstart();
    else if (c.solver === "standardPgs") ip.switchToStandardPgsSolver();

    // ---- walls: thick static slabs that extend OUTWARD from a fixed inner
    // face. Thickness never changes the interior dynamics (so dataset parity is
    // preserved), but a thick wall a fast block cannot skip over in one step is
    // the robust cure for tunnelling; CCD on the block is the belt-and-braces. ----
    if (c.walls) {
      const inLo = 7; // inner face (matches the original 4px-wall inner face)
      const inHi = c.worldSize - 8;
      const t = c.wallThickness; // half-thickness, extends outward
      const mid = c.worldSize / 2;
      const span = c.worldSize / 2 + t; // half-length: covers the side + corners
      const wallBody = this.world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
      const segs: [number, number, number, number][] = [
        [mid, inLo - t, span, t], // bottom
        [mid, inHi + t, span, t], // top
        [inLo - t, mid, t, span], // left
        [inHi + t, mid, t, span], // right
      ];
      for (const [x, y, hx, hy] of segs) {
        const cd = RAPIER.ColliderDesc.cuboid(hx, hy)
          .setTranslation(x, y)
          .setFriction(c.contactFriction)
          .setRestitution(c.contactRestitution);
        this.world.createCollider(cd, wallBody);
      }
    }

    // ---- agent circle ----
    // "kinematic": velocity-imposed, unstoppable (reference / dataset).
    // "dynamic": finite-mass body driven by a PD *force*, so walls/blocks can
    //   stop it — it cannot crush the T through a wall or tunnel through it.
    this.agentVel = { x: 0, y: 0 };
    const agentDesc =
      c.agentMode === "dynamic"
        ? RAPIER.RigidBodyDesc.dynamic()
            .setTranslation(agent[0], agent[1])
            .setCcdEnabled(c.agentCcd)
            .setAdditionalMassProperties(c.agentMass, { x: 0, y: 0 }, c.agentMass * c.agentRadius * c.agentRadius)
        : RAPIER.RigidBodyDesc.kinematicVelocityBased()
            .setTranslation(agent[0], agent[1])
            .setCcdEnabled(c.agentCcd);
    this.agent = this.world.createRigidBody(agentDesc);
    this.world.createCollider(
      RAPIER.ColliderDesc.ball(c.agentRadius)
        .setDensity(0)
        .setFriction(c.contactFriction)
        .setRestitution(c.contactRestitution),
      this.agent
    );

    // ---- block: dynamic T ----
    // "reference": reproduce the pymunk env exactly, including its T-inertia
    //   copy-paste bug (inertia 3000) and its assumption that the centre of
    //   gravity is the un-weighted mean of the two box centroids (body origin).
    //   Use this to match the lerobot/pusht dataset.
    // "physical": let Rapier derive mass/COM/inertia from the actual geometry
    //   (uniform density) -> correct rigid-body dynamics (COM (0,-4.29),
    //   inertia ~1932 for mass 1). This is our product default.
    const physical = c.blockMassMode === "physical";
    const k = c.sizeScale;
    const totalArea = T_BOXES.reduce((s, b) => s + 4 * b.hx * b.hy, 0) * k * k;
    const density = physical ? c.blockMass / totalArea : 0;

    const bd = RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(blockPos[0], blockPos[1])
      .setRotation(blockAngle)
      .setLinearDamping(c.linearDamping)
      .setAngularDamping(c.angularDamping)
      .setCcdEnabled(c.blockCcd);
    if (!physical) {
      // mass, COM at body origin (0,0) = keypoint centroid, fixed inertia
      bd.setAdditionalMassProperties(c.blockMass, { x: 0, y: 0 }, c.blockAngularInertia);
    }
    this.block = this.world.createRigidBody(bd);

    for (const box of T_BOXES) {
      this.world.createCollider(
        RAPIER.ColliderDesc.cuboid(box.hx * k, box.hy * k)
          .setTranslation(box.cx * k, box.cy * k)
          .setDensity(density)
          .setFriction(c.contactFriction)
          .setRestitution(c.contactRestitution),
        this.block
      );
    }
  }

  /** One physics substep (dt) with PD control toward `action`. */
  private _substep(action: [number, number]): void {
    const c = this.cfg;
    const p = this.agent.translation();

    if (c.agentMode === "dynamic") {
      // Force-driven servo: F = m*(kp*(target-pos) + kv*(0-v)). The body is
      // dynamic, so walls and the block genuinely resist it (no crush-through,
      // no tunnelling). Read the *actual* velocity, since contacts change it.
      let v = this.agent.linvel();
      if (c.maxAgentSpeed > 0) {
        const vmag = Math.hypot(v.x, v.y);
        if (vmag > c.maxAgentSpeed) {
          const s = c.maxAgentSpeed / vmag;
          this.agent.setLinvel({ x: v.x * s, y: v.y * s }, true);
          v = this.agent.linvel();
        }
      }
      let fx = c.agentMass * (c.kp * (action[0] - p.x) + c.kv * (0 - v.x));
      let fy = c.agentMass * (c.kp * (action[1] - p.y) + c.kv * (0 - v.y));
      if (c.agentMaxForce > 0) {
        const fmag = Math.hypot(fx, fy);
        if (fmag > c.agentMaxForce) {
          const s = c.agentMaxForce / fmag;
          fx *= s;
          fy *= s;
        }
      }
      this.agent.resetForces(false);
      this.agent.addForce({ x: fx, y: fy }, true);
    } else {
      // Kinematic velocity-imposed control (matches the pymunk env exactly).
      let vx = this.agentVel.x + c.kp * (action[0] - p.x) * this.dt + c.kv * (0 - this.agentVel.x) * this.dt;
      let vy = this.agentVel.y + c.kp * (action[1] - p.y) * this.dt + c.kv * (0 - this.agentVel.y) * this.dt;
      if (c.maxAgentSpeed > 0) {
        const vmag = Math.hypot(vx, vy);
        if (vmag > c.maxAgentSpeed) {
          const s = c.maxAgentSpeed / vmag;
          vx *= s;
          vy *= s;
        }
      }
      this.agentVel.x = vx;
      this.agentVel.y = vy;
      this.agent.setLinvel({ x: vx, y: vy }, true);
    }

    // damping=0: block velocity reset before the solve, re-acquired by contact.
    if (c.resetBlockVelEachStep) {
      this.block.setLinvel({ x: 0, y: 0 }, true);
      this.block.setAngvel(0, true);
    }
    const prevA = { x: p.x, y: p.y };
    this.world.step();
    if (c.boundaryGuard) {
      this._enforceBoundary();
      this._noPassThrough(prevA);
    }
  }

  /**
   * Signed distance from a world point to the T's real surface (negative =
   * inside the union). Uses the outline polygon, NOT the two boxes: a min over
   * per-box distances reads ~0 on the interior bar/stem seam, which left the
   * pass-through guard blind exactly where the two rectangles connect.
   */
  private _tDist(wx: number, wy: number): number {
    const bp = this.block.translation();
    const a = this.block.rotation();
    const ca = Math.cos(-a);
    const sa = Math.sin(-a);
    const dx = wx - bp.x;
    const dy = wy - bp.y;
    const k = this.cfg.sizeScale;
    const px = (ca * dx - sa * dy) / k;
    const py = (sa * dx + ca * dy) / k;
    let inside = false;
    let d = Infinity;
    for (let i = 0, j = T_OUTLINE.length - 1; i < T_OUTLINE.length; j = i++) {
      const [xi, yi] = T_OUTLINE[i];
      const [xj, yj] = T_OUTLINE[j];
      // even-odd crossing test for the sign
      if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
      // distance to the edge segment
      const l2 = (xj - xi) * (xj - xi) + (yj - yi) * (yj - yi);
      const t = Math.max(0, Math.min(1, ((px - xi) * (xj - xi) + (py - yi) * (yj - yi)) / l2));
      d = Math.min(d, Math.hypot(px - (xi + t * (xj - xi)), py - (yi + t * (yj - yi))));
    }
    // body-frame coords were divided by sizeScale, so scale the distance back
    return (inside ? -d : d) * k;
  }

  /**
   * Position-level invariant #2: the agent's centre never ends a substep inside
   * the T. When the block is pinned (wall/corner + boundary guard) the agent's
   * unbounded PD force drives it past the T's 15 px half-thickness in a couple
   * of substeps; the solver then ejects it out the FAR side — a visible
   * pass-through that no stiffness/force-cap setting prevents (measured). Fix:
   * walk the centre back along this substep's motion to the surface it entered
   * through. No-op whenever the centre is outside the T.
   */
  private _noPassThrough(prev: { x: number; y: number }): void {
    const cur = this.agent.translation();
    const margin = 0.5;
    if (this._tDist(cur.x, cur.y) >= 0) return;

    if (this._tDist(prev.x, prev.y) > margin) {
      // bisect along cur -> prev for the point at distance `margin` outside
      let lo = 0; // cur (inside)
      let hi = 1; // prev (outside)
      for (let i = 0; i < 16; i++) {
        const m = (lo + hi) / 2;
        const x = cur.x + (prev.x - cur.x) * m;
        const y = cur.y + (prev.y - cur.y) * m;
        if (this._tDist(x, y) < margin) lo = m;
        else hi = m;
      }
      this.agent.setTranslation(
        { x: cur.x + (prev.x - cur.x) * hi, y: cur.y + (prev.y - cur.y) * hi },
        true
      );
    } else {
      // block moved/rotated onto the agent: eject along the local outward normal
      const d0 = this._tDist(cur.x, cur.y);
      const e = 0.5;
      let nx = (this._tDist(cur.x + e, cur.y) - d0) / e;
      let ny = (this._tDist(cur.x, cur.y + e) - d0) / e;
      const n = Math.hypot(nx, ny) || 1;
      nx /= n;
      ny /= n;
      this.agent.setTranslation(
        { x: cur.x + nx * (margin - d0), y: cur.y + ny * (margin - d0) },
        true
      );
    }
  }

  /**
   * Position-level invariant: no body ever leaves the play area.
   * Contact-solver contacts are compliant springs, so a fast heavy agent can
   * ratchet the block INTO a wall over repeated impacts (CCD can't help once
   * bodies already overlap). Instead of fighting that with stiffness (which
   * degrades the validated contact behaviour), we project the block (and the
   * agent) back inside the walls' inner faces after every substep. This is a
   * no-op unless something has already penetrated a wall, so it never affects
   * interior dynamics.
   */
  private _enforceBoundary(): void {
    const lo = 7;
    const hi = this.cfg.worldSize - 8;

    // block: translate so all keypoints are inside [lo, hi]
    const kp = this.blockKeypoints();
    let minx = Infinity;
    let miny = Infinity;
    let maxx = -Infinity;
    let maxy = -Infinity;
    for (let i = 0; i < kp.length; i += 2) {
      minx = Math.min(minx, kp[i]);
      maxx = Math.max(maxx, kp[i]);
      miny = Math.min(miny, kp[i + 1]);
      maxy = Math.max(maxy, kp[i + 1]);
    }
    let dx = 0;
    let dy = 0;
    if (minx < lo) dx = lo - minx;
    else if (maxx > hi) dx = hi - maxx;
    if (miny < lo) dy = lo - miny;
    else if (maxy > hi) dy = hi - maxy;
    if (dx !== 0 || dy !== 0) {
      const t = this.block.translation();
      this.block.setTranslation({ x: t.x + dx, y: t.y + dy }, true);
    }

    // agent: clamp its centre so the circle stays inside
    const r = this.cfg.agentRadius;
    const p = this.agent.translation();
    const cx = Math.max(lo + r, Math.min(hi - r, p.x));
    const cy = Math.max(lo + r, Math.min(hi - r, p.y));
    if (cx !== p.x || cy !== p.y) this.agent.setTranslation({ x: cx, y: cy }, true);
  }

  /** Advance one control step (`substeps` physics steps) with a fixed target.
   *  This is the unit used for dataset replay / recording (control_hz). */
  step(action: [number, number]): void {
    for (let i = 0; i < this.substeps; i++) this._substep(action);
  }

  /** Advance `n` physics substeps (for a real-time render loop). */
  advance(action: [number, number], n: number): void {
    for (let i = 0; i < n; i++) this._substep(action);
  }

  agentPos(): [number, number] {
    const p = this.agent.translation();
    return [p.x, p.y];
  }

  blockPose(): BlockPose {
    const p = this.block.translation();
    return { x: p.x, y: p.y, angle: this.block.rotation() };
  }

  /** World-space keypoints (flat [x0,y0,...]) matching the dataset layout. */
  blockKeypoints(): number[] {
    const p = this.block.translation();
    const a = this.block.rotation();
    const k = this.cfg.sizeScale;
    const ca = Math.cos(a);
    const sa = Math.sin(a);
    const out: number[] = [];
    for (const [bx, by] of KEYPOINTS_BODY) {
      out.push(p.x + (ca * bx - sa * by) * k, p.y + (sa * bx + ca * by) * k);
    }
    return out;
  }
}
