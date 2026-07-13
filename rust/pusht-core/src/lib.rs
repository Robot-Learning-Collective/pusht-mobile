//! PushT physics on Rapier2D — Rust port of `src/sim/pushtSim.ts`.
//!
//! Two modes:
//! - `Reference`: bug-for-bug port of the original pymunk env (gym-pusht /
//!   diffusion_policy), validated against the lerobot/pusht dataset.
//!   Kinematic PD agent (unstoppable), block inertia 3000 (the original's
//!   moment_for_poly copy-paste bug), dataset-size geometry.
//! - `Product`: what the game ships. Physically-derived block mass properties,
//!   a force-driven dynamic agent that walls and the block can actually stop,
//!   CCD, position-level guards against wall escapes / pass-through, and
//!   1.5x geometry for phone playability.
//!
//! Shared behaviour (both modes, matching the reference env):
//! - 512x512 px world, gravity (0,0), frictionless contacts
//! - 100 Hz physics, 10 substeps per 10 Hz control step
//! - PD control: v += (kp*(target - pos) + kv*(0 - v)) * dt
//! - pymunk `space.damping = 0` fully resets the block's velocity every step;
//!   replicated by zeroing the block velocity before every substep.
//!
//! The engine runs in f32 (same as the wasm build the data was collected on);
//! all control math and guard math is done in f64, exactly like the JS side.

use std::num::NonZeroUsize;

use rapier2d::prelude::*;

pub const WORLD: f64 = 512.0;
pub const CONTROL_HZ: u32 = 10;
pub const WALL_LO: f64 = 7.0; // walls' inner faces
pub const WALL_HI: f64 = WORLD - 8.0;

pub const PRODUCT_SCALE: f64 = 1.5;
pub const PRODUCT_AGENT_RADIUS: f64 = 22.5;
const REFERENCE_SCALE: f64 = 1.0;
const REFERENCE_AGENT_RADIUS: f64 = 15.0;

pub const DT: f64 = 0.01; // 100 Hz
const SUBSTEPS: u32 = 10;
const KP: f64 = 100.0;
const KV: f64 = 20.0;
const AGENT_MASS: f64 = 60.0; // product agent: heavy enough to push like the reference one
const BLOCK_MASS: f64 = 1.0;
const REFERENCE_INERTIA: f64 = 3000.0; // the original env's buggy value
const WALL_THICKNESS: f64 = 40.0; // half-thickness; slabs extend OUTWARD from the inner face
const GUARD_MARGIN: f64 = 0.5;

/// The T as two axis-aligned boxes in body frame at unit scale
/// (origin = COM = keypoint centroid): (cx, cy, hx, hy).
pub const T_BOXES: [(f64, f64, f64, f64); 2] = [
    (0.0, -30.0, 60.0, 15.0), // bar  (x[-60,60], y[-45,-15])
    (0.0, 30.0, 15.0, 45.0),  // stem (x[-15,15], y[-15,75])
];

/// The T's outline polygon (the real union boundary; the bar/stem seam is
/// interior and NOT part of it).
pub const T_OUTLINE: [(f64, f64); 8] = [
    (-60.0, -45.0),
    (60.0, -45.0),
    (60.0, -15.0),
    (15.0, -15.0),
    (15.0, 75.0),
    (-15.0, 75.0),
    (-15.0, -15.0),
    (-60.0, -15.0),
];

/// The 8 T keypoints in body frame, in lerobot/pusht_keypoints index order.
pub const KEYPOINTS_BODY: [(f64, f64); 8] = [
    (-60.0, -45.0),
    (60.0, -45.0),
    (60.0, -15.0),
    (-60.0, -15.0),
    (-15.0, -15.0),
    (15.0, -15.0),
    (15.0, 75.0),
    (-15.0, 75.0),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Product,
    Reference,
}

#[derive(Clone, Copy, Debug)]
pub struct BlockPose {
    pub x: f64,
    pub y: f64,
    pub angle: f64,
}

/// True if a world point lies inside the T placed at the given pose.
pub fn point_in_t(px: f64, py: f64, pose: &BlockPose, scale: f64) -> bool {
    let ca = (-pose.angle).cos();
    let sa = (-pose.angle).sin();
    let dx = px - pose.x;
    let dy = py - pose.y;
    let bx = (ca * dx - sa * dy) / scale; // into unit-scale body frame
    let by = (sa * dx + ca * dy) / scale;
    T_BOXES
        .iter()
        .any(|&(cx, cy, hx, hy)| (bx - cx).abs() <= hx && (by - cy).abs() <= hy)
}

pub struct PushTSim {
    pub mode: Mode,
    pub scale: f64,
    pub agent_radius: f64,

    pipeline: PhysicsPipeline,
    ip: IntegrationParameters,
    islands: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,

    agent: RigidBodyHandle,
    block: RigidBodyHandle,
    agent_vel: (f64, f64), // reference mode: manually integrated velocity
}

impl PushTSim {
    pub fn new(mode: Mode, agent: (f64, f64), block_pos: (f64, f64), block_angle: f64) -> Self {
        let (scale, agent_radius) = match mode {
            Mode::Product => (PRODUCT_SCALE, PRODUCT_AGENT_RADIUS),
            Mode::Reference => (REFERENCE_SCALE, REFERENCE_AGENT_RADIUS),
        };

        let mut ip = IntegrationParameters::default();
        ip.dt = DT as f32;
        ip.length_unit = 10.0; // keeps solver tolerances well-scaled for px units
        ip.num_solver_iterations = NonZeroUsize::new(10).unwrap(); // match pymunk's default

        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();

        let zero_friction = |cb: ColliderBuilder| cb.friction(0.0).restitution(0.0);

        // Walls: thick slabs extending outward from fixed inner faces.
        let t = WALL_THICKNESS;
        let mid = WORLD / 2.0;
        let span = WORLD / 2.0 + t; // covers the side plus both corners
        let walls = bodies.insert(RigidBodyBuilder::fixed());
        for (x, y, hx, hy) in [
            (mid, WALL_LO - t, span, t),
            (mid, WALL_HI + t, span, t),
            (WALL_LO - t, mid, t, span),
            (WALL_HI + t, mid, t, span),
        ] {
            colliders.insert_with_parent(
                zero_friction(ColliderBuilder::cuboid(hx as f32, hy as f32))
                    .translation(vector![x as f32, y as f32])
                    .build(),
                walls,
                &mut bodies,
            );
        }

        // Agent circle. Reference: kinematic (velocity-imposed, unstoppable).
        // Product: dynamic with finite mass, driven by a PD force; CCD on.
        let agent_body = match mode {
            Mode::Product => RigidBodyBuilder::dynamic()
                .translation(vector![agent.0 as f32, agent.1 as f32])
                .ccd_enabled(true)
                .additional_mass_properties(MassProperties::new(
                    point![0.0, 0.0],
                    AGENT_MASS as f32,
                    (AGENT_MASS * agent_radius * agent_radius) as f32,
                )),
            Mode::Reference => RigidBodyBuilder::kinematic_velocity_based()
                .translation(vector![agent.0 as f32, agent.1 as f32]),
        };
        let agent_handle = bodies.insert(agent_body);
        colliders.insert_with_parent(
            zero_friction(ColliderBuilder::ball(agent_radius as f32))
                .density(0.0)
                .build(),
            agent_handle,
            &mut bodies,
        );

        // Block: dynamic T. Reference: mass 1 with COM at the keypoint centroid
        // and the original's buggy inertia. Product: mass/COM/inertia derived
        // from the actual geometry at uniform density.
        let k = scale;
        let mut block_builder = RigidBodyBuilder::dynamic()
            .translation(vector![block_pos.0 as f32, block_pos.1 as f32])
            .rotation(block_angle as f32)
            .ccd_enabled(true);
        if mode == Mode::Reference {
            block_builder = block_builder.additional_mass_properties(MassProperties::new(
                point![0.0, 0.0],
                BLOCK_MASS as f32,
                REFERENCE_INERTIA as f32,
            ));
        }
        let block_handle = bodies.insert(block_builder);
        let area: f64 = T_BOXES.iter().map(|&(_, _, hx, hy)| 4.0 * hx * hy).sum::<f64>() * k * k;
        let density = match mode {
            Mode::Product => BLOCK_MASS / area,
            Mode::Reference => 0.0,
        };
        for (cx, cy, hx, hy) in T_BOXES {
            colliders.insert_with_parent(
                zero_friction(ColliderBuilder::cuboid((hx * k) as f32, (hy * k) as f32))
                    .translation(vector![(cx * k) as f32, (cy * k) as f32])
                    .density(density as f32)
                    .build(),
                block_handle,
                &mut bodies,
            );
        }

        PushTSim {
            mode,
            scale,
            agent_radius,
            pipeline: PhysicsPipeline::new(),
            ip,
            islands: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            bodies,
            colliders,
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            agent: agent_handle,
            block: block_handle,
            agent_vel: (0.0, 0.0),
        }
    }

    /// Fresh world at the given initial state (mirrors `reset()` on the JS side,
    /// which frees and rebuilds the whole world).
    pub fn reset(&mut self, agent: (f64, f64), block_pos: (f64, f64), block_angle: f64) {
        *self = PushTSim::new(self.mode, agent, block_pos, block_angle);
    }

    /// Advance one 10 Hz control step (the unit used for dataset replay).
    pub fn step(&mut self, action: (f64, f64)) {
        for _ in 0..SUBSTEPS {
            self.substep(action);
        }
    }

    /// Advance `n` physics substeps (for a real-time render loop).
    pub fn advance(&mut self, action: (f64, f64), n: u32) {
        for _ in 0..n {
            self.substep(action);
        }
    }

    pub fn agent_pos(&self) -> (f64, f64) {
        let p = self.bodies[self.agent].translation();
        (p.x as f64, p.y as f64)
    }

    pub fn block_pose(&self) -> BlockPose {
        let b = &self.bodies[self.block];
        let p = b.translation();
        BlockPose {
            x: p.x as f64,
            y: p.y as f64,
            angle: b.rotation().angle() as f64,
        }
    }

    /// World-space keypoints (flat [x0,y0,...]) in the dataset's layout.
    pub fn block_keypoints(&self) -> [f64; 16] {
        let pose = self.block_pose();
        let ca = pose.angle.cos();
        let sa = pose.angle.sin();
        let mut out = [0.0; 16];
        for (i, (bx, by)) in KEYPOINTS_BODY.iter().enumerate() {
            out[2 * i] = pose.x + (ca * bx - sa * by) * self.scale;
            out[2 * i + 1] = pose.y + (sa * bx + ca * by) * self.scale;
        }
        out
    }

    fn substep(&mut self, action: (f64, f64)) {
        let p = self.agent_pos();

        match self.mode {
            Mode::Product => {
                // Force-driven servo, F = m*(kp*(target-pos) + kv*(0-v)).
                // Read the real velocity — contacts change it.
                let body = &mut self.bodies[self.agent];
                let v = *body.linvel();
                body.reset_forces(false);
                body.add_force(
                    vector![
                        (AGENT_MASS * (KP * (action.0 - p.0) - KV * v.x as f64)) as f32,
                        (AGENT_MASS * (KP * (action.1 - p.1) - KV * v.y as f64)) as f32
                    ],
                    true,
                );
            }
            Mode::Reference => {
                // Kinematic velocity-imposed control, matching pymunk exactly.
                self.agent_vel.0 += (KP * (action.0 - p.0) - KV * self.agent_vel.0) * DT;
                self.agent_vel.1 += (KP * (action.1 - p.1) - KV * self.agent_vel.1) * DT;
                self.bodies[self.agent]
                    .set_linvel(vector![self.agent_vel.0 as f32, self.agent_vel.1 as f32], true);
            }
        }

        // pymunk damping=0: block velocity resets every step (never coasts).
        let block = &mut self.bodies[self.block];
        block.set_linvel(vector![0.0, 0.0], true);
        block.set_angvel(0.0, true);

        let prev = p;
        self.pipeline.step(
            &vector![0.0, 0.0],
            &self.ip,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );

        if self.mode == Mode::Product {
            self.keep_in_bounds();
            self.no_pass_through(prev);
        }
    }

    /// Position-level invariant #1: nothing leaves the play area. After every
    /// substep, project the block (and the agent) back inside the walls' inner
    /// faces. A no-op unless something has already penetrated.
    fn keep_in_bounds(&mut self) {
        let kp = self.block_keypoints();
        let (mut minx, mut miny, mut maxx, mut maxy) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for i in (0..kp.len()).step_by(2) {
            minx = minx.min(kp[i]);
            maxx = maxx.max(kp[i]);
            miny = miny.min(kp[i + 1]);
            maxy = maxy.max(kp[i + 1]);
        }
        let dx = if minx < WALL_LO {
            WALL_LO - minx
        } else if maxx > WALL_HI {
            WALL_HI - maxx
        } else {
            0.0
        };
        let dy = if miny < WALL_LO {
            WALL_LO - miny
        } else if maxy > WALL_HI {
            WALL_HI - maxy
        } else {
            0.0
        };
        if dx != 0.0 || dy != 0.0 {
            let block = &mut self.bodies[self.block];
            let t = *block.translation();
            block.set_translation(vector![t.x + dx as f32, t.y + dy as f32], true);
        }

        let r = self.agent_radius;
        let p = self.agent_pos();
        let cx = p.0.clamp(WALL_LO + r, WALL_HI - r);
        let cy = p.1.clamp(WALL_LO + r, WALL_HI - r);
        if cx != p.0 || cy != p.1 {
            self.bodies[self.agent].set_translation(vector![cx as f32, cy as f32], true);
        }
    }

    /// Position-level invariant #2: the agent's centre never ends a substep
    /// inside the T. Walk the centre back along this substep's motion to the
    /// surface it entered through, or eject along the outward normal if the
    /// block moved onto the agent.
    fn no_pass_through(&mut self, prev: (f64, f64)) {
        let cur = self.agent_pos();
        if self.t_dist(cur.0, cur.1) >= 0.0 {
            return;
        }

        if self.t_dist(prev.0, prev.1) > GUARD_MARGIN {
            // Bisect along cur -> prev for the point GUARD_MARGIN outside the T.
            let mut lo = 0.0; // cur (inside)
            let mut hi = 1.0; // prev (outside)
            for _ in 0..16 {
                let m = (lo + hi) / 2.0;
                if self.t_dist(cur.0 + (prev.0 - cur.0) * m, cur.1 + (prev.1 - cur.1) * m) < GUARD_MARGIN {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            self.bodies[self.agent].set_translation(
                vector![
                    (cur.0 + (prev.0 - cur.0) * hi) as f32,
                    (cur.1 + (prev.1 - cur.1) * hi) as f32
                ],
                true,
            );
        } else {
            // Block moved/rotated onto the agent: eject along the outward normal.
            let d0 = self.t_dist(cur.0, cur.1);
            let e = 0.5;
            let mut nx = (self.t_dist(cur.0 + e, cur.1) - d0) / e;
            let mut ny = (self.t_dist(cur.0, cur.1 + e) - d0) / e;
            let n = nx.hypot(ny);
            let n = if n == 0.0 { 1.0 } else { n };
            nx /= n;
            ny /= n;
            self.bodies[self.agent].set_translation(
                vector![
                    (cur.0 + nx * (GUARD_MARGIN - d0)) as f32,
                    (cur.1 + ny * (GUARD_MARGIN - d0)) as f32
                ],
                true,
            );
        }
    }

    /// Signed distance from a world point to the T's outline (negative = inside).
    fn t_dist(&self, wx: f64, wy: f64) -> f64 {
        let pose = self.block_pose();
        let ca = (-pose.angle).cos();
        let sa = (-pose.angle).sin();
        let dx = wx - pose.x;
        let dy = wy - pose.y;
        let px = (ca * dx - sa * dy) / self.scale;
        let py = (sa * dx + ca * dy) / self.scale;
        let mut inside = false;
        let mut d = f64::INFINITY;
        let n = T_OUTLINE.len();
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = T_OUTLINE[i];
            let (xj, yj) = T_OUTLINE[j];
            if (yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi {
                inside = !inside;
            }
            let l2 = (xj - xi).powi(2) + (yj - yi).powi(2);
            let t = (((px - xi) * (xj - xi) + (py - yi) * (yj - yi)) / l2).clamp(0.0, 1.0);
            d = d.min((px - (xi + t * (xj - xi))).hypot(py - (yi + t * (yj - yi))));
            j = i;
        }
        (if inside { -d } else { d }) * self.scale
    }
}

// ---------------------------------------------------------------------------
// Game logic (port of src/game/pushtGame.ts)
// ---------------------------------------------------------------------------

/// Reference PushT goal pose: centred, rotated 45 deg.
pub const GOAL_POSE: BlockPose = BlockPose {
    x: WORLD / 2.0,
    y: WORLD / 2.0,
    angle: std::f64::consts::FRAC_PI_4,
};

pub const SUCCESS_THRESHOLD: f64 = 0.95;

fn inside_body_t(bx: f64, by: f64) -> bool {
    // unit scale: bar x[-60,60] y[-45,-15]; stem x[-15,15] y[-15,75]
    (bx.abs() <= 60.0 && by >= -45.0 && by <= -15.0) || (bx.abs() <= 15.0 && by >= -15.0 && by <= 75.0)
}

/// Grid of world-space sample points inside the goal T (for coverage),
/// at product scale.
pub fn goal_samples(pose: &BlockPose) -> Vec<(f64, f64)> {
    let size = PRODUCT_SCALE;
    let step = 3.0 * size;
    let ca = pose.angle.cos();
    let sa = pose.angle.sin();
    let mut pts = Vec::new();
    // Sample at cell centres so points never sit exactly on the T boundary.
    let mut bx = -60.0 * size + step / 2.0;
    while bx < 60.0 * size {
        let mut by = -45.0 * size + step / 2.0;
        while by < 75.0 * size {
            if inside_body_t(bx / size, by / size) {
                pts.push((pose.x + ca * bx - sa * by, pose.y + sa * bx + ca * by));
            }
            by += step;
        }
        bx += step;
    }
    pts
}

/// Fraction of the goal area covered by the block T (product scale).
pub fn coverage(block: &BlockPose, samples: &[(f64, f64)]) -> f64 {
    let inside = samples
        .iter()
        .filter(|&&(x, y)| point_in_t(x, y, block, PRODUCT_SCALE))
        .count();
    inside as f64 / samples.len() as f64
}

/// Deterministic RNG (mulberry32) for seedable resets.
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Rng(seed)
    }

    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6D2B79F5);
        let a = self.0;
        let mut t = (a ^ (a >> 15)).wrapping_mul(a | 1);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61)) ^ t;
        (t ^ (t >> 14)) as f64 / 4294967296.0
    }
}

pub struct InitState {
    pub agent: (f64, f64),
    pub block_pos: (f64, f64),
    pub block_angle: f64,
}

/// Uniform random reset (product geometry): the T fully inside the walls,
/// not on top of the agent. Port of randomInit() in pushtGame.ts.
pub fn random_init(rng: &mut Rng) -> InitState {
    let t_bound = 15.0_f64.hypot(75.0) * PRODUCT_SCALE; // T bounding radius about its COM
    let r = PRODUCT_AGENT_RADIUS;
    let u = |lo: f64, hi: f64, rng: &mut Rng| lo + rng.next_f64() * (hi - lo);
    loop {
        let agent = (
            u(WALL_LO + r + 2.0, WALL_HI - r - 2.0, rng),
            u(WALL_LO + r + 2.0, WALL_HI - r - 2.0, rng),
        );
        let block_pos = (
            u(WALL_LO + t_bound + 2.0, WALL_HI - t_bound - 2.0, rng),
            u(WALL_LO + t_bound + 2.0, WALL_HI - t_bound - 2.0, rng),
        );
        if (agent.0 - block_pos.0).hypot(agent.1 - block_pos.1) >= t_bound + r + 10.0 {
            let block_angle = u(-std::f64::consts::PI, std::f64::consts::PI, rng);
            return InitState {
                agent,
                block_pos,
                block_angle,
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Pose helpers (port of eval/pose.ts)
// ---------------------------------------------------------------------------

/// Recover block pose from 8 world keypoints via 2D Procrustes alignment
/// against the canonical body-frame template.
pub fn pose_from_keypoints(flat: &[f64]) -> BlockPose {
    let n = KEYPOINTS_BODY.len();
    let mut cx = 0.0;
    let mut cy = 0.0;
    for i in 0..n {
        cx += flat[2 * i];
        cy += flat[2 * i + 1];
    }
    cx /= n as f64;
    cy /= n as f64;

    let mut s_cross = 0.0;
    let mut s_dot = 0.0;
    for i in 0..n {
        let (tx, ty) = KEYPOINTS_BODY[i];
        let kx = flat[2 * i] - cx;
        let ky = flat[2 * i + 1] - cy;
        s_cross += tx * ky - ty * kx;
        s_dot += tx * kx + ty * ky;
    }
    BlockPose {
        x: cx,
        y: cy,
        angle: s_cross.atan2(s_dot),
    }
}

pub fn mean_keypoint_error(sim_flat: &[f64], gt_flat: &[f64]) -> f64 {
    let n = sim_flat.len() / 2;
    let mut s = 0.0;
    for i in 0..n {
        let dx = sim_flat[2 * i] - gt_flat[2 * i];
        let dy = sim_flat[2 * i + 1] - gt_flat[2 * i + 1];
        s += dx.hypot(dy);
    }
    s / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Product-mode smoke test: seeded reset, drive the agent at the block,
    /// everything stays in bounds and the block actually moves.
    #[test]
    fn product_mode_pushes_block() {
        let mut rng = Rng::new(42);
        let init = random_init(&mut rng);
        let mut sim = PushTSim::new(Mode::Product, init.agent, init.block_pos, init.block_angle);
        let b0 = sim.block_pose();

        for _ in 0..100 {
            // aim slightly past the block's centre so we push through it
            let b = sim.block_pose();
            sim.step((b.x, b.y));
            let (ax, ay) = sim.agent_pos();
            assert!(ax >= WALL_LO && ax <= WALL_HI && ay >= WALL_LO && ay <= WALL_HI);
            let kp = sim.block_keypoints();
            for i in (0..kp.len()).step_by(2) {
                assert!(kp[i] >= WALL_LO - 1.0 && kp[i] <= WALL_HI + 1.0);
                assert!(kp[i + 1] >= WALL_LO - 1.0 && kp[i + 1] <= WALL_HI + 1.0);
            }
        }
        let b1 = sim.block_pose();
        assert!((b1.x - b0.x).hypot(b1.y - b0.y) > 5.0, "block did not move");
    }

    /// Coverage is ~1 when the block sits exactly on the goal, ~0 far away.
    #[test]
    fn coverage_sanity() {
        let samples = goal_samples(&GOAL_POSE);
        assert!(coverage(&GOAL_POSE, &samples) >= SUCCESS_THRESHOLD);
        let far = BlockPose { x: 100.0, y: 100.0, angle: 0.0 };
        assert!(coverage(&far, &samples) < 0.5);
    }

    /// Seeded RNG is deterministic and random_init respects its constraints.
    #[test]
    fn seeded_init_deterministic() {
        let a = random_init(&mut Rng::new(7));
        let b = random_init(&mut Rng::new(7));
        assert_eq!(a.agent, b.agent);
        assert_eq!(a.block_pos, b.block_pos);
        assert_eq!(a.block_angle, b.block_angle);
        let t_bound = 15.0_f64.hypot(75.0) * PRODUCT_SCALE;
        let d = (a.agent.0 - a.block_pos.0).hypot(a.agent.1 - a.block_pos.1);
        assert!(d >= t_bound + PRODUCT_AGENT_RADIUS + 10.0);
    }
}
