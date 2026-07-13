//! wasm-bindgen bindings over pusht-core for the browser UI.
//!
//! Build: wasm-pack build rust/pusht-wasm --target web --release
//! (or `npm run build:wasm` from the repo root)

use std::sync::OnceLock;

use pusht_core as core;
use wasm_bindgen::prelude::*;

fn samples() -> &'static Vec<(f64, f64)> {
    static S: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    S.get_or_init(|| core::goal_samples(&core::GOAL_POSE))
}

/// Product-mode PushT simulation (what the game ships).
#[wasm_bindgen]
pub struct Sim(core::PushTSim);

#[wasm_bindgen]
impl Sim {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Sim {
        // Placeholder state; the game always calls reset() before use.
        Sim(core::PushTSim::new(core::Mode::Product, (256.0, 256.0), (128.0, 128.0), 0.0))
    }

    pub fn reset(&mut self, ax: f64, ay: f64, bx: f64, by: f64, block_angle: f64) {
        self.0.reset((ax, ay), (bx, by), block_angle);
    }

    /// Advance `n` physics substeps (100 Hz) toward the target action.
    pub fn advance(&mut self, tx: f64, ty: f64, n: u32) {
        self.0.advance((tx, ty), n);
    }

    /// One 10 Hz control step (10 substeps).
    pub fn step(&mut self, tx: f64, ty: f64) {
        self.0.step((tx, ty));
    }

    #[wasm_bindgen(js_name = agentPos)]
    pub fn agent_pos(&self) -> Vec<f64> {
        let p = self.0.agent_pos();
        vec![p.0, p.1]
    }

    /// [x, y, angle]
    #[wasm_bindgen(js_name = blockPose)]
    pub fn block_pose(&self) -> Vec<f64> {
        let b = self.0.block_pose();
        vec![b.x, b.y, b.angle]
    }

    /// Fraction of the goal T covered by the block.
    pub fn coverage(&self) -> f64 {
        core::coverage(&self.0.block_pose(), samples())
    }

    #[wasm_bindgen(getter)]
    pub fn dt(&self) -> f64 {
        core::DT
    }
}

/// Seeded uniform random reset: [ax, ay, bx, by, blockAngle].
#[wasm_bindgen(js_name = randomInit)]
pub fn random_init(seed: u32) -> Vec<f64> {
    let s = core::random_init(&mut core::Rng::new(seed));
    vec![s.agent.0, s.agent.1, s.block_pos.0, s.block_pos.1, s.block_angle]
}

// ---- constants for the UI ----

#[wasm_bindgen(js_name = world)]
pub fn world() -> f64 {
    core::WORLD
}

#[wasm_bindgen(js_name = controlHz)]
pub fn control_hz() -> u32 {
    core::CONTROL_HZ
}

#[wasm_bindgen(js_name = successThreshold)]
pub fn success_threshold() -> f64 {
    core::SUCCESS_THRESHOLD
}

#[wasm_bindgen(js_name = productScale)]
pub fn product_scale() -> f64 {
    core::PRODUCT_SCALE
}

#[wasm_bindgen(js_name = agentRadius)]
pub fn agent_radius() -> f64 {
    core::PRODUCT_AGENT_RADIUS
}

/// [x, y, angle]
#[wasm_bindgen(js_name = goalPose)]
pub fn goal_pose() -> Vec<f64> {
    vec![core::GOAL_POSE.x, core::GOAL_POSE.y, core::GOAL_POSE.angle]
}

/// The T as two boxes in body frame at unit scale, flat [cx, cy, hx, hy] x2
/// (for rendering).
#[wasm_bindgen(js_name = tBoxes)]
pub fn t_boxes() -> Vec<f64> {
    core::T_BOXES.iter().flat_map(|&(cx, cy, hx, hy)| [cx, cy, hx, hy]).collect()
}
