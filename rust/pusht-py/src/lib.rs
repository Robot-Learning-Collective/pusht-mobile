//! PyO3 bindings over pusht-core: the `pusht_core` Python module.
//!
//! Build: maturin build --release -m rust/pusht-py/Cargo.toml
//! The gymnasium wrapper lives in python/pusht_mobile_env.

use pusht_core_rs as core;
use pyo3::prelude::*;

/// PushT simulation. mode: "product" (the game/env) or "reference"
/// (bug-for-bug pymunk port used for dataset validation).
#[pyclass]
struct PushTSim {
    inner: core::PushTSim,
    samples: Vec<(f64, f64)>,
}

#[pymethods]
impl PushTSim {
    #[new]
    #[pyo3(signature = (mode = "product"))]
    fn new(mode: &str) -> PyResult<Self> {
        let mode = match mode {
            "product" => core::Mode::Product,
            "reference" => core::Mode::Reference,
            m => return Err(pyo3::exceptions::PyValueError::new_err(format!("unknown mode: {m}"))),
        };
        Ok(PushTSim {
            inner: core::PushTSim::new(mode, (256.0, 256.0), (128.0, 128.0), 0.0),
            samples: core::goal_samples(&core::GOAL_POSE),
        })
    }

    fn reset(&mut self, agent: (f64, f64), block_pos: (f64, f64), block_angle: f64) {
        self.inner.reset(agent, block_pos, block_angle);
    }

    /// One 10 Hz control step (10 physics substeps at 100 Hz).
    fn step(&mut self, action: (f64, f64)) {
        self.inner.step(action);
    }

    fn agent_pos(&self) -> (f64, f64) {
        self.inner.agent_pos()
    }

    /// (x, y, angle)
    fn block_pose(&self) -> (f64, f64, f64) {
        let b = self.inner.block_pose();
        (b.x, b.y, b.angle)
    }

    /// Fraction of the goal T covered by the block (product-scale goal).
    fn coverage(&self) -> f64 {
        core::coverage(&self.inner.block_pose(), &self.samples)
    }

    /// Seeded uniform random reset -> ((ax, ay), (bx, by), block_angle).
    #[staticmethod]
    fn random_init(seed: u32) -> ((f64, f64), (f64, f64), f64) {
        let s = core::random_init(&mut core::Rng::new(seed));
        (s.agent, s.block_pos, s.block_angle)
    }
}

#[pymodule]
fn pusht_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PushTSim>()?;
    m.add("WORLD", core::WORLD)?;
    m.add("CONTROL_HZ", core::CONTROL_HZ)?;
    m.add("DT", core::DT)?;
    m.add("SUCCESS_THRESHOLD", core::SUCCESS_THRESHOLD)?;
    m.add("PRODUCT_SCALE", core::PRODUCT_SCALE)?;
    m.add("PRODUCT_AGENT_RADIUS", core::PRODUCT_AGENT_RADIUS)?;
    m.add("GOAL_POSE", (core::GOAL_POSE.x, core::GOAL_POSE.y, core::GOAL_POSE.angle))?;
    m.add("T_OUTLINE", core::T_OUTLINE.to_vec())?;
    Ok(())
}
