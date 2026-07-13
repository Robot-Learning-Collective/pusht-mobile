//! Deterministic product-mode trajectory dump, for native-vs-wasm parity
//! checks (compare with eval/wasmParity.mjs output).

use pusht_core::*;

fn main() {
    let mut rng = Rng::new(42);
    let init = random_init(&mut rng);
    let mut sim = PushTSim::new(Mode::Product, init.agent, init.block_pos, init.block_angle);
    println!(
        "init agent=({:.12},{:.12}) block=({:.12},{:.12},{:.12})",
        init.agent.0, init.agent.1, init.block_pos.0, init.block_pos.1, init.block_angle
    );
    for t in 0..200u32 {
        let b = sim.block_pose();
        sim.step((b.x, b.y)); // chase the block's centre: guarantees rich contact
        if (t + 1) % 50 == 0 {
            let (ax, ay) = sim.agent_pos();
            let b = sim.block_pose();
            println!(
                "t={} agent=({:.12},{:.12}) block=({:.12},{:.12},{:.12})",
                t + 1,
                ax,
                ay,
                b.x,
                b.y,
                b.angle
            );
        }
    }
    let samples = goal_samples(&GOAL_POSE);
    println!("coverage={:.12}", coverage(&sim.block_pose(), &samples));
}
