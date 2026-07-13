//! Open-loop replay of lerobot/pusht dataset episodes through the Rust sim in
//! Reference mode; prints the error distribution vs the recorded ground truth.
//! Mirrors eval/fullEval.ts so the numbers are directly comparable.
//!
//!   cargo run --release --bin replay -- ../../data/episodes_all.json

use std::env;
use std::fs;

use pusht_core::{mean_keypoint_error, pose_from_keypoints, Mode, PushTSim};
use serde::Deserialize;

#[derive(Deserialize)]
struct Data {
    episodes: Vec<Episode>,
}

#[derive(Deserialize)]
struct Episode {
    episode_index: u32,
    n_frames: usize,
    agent: Vec<[f64; 2]>,
    keypoints: Vec<Vec<f64>>,
    action: Vec<[f64; 2]>,
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| "data/episodes_all.json".into());
    let data: Data = serde_json::from_str(&fs::read_to_string(&path).expect("read data file")).expect("parse json");

    let mut per_ep: Vec<(u32, f64, f64)> = Vec::new(); // (ep, kpMean, agentRmse)
    for ep in &data.episodes {
        let p0 = pose_from_keypoints(&ep.keypoints[0]);
        let mut sim = PushTSim::new(
            Mode::Reference,
            (ep.agent[0][0], ep.agent[0][1]),
            (p0.x, p0.y),
            p0.angle,
        );

        let mut agent_sq = 0.0;
        let mut kp_sum = 0.0;
        let n = ep.n_frames - 1;
        for t in 0..n {
            sim.step((ep.action[t][0], ep.action[t][1]));
            let (ax, ay) = sim.agent_pos();
            agent_sq += (ax - ep.agent[t + 1][0]).powi(2) + (ay - ep.agent[t + 1][1]).powi(2);
            kp_sum += mean_keypoint_error(&sim.block_keypoints(), &ep.keypoints[t + 1]);
        }
        per_ep.push((ep.episode_index, kp_sum / n as f64, (agent_sq / n as f64).sqrt()));
    }

    let mut kp: Vec<f64> = per_ep.iter().map(|s| s.1).collect();
    kp.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| {
        let i = (kp.len() - 1) as f64 * p;
        let (lo, hi) = (i.floor() as usize, i.ceil() as usize);
        kp[lo] + (kp[hi] - kp[lo]) * (i - i.floor())
    };
    let frac = |thr: f64| kp.iter().filter(|&&v| v <= thr).count() as f64 / kp.len() as f64 * 100.0;

    println!("{} episodes, open-loop rollout, reference mode (rust core)", kp.len());
    println!(
        "agent RMSE max:       {:.2e} px",
        per_ep.iter().map(|s| s.2).fold(0.0_f64, f64::max)
    );
    println!(
        "block kpMeanErr (px): mean {:.2}  median {:.3}  p90 {:.2}  p95 {:.2}  p99 {:.2}  max {:.2}",
        kp.iter().sum::<f64>() / kp.len() as f64,
        pct(0.5),
        pct(0.9),
        pct(0.95),
        pct(0.99),
        kp[kp.len() - 1]
    );
    println!(
        "episodes <=0.5px {:.1}%   <=1px {:.1}%   <=2px {:.1}%   <=5px {:.1}%",
        frac(0.5),
        frac(1.0),
        frac(2.0),
        frac(5.0)
    );
    let mut worst = per_ep.clone();
    worst.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let worst5: Vec<String> = worst.iter().take(5).map(|s| format!("ep{}={:.1}px", s.0, s.1)).collect();
    println!("worst 5: {}", worst5.join("  "));
}
