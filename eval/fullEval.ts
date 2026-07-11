import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { PushTSim, initRapier, REFERENCE_CONFIG, type SimConfig } from "../src/sim/pushtSim.js";
import { poseFromKeypoints, angleDiff, meanKeypointError } from "./pose.js";

const r2d = (r: number) => (r * 180) / Math.PI;
const DATA = resolve("/home/zaringleb/pusht-mobile", process.env.EPISODES ?? "data/episodes_all.json");

interface EpStat {
  ep: number;
  frames: number;
  agentRmse: number;
  kpMean: number;
  kpMax: number;
  angMean: number;
  travel: number;
  rot: number;
}

function pct(sorted: number[], p: number): number {
  const i = (sorted.length - 1) * p;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

export function evalAll(cfg: Partial<SimConfig>): EpStat[] {
  const { episodes } = JSON.parse(readFileSync(DATA, "utf8")) as { episodes: any[] };
  const sim = new PushTSim({ ...REFERENCE_CONFIG, ...cfg });
  const stats: EpStat[] = [];
  for (const ep of episodes) {
    const N = ep.n_frames;
    if (N < 2) continue;
    const poses = ep.keypoints.map((k: number[]) => poseFromKeypoints(k));
    let travel = 0;
    let rot = 0;
    for (let t = 1; t < N; t++) {
      travel += Math.hypot(poses[t].x - poses[t - 1].x, poses[t].y - poses[t - 1].y);
      rot += Math.abs(r2d(angleDiff(poses[t].angle, poses[t - 1].angle)));
    }
    sim.reset(ep.agent[0], [poses[0].x, poses[0].y], poses[0].angle);
    let aSq = 0;
    let kpSum = 0;
    let kpMax = 0;
    let angSum = 0;
    for (let t = 0; t < N - 1; t++) {
      sim.step(ep.action[t]);
      const sp = sim.blockPose();
      const kpe = meanKeypointError(sim.blockKeypoints(), ep.keypoints[t + 1]);
      kpSum += kpe;
      kpMax = Math.max(kpMax, kpe);
      angSum += Math.abs(r2d(angleDiff(sp.angle, poses[t + 1].angle)));
      const ae = Math.hypot(sim.agentPos()[0] - ep.agent[t + 1][0], sim.agentPos()[1] - ep.agent[t + 1][1]);
      aSq += ae * ae;
    }
    stats.push({
      ep: ep.episode_index,
      frames: N,
      agentRmse: Math.sqrt(aSq / (N - 1)),
      kpMean: kpSum / (N - 1),
      kpMax,
      angMean: angSum / (N - 1),
      travel,
      rot,
    });
  }
  return stats;
}

export function report(label: string, stats: EpStat[]): void {
  const kp = stats.map((s) => s.kpMean).sort((a, b) => a - b);
  const agent = stats.map((s) => s.agentRmse).sort((a, b) => a - b);
  const n = stats.length;
  const mean = kp.reduce((a, b) => a + b, 0) / n;
  const frac = (thr: number) => stats.filter((s) => s.kpMean <= thr).length / n;

  console.log(`\n=== ${label}: full dataset, ${n} episodes (open-loop rollout) ===`);
  console.log(`agent RMSE:           max ${agent[n - 1].toExponential(2)} px  (kinematic+PD is deterministic)`);
  console.log(`block kpMeanErr (px): mean ${mean.toFixed(2)}  median ${pct(kp, 0.5).toFixed(3)}  p90 ${pct(kp, 0.9).toFixed(2)}  p95 ${pct(kp, 0.95).toFixed(2)}  p99 ${pct(kp, 0.99).toFixed(2)}  max ${kp[n - 1].toFixed(2)}`);
  console.log(
    `fraction of episodes:  <=0.5px ${(frac(0.5) * 100).toFixed(1)}%   <=1px ${(frac(1) * 100).toFixed(1)}%   <=2px ${(frac(2) * 100).toFixed(1)}%   <=5px ${(frac(5) * 100).toFixed(1)}%`,
  );
  const worst = [...stats].sort((a, b) => b.kpMean - a.kpMean).slice(0, 10);
  console.log(`worst 10 episodes (ep: kpMean px / kpMax px / angMean deg / travel px / rot deg):`);
  for (const s of worst) {
    console.log(
      `  ${String(s.ep).padStart(3)}: ${s.kpMean.toFixed(2).padStart(6)} / ${s.kpMax
        .toFixed(1)
        .padStart(6)} / ${s.angMean.toFixed(2).padStart(6)} / ${s.travel.toFixed(0).padStart(5)} / ${s.rot
        .toFixed(0)
        .padStart(4)}`,
    );
  }
}

const variant = process.env.VARIANT ?? "default";
const cfgs: Record<string, Partial<SimConfig>> = {
  default: {},
  natFreq10: { contactNaturalFrequency: 10 },
};
await initRapier();
report(variant, evalAll(cfgs[variant] ?? {}));
