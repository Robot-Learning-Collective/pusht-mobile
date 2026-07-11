/**
 * Open-loop replay of ALL dataset episodes through the sim in "reference" mode;
 * prints the error distribution and the worst offenders.
 *
 *   npx tsx eval/fullEval.ts [data/episodes_all.json]
 */
import { readFileSync } from "node:fs";
import { PushTSim, initRapier } from "../src/sim/pushtSim.js";
import { poseFromKeypoints, meanKeypointError } from "./pose.js";

await initRapier();
const path = process.argv[2] ?? "data/episodes_all.json";
const { episodes } = JSON.parse(readFileSync(path, "utf8")) as { episodes: any[] };
const sim = new PushTSim("reference");

const perEp: { ep: number; kpMean: number; agentRmse: number }[] = [];
for (const ep of episodes) {
  const poses = ep.keypoints.map(poseFromKeypoints);
  sim.reset(ep.agent[0], [poses[0].x, poses[0].y], poses[0].angle);
  let agentSq = 0;
  let kpSum = 0;
  const n = ep.n_frames - 1;
  for (let t = 0; t < n; t++) {
    sim.step(ep.action[t]);
    const [ax, ay] = sim.agentPos();
    agentSq += (ax - ep.agent[t + 1][0]) ** 2 + (ay - ep.agent[t + 1][1]) ** 2;
    kpSum += meanKeypointError(sim.blockKeypoints(), ep.keypoints[t + 1]);
  }
  perEp.push({ ep: ep.episode_index, kpMean: kpSum / n, agentRmse: Math.sqrt(agentSq / n) });
}

const kp = perEp.map((s) => s.kpMean).sort((a, b) => a - b);
const pct = (p: number) => {
  const i = (kp.length - 1) * p;
  return kp[Math.floor(i)] + (kp[Math.ceil(i)] - kp[Math.floor(i)]) * (i - Math.floor(i));
};
const frac = (thr: number) => ((kp.filter((v) => v <= thr).length / kp.length) * 100).toFixed(1);

console.log(`${kp.length} episodes, open-loop rollout, reference mode`);
console.log(`agent RMSE max:       ${Math.max(...perEp.map((s) => s.agentRmse)).toExponential(2)} px`);
console.log(
  `block kpMeanErr (px): mean ${(kp.reduce((a, b) => a + b) / kp.length).toFixed(2)}  median ${pct(0.5).toFixed(
    3
  )}  p90 ${pct(0.9).toFixed(2)}  p95 ${pct(0.95).toFixed(2)}  p99 ${pct(0.99).toFixed(2)}  max ${kp[
    kp.length - 1
  ].toFixed(2)}`
);
console.log(`episodes <=0.5px ${frac(0.5)}%   <=1px ${frac(1)}%   <=2px ${frac(2)}%   <=5px ${frac(5)}%`);
console.log("worst 5:", [...perEp].sort((a, b) => b.kpMean - a.kpMean).slice(0, 5).map((s) => `ep${s.ep}=${s.kpMean.toFixed(1)}px`).join("  "));
