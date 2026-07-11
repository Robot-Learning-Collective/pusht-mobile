/**
 * Replay recorded lerobot/pusht episodes through the sim in "reference" mode
 * and print per-episode differences from the ground truth.
 *
 *   npx tsx eval/runEval.ts [data/episodes.json]
 */
import { readFileSync } from "node:fs";
import { PushTSim, initRapier } from "../src/sim/pushtSim.js";
import { poseFromKeypoints, angleDiff, meanKeypointError } from "./pose.js";

interface Episode {
  episode_index: number;
  n_frames: number;
  agent: number[][];
  keypoints: number[][];
  action: number[][];
}

const rad2deg = (r: number) => (r * 180) / Math.PI;
const fmt = (n: number, w = 7, d = 2) => n.toFixed(d).padStart(w);

await initRapier();
const path = process.argv[2] ?? "data/episodes.json";
const { episodes } = JSON.parse(readFileSync(path, "utf8")) as { episodes: Episode[] };
const sim = new PushTSim("reference");

console.log("ep      frames  agentRMSE  blkPosMean  blkAngMean(deg)  kpMean(px)  kpFinal");
let kpTotal = 0;
for (const ep of episodes) {
  const poses = ep.keypoints.map(poseFromKeypoints);
  sim.reset(ep.agent[0] as [number, number], [poses[0].x, poses[0].y], poses[0].angle);

  let agentSq = 0;
  let posSum = 0;
  let angSum = 0;
  let kpSum = 0;
  let kpFinal = 0;
  const n = ep.n_frames - 1;
  for (let t = 0; t < n; t++) {
    sim.step(ep.action[t] as [number, number]);
    const [ax, ay] = sim.agentPos();
    agentSq += (ax - ep.agent[t + 1][0]) ** 2 + (ay - ep.agent[t + 1][1]) ** 2;
    const gt = poses[t + 1];
    const sp = sim.blockPose();
    posSum += Math.hypot(sp.x - gt.x, sp.y - gt.y);
    angSum += Math.abs(rad2deg(angleDiff(sp.angle, gt.angle)));
    kpFinal = meanKeypointError(sim.blockKeypoints(), ep.keypoints[t + 1]);
    kpSum += kpFinal;
  }
  kpTotal += kpSum / n;
  console.log(
    `${String(ep.episode_index).padStart(4)}  ${String(ep.n_frames).padStart(6)}  ${fmt(
      Math.sqrt(agentSq / n)
    )}  ${fmt(posSum / n)}      ${fmt(angSum / n)}       ${fmt(kpSum / n)}  ${fmt(kpFinal)}`
  );
}
console.log(`\nmean keypoint error over ${episodes.length} episodes: ${(kpTotal / episodes.length).toFixed(3)} px`);
