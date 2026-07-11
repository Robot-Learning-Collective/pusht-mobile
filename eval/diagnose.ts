import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { PushTSim, initRapier, REFERENCE_CONFIG } from "../src/sim/pushtSim.js";
import { poseFromKeypoints, angleDiff, meanKeypointError } from "./pose.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA = resolve(__dirname, "../data/episodes.json");
const r2d = (r: number) => (r * 180) / Math.PI;

await initRapier();
const { episodes } = JSON.parse(readFileSync(DATA, "utf8")) as { episodes: any[] };
const sim = new PushTSim(REFERENCE_CONFIG);

console.log(
  "ep    | GT block travel | GT total rot | maxBlkPosErr | maxBlkAngErr | maxKpErr | success",
);
for (const ep of episodes) {
  const N = ep.n_frames;
  const poses = ep.keypoints.map((k: number[]) => poseFromKeypoints(k));
  // ground-truth cumulative motion
  let travel = 0;
  let rot = 0;
  for (let t = 1; t < N; t++) {
    travel += Math.hypot(poses[t].x - poses[t - 1].x, poses[t].y - poses[t - 1].y);
    rot += Math.abs(r2d(angleDiff(poses[t].angle, poses[t - 1].angle)));
  }

  sim.reset(ep.agent[0], [poses[0].x, poses[0].y], poses[0].angle);
  let maxPos = 0;
  let maxAng = 0;
  let maxKp = 0;
  for (let t = 0; t < N - 1; t++) {
    sim.step(ep.action[t]);
    const gt = poses[t + 1];
    const sp = sim.blockPose();
    maxPos = Math.max(maxPos, Math.hypot(sp.x - gt.x, sp.y - gt.y));
    maxAng = Math.max(maxAng, Math.abs(r2d(angleDiff(sp.angle, gt.angle))));
    maxKp = Math.max(maxKp, meanKeypointError(sim.blockKeypoints(), ep.keypoints[t + 1]));
  }
  const success = ep.success.some((s: boolean) => s);
  console.log(
    `${String(ep.episode_index).padStart(4)}  | ${travel.toFixed(1).padStart(9)} px    | ${rot
      .toFixed(0)
      .padStart(6)} deg   | ${maxPos.toFixed(2).padStart(8)} px  | ${maxAng
      .toFixed(2)
      .padStart(8)} deg | ${maxKp.toFixed(2).padStart(6)} | ${success}`,
  );
}
