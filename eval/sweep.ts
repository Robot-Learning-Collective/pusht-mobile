import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { PushTSim, initRapier, REFERENCE_CONFIG, type SimConfig } from "../src/sim/pushtSim.js";
import { poseFromKeypoints, angleDiff, meanKeypointError } from "./pose.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA = resolve(__dirname, "../data/episodes.json");
const r2d = (r: number) => (r * 180) / Math.PI;

await initRapier();
const { episodes } = JSON.parse(readFileSync(DATA, "utf8")) as { episodes: any[] };

function objective(cfg: Partial<SimConfig>) {
  const sim = new PushTSim({ ...REFERENCE_CONFIG, ...cfg });
  let kpSum = 0;
  let angSum = 0;
  let cnt = 0;
  for (const ep of episodes) {
    const N = ep.n_frames;
    const poses = ep.keypoints.map((k: number[]) => poseFromKeypoints(k));
    sim.reset(ep.agent[0], [poses[0].x, poses[0].y], poses[0].angle);
    for (let t = 0; t < N - 1; t++) {
      sim.step(ep.action[t]);
      kpSum += meanKeypointError(sim.blockKeypoints(), ep.keypoints[t + 1]);
      angSum += Math.abs(r2d(angleDiff(sim.blockPose().angle, poses[t + 1].angle)));
      cnt++;
    }
  }
  return { kp: kpSum / cnt, ang: angSum / cnt };
}

const variants: [string, Partial<SimConfig>][] = [
  ["DEFAULT", {}],
  ["resetBlockVel=false (coasting)", { resetBlockVelEachStep: false }],
  ["resetVel=false + linDamp=5", { resetBlockVelEachStep: false, linearDamping: 5, angularDamping: 5 }],
  ["solverIters=4", { numSolverIterations: 4 }],
  ["solverIters=16", { numSolverIterations: 16 }],
  ["internalPgs=4", { numInternalPgsIterations: 4 }],
  ["lengthUnit=1", { lengthUnit: 1 }],
  ["lengthUnit=10", { lengthUnit: 10 }],
  ["angInertia=1500", { blockAngularInertia: 1500 }],
  ["angInertia=7875 (no bug)", { blockAngularInertia: 7875 }],
  ["angInertia=300", { blockAngularInertia: 300 }],
  ["friction=0.3", { contactFriction: 0.3 }],
  ["friction=1.0", { contactFriction: 1.0 }],
  ["restitution=0.2", { contactRestitution: 0.2 }],
  ["mass=10", { blockMass: 10 }],
  ["mass=0.1", { blockMass: 0.1 }],
];

console.log("variant                            | kpMeanErr(px) | angMeanErr(deg)");
let best = Infinity;
let bestName = "";
for (const [name, cfg] of variants) {
  const o = objective(cfg);
  if (o.kp < best) {
    best = o.kp;
    bestName = name;
  }
  console.log(`${name.padEnd(34)} | ${o.kp.toFixed(4).padStart(11)}   | ${o.ang.toFixed(4).padStart(11)}`);
}
console.log(`\nBEST: ${bestName} (kpMeanErr=${best.toFixed(4)} px)`);
