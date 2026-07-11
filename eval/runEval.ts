import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { PushTSim, initRapier, REFERENCE_CONFIG, type SimConfig } from "../src/sim/pushtSim.js";
import { poseFromKeypoints, angleDiff, meanKeypointError } from "./pose.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA = resolve(__dirname, "..", process.env.EPISODES ?? "data/episodes.json");

interface Episode {
  episode_index: number;
  n_frames: number;
  agent: number[][];
  keypoints: number[][];
  action: number[][];
  success: boolean[];
}

interface EpResult {
  ep: number;
  nFrames: number;
  agentRmse: number;
  agentMax: number;
  blockPosMean: number;
  blockPosFinal: number;
  blockAngMeanDeg: number;
  blockAngFinalDeg: number;
  kpMean: number;
  kpFinal: number;
}

function rad2deg(r: number): number {
  return (r * 180) / Math.PI;
}

export function evalEpisode(sim: PushTSim, ep: Episode): EpResult {
  const N = ep.n_frames;
  const gt0 = poseFromKeypoints(ep.keypoints[0]);
  sim.reset(ep.agent[0] as [number, number], [gt0.x, gt0.y], gt0.angle);

  let agentSq = 0;
  let agentMax = 0;
  let blockPosSum = 0;
  let blockAngSum = 0;
  let kpSum = 0;
  let blockPosFinal = 0;
  let blockAngFinal = 0;
  let kpFinal = 0;
  let count = 0;

  for (let t = 0; t < N - 1; t++) {
    sim.step(ep.action[t] as [number, number]);
    const next = t + 1;

    // agent error
    const [ax, ay] = sim.agentPos();
    const dax = ax - ep.agent[next][0];
    const day = ay - ep.agent[next][1];
    const ae = Math.hypot(dax, day);
    agentSq += ae * ae;
    agentMax = Math.max(agentMax, ae);

    // block error
    const gt = poseFromKeypoints(ep.keypoints[next]);
    const sp = sim.blockPose();
    const bpe = Math.hypot(sp.x - gt.x, sp.y - gt.y);
    const bae = Math.abs(rad2deg(angleDiff(sp.angle, gt.angle)));
    const kpe = meanKeypointError(sim.blockKeypoints(), ep.keypoints[next]);

    blockPosSum += bpe;
    blockAngSum += bae;
    kpSum += kpe;
    blockPosFinal = bpe;
    blockAngFinal = bae;
    kpFinal = kpe;
    count++;
  }

  return {
    ep: ep.episode_index,
    nFrames: N,
    agentRmse: Math.sqrt(agentSq / count),
    agentMax,
    blockPosMean: blockPosSum / count,
    blockPosFinal,
    blockAngMeanDeg: blockAngSum / count,
    blockAngFinalDeg: blockAngFinal,
    kpMean: kpSum / count,
    kpFinal,
  };
}

function fmt(n: number, w = 7, d = 2): string {
  return n.toFixed(d).padStart(w);
}

export async function runEval(cfg: Partial<SimConfig> = {}, label = "reference"): Promise<number> {
  await initRapier();
  const { episodes } = JSON.parse(readFileSync(DATA, "utf8")) as { episodes: Episode[] };
  // Validation is against the dataset -> reproduce the reference (buggy) env.
  const sim = new PushTSim({ ...REFERENCE_CONFIG, ...cfg });

  const results = episodes.map((ep) => evalEpisode(sim, ep));

  console.log(`\n=== config: ${label} ===`);
  console.log(
    "ep      frames  agentRMSE agentMax  blkPosMean blkPosFin  blkAngMean blkAngFin  kpMean  kpFinal",
  );
  for (const r of results) {
    console.log(
      `${String(r.ep).padStart(4)}  ${String(r.nFrames).padStart(6)}  ${fmt(r.agentRmse)}  ${fmt(
        r.agentMax,
      )}  ${fmt(r.blockPosMean)}    ${fmt(r.blockPosFinal)}  ${fmt(r.blockAngMeanDeg)}    ${fmt(
        r.blockAngFinalDeg,
      )}  ${fmt(r.kpMean)}  ${fmt(r.kpFinal)}`,
    );
  }
  const mean = (f: (r: EpResult) => number) => results.reduce((s, r) => s + f(r), 0) / results.length;
  console.log(
    `MEAN          ${fmt(mean((r) => r.agentRmse))}  ${fmt(mean((r) => r.agentMax))}  ${fmt(
      mean((r) => r.blockPosMean),
    )}    ${fmt(mean((r) => r.blockPosFinal))}  ${fmt(mean((r) => r.blockAngMeanDeg))}    ${fmt(
      mean((r) => r.blockAngFinalDeg),
    )}  ${fmt(mean((r) => r.kpMean))}  ${fmt(mean((r) => r.kpFinal))}`,
  );
  // primary objective: mean keypoint error over all frames (captures pos+angle)
  return mean((r) => r.kpMean);
}

// Run with defaults when invoked directly.
const isMain = process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1]);
if (isMain) {
  runEval().then((obj) => {
    console.log(`\nprimary objective (mean keypoint error, px): ${obj.toFixed(3)}`);
  });
}
