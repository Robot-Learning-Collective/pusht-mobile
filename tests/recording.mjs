// Uses the same recorder and WASM as the page. Run via test_recording.py.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import init, { Sim, randomInit, goalPose, controlHz, productScale, agentRadius, successThreshold } from "../rust/pusht-wasm/pkg/pusht_wasm.js";
const { Recording } = await import(pathToFileURL(process.argv[2]));
const wasm = readFileSync(new URL("../rust/pusht-wasm/pkg/pusht_wasm_bg.wasm", import.meta.url));
await init({ module_or_path: wasm });
const goal = [...goalPose()];
const episodes = [];
for (let seed = 0; seed < 5; seed++) {
  const s = [...randomInit(seed)];
  const sim = new Sim();
  const replay = new Sim();
  sim.reset(...s);
  replay.reset(...s);
  const recording = new Recording(sim, controlHz());
  for (let step = 0; step < 100; step++) {
    // First 3 steps are stationary. Then push into the block and toward walls.
    const b = sim.blockPose();
    const action = step < 3 ? s.slice(0, 2) : step < 60 ? [b[0], b[1]] : [490, 490];
    const before = recording.state();
    for (let sub = 0; sub < 10; sub++) {
      // Finger moves between physics ticks: the sampled target MUST stay held.
      assert.equal(recording.tick(sub === 0 ? action : [30 + sub, 480 - sub]), sub === 9);
    }
    assert.deepEqual(recording.frames[step], { ...before, step, timestamp: step / 10, action });
    replay.step(...action);
    assert.deepEqual([...sim.agentPos(), ...sim.blockPose()], [...replay.agentPos(), ...replay.blockPose()]);
  }
  episodes.push({
    schemaVersion: 2, name: "automated-test", engine: "pusht-core@0.1.0 (rapier2d 0.22.0, wasm)",
    simulatorSha256: createHash("sha256").update(wasm).digest("hex"), mode: "product",
    dt: sim.dt, controlHz: controlHz(), scale: productScale(), agentRadius: agentRadius(),
    goalPose: { x: goal[0], y: goal[1], angle: goal[2] }, successThreshold: successThreshold(),
    init: { agent: s.slice(0, 2), blockPos: s.slice(2, 4), blockAngle: s[4], seed },
    // Export tests also cover unsuccessful trajectories; collector tests below
    // use a genuinely solved fixture and reject success:false.
    success: sim.coverage() >= successThreshold(), nFrames: recording.frames.length,
    frames: recording.frames, finalState: recording.state(), finalTimestamp: recording.frames.length / controlHz(),
  });
}
// A solved, stationary fixture for exercising the success-only upload API.
const sim = new Sim();
sim.reset(50, 50, ...goal);
const rec = new Recording(sim, controlHz());
for (let i = 0; i < 30; i++) rec.tick([50, 50]);
assert.ok(sim.coverage() >= successThreshold());
const solved = { ...episodes[0], init: { agent: [50, 50], blockPos: goal.slice(0, 2), blockAngle: goal[2], seed: 0 }, success: true,
  frames: rec.frames, nFrames: rec.frames.length, finalState: rec.state(), finalTimestamp: rec.frames.length / 10 };
writeFileSync(process.argv[3], JSON.stringify({ episodes, solved }));
console.log("Recorder: 5 × 100 control steps match replay; stationary frames retained.");
