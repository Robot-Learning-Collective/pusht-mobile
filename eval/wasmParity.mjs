// Native-vs-wasm parity check: runs the identical scenario as
// `cargo run --release --bin traj` and prints in the same format.
//
//   node eval/wasmParity.mjs        (after `npm run build:wasm`)

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import init, { Sim, randomInit } from "../rust/pusht-wasm/pkg/pusht_wasm.js";

const wasmPath = fileURLToPath(new URL("../rust/pusht-wasm/pkg/pusht_wasm_bg.wasm", import.meta.url));
await init({ module_or_path: readFileSync(wasmPath) });

const f = (n) => n.toFixed(12);
const s = randomInit(42);
console.log(`init agent=(${f(s[0])},${f(s[1])}) block=(${f(s[2])},${f(s[3])},${f(s[4])})`);

const sim = new Sim();
sim.reset(s[0], s[1], s[2], s[3], s[4]);
for (let t = 0; t < 200; t++) {
  const b = sim.blockPose();
  sim.step(b[0], b[1]);
  if ((t + 1) % 50 === 0) {
    const [ax, ay] = sim.agentPos();
    const p = sim.blockPose();
    console.log(`t=${t + 1} agent=(${f(ax)},${f(ay)}) block=(${f(p[0])},${f(p[1])},${f(p[2])})`);
  }
}
console.log(`coverage=${f(sim.coverage())}`);
