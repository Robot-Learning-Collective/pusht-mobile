import { initRapier, type SimConfig } from "../src/sim/pushtSim.js";
import { evalAll } from "./fullEval.js";

function pct(sorted: number[], p: number): number {
  const i = (sorted.length - 1) * p;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

const variants: [string, Partial<SimConfig>][] = [
  ["default", {}],
  ["natFreq=40", { contactNaturalFrequency: 40 }],
  ["natFreq=20", { contactNaturalFrequency: 20 }],
  ["natFreq=10", { contactNaturalFrequency: 10 }],
  ["natFreq=5", { contactNaturalFrequency: 5 }],
  ["linDamp=2", { linearDamping: 2, angularDamping: 2 }],
  ["solverIters=20", { numSolverIterations: 20 }],
  ["internalPgs=8", { numInternalPgsIterations: 8 }],
  ["angInertia=4000", { blockAngularInertia: 4000 }],
  ["natFreq=10+iters=20", { contactNaturalFrequency: 10, numSolverIterations: 20 }],
];

await initRapier();
console.log(
  "variant                | mean  | median | p90  | p95  | p99   | max   | <=0.5px | <=2px | <=5px",
);
for (const [name, cfg] of variants) {
  const stats = evalAll(cfg);
  const kp = stats.map((s) => s.kpMean).sort((a, b) => a - b);
  const n = kp.length;
  const mean = kp.reduce((a, b) => a + b, 0) / n;
  const frac = (t: number) => ((stats.filter((s) => s.kpMean <= t).length / n) * 100).toFixed(1);
  console.log(
    `${name.padEnd(22)} | ${mean.toFixed(2).padStart(5)} | ${pct(kp, 0.5).toFixed(3).padStart(6)} | ${pct(
      kp,
      0.9,
    )
      .toFixed(2)
      .padStart(4)} | ${pct(kp, 0.95).toFixed(2).padStart(4)} | ${pct(kp, 0.99)
      .toFixed(2)
      .padStart(5)} | ${kp[n - 1].toFixed(2).padStart(5)} | ${frac(0.5).padStart(6)}% | ${frac(2).padStart(
      5,
    )}% | ${frac(5).padStart(5)}%`,
  );
}
