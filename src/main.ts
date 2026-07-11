import { Application, Container, Graphics } from "pixi.js";
import {
  PushTSim,
  initRapier,
  DEFAULT_CONFIG,
  T_BOXES,
  type BlockPose,
} from "./sim/pushtSim.js";
import {
  WORLD,
  GOAL_POSE,
  SUCCESS_THRESHOLD,
  randomInit,
  goalSamples,
  coverage,
  EpisodeRecorder,
  type InitState,
} from "./game/pushtGame.js";

// ---------- colours ----------
const C = {
  bg: 0x1b1e23,
  board: 0x2a2f37,
  goal: 0x35c46a,
  block: 0x9aa6b2,
  blockSolved: 0x35c46a,
  agent: 0x4a90e2,
  target: 0xe0e0e0,
};

const samples = goalSamples(GOAL_POSE); // step scales with sizeScale (constant sample count)

// ---------- pixi setup ----------
const app = new Application();
const host = document.getElementById("board")!;
await app.init({ background: C.bg, antialias: true, resizeTo: host });
host.appendChild(app.canvas);

const root = new Container();
app.stage.addChild(root);

function layout() {
  const size = Math.min(app.renderer.width, app.renderer.height);
  const s = size / WORLD;
  root.scale.set(s);
  root.position.set((app.renderer.width - size) / 2, (app.renderer.height - size) / 2);
}
app.renderer.on("resize", layout);

// board background
const board = new Graphics().rect(0, 0, WORLD, WORLD).fill(C.board);
root.addChild(board);

// goal silhouette
const goal = drawT(C.goal, 0.28);
goal.position.set(GOAL_POSE.x, GOAL_POSE.y);
goal.rotation = GOAL_POSE.angle;
root.addChild(goal);

// block
const blockG = drawT(C.block, 1);
root.addChild(blockG);

// agent
const agentG = new Graphics().circle(0, 0, DEFAULT_CONFIG.agentRadius).fill(C.agent);
root.addChild(agentG);

// target marker (where the finger is)
const targetG = new Graphics().circle(0, 0, 5).stroke({ width: 2, color: C.target, alpha: 0.8 });
targetG.visible = false;
root.addChild(targetG);

layout();

function drawT(color: number, alpha: number): Container {
  const c = new Container();
  const g = new Graphics();
  const k = DEFAULT_CONFIG.sizeScale;
  for (const b of T_BOXES) g.rect((b.cx - b.hx) * k, (b.cy - b.hy) * k, b.hx * 2 * k, b.hy * 2 * k);
  g.fill({ color, alpha });
  c.addChild(g);
  return c;
}

// ---------- simulation state ----------
await initRapier();
// Product uses the physically-correct block dynamics (DEFAULT_CONFIG); the
// dataset-matching reference config lives in the eval harness.
const sim = new PushTSim();
const cfg = sim.cfg;

let init: InitState;
let started = false;
let solved = false;
let target: [number, number] = [0, 0];
let recorder: EpisodeRecorder | null = null;
let acc = 0;
let recAcc = 0;
let episodeCount = 0;

const elStatus = document.getElementById("status")!;
const elCount = document.getElementById("count")!;
const elCoverage = document.getElementById("coverage")!;
const btnRestart = document.getElementById("restart") as HTMLButtonElement;
const btnSave = document.getElementById("save") as HTMLButtonElement;

function setStatus(text: string, kind: "info" | "good" | "bad" = "info") {
  elStatus.textContent = text;
  elStatus.className = kind;
}

function reset() {
  init = randomInit();
  sim.reset(init.agent, init.blockPos, init.blockAngle);
  target = [...init.agent];
  started = false;
  solved = false;
  recorder = null;
  acc = 0;
  recAcc = 0;
  btnSave.disabled = true;
  setStatus("Touch the robot to start");
  render();
}

function startEpisode() {
  if (started) return;
  started = true;
  recorder = new EpisodeRecorder(init, cfg, sim.dt, DEFAULT_CONFIG.controlHz, performance.now());
  setStatus("Push the T onto the green target");
}

// ---------- input ----------
const R = DEFAULT_CONFIG.agentRadius;
const clamp = (v: number) => Math.max(R, Math.min(WORLD - R, v));

function toWorld(clientX: number, clientY: number): [number, number] {
  const r = app.canvas.getBoundingClientRect();
  const px = ((clientX - r.left) / r.width) * app.renderer.width;
  const py = ((clientY - r.top) / r.height) * app.renderer.height;
  // The agent is kinematic (physics can't stop it leaving the board), so we
  // clamp its target into the play area — this keeps the robot on-board and
  // stops it from crushing the T through a wall from outside.
  return [clamp((px - root.position.x) / root.scale.x), clamp((py - root.position.y) / root.scale.y)];
}

function pointerDown(e: PointerEvent) {
  e.preventDefault();
  // Capture the pointer: move events keep firing even when the finger leaves
  // the canvas (footer, screen edge), so the target can reach the borders and
  // the robot stays responsive there.
  app.canvas.setPointerCapture(e.pointerId);
  const [wx, wy] = toWorld(e.clientX, e.clientY);
  if (!started) {
    const [ax, ay] = sim.agentPos();
    // start only if the touch is on/near the robot
    if (Math.hypot(wx - ax, wy - ay) <= DEFAULT_CONFIG.agentRadius * 2.2) {
      target = [wx, wy];
      startEpisode();
    }
    return;
  }
  target = [wx, wy];
}
function pointerMove(e: PointerEvent) {
  if (!started || solved) return;
  e.preventDefault();
  target = toWorld(e.clientX, e.clientY);
}

app.canvas.style.touchAction = "none";
app.canvas.addEventListener("pointerdown", pointerDown);
app.canvas.addEventListener("pointermove", pointerMove);

btnRestart.addEventListener("click", reset);
btnSave.addEventListener("click", saveEpisode);

function saveEpisode() {
  if (!recorder) return;
  const blob = new Blob([recorder.toJSON()], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `pusht_episode_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ---------- loop ----------
const MAX_CATCHUP = 0.1; // s, avoid spiral of death
app.ticker.add((tk) => {
  const dtReal = Math.min(tk.deltaMS / 1000, MAX_CATCHUP);
  if (started && !solved) {
    acc += dtReal;
    const n = Math.floor(acc / sim.dt);
    if (n > 0) {
      sim.advance(target, n);
      acc -= n * sim.dt;
      // record at control_hz
      recAcc += n * sim.dt;
      const recDt = 1 / DEFAULT_CONFIG.controlHz;
      while (recAcc >= recDt && recorder) {
        recAcc -= recDt;
        const [ax, ay] = sim.agentPos();
        const bp = sim.blockPose();
        recorder.push({ agent: [ax, ay], block: [bp.x, bp.y, bp.angle], action: [...target] });
      }
    }
    const cov = coverage(sim.blockPose(), samples);
    elCoverage.textContent = `coverage ${(cov * 100).toFixed(0)}%`;
    if (cov >= SUCCESS_THRESHOLD) {
      solved = true;
      if (recorder) recorder.success = true;
      episodeCount++;
      elCount.textContent = String(episodeCount);
      btnSave.disabled = false;
      setStatus("Solved! 🎉  Save or Restart", "good");
    }
  }
  render();
});

function render() {
  const bp: BlockPose = sim.blockPose();
  blockG.position.set(bp.x, bp.y);
  blockG.rotation = bp.angle;
  (blockG.children[0] as Graphics).tint = solved ? C.blockSolved : 0xffffff;

  const [ax, ay] = sim.agentPos();
  agentG.position.set(ax, ay);

  targetG.visible = started && !solved;
  if (targetG.visible) targetG.position.set(target[0], target[1]);
}

reset();
