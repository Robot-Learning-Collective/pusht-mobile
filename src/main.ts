import { Application, Container, Graphics } from "pixi.js";
import {
  PushTSim,
  initRapier,
  CONTROL_HZ,
  PRODUCT,
  T_BOXES,
  WORLD,
  type BlockPose,
} from "./sim/pushtSim.js";
import {
  GOAL_POSE,
  SUCCESS_THRESHOLD,
  randomInit,
  goalSamples,
  coverage,
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

const samples = goalSamples();

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

function drawT(color: number, alpha: number): Graphics {
  const g = new Graphics();
  const k = PRODUCT.scale;
  for (const b of T_BOXES) g.rect((b.cx - b.hx) * k, (b.cy - b.hy) * k, b.hx * 2 * k, b.hy * 2 * k);
  g.fill({ color, alpha });
  return g;
}

const board = new Graphics().rect(0, 0, WORLD, WORLD).fill(C.board);
root.addChild(board);

const goal = drawT(C.goal, 0.28);
goal.position.set(GOAL_POSE.x, GOAL_POSE.y);
goal.rotation = GOAL_POSE.angle;
root.addChild(goal);

const blockG = drawT(C.block, 1);
root.addChild(blockG);

const agentG = new Graphics().circle(0, 0, PRODUCT.agentRadius).fill(C.agent);
root.addChild(agentG);

// target marker (where the finger is)
const targetG = new Graphics().circle(0, 0, 5).stroke({ width: 2, color: C.target, alpha: 0.8 });
targetG.visible = false;
root.addChild(targetG);

layout();

// ---------- simulation state ----------
await initRapier();
const sim = new PushTSim("product");

interface Frame {
  agent: [number, number];
  block: [number, number, number]; // x, y, angle
  action: [number, number];
}

let init: InitState;
let started = false;
let solved = false;
let target: [number, number] = [0, 0];
let frames: Frame[] = [];
let acc = 0;
let recAcc = 0;
let episodeCount = 0;

const elStatus = document.getElementById("status")!;
const elCount = document.getElementById("count")!;
const elCoverage = document.getElementById("coverage")!;
const btnRestart = document.getElementById("restart") as HTMLButtonElement;
const btnSave = document.getElementById("save") as HTMLButtonElement;

function setStatus(text: string, kind: "info" | "good" = "info") {
  elStatus.textContent = text;
  elStatus.className = kind;
}

function reset() {
  init = randomInit();
  sim.reset(init.agent, init.blockPos, init.blockAngle);
  target = [...init.agent];
  started = false;
  solved = false;
  frames = [];
  acc = 0;
  recAcc = 0;
  btnSave.disabled = true;
  setStatus("Touch the robot to start");
  render();
}

// ---------- input ----------
const R = PRODUCT.agentRadius;
const clamp = (v: number) => Math.max(R, Math.min(WORLD - R, v));

function toWorld(e: PointerEvent): [number, number] {
  const r = app.canvas.getBoundingClientRect();
  const px = ((e.clientX - r.left) / r.width) * app.renderer.width;
  const py = ((e.clientY - r.top) / r.height) * app.renderer.height;
  // Clamp the target into the play area so the robot can't be led off-board.
  return [clamp((px - root.position.x) / root.scale.x), clamp((py - root.position.y) / root.scale.y)];
}

app.canvas.style.touchAction = "none";

app.canvas.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  // Capture the pointer: move events keep firing even when the finger leaves
  // the canvas, so the target can reach the borders at full speed.
  app.canvas.setPointerCapture(e.pointerId);
  const [wx, wy] = toWorld(e);
  if (!started) {
    const [ax, ay] = sim.agentPos();
    if (Math.hypot(wx - ax, wy - ay) <= R * 2.2) {
      // start only when the touch is on/near the robot
      target = [wx, wy];
      started = true;
      setStatus("Push the T onto the green target");
    }
    return;
  }
  target = [wx, wy];
});

app.canvas.addEventListener("pointermove", (e) => {
  if (!started || solved) return;
  e.preventDefault();
  target = toWorld(e);
});

btnRestart.addEventListener("click", reset);
btnSave.addEventListener("click", saveEpisode);

function saveEpisode() {
  const episode = {
    engine: "rapier2d-compat@0.14.0",
    dt: sim.dt,
    controlHz: CONTROL_HZ,
    scale: PRODUCT.scale,
    agentRadius: PRODUCT.agentRadius,
    goalPose: GOAL_POSE,
    successThreshold: SUCCESS_THRESHOLD,
    init,
    success: solved,
    nFrames: frames.length,
    frames,
  };
  const blob = new Blob([JSON.stringify(episode)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `pusht_episode_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ---------- loop ----------
const MAX_CATCHUP = 0.1; // s, avoid spiral of death
const REC_DT = 1 / CONTROL_HZ;

app.ticker.add((tk) => {
  if (started && !solved) {
    acc += Math.min(tk.deltaMS / 1000, MAX_CATCHUP);
    const n = Math.floor(acc / sim.dt);
    if (n > 0) {
      sim.advance(target, n);
      acc -= n * sim.dt;
      recAcc += n * sim.dt;
      while (recAcc >= REC_DT) {
        recAcc -= REC_DT;
        const bp = sim.blockPose();
        frames.push({ agent: sim.agentPos(), block: [bp.x, bp.y, bp.angle], action: [...target] });
      }
    }
    const cov = coverage(sim.blockPose(), samples);
    elCoverage.textContent = `coverage ${(cov * 100).toFixed(0)}%`;
    if (cov >= SUCCESS_THRESHOLD) {
      solved = true;
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
  blockG.tint = solved ? C.blockSolved : 0xffffff;

  const [ax, ay] = sim.agentPos();
  agentG.position.set(ax, ay);

  targetG.visible = started && !solved;
  if (targetG.visible) targetG.position.set(target[0], target[1]);
}

reset();
