import { Application, Container, Graphics } from "pixi.js";
import wasmInit, {
  Sim,
  randomInit,
  world,
  controlHz,
  successThreshold,
  productScale,
  agentRadius,
  goalPose,
  tBoxes,
} from "../rust/pusht-wasm/pkg/pusht_wasm.js";

await wasmInit();

const ENGINE = "pusht-core@0.1.0 (rapier2d 0.22.0, wasm)";
const WORLD = world();
const CONTROL_HZ = controlHz();
const SUCCESS_THRESHOLD = successThreshold();
const SCALE = productScale();
const AGENT_RADIUS = agentRadius();
const GOAL = goalPose(); // [x, y, angle]
const T_BOXES = tBoxes(); // flat [cx, cy, hx, hy] x2, unit scale

interface BlockPose {
  x: number;
  y: number;
  angle: number;
}

interface InitState {
  agent: [number, number];
  blockPos: [number, number];
  blockAngle: number;
  seed: number;
}

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
  const k = SCALE;
  for (let i = 0; i < T_BOXES.length; i += 4) {
    const [cx, cy, hx, hy] = [T_BOXES[i], T_BOXES[i + 1], T_BOXES[i + 2], T_BOXES[i + 3]];
    g.rect((cx - hx) * k, (cy - hy) * k, hx * 2 * k, hy * 2 * k);
  }
  g.fill({ color, alpha });
  return g;
}

const board = new Graphics().rect(0, 0, WORLD, WORLD).fill(C.board);
root.addChild(board);

const goal = drawT(C.goal, 0.28);
goal.position.set(GOAL[0], GOAL[1]);
goal.rotation = GOAL[2];
root.addChild(goal);

const blockG = drawT(C.block, 1);
root.addChild(blockG);

const agentG = new Graphics().circle(0, 0, AGENT_RADIUS).fill(C.agent);
root.addChild(agentG);

// target marker (where the finger is)
const targetG = new Graphics().circle(0, 0, 5).stroke({ width: 2, color: C.target, alpha: 0.8 });
targetG.visible = false;
root.addChild(targetG);

layout();

// ---------- simulation state ----------
const sim = new Sim();

function blockPose(): BlockPose {
  const b = sim.blockPose();
  return { x: b[0], y: b[1], angle: b[2] };
}

function agentPos(): [number, number] {
  const p = sim.agentPos();
  return [p[0], p[1]];
}

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
const elName = document.getElementById("name") as HTMLInputElement;

function setStatus(text: string, kind: "info" | "good" = "info") {
  elStatus.textContent = text;
  elStatus.className = kind;
}

// ---------- result dialog ----------
const dlg = document.getElementById("dialog") as HTMLDialogElement;
const dlgTitle = document.getElementById("dlg-title")!;
const dlgMsg = document.getElementById("dlg-msg")!;
const dlgStats = document.querySelector<HTMLElement>(".dlg-stats")!;
const dlgTrimmed = document.getElementById("dlg-trimmed")!;
const dlgDuration = document.getElementById("dlg-duration")!;
const dlgSpeed = document.getElementById("dlg-speed")!;
const dlgTotalLabel = document.getElementById("dlg-total-label")!;
const dlgTotal = document.getElementById("dlg-total")!;
const btnDlgPrimary = document.getElementById("dlg-primary") as HTMLButtonElement;
const btnDlgSecondary = document.getElementById("dlg-secondary") as HTMLButtonElement;

interface EpisodeStats {
  trimmed: string;
  duration: string;
  speed: string;
}

interface DialogOpts {
  good: boolean;
  title: string;
  msg?: string;
  stats?: EpisodeStats;
  total?: number;
  primary: { label: string; action: () => void };
  secondary?: { label: string; action: () => void };
}

let dlgPrimaryAction = () => {};
let dlgSecondaryAction: (() => void) | undefined;
let dlgPendingAction: (() => void) | null = null;

function showDialog(o: DialogOpts) {
  dlgTitle.textContent = o.title;
  dlgTitle.className = o.good ? "good" : "bad";
  dlgMsg.textContent = o.msg ?? "";
  dlgMsg.style.display = o.msg ? "" : "none";
  dlgStats.style.display = o.stats ? "" : "none";
  if (o.stats) {
    dlgTrimmed.textContent = o.stats.trimmed;
    dlgDuration.textContent = o.stats.duration;
    dlgSpeed.textContent = o.stats.speed;
  }
  const showTotal = o.total !== undefined;
  dlgTotalLabel.style.display = showTotal ? "" : "none";
  dlgTotal.style.display = showTotal ? "" : "none";
  if (showTotal) dlgTotal.textContent = String(o.total);
  btnDlgPrimary.textContent = o.primary.label;
  btnDlgSecondary.style.display = o.secondary ? "" : "none";
  if (o.secondary) btnDlgSecondary.textContent = o.secondary.label;
  dlgPrimaryAction = o.primary.action;
  dlgSecondaryAction = o.secondary?.action;
  dlg.showModal();
}

btnDlgPrimary.addEventListener("click", () => {
  dlgPendingAction = dlgPrimaryAction;
  dlg.close();
});
btnDlgSecondary.addEventListener("click", () => {
  dlgPendingAction = dlgSecondaryAction ?? null;
  dlg.close();
});
// Runs on any close, incl. Esc: fall back to the dismiss-safe action.
dlg.addEventListener("close", () => {
  const a = dlgPendingAction ?? dlgSecondaryAction ?? dlgPrimaryAction;
  dlgPendingAction = null;
  a();
});

// ---------- participant name ----------
const NAME_KEY = "pusht_name";
elName.value = localStorage.getItem(NAME_KEY) ?? "";
elName.addEventListener("input", () => {
  localStorage.setItem(NAME_KEY, elName.value.trim());
  elName.classList.remove("missing");
});
elName.addEventListener("keydown", (e) => {
  if (e.key === "Enter") elName.blur();
});

function reset() {
  const seed = (Math.random() * 0x100000000) >>> 0;
  const s = randomInit(seed);
  init = { agent: [s[0], s[1]], blockPos: [s[2], s[3]], blockAngle: s[4], seed };
  sim.reset(s[0], s[1], s[2], s[3], s[4]);
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
const R = AGENT_RADIUS;
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
    const [ax, ay] = agentPos();
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

// Pre-send validation thresholds (world units; board is 512×512, frames at 10 Hz).
const STATIC_EPS = 0.01; // below this, a position is considered unchanged
const MAX_FRAME_DIST = 150; // max agent travel between two consecutive frames
const MAX_DURATION_S = 20; // max episode length after static frames are trimmed

/** Drop frames where nothing moved at all (agent, block and finger target). */
function dropStaticFrames(all: Frame[]): Frame[] {
  const kept: Frame[] = [];
  for (const f of all) {
    const p = kept[kept.length - 1];
    if (
      p &&
      Math.hypot(f.agent[0] - p.agent[0], f.agent[1] - p.agent[1]) < STATIC_EPS &&
      Math.hypot(f.block[0] - p.block[0], f.block[1] - p.block[1]) < STATIC_EPS &&
      Math.abs(f.block[2] - p.block[2]) < STATIC_EPS &&
      Math.hypot(f.action[0] - p.action[0], f.action[1] - p.action[1]) < STATIC_EPS
    ) {
      continue;
    }
    kept.push(f);
  }
  return kept;
}

async function saveEpisode() {
  const kept = dropStaticFrames(frames);
  const removed = frames.length - kept.length;

  let maxDist = 0;
  for (let i = 1; i < kept.length; i++) {
    const d = Math.hypot(kept[i].agent[0] - kept[i - 1].agent[0], kept[i].agent[1] - kept[i - 1].agent[1]);
    if (d > maxDist) maxDist = d;
  }
  const durationS = kept.length / CONTROL_HZ;
  const stats: EpisodeStats = {
    trimmed: `${removed} of ${frames.length}`,
    duration: `${durationS.toFixed(1)} s`,
    speed: `${Math.round(maxDist)} px / frame`,
  };

  const name = elName.value.trim();
  if (!name) {
    showDialog({
      good: false,
      title: "Name required",
      msg: "Enter your name (top right) so we can track who sent this episode.",
      primary: {
        label: "OK",
        action: () => {
          elName.classList.add("missing");
          elName.focus();
        },
      },
    });
    return;
  }

  if (maxDist > MAX_FRAME_DIST) {
    btnSave.disabled = true;
    showDialog({
      good: false,
      title: "Not sent — too fast",
      msg:
        `The robot jumped ${Math.round(maxDist)} px between two frames (limit ${MAX_FRAME_DIST}). ` +
        "Teleport-style moves make poor training data — move smoothly next time.",
      stats,
      primary: { label: "New episode", action: reset },
    });
    return;
  }

  if (durationS > MAX_DURATION_S) {
    btnSave.disabled = true;
    showDialog({
      good: false,
      title: "Not sent — too long",
      msg:
        `${Math.round(durationS)} s of motion is over the ${MAX_DURATION_S} s limit. ` +
        "Long episodes make poor training data — try to solve it faster.",
      stats,
      primary: { label: "New episode", action: reset },
    });
    return;
  }

  btnSave.disabled = true;
  btnSave.textContent = "Sending…";
  const episode = {
    name,
    engine: ENGINE,
    dt: sim.dt,
    controlHz: CONTROL_HZ,
    scale: SCALE,
    agentRadius: AGENT_RADIUS,
    goalPose: { x: GOAL[0], y: GOAL[1], angle: GOAL[2] },
    successThreshold: SUCCESS_THRESHOLD,
    init,
    success: solved,
    staticFramesRemoved: removed,
    nFrames: kept.length,
    frames: kept,
  };
  try {
    const res = await fetch("/api/episodes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(episode),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const { total } = await res.json();
    showDialog({
      good: true,
      title: "Saved 🎉",
      msg: `Nice one, ${name}!`,
      stats,
      total,
      primary: { label: "New episode", action: reset },
    });
  } catch {
    showDialog({
      good: false,
      title: "Upload failed",
      msg: "Could not reach the server — check your connection.",
      stats,
      primary: { label: "Retry", action: () => void saveEpisode() },
      secondary: {
        label: "Close",
        action: () => {
          btnSave.disabled = false;
        },
      },
    });
  } finally {
    btnSave.textContent = "Save episode";
  }
}

// ---------- loop ----------
const MAX_CATCHUP = 0.1; // s, avoid spiral of death
const REC_DT = 1 / CONTROL_HZ;

function pushFrame() {
  const bp = blockPose();
  frames.push({ agent: agentPos(), block: [bp.x, bp.y, bp.angle], action: [...target] });
}

app.ticker.add((tk) => {
  if (started && !solved) {
    acc += Math.min(tk.deltaMS / 1000, MAX_CATCHUP);
    // Step one physics tick at a time so success is detected at the exact
    // step it happens and no frames are recorded past it.
    while (acc >= sim.dt && !solved) {
      sim.advance(target[0], target[1], 1);
      acc -= sim.dt;
      recAcc += sim.dt;
      let recorded = false;
      if (recAcc >= REC_DT) {
        recAcc -= REC_DT;
        pushFrame();
        recorded = true;
      }
      if (sim.coverage() >= SUCCESS_THRESHOLD) {
        solved = true;
        if (!recorded) pushFrame(); // always capture the final solved state
        episodeCount++;
        elCount.textContent = String(episodeCount);
        btnSave.disabled = false;
        setStatus("Solved! 🎉  Save or Restart", "good");
      }
    }
    elCoverage.textContent = `coverage ${(sim.coverage() * 100).toFixed(0)}%`;
  }
  render();
});

function render() {
  const bp = blockPose();
  blockG.position.set(bp.x, bp.y);
  blockG.rotation = bp.angle;
  blockG.tint = solved ? C.blockSolved : 0xffffff;

  const [ax, ay] = agentPos();
  agentG.position.set(ax, ay);

  targetG.visible = started && !solved;
  if (targetG.visible) targetG.position.set(target[0], target[1]);
}

reset();
