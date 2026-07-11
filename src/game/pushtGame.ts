import { pointInT, DEFAULT_CONFIG, type BlockPose, type SimConfig } from "../sim/pushtSim.js";

export const WORLD = 512;

// Product geometry scale (2x-size T/robot for phones); the sim carries the
// same factor in DEFAULT_CONFIG.sizeScale.
export const SIZE = DEFAULT_CONFIG.sizeScale;

// Reference PushT goal pose: centred, rotated 45 deg.
export const GOAL_POSE: BlockPose = { x: WORLD / 2, y: WORLD / 2, angle: Math.PI / 4 };

export const SUCCESS_THRESHOLD = 0.95;

// Bounding radius of the scaled T about its COM (farthest outline vertex:
// stem bottom corners (±15, 75) -> ~76.5 per unit scale), plus wall margin.
const T_BOUND = Math.hypot(15, 75) * SIZE;
const WALL_LO = 7; // walls' inner faces in the sim
const WALL_HI = WORLD - 8;

export interface InitState {
  agent: [number, number];
  blockPos: [number, number];
  blockAngle: number;
}

/** Reference-style uniform reset, adapted so the scaled T spawns fully inside
 *  the walls and not on top of the agent. */
export function randomInit(): InitState {
  const u = (lo: number, hi: number) => lo + Math.random() * (hi - lo);
  const bLo = WALL_LO + T_BOUND + 2;
  const bHi = WALL_HI - T_BOUND - 2;
  const r = DEFAULT_CONFIG.agentRadius;
  const aLo = WALL_LO + r + 2;
  const aHi = WALL_HI - r - 2;
  const minSep = T_BOUND + r + 10;
  let agent: [number, number];
  let blockPos: [number, number];
  do {
    agent = [u(aLo, aHi), u(aLo, aHi)];
    blockPos = [u(bLo, bHi), u(bLo, bHi)];
  } while (Math.hypot(agent[0] - blockPos[0], agent[1] - blockPos[1]) < minSep);
  return { agent, blockPos, blockAngle: u(-Math.PI, Math.PI) };
}

/** Grid of world-space sample points that fall inside the goal T. */
export function goalSamples(pose: BlockPose = GOAL_POSE, step = 3 * SIZE): number[][] {
  const pts: number[][] = [];
  // unit body-frame bounding box of the T: x[-60,60], y[-45,75], scaled by SIZE
  const ca = Math.cos(pose.angle);
  const sa = Math.sin(pose.angle);
  // Sample at cell centres so points never sit exactly on the T boundary
  // (otherwise float round-trip during rotation drops ~5% of them and a
  //  perfectly placed block scores <95%).
  for (let bx = (-60 + step / 2 / SIZE) * SIZE; bx < 60 * SIZE; bx += step) {
    for (let by = (-45 + step / 2 / SIZE) * SIZE; by < 75 * SIZE; by += step) {
      if (!insideBodyT(bx / SIZE, by / SIZE)) continue;
      pts.push([pose.x + ca * bx - sa * by, pose.y + sa * bx + ca * by]);
    }
  }
  return pts;
}

function insideBodyT(bx: number, by: number): boolean {
  // unit scale: bar x[-60,60] y[-45,-15]; stem x[-15,15] y[-15,75]
  if (Math.abs(bx) <= 60 && by >= -45 && by <= -15) return true;
  if (Math.abs(bx) <= 15 && by >= -15 && by <= 75) return true;
  return false;
}

/** Fraction of goal area covered by the block T (overlap coverage). */
export function coverage(block: BlockPose, samples: number[][]): number {
  if (samples.length === 0) return 0;
  let inside = 0;
  for (const [x, y] of samples) if (pointInT(x, y, block, SIZE)) inside++;
  return inside / samples.length;
}

export interface RecordedFrame {
  agent: [number, number];
  block: [number, number, number]; // x, y, angle
  action: [number, number];
}

export class EpisodeRecorder {
  engine = "rapier2d-compat@0.14.0";
  startedAt: number;
  init: InitState;
  config: Partial<SimConfig>;
  dt: number;
  controlHz: number;
  frames: RecordedFrame[] = [];
  success = false;

  constructor(init: InitState, config: Partial<SimConfig>, dt: number, controlHz: number, startedAt: number) {
    this.init = init;
    this.config = config;
    this.dt = dt;
    this.controlHz = controlHz;
    this.startedAt = startedAt;
  }

  push(f: RecordedFrame): void {
    this.frames.push(f);
  }

  toJSON(): string {
    return JSON.stringify(
      {
        engine: this.engine,
        startedAt: this.startedAt,
        controlHz: this.controlHz,
        dt: this.dt,
        config: this.config,
        goalPose: GOAL_POSE,
        successThreshold: SUCCESS_THRESHOLD,
        init: this.init,
        success: this.success,
        nFrames: this.frames.length,
        frames: this.frames,
      },
      null,
      0,
    );
  }
}
