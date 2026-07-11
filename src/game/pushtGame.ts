import { pointInT, type BlockPose, type SimConfig } from "../sim/pushtSim.js";

export const WORLD = 512;

// Reference PushT goal pose: centred, rotated 45 deg.
export const GOAL_POSE: BlockPose = { x: WORLD / 2, y: WORLD / 2, angle: Math.PI / 4 };

export const SUCCESS_THRESHOLD = 0.95;

export interface InitState {
  agent: [number, number];
  blockPos: [number, number];
  blockAngle: number;
}

/** Match the reference reset ranges (uniform), with a light overlap guard. */
export function randomInit(): InitState {
  const u = (lo: number, hi: number) => lo + Math.random() * (hi - lo);
  let agent: [number, number];
  let blockPos: [number, number];
  do {
    agent = [u(50, 450), u(50, 450)];
    blockPos = [u(100, 400), u(100, 400)];
  } while (Math.hypot(agent[0] - blockPos[0], agent[1] - blockPos[1]) < 60);
  return { agent, blockPos, blockAngle: u(-Math.PI, Math.PI) };
}

/** Grid of world-space sample points that fall inside the goal T. */
export function goalSamples(pose: BlockPose = GOAL_POSE, step = 3): number[][] {
  const pts: number[][] = [];
  // body-frame bounding box of the T: x[-60,60], y[-45,75]
  const ca = Math.cos(pose.angle);
  const sa = Math.sin(pose.angle);
  // Sample at cell centres so points never sit exactly on the T boundary
  // (otherwise float round-trip during rotation drops ~5% of them and a
  //  perfectly placed block scores <95%).
  for (let bx = -60 + step / 2; bx < 60; bx += step) {
    for (let by = -45 + step / 2; by < 75; by += step) {
      if (!insideBodyT(bx, by)) continue;
      pts.push([pose.x + ca * bx - sa * by, pose.y + sa * bx + ca * by]);
    }
  }
  return pts;
}

function insideBodyT(bx: number, by: number): boolean {
  // bar x[-60,60] y[-45,-15]; stem x[-15,15] y[-15,75]
  if (Math.abs(bx) <= 60 && by >= -45 && by <= -15) return true;
  if (Math.abs(bx) <= 15 && by >= -15 && by <= 75) return true;
  return false;
}

/** Fraction of goal area covered by the block T (overlap coverage). */
export function coverage(block: BlockPose, samples: number[][]): number {
  if (samples.length === 0) return 0;
  let inside = 0;
  for (const [x, y] of samples) if (pointInT(x, y, block)) inside++;
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
