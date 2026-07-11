import { pointInT, PRODUCT, WORLD, WALL, type BlockPose } from "../sim/pushtSim.js";

// Reference PushT goal pose: centred, rotated 45 deg.
export const GOAL_POSE: BlockPose = { x: WORLD / 2, y: WORLD / 2, angle: Math.PI / 4 };

export const SUCCESS_THRESHOLD = 0.95;

const SIZE = PRODUCT.scale;
const T_BOUND = Math.hypot(15, 75) * SIZE; // T bounding radius about its COM

export interface InitState {
  agent: [number, number];
  blockPos: [number, number];
  blockAngle: number;
}

/** Uniform random reset: the T fully inside the walls, not on top of the agent. */
export function randomInit(): InitState {
  const u = (lo: number, hi: number) => lo + Math.random() * (hi - lo);
  const r = PRODUCT.agentRadius;
  let agent: [number, number];
  let blockPos: [number, number];
  do {
    agent = [u(WALL.lo + r + 2, WALL.hi - r - 2), u(WALL.lo + r + 2, WALL.hi - r - 2)];
    blockPos = [u(WALL.lo + T_BOUND + 2, WALL.hi - T_BOUND - 2), u(WALL.lo + T_BOUND + 2, WALL.hi - T_BOUND - 2)];
  } while (Math.hypot(agent[0] - blockPos[0], agent[1] - blockPos[1]) < T_BOUND + r + 10);
  return { agent, blockPos, blockAngle: u(-Math.PI, Math.PI) };
}

/** Grid of world-space sample points inside the goal T (for coverage). */
export function goalSamples(pose: BlockPose = GOAL_POSE): number[][] {
  const step = 3 * SIZE;
  const ca = Math.cos(pose.angle);
  const sa = Math.sin(pose.angle);
  const pts: number[][] = [];
  // Sample at cell centres so points never sit exactly on the T boundary
  // (float round-trip during rotation would drop ~5% of them and a perfectly
  // placed block would score <95%).
  for (let bx = -60 * SIZE + step / 2; bx < 60 * SIZE; bx += step) {
    for (let by = -45 * SIZE + step / 2; by < 75 * SIZE; by += step) {
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

/** Fraction of the goal area covered by the block T. */
export function coverage(block: BlockPose, samples: number[][]): number {
  let inside = 0;
  for (const [x, y] of samples) if (pointInT(x, y, block, SIZE)) inside++;
  return inside / samples.length;
}
