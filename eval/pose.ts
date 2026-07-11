import { KEYPOINTS_BODY, type BlockPose } from "../src/sim/pushtSim.js";

/**
 * Recover block pose (centroid position + rotation) from 8 world keypoints via
 * 2D Procrustes alignment against the canonical body-frame template.
 * The template centroid is at the origin, so the translation is just the
 * centroid of the observed keypoints, and the rotation has the closed form
 *   theta = atan2( sum(Tx*Kcy - Ty*Kcx), sum(Tx*Kcx + Ty*Kcy) )
 */
export function poseFromKeypoints(flat: number[]): BlockPose {
  const n = KEYPOINTS_BODY.length;
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < n; i++) {
    cx += flat[2 * i];
    cy += flat[2 * i + 1];
  }
  cx /= n;
  cy /= n;

  let sCross = 0;
  let sDot = 0;
  for (let i = 0; i < n; i++) {
    const tx = KEYPOINTS_BODY[i][0];
    const ty = KEYPOINTS_BODY[i][1];
    const kx = flat[2 * i] - cx;
    const ky = flat[2 * i + 1] - cy;
    sCross += tx * ky - ty * kx;
    sDot += tx * kx + ty * ky;
  }
  return { x: cx, y: cy, angle: Math.atan2(sCross, sDot) };
}

export function angleDiff(a: number, b: number): number {
  return Math.atan2(Math.sin(a - b), Math.cos(a - b));
}

export function meanKeypointError(simFlat: number[], gtFlat: number[]): number {
  let s = 0;
  const n = simFlat.length / 2;
  for (let i = 0; i < n; i++) {
    const dx = simFlat[2 * i] - gtFlat[2 * i];
    const dy = simFlat[2 * i + 1] - gtFlat[2 * i + 1];
    s += Math.hypot(dx, dy);
  }
  return s / n;
}
