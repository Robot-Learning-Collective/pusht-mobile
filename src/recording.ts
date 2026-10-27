/** One row is the state BEFORE an action held for one complete control step. */
export interface State {
  agent: [number, number];
  block: [number, number, number];
}
export interface Frame extends State {
  step: number;
  timestamp: number; // simulation seconds, not wall time
  action: [number, number];
}
interface Simulator {
  readonly dt: number;
  agentPos(): Float64Array;
  blockPose(): Float64Array;
  advance(x: number, y: number, n: number): void;
}

export class Recording {
  frames: Frame[] = [];
  private substep = 0;
  private pending: Frame | undefined;
  private readonly substeps: number;

  constructor(private sim: Simulator, private hz: number) {
    this.substeps = Math.round(1 / (hz * sim.dt));
    if (Math.abs(this.substeps * sim.dt * hz - 1) > 1e-9) {
      throw new Error("Control interval must contain whole physics steps");
    }
  }

  state(): State {
    const a = this.sim.agentPos();
    const b = this.sim.blockPose();
    return { agent: [a[0], a[1]], block: [b[0], b[1], b[2]] };
  }

  /** Advance smoothly at physics rate; return true at a control boundary. */
  tick(target: [number, number]): boolean {
    if (this.substep === 0) {
      const step = this.frames.length;
      this.pending = { ...this.state(), step, timestamp: step / this.hz, action: [...target] };
    }
    const action = this.pending!.action;
    this.sim.advance(action[0], action[1], 1);
    if (++this.substep < this.substeps) return false;
    this.frames.push(this.pending!);
    this.substep = 0;
    return true;
  }
}
