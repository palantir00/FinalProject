import { clamp } from "../../utils/math";

/**
 * One Euro Filter (Casiez et al.) – stabilizacja sygnału w czasie rzeczywistym.
 * Wymóg ze specyfikacji: "Stabilizacja: One-Euro Filter + ...", cel FPS/latency.
 *
 * Referencja koncepcyjna (implementacja własna – bez zależności):
 * - low-pass z adaptacyjnym cutoff zależnym od pochodnej
 */

function alpha(dt: number, cutoffHz: number) {
  const tau = 1.0 / (2.0 * Math.PI * cutoffHz);
  return 1.0 / (1.0 + tau / dt);
}

class LowPass {
  private y: number | null = null;

  filter(x: number, a: number) {
    if (this.y === null) this.y = x;
    this.y = a * x + (1 - a) * this.y;
    return this.y;
  }

  last() {
    return this.y;
  }
}

export class OneEuroFilter1D {
  private xFilter = new LowPass();
  private dxFilter = new LowPass();
  private lastT: number | null = null;

  constructor(
    private readonly minCutoffHz = 1.0,
    private readonly beta = 0.0,
    private readonly dCutoffHz = 1.0
  ) {}

  reset() {
    this.xFilter = new LowPass();
    this.dxFilter = new LowPass();
    this.lastT = null;
  }

  /**
   * @param x value
   * @param tMs timestamp in ms (performance.now()).
   */
  filter(x: number, tMs: number) {
    if (this.lastT === null) {
      this.lastT = tMs;
      return this.xFilter.filter(x, 1);
    }

    const dt = clamp((tMs - this.lastT) / 1000.0, 1 / 240, 1); // clamp for stability
    this.lastT = tMs;

    const prevX = this.xFilter.last() ?? x;
    const dx = (x - prevX) / dt;

    const aD = alpha(dt, this.dCutoffHz);
    const edx = this.dxFilter.filter(dx, aD);

    const cutoff = this.minCutoffHz + this.beta * Math.abs(edx);
    const aX = alpha(dt, cutoff);
    return this.xFilter.filter(x, aX);
  }
}

export class OneEuroFilterVec3 {
  private fx: OneEuroFilter1D;
  private fy: OneEuroFilter1D;
  private fz: OneEuroFilter1D;

  constructor(minCutoffHz = 1.0, beta = 0.0, dCutoffHz = 1.0) {
    this.fx = new OneEuroFilter1D(minCutoffHz, beta, dCutoffHz);
    this.fy = new OneEuroFilter1D(minCutoffHz, beta, dCutoffHz);
    this.fz = new OneEuroFilter1D(minCutoffHz, beta, dCutoffHz);
  }

  reset() {
    this.fx.reset();
    this.fy.reset();
    this.fz.reset();
  }

  filter(v: { x: number; y: number; z: number }, tMs: number) {
    return {
      x: this.fx.filter(v.x, tMs),
      y: this.fy.filter(v.y, tMs),
      z: this.fz.filter(v.z, tMs)
    };
  }
}

