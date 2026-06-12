import type { Vec2 } from "../utils/math";

// Zgodnie ze spec (docs/SPEC_EXTRACTED.txt: "FSM do sterowania trybami (Create / Transform / Measure / Slice / CSG)").
export type Mode = "Idle" | "Create" | "Transform" | "Measure" | "Slice" | "Lathe";

export type GestureName = "None" | "Pinch" | "Move" | "Rotate" | "Scale" | "Slice" | "MeasureL" | "Point" | "Sketch";

export type GestureFrame = {
  tMs: number;
  mode: Mode;
  gesture: GestureName;
  // Normalized screen-space deltas (approx; mapping layer can refine later)
  moveDelta?: Vec2;
  rotateDelta?: { x: number; y: number }; // x = rotacja wokół Y (lewo/prawo), y = rotacja wokół X (góra/dół)
  scaleFactor?: number;
  sliceDelta?: number;
  // UI / telemetry
  pinchStrength?: number; // 0..1
  handsDetected: number;
  // Point gesture: pozycja koniuszka palca wskazującego (dla wirtualnego kursora)
  pointPosition?: Vec2; // znormalizowane współrzędne (0..1) koniuszka palca wskazującego
  // Debug: stan palców (dla debugowania Point)
  fingerState?: {
    thumb: boolean;
    index: boolean;
    middle: boolean;
    ring: boolean;
    pinky: boolean;
  };
};

