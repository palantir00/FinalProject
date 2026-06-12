// Minimal ambient typings to keep TS happy if MediaPipe packages ship without full types.
// If upstream types are present, TS will prefer them and these won't be used.

declare module "@mediapipe/hands" {
  export type NormalizedLandmark = { x: number; y: number; z: number };
  export type NormalizedLandmarkList = NormalizedLandmark[];
  export type Classification = { label: "Left" | "Right"; score: number };
  export type ClassificationList = { classification: Classification[] };
  export type Results = {
    image: HTMLVideoElement | HTMLCanvasElement | ImageBitmap;
    multiHandLandmarks?: NormalizedLandmarkList[];
    // MediaPipe w praktyce zwraca różne “kształty” danych między wersjami/buildami.
    // Obsługujemy najczęstsze:
    // - [{label, score}, ...]
    // - [{classification: [{label, score}, ...]}, ...]
    multiHandedness?: Array<Classification | ClassificationList>;
  };

  export class Hands {
    constructor(opts: { locateFile: (file: string) => string });
    setOptions(options: Record<string, unknown>): void;
    onResults(cb: (results: Results) => void): void;
    send(inputs: { image: HTMLVideoElement | HTMLCanvasElement | ImageBitmap }): Promise<void>;
    close(): void;
  }
}

declare module "@mediapipe/camera_utils" {
  export class Camera {
    constructor(
      video: HTMLVideoElement,
      opts: { onFrame: () => Promise<void> | void; width?: number; height?: number }
    );
    start(): void;
    stop(): void;
  }
}

