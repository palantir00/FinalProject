import { Hands, type Results, type NormalizedLandmarkList, type Classification } from "@mediapipe/hands";

/**
 * Pipeline wideo + MediaPipe Hands.
 *
 * Spec (sekcja 5/9): `src/cv` odpowiada za MediaPipe, filtry, mapping.
 * Cel: minimalny, czytelny wrapper emitujący landmarki + handedness.
 */

export type HandednessLabel = "Left" | "Right";
export type HandFrame = {
  tMs: number; // performance.now()
  hands: Array<{
    handedness: HandednessLabel;
    score: number;
    landmarks: NormalizedLandmarkList;
  }>;
};

export type HandTrackerConfig = {
  maxHands: 1 | 2;
  width: number;
  height: number;
  minDetectionConfidence: number;
  minTrackingConfidence: number;
};

export type HandTrackerDebug = {
  videoReadyState: number;
  videoSize: { w: number; h: number };
  sentFrames: number;
  resultsFrames: number;
  lastSendMs: number | null;
  lastResultMs: number | null;
  lastError: string | null;
};

const DEFAULT_CONFIG: HandTrackerConfig = {
  maxHands: 2,
  width: 640,
  height: 360,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7
};

// Pinujemy wersję CDN do tej z `app/package.json` (eliminuje mismatch assetów).
const MEDIAPIPE_HANDS_VERSION = "0.4.1675469240";

export class HandTracker {
  private hands: Hands | null = null;
  private stream: MediaStream | null = null;
  private raf = 0;
  private onFrameCb: ((frame: HandFrame) => void) | null = null;
  private cfg: HandTrackerConfig;
  private dbg: HandTrackerDebug = {
    videoReadyState: 0,
    videoSize: { w: 0, h: 0 },
    sentFrames: 0,
    resultsFrames: 0,
    lastSendMs: null,
    lastResultMs: null,
    lastError: null
  };
  private inFlight = false;
  private lastTickMs = 0;

  constructor(cfg: Partial<HandTrackerConfig> = {}) {
    this.cfg = { ...DEFAULT_CONFIG, ...cfg };
  }

  onFrame(cb: (frame: HandFrame) => void) {
    this.onFrameCb = cb;
  }

  getDebug(): HandTrackerDebug {
    return { ...this.dbg, videoSize: { ...this.dbg.videoSize } };
  }

  async start(videoEl: HTMLVideoElement) {
    // 1) Kamera (jawny getUserMedia – mniej “magii” niż camera_utils, łatwiej debugować)
    // Jeśli nie ma klatek wideo, MediaPipe nie zwróci żadnych landmarków.
    videoEl.autoplay = true;
    videoEl.muted = true; // wymagane przez politykę autoplay w wielu przeglądarkach
    videoEl.playsInline = true;

    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: this.cfg.width },
        height: { ideal: this.cfg.height },
        facingMode: "user"
      },
      audio: false
    });

    videoEl.srcObject = this.stream;
    await videoEl.play();
    this.dbg.lastError = null;

    // MediaPipe assets (WASM + model) – CDN; lokalnie można później przenieść do /public.
    this.hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${MEDIAPIPE_HANDS_VERSION}/${file}`
    });

    this.hands.setOptions({
      maxNumHands: this.cfg.maxHands,
      modelComplexity: 1,
      selfieMode: true,
      minDetectionConfidence: this.cfg.minDetectionConfidence,
      minTrackingConfidence: this.cfg.minTrackingConfidence
    });

    this.hands.onResults((results) => this.handleResults(results));

    // 2) Pętla wysyłania klatek do MediaPipe (docelowo throttle do 30 FPS)
    const loop = async () => {
      this.raf = requestAnimationFrame(loop);
      if (!this.hands) return;
      this.dbg.videoReadyState = videoEl.readyState;
      this.dbg.videoSize = { w: videoEl.videoWidth || 0, h: videoEl.videoHeight || 0 };
      if (videoEl.readyState < 2) return; // brak danych

      // Krytyczne: nie wolno robić równoległych `send()` – potrafi to “zabić” pipeline i daje Hands=0.
      if (this.inFlight) return;

      // Throttle do ~30 FPS (cel specyfikacji). Pozwala też uniknąć przeciążenia CPU.
      const now = performance.now();
      if (now - this.lastTickMs < 33) return;
      this.lastTickMs = now;

      this.inFlight = true;
      try {
        this.dbg.sentFrames += 1;
        this.dbg.lastSendMs = performance.now();
        await this.hands.send({ image: videoEl });
      } catch (e) {
        // Nie ukrywamy – to jest główna diagnostyka, gdy mamy podgląd a nadal Hands=0.
        this.dbg.lastError = e instanceof Error ? e.message : String(e);
      } finally {
        this.inFlight = false;
      }
    };
    loop();
  }

  stop() {
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.hands?.close();
    this.hands = null;
    this.inFlight = false;
  }

  private handleResults(results: Results) {
    const tMs = performance.now();
    this.dbg.resultsFrames += 1;
    this.dbg.lastResultMs = tMs;

    const lm = results.multiHandLandmarks ?? [];
    const handednessRaw = results.multiHandedness ?? [];

    const hands = lm.map((landmarks, i) => {
      // WAŻNE: poprzednio zakładaliśmy format `handednessRaw[i][0]`, co w wielu buildach MediaPipe jest nieprawdą.
      // Efekt uboczny: score=0 -> filtr usuwał wszystkie dłonie => Hands=0 mimo że landmarki były obecne.
      const hAny = handednessRaw[i] as any;

      const pick = (x: any): Classification | null => {
        if (!x) return null;
        if (typeof x.label === "string" && typeof x.score === "number") return x as Classification;
        if (Array.isArray(x) && x[0] && typeof x[0].label === "string") return x[0] as Classification;
        if (Array.isArray(x.classification) && x.classification[0]) return x.classification[0] as Classification;
        return null;
      };

      const h = pick(hAny);
      return {
        handedness: ((h?.label ?? "Right") as "Left" | "Right"),
        // Jeśli nie mamy handedness (różny format) – nie “karzemy” tego wyniku score=0.
        score: h?.score ?? 1,
        landmarks
      };
    });

    this.onFrameCb?.({ tMs, hands });
  }
}

