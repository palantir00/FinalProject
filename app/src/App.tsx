import { useEffect, useMemo, useRef, useState } from "react";
import { HandTracker } from "./cv/HandTracker";
import { GestureEngine } from "./gestures/GestureEngine";
import type { GestureFrame } from "./gestures/types";
import { SceneManager } from "./three/SceneManager";
import { Hud } from "./ui/Hud";
import { TutorialOverlay } from "./ui/TutorialOverlay";
import { SolidsMenu } from "./ui/SolidsMenu";
import { SolidParamsForm } from "./ui/SolidParamsForm";
import { VirtualCursor } from "./ui/VirtualCursor";
import { ExportPanel } from "./ui/ExportPanel";
import { LatheOverlay } from "./ui/LatheOverlay";
import { exportSTL, exportOBJ } from "./export/exporters";
import { pathToLathe } from "./geometry/lathe";
import type { SolidType } from "./geometry/types";
import { CSGPanel } from "./ui/CSGPanel";
import type { CSGOperation } from "./three/SceneManager";

export function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const sceneRef = useRef<SceneManager | null>(null);
  const trackerRef = useRef<HandTracker | null>(null);
  const engineRef = useRef<GestureEngine | null>(null);

  const [started, setStarted] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [showCamera, setShowCamera] = useState(true);
  const [gesture, setGesture] = useState<GestureFrame>({ tMs: 0, mode: "Idle", gesture: "None", handsDetected: 0 });
  const [measureEnabled, setMeasureEnabled] = useState(false);
  const [showSolidsMenu, setShowSolidsMenu] = useState(false);
  const [showParamsForm, setShowParamsForm] = useState(false);
  const [selectedSolidType, setSelectedSolidType] = useState<SolidType | null>(null);
  const [hasSolid, setHasSolid] = useState(true); // default cube always present
  const [meshCount, setMeshCount] = useState(1);
  const [tick, setTick] = useState(0); // for HUD telemetry refresh

  // Sketch / Lathe — rysowanie profilu bryły obrotowej
  const sketchPathRef = useRef<{ x: number; y: number }[]>([]);
  const [sketchOverlay, setSketchOverlay] = useState<{ x: number; y: number }[]>([]);
  const [sketchActive, setSketchActive] = useState(false);
  // sketchModeRef mirrors sketchActive for use inside the onFrame closure (avoids stale state)
  const sketchModeRef = useRef(false);
  const prevGestureRef = useRef<string>("None");
  const SKETCH_MIN_POINTS = 12;
  const [latheInfo, setLatheInfo] = useState<{ formula: string; volume: number; area: number } | null>(null);

  const tracker = useMemo(() => new HandTracker({ maxHands: 2, width: 640, height: 360 }), []);
  const engine = useMemo(() => new GestureEngine(), []);

  useEffect(() => {
    trackerRef.current = tracker;
    engineRef.current = engine;
  }, [tracker, engine]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const sm = new SceneManager(canvas);
    sceneRef.current = sm;
    sm.setOnMeshCountChange((count) => {
      setMeshCount(count);
      setHasSolid(count > 0);
    });
    sm.start();

    const id = window.setInterval(() => setTick((t) => t + 1), 250);
    return () => {
      window.clearInterval(id);
      sm.dispose();
      sceneRef.current = null;
    };
  }, []);

  const startCamera = async () => {
    if (started) return;
    const video = videoRef.current;
    if (!video) return;
    setCameraError(null);

    tracker.onFrame((hf) => {
      // Interpretacja gestów (FSM + filtry)
      const gf = engine.update(hf);
      setGesture(gf);
      setMeasureEnabled(engine.isMeasureEnabled());

      // ── Sketch: toggle trybu rysowania gestem V, rysowanie gestem Point ────
      const prevG = prevGestureRef.current;
      prevGestureRef.current = gf.gesture;

      // Gest V (Sketch) na zboczu rosnącym = toggle trybu
      if (gf.gesture === "Sketch" && prevG !== "Sketch") {
        if (!sketchModeRef.current) {
          // Wejście w tryb — czyścimy poprzedni profil
          sketchModeRef.current = true;
          sketchPathRef.current = [];
          setSketchActive(true);
          setSketchOverlay([]);
        } else {
          // Wyjście z trybu — generujemy bryłę z nagranego profilu
          sketchModeRef.current = false;
          setSketchActive(false);
          const path = sketchPathRef.current;
          if (path.length >= SKETCH_MIN_POINTS) {
            try {
              const result = pathToLathe(path);
              sceneRef.current?.createLathe(result.lathePoints);
              // hasSolid / meshCount updated via onMeshCountChange
              setLatheInfo({ formula: result.formula, volume: result.volume, area: result.lateralArea });
            } catch {
              // zdegenerowany profil — ignorujemy
            }
          }
          sketchPathRef.current = [];
          setSketchOverlay([]);
        }
      }

      // Gdy tryb aktywny + gest Point → nagrywamy pozycję koniuszka palca wskazującego
      if (sketchModeRef.current && gf.gesture === "Point" && gf.pointPosition) {
        const pt = gf.pointPosition;
        const path = sketchPathRef.current;
        const last = path[path.length - 1];
        const tooClose = last && (pt.x - last.x) ** 2 + (pt.y - last.y) ** 2 < 0.0001;
        if (!tooClose) {
          sketchPathRef.current = [...path.slice(-89), pt];
          if (sketchPathRef.current.length % 3 === 0) {
            setSketchOverlay([...sketchPathRef.current]);
          }
        }
      }
      // ────────────────────────────────────────────────────────────────────────
      
      // Menu brył: 
      // - Otwórz gdy gesture === "Pinch" (pinch wykryty)
      // - Pozostaw otwarte gdy gesture === "Point" (użytkownik wskazuje palcem w menu)
      // - Zamknij gdy użytkownik zrobi inny gest (nie Pinch, nie Point)
      setShowSolidsMenu((prev) => {
        if (gf.gesture === "Pinch") {
          return true; // Otwórz menu gdy Pinch
        } else if (gf.gesture === "Point") {
          return prev; // Pozostaw otwarte gdy Point (zachowaj poprzedni stan - jeśli było otwarte, zostaje otwarte)
        } else if (prev) {
          return false; // Zamknij gdy inny gest (nie Pinch, nie Point)
        }
        return prev; // W przeciwnym razie zachowaj stan
      });
      
      sceneRef.current?.setGestureFrame(gf);
    });

    try {
      await tracker.start(video);
      setStarted(true);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Nie udało się uruchomić kamery.";
      setCameraError(msg);
      setStarted(false);
    }
  };

  const telemetry = sceneRef.current?.getTelemetry() ?? { fps: 0, lastLatencyMs: 0 };
  const metrics = sceneRef.current?.getCubeMetrics() ?? { area: 0, volume: 0 };
  const cv = trackerRef.current?.getDebug?.();

  const handleCSGOperation = (op: CSGOperation) => {
    sceneRef.current?.performCSG(op);
    // meshCount updated via onMeshCountChange callback
  };

  // tick is only used to trigger re-render for telemetry changes (fps/latency)
  void tick;

  return (
    <div className="appShell">
      <div className="viewport">
        <canvas ref={canvasRef} className="glCanvas" />

        <Hud gesture={gesture} telemetry={telemetry} metrics={metrics} measureEnabled={measureEnabled} cv={cv} />
        <TutorialOverlay />
        <SolidsMenu
          visible={showSolidsMenu}
          cursorPosition={gesture.gesture === "Point" ? gesture.pointPosition || null : null}
          onSelect={(solid) => {
            // Po wyborze bryły z menu, pokazujemy formularz parametrów
            setSelectedSolidType(solid);
            setShowSolidsMenu(false);
            setShowParamsForm(true);
          }}
        />
        {selectedSolidType && (
          <SolidParamsForm
            solidType={selectedSolidType}
            visible={showParamsForm}
            cursorPosition={gesture.gesture === "Point" ? gesture.pointPosition || null : null}
            onConfirm={(params) => {
              sceneRef.current?.createSolid(selectedSolidType, params);
              setShowParamsForm(false);
              setSelectedSolidType(null);
              // meshCount / hasSolid are updated via onMeshCountChange callback
            }}
            onCancel={() => {
              setShowParamsForm(false);
              setSelectedSolidType(null);
              // Przywracamy menu brył jeśli anulowano
              setShowSolidsMenu(true);
            }}
          />
        )}
        <VirtualCursor
          position={gesture.gesture === "Point" ? gesture.pointPosition || null : null}
        />

        <LatheOverlay path={sketchOverlay} active={sketchActive} />

        <CSGPanel meshCount={meshCount} onOperation={handleCSGOperation} />

        <div style={styles.exportWrap}>
          <ExportPanel
            disabled={!hasSolid}
            onExportSTL={() => {
              const mesh = sceneRef.current?.getMesh();
              if (mesh) exportSTL(mesh, "solid.stl");
            }}
            onExportOBJ={() => {
              const mesh = sceneRef.current?.getMesh();
              if (mesh) exportOBJ(mesh, "solid.obj");
            }}
          />
        </div>

        {/* Podgląd kamery (MVP) – pomaga zweryfikować, czy stream działa */}
        <div style={styles.camWrap}>
          <button style={styles.camBtn} onClick={() => setShowCamera((s) => !s)}>
            {showCamera ? "Ukryj kamerę" : "Pokaż kamerę"}
          </button>
          <video
            ref={videoRef}
            style={{
              ...styles.camVideo,
              display: showCamera ? "block" : "none"
            }}
          />
        </div>

        {!started && (
          <div style={styles.centerGate}>
            <div style={styles.gatePanel}>
              <div style={styles.gateTitle}>Włącz kamerę</div>
              <div style={styles.gateDesc}>
                Aplikacja używa MediaPipe Hands do śledzenia dłoni. Kliknij start i zaakceptuj uprawnienia kamery.
              </div>
              {cameraError && <div style={styles.gateErr}>{cameraError}</div>}
              <button style={styles.gateBtn} onClick={startCamera}>
                Start
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  centerGate: {
    position: "absolute",
    inset: 0,
    display: "grid",
    placeItems: "center",
    pointerEvents: "auto"
  },
  gatePanel: {
    width: 420,
    borderRadius: 14,
    background: "rgba(20, 24, 34, 0.82)",
    border: "1px solid rgba(255,255,255,0.14)",
    backdropFilter: "blur(10px)",
    padding: 16
  },
  gateTitle: { fontWeight: 800, fontSize: 18, marginBottom: 8 },
  gateDesc: { color: "rgba(255,255,255,0.72)", fontSize: 13, lineHeight: "18px", marginBottom: 12 },
  gateBtn: {
    border: "1px solid rgba(255,255,255,0.18)",
    background: "linear-gradient(135deg, rgba(124,58,237,0.85), rgba(99,102,241,0.65))",
    color: "white",
    borderRadius: 12,
    padding: "10px 14px",
    cursor: "pointer",
    fontWeight: 700
  },
  gateErr: {
    marginBottom: 10,
    padding: "8px 10px",
    borderRadius: 10,
    border: "1px solid rgba(245,158,11,0.35)",
    background: "rgba(245,158,11,0.10)",
    color: "rgba(255,255,255,0.85)",
    fontSize: 12,
    lineHeight: "16px"
  },
  exportWrap: {
    position: "absolute",
    bottom: 70,
    left: 12,
    pointerEvents: "auto"
  },
  camWrap: {
    position: "absolute",
    right: 12,
    bottom: 12,
    display: "grid",
    gap: 8,
    pointerEvents: "auto",
    justifyItems: "end"
  },
  camBtn: {
    border: "1px solid rgba(255,255,255,0.16)",
    background: "rgba(255,255,255,0.06)",
    color: "rgba(255,255,255,0.90)",
    borderRadius: 999,
    padding: "8px 12px",
    cursor: "pointer"
  },
  camVideo: {
    width: 240,
    height: 135,
    borderRadius: 12,
    border: "1px solid rgba(255,255,255,0.14)",
    background: "rgba(0,0,0,0.25)",
    objectFit: "cover",
    transform: "scaleX(-1)" // selfie preview
  }
};

