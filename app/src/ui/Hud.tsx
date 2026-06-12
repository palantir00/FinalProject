import type { GestureFrame } from "../gestures/types";
import type { SceneTelemetry } from "../three/SceneManager";
import type { SolidMetrics } from "../geometry/analytics";
import type { HandTrackerDebug } from "../cv/HandTracker";

type Props = {
  gesture: GestureFrame;
  telemetry: SceneTelemetry;
  metrics: SolidMetrics;
  measureEnabled: boolean;
  cv?: HandTrackerDebug;
};

function fmt(n: number) {
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 1000) return n.toFixed(0);
  if (Math.abs(n) >= 10) return n.toFixed(2);
  return n.toFixed(3);
}

export function Hud({ gesture, telemetry, metrics, measureEnabled, cv }: Props) {
  // Spec: "UI: tutorial gestów, podgląd FPS/latency" + "Measure: gest L → etykiety".
  return (
    <div style={styles.wrap}>
      <div style={styles.panel}>
        <div style={styles.title}>Gesture-to-Geometry (MVP)</div>
        <div style={styles.row}>
          <span style={styles.k}>Mode</span>
          <span style={styles.v}>{gesture.mode}</span>
        </div>
        <div style={styles.row}>
          <span style={styles.k}>Gesture</span>
          <span
            style={{
              ...styles.v,
              color: gesture.gesture === "Point" ? "rgba(34,197,94,0.95)" : styles.v.color
            }}
          >
            {gesture.gesture}
            {gesture.gesture === "Point" && gesture.pointPosition && (
              <span style={{ fontSize: "10px", marginLeft: "8px", opacity: 0.7 }}>
                ({gesture.pointPosition.x.toFixed(2)}, {gesture.pointPosition.y.toFixed(2)})
              </span>
            )}
          </span>
        </div>
        <div style={styles.row}>
          <span style={styles.k}>Hands</span>
          <span style={styles.v}>{gesture.handsDetected}</span>
        </div>
        <div style={styles.row}>
          <span style={styles.k}>Pinch</span>
          <span
            style={{
              ...styles.v,
              color:
                gesture.pinchStrength != null && gesture.pinchStrength > 0.25
                  ? "rgba(34,197,94,0.95)"
                  : gesture.pinchStrength != null && gesture.pinchStrength > 0.1
                    ? "rgba(234,179,8,0.9)"
                    : styles.v.color
            }}
          >
            {gesture.pinchStrength != null ? fmt(gesture.pinchStrength) : "—"}
          </span>
        </div>
        {gesture.pinchStrength != null && gesture.pinchStrength > 0.1 && (
          <div style={styles.debugHint}>
            {gesture.pinchStrength > 0.25 ? "✓ Pinch wykryty (Create)" : "⚠ Pinch słaby (<0.25)"}
          </div>
        )}
        {gesture.fingerState && (
          <div style={styles.debugHint}>
            Palce: T:{gesture.fingerState.thumb ? "✓" : "✗"} I:{gesture.fingerState.index ? "✓" : "✗"} M:{gesture.fingerState.middle ? "✓" : "✗"} R:{gesture.fingerState.ring ? "✓" : "✗"} P:{gesture.fingerState.pinky ? "✓" : "✗"}
            {gesture.fingerState.index && !gesture.fingerState.thumb && !gesture.fingerState.middle && !gesture.fingerState.ring && !gesture.fingerState.pinky && (
              <span style={{ color: "rgba(34,197,94,0.95)", marginLeft: "8px" }}>→ Point powinien być wykryty!</span>
            )}
          </div>
        )}

        <div style={styles.sep} />

        <div style={styles.row}>
          <span style={styles.k}>FPS</span>
          <span style={styles.v}>{telemetry.fps || "—"}</span>
        </div>
        <div style={styles.row}>
          <span style={styles.k}>Latency</span>
          <span style={styles.v}>{telemetry.lastLatencyMs ? `${telemetry.lastLatencyMs.toFixed(0)} ms` : "—"}</span>
        </div>

        <div style={styles.sep} />

        <div style={styles.row}>
          <span style={styles.k}>CV frames</span>
          <span style={styles.v}>{cv ? `${cv.resultsFrames}/${cv.sentFrames}` : "—"}</span>
        </div>
        <div style={styles.row}>
          <span style={styles.k}>Video</span>
          <span style={styles.v}>
            {cv ? `${cv.videoSize.w}×${cv.videoSize.h} (rs=${cv.videoReadyState})` : "—"}
          </span>
        </div>
        {cv?.lastError && (
          <div style={styles.err} title={cv.lastError}>
            {cv.lastError}
          </div>
        )}

        <div style={styles.sep} />

        <div style={styles.row}>
          <span style={styles.k}>Labels</span>
          <span style={styles.v}>{measureEnabled ? "ON" : "OFF"}</span>
        </div>
        {measureEnabled && (
          <>
            <div style={styles.row}>
              <span style={styles.k}>Area</span>
              <span style={styles.v}>{fmt(metrics.area)}</span>
            </div>
            <div style={styles.row}>
              <span style={styles.k}>Volume</span>
              <span style={styles.v}>{fmt(metrics.volume)}</span>
            </div>
          </>
        )}
      </div>

      <div style={styles.hint}>
        Tip: jeśli kamera nie działa w przeglądarce, upewnij się że strona ma uprawnienia do kamery (HTTPS / localhost).
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  wrap: {
    position: "absolute",
    top: 12,
    left: 12,
    display: "grid",
    gap: 10,
    pointerEvents: "none"
  },
  panel: {
    width: 280,
    borderRadius: 12,
    background: "rgba(20, 24, 34, 0.72)",
    border: "1px solid rgba(255,255,255,0.12)",
    backdropFilter: "blur(8px)",
    padding: 12
  },
  title: { fontWeight: 700, letterSpacing: 0.3, marginBottom: 10 },
  row: { display: "flex", justifyContent: "space-between", gap: 12, fontSize: 12, lineHeight: "18px" },
  k: { color: "rgba(255,255,255,0.68)" },
  v: { color: "rgba(255,255,255,0.92)", fontVariantNumeric: "tabular-nums" },
  sep: { height: 1, background: "rgba(255,255,255,0.10)", margin: "10px 0" },
  err: {
    marginTop: 8,
    fontSize: 11,
    lineHeight: "15px",
    color: "rgba(255,255,255,0.86)",
    border: "1px solid rgba(245,158,11,0.35)",
    background: "rgba(245,158,11,0.10)",
    borderRadius: 10,
    padding: "8px 10px",
    maxHeight: 60,
    overflow: "hidden",
    textOverflow: "ellipsis"
  },
  hint: {
    width: 280,
    fontSize: 11,
    color: "rgba(255,255,255,0.55)",
    lineHeight: "16px",
    background: "rgba(20, 24, 34, 0.40)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 10,
    padding: "10px 12px"
  },
  debugHint: {
    fontSize: 10,
    color: "rgba(255,255,255,0.75)",
    lineHeight: "14px",
    background: "rgba(124,58,237,0.15)",
    border: "1px solid rgba(124,58,237,0.25)",
    borderRadius: 8,
    padding: "6px 8px",
    marginTop: 4
  }
};

