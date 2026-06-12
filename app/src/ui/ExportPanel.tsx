import { useState } from "react";

type Props = {
  onExportSTL: () => void;
  onExportOBJ: () => void;
  disabled?: boolean;
};

export function ExportPanel({ onExportSTL, onExportOBJ, disabled }: Props) {
  const [feedback, setFeedback] = useState<string | null>(null);

  const handle = (fn: () => void, label: string) => {
    fn();
    setFeedback(`Pobrano ${label}`);
    setTimeout(() => setFeedback(null), 2000);
  };

  return (
    <div style={styles.panel}>
      <div style={styles.title}>Eksport</div>
      <div style={styles.row}>
        <button
          style={{ ...styles.btn, opacity: disabled ? 0.4 : 1 }}
          disabled={disabled}
          onClick={() => handle(onExportSTL, "STL")}
        >
          STL
        </button>
        <button
          style={{ ...styles.btn, opacity: disabled ? 0.4 : 1 }}
          disabled={disabled}
          onClick={() => handle(onExportOBJ, "OBJ")}
        >
          OBJ
        </button>
      </div>
      {feedback && <div style={styles.feedback}>{feedback}</div>}
      {disabled && <div style={styles.hint}>Najpierw utwórz bryłę (Pinch)</div>}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    width: 160,
    borderRadius: 12,
    background: "rgba(20, 24, 34, 0.72)",
    border: "1px solid rgba(255,255,255,0.12)",
    backdropFilter: "blur(8px)",
    padding: 12,
    pointerEvents: "auto"
  },
  title: {
    fontWeight: 700,
    fontSize: 12,
    letterSpacing: 0.3,
    marginBottom: 10,
    color: "rgba(255,255,255,0.92)"
  },
  row: { display: "flex", gap: 8 },
  btn: {
    flex: 1,
    border: "1px solid rgba(255,255,255,0.18)",
    background: "linear-gradient(135deg, rgba(124,58,237,0.75), rgba(99,102,241,0.55))",
    color: "white",
    borderRadius: 8,
    padding: "7px 0",
    cursor: "pointer",
    fontWeight: 700,
    fontSize: 12
  },
  feedback: {
    marginTop: 8,
    fontSize: 11,
    color: "rgba(34,197,94,0.95)",
    textAlign: "center"
  },
  hint: {
    marginTop: 8,
    fontSize: 10,
    color: "rgba(255,255,255,0.45)",
    lineHeight: "14px"
  }
};
