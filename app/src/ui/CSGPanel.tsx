import type { CSGOperation } from "../three/SceneManager";

type Props = {
  meshCount: number;
  onOperation: (op: CSGOperation) => void;
};

const OPS: { op: CSGOperation; label: string; symbol: string; desc: string }[] = [
  { op: "union",       label: "Suma",            symbol: "A ∪ B", desc: "polacz obie bryly w jedna" },
  { op: "subtraction", label: "Roznica",          symbol: "A − B", desc: "odejmij B od A" },
  { op: "intersection",label: "Czesc wspolna",   symbol: "A ∩ B", desc: "zostaw tylko czesc wspolna" }
];

export function CSGPanel({ meshCount, onOperation }: Props) {
  if (meshCount < 2) return null;

  return (
    <div style={styles.wrap}>
      <div style={styles.header}>
        <div style={styles.title}>Operacje CSG</div>
        <div style={styles.hint}>{meshCount} bryly w scenie — kliknij aby polaczyc</div>
      </div>
      <div style={styles.hint2}>
        Kliknij bryle aby zmienic aktywna (podswietlona = aktywna)
      </div>
      <div style={styles.ops}>
        {OPS.map(({ op, label, symbol, desc }) => (
          <button
            key={op}
            style={styles.btn}
            data-gesture-target
            onClick={() => onOperation(op)}
          >
            <span style={styles.symbol}>{symbol}</span>
            <span style={styles.label}>{label}</span>
            <span style={styles.desc}>{desc}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  wrap: {
    position: "absolute",
    bottom: 70,
    right: 12,
    width: 220,
    borderRadius: 12,
    background: "rgba(20, 24, 34, 0.82)",
    border: "1px solid rgba(124,58,237,0.4)",
    backdropFilter: "blur(8px)",
    padding: 12,
    pointerEvents: "auto"
  },
  header: {
    marginBottom: 4
  },
  title: {
    fontWeight: 700,
    fontSize: 13,
    color: "rgba(167,139,250,1)"
  },
  hint: {
    fontSize: 10,
    color: "rgba(255,255,255,0.5)",
    marginTop: 2
  },
  hint2: {
    fontSize: 10,
    color: "rgba(255,255,255,0.4)",
    marginBottom: 8,
    lineHeight: "14px"
  },
  ops: {
    display: "grid",
    gap: 6
  },
  btn: {
    display: "grid",
    gridTemplateColumns: "auto 1fr",
    gridTemplateRows: "auto auto",
    gap: "0 8px",
    alignItems: "center",
    padding: "8px 10px",
    borderRadius: 8,
    border: "1px solid rgba(124,58,237,0.35)",
    background: "rgba(124,58,237,0.12)",
    color: "rgba(255,255,255,0.9)",
    cursor: "pointer",
    textAlign: "left" as const
  },
  symbol: {
    gridRow: "1 / 3",
    fontSize: 18,
    fontWeight: 700,
    color: "rgba(167,139,250,1)",
    minWidth: 44
  },
  label: {
    fontSize: 11,
    fontWeight: 700
  },
  desc: {
    fontSize: 10,
    color: "rgba(255,255,255,0.5)",
    lineHeight: "14px"
  }
};
