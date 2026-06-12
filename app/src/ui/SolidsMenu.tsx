import { useEffect, useRef, useState } from "react";
import type { SolidType } from "../geometry/types";
import type { Vec2 } from "../utils/math";

/**
 * Menu brył (spec: "Create: pinch → menu brył (kółko/siatka ikon)").
 * 
 * Pokazuje się gdy mode === "Create" i pozwala wybrać typ bryły do utworzenia.
 * Obsługuje gesty: palec wskazujący do nawigacji, przytrzymanie 2 sekundy do wyboru.
 */

type Props = {
  visible: boolean;
  cursorPosition: Vec2 | null; // pozycja wirtualnego kursora (znormalizowane 0..1)
  onSelect: (solid: SolidType) => void;
};

const SOLIDS: Array<{ type: SolidType; label: string; icon: string }> = [
  { type: "cube", label: "Sześcian", icon: "⬜" },
  { type: "sphere", label: "Kula", icon: "⚪" },
  { type: "cylinder", label: "Walec", icon: "⭕" },
  { type: "cone", label: "Stożek", icon: "🔺" },
  { type: "pyramid", label: "Ostrosłup", icon: "🔺" },
  { type: "box", label: "Prostopadłościan", icon: "📦" }
];

export function SolidsMenu({ visible, cursorPosition, onSelect }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredSolid, setHoveredSolid] = useState<SolidType | null>(null);
  const hoverTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (!visible || !cursorPosition || !containerRef.current) {
      setHoveredSolid(null);
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }
      return;
    }

    // Konwertuj pozycję kursora na piksele
    const cursorX = cursorPosition.x * window.innerWidth;
    const cursorY = cursorPosition.y * window.innerHeight;

    // Sprawdź czy kursor jest nad kontenerem menu
    const rect = containerRef.current.getBoundingClientRect();
    if (cursorX < rect.left || cursorX > rect.right || cursorY < rect.top || cursorY > rect.bottom) {
      setHoveredSolid(null);
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }
      return;
    }

    // Znajdź element pod kursorem
    const buttons = containerRef.current.querySelectorAll<HTMLElement>("[data-solid-type]");
    let foundSolid: SolidType | null = null;

    for (const btn of buttons) {
      const btnRect = btn.getBoundingClientRect();
      if (
        cursorX >= btnRect.left &&
        cursorX <= btnRect.right &&
        cursorY >= btnRect.top &&
        cursorY <= btnRect.bottom
      ) {
        foundSolid = (btn.dataset.solidType as SolidType) || null;
        break;
      }
    }

    if (foundSolid !== hoveredSolid) {
      // Nowy element - reset timera
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }

      setHoveredSolid(foundSolid);

      if (foundSolid) {
        // Start timera dla select (2 sekundy)
        hoverTimerRef.current = window.setTimeout(() => {
          onSelect(foundSolid!);
          hoverTimerRef.current = null;
        }, 2000);
      }
    }
  }, [cursorPosition, visible, hoveredSolid, onSelect]);

  useEffect(() => {
    return () => {
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
      }
    };
  }, []);

  if (!visible) return null;

  return (
    <div ref={containerRef} style={styles.wrap}>
      <div style={styles.title}>Wybierz bryłę</div>
      <div style={styles.grid}>
        {SOLIDS.map((s) => (
          <button
            key={s.type}
            data-solid-type={s.type}
            data-gesture-target
            style={{
              ...styles.item,
              ...(hoveredSolid === s.type ? styles.itemHovered : {})
            }}
            onClick={() => onSelect(s.type)} // Nadal działa z myszką jako fallback
          >
            <div style={styles.icon}>{s.icon}</div>
            <div style={styles.label}>{s.label}</div>
          </button>
        ))}
      </div>
      <div style={styles.hint}>
        {hoveredSolid
          ? `Przytrzymaj 2 sekundy aby wybrać: ${SOLIDS.find((s) => s.type === hoveredSolid)?.label}`
          : "Wskaż palcem wskazującym i przytrzymaj 2 sekundy"}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  wrap: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    width: 480,
    borderRadius: 16,
    background: "rgba(20, 24, 34, 0.92)",
    border: "1px solid rgba(124,58,237,0.35)",
    backdropFilter: "blur(12px)",
    padding: 20,
    pointerEvents: "auto",
    boxShadow: "0 8px 32px rgba(0,0,0,0.4)"
  },
  title: {
    fontWeight: 700,
    fontSize: 18,
    marginBottom: 16,
    textAlign: "center",
    color: "rgba(255,255,255,0.95)"
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: 12
  },
  item: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    padding: "16px 12px",
    borderRadius: 12,
    background: "rgba(124,58,237,0.10)",
    border: "1px solid rgba(124,58,237,0.20)",
    cursor: "pointer",
    transition: "all 0.15s ease",
    color: "rgba(255,255,255,0.90)"
  },
  itemHovered: {
    background: "rgba(124,58,237,0.35)",
    borderColor: "rgba(124,58,237,0.6)",
    boxShadow: "0 0 16px rgba(124,58,237,0.4)",
    transform: "scale(1.05)"
  },
  icon: {
    fontSize: 32,
    lineHeight: 1
  },
  label: {
    fontSize: 12,
    fontWeight: 500,
    textAlign: "center"
  },
  hint: {
    marginTop: 16,
    fontSize: 11,
    color: "rgba(255,255,255,0.55)",
    textAlign: "center",
    lineHeight: "16px"
  }
};
