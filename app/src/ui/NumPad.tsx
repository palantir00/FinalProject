import { useEffect, useRef, useState } from "react";
import type { Vec2 } from "../utils/math";

/**
 * Klawiatura numeryczna obsługiwana gestami.
 * Użytkownik wskazuje palcem wskazującym na cyfrę, przytrzymuje 2 sekundy, aby wybrać.
 */

type Props = {
  visible: boolean;
  cursorPosition: Vec2 | null; // pozycja wirtualnego kursora (znormalizowane 0..1)
  onDigit: (digit: string) => void;
  onBackspace: () => void;
  onClear: () => void;
  onConfirm: () => void;
};

const DIGITS = [
  ["1", "2", "3"],
  ["4", "5", "6"],
  ["7", "8", "9"],
  ["", "0", "."]
];

export function NumPad({ visible, cursorPosition, onDigit, onBackspace, onClear, onConfirm }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);
  const hoverTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (!visible || !cursorPosition || !containerRef.current) {
      setHoveredKey(null);
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }
      return;
    }

    // Konwertuj pozycję kursora na piksele
    const cursorX = cursorPosition.x * window.innerWidth;
    const cursorY = cursorPosition.y * window.innerHeight;

    // Sprawdź czy kursor jest nad kontenerem klawiatury
    const rect = containerRef.current.getBoundingClientRect();
    if (cursorX < rect.left || cursorX > rect.right || cursorY < rect.top || cursorY > rect.bottom) {
      setHoveredKey(null);
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }
      return;
    }

    // Znajdź klawisz pod kursorem
    const buttons = containerRef.current.querySelectorAll<HTMLElement>("[data-key]");
    let foundKey: string | null = null;

    for (const btn of buttons) {
      const btnRect = btn.getBoundingClientRect();
      if (
        cursorX >= btnRect.left &&
        cursorX <= btnRect.right &&
        cursorY >= btnRect.top &&
        cursorY <= btnRect.bottom
      ) {
        foundKey = btn.dataset.key || null;
        break;
      }
    }

    if (foundKey !== hoveredKey) {
      // Nowy klawisz - reset timera
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }

      setHoveredKey(foundKey);

      if (foundKey) {
        // Start timera dla select (2 sekundy)
        hoverTimerRef.current = window.setTimeout(() => {
          handleKeySelect(foundKey!);
          hoverTimerRef.current = null;
        }, 2000);
      }
    }
  }, [cursorPosition, visible, hoveredKey]);

  const handleKeySelect = (key: string) => {
    if (key === "backspace") {
      onBackspace();
    } else if (key === "clear") {
      onClear();
    } else if (key === "confirm") {
      onConfirm();
    } else if (key && key !== "") {
      onDigit(key);
    }
  };

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
      <div style={styles.grid}>
        {DIGITS.map((row, i) => (
          <div key={i} style={styles.row}>
            {row.map((digit, j) => {
              if (digit === "" && i === 3 && j === 0) {
                // Lewy dolny róg - Backspace
                return (
                  <button
                    key={`${i}-${j}`}
                    data-key="backspace"
                    style={{
                      ...styles.key,
                      ...(hoveredKey === "backspace" ? styles.keyHovered : {})
                    }}
                  >
                    ⌫
                  </button>
                );
              }
              if (digit !== "") {
                return (
                  <button
                    key={`${i}-${j}`}
                    data-key={digit}
                    style={{
                      ...styles.key,
                      ...(hoveredKey === digit ? styles.keyHovered : {})
                    }}
                  >
                    {digit}
                  </button>
                );
              }
              return <div key={`${i}-${j}`} style={styles.key} />;
            })}
          </div>
        ))}
      </div>
      <div style={styles.actions}>
        <button
          data-key="clear"
          style={{
            ...styles.actionBtn,
            ...(hoveredKey === "clear" ? styles.keyHovered : {})
          }}
        >
          Wyczyść
        </button>
        <button
          data-key="confirm"
          style={{
            ...styles.actionBtn,
            ...styles.confirmBtn,
            ...(hoveredKey === "confirm" ? styles.keyHovered : {})
          }}
        >
          OK
        </button>
      </div>
      {hoveredKey && (
        <div style={styles.hint}>
          Przytrzymaj 2 sekundy aby wybrać: {hoveredKey === "backspace" ? "⌫" : hoveredKey === "clear" ? "Wyczyść" : hoveredKey === "confirm" ? "OK" : hoveredKey}
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  wrap: {
    width: 240,
    borderRadius: 12,
    background: "rgba(20, 24, 34, 0.95)",
    border: "1px solid rgba(124,58,237,0.35)",
    backdropFilter: "blur(12px)",
    padding: 16,
    boxShadow: "0 8px 32px rgba(0,0,0,0.5)"
  },
  grid: {
    display: "flex",
    flexDirection: "column",
    gap: 8
  },
  row: {
    display: "flex",
    gap: 8
  },
  key: {
    flex: 1,
    minWidth: 60,
    height: 50,
    borderRadius: 8,
    background: "rgba(255,255,255,0.08)",
    border: "1px solid rgba(255,255,255,0.15)",
    color: "rgba(255,255,255,0.95)",
    fontSize: 18,
    fontWeight: 600,
    cursor: "pointer",
    transition: "all 0.2s ease",
    display: "flex",
    alignItems: "center",
    justifyContent: "center"
  },
  keyHovered: {
    background: "rgba(124,58,237,0.35)",
    borderColor: "rgba(124,58,237,0.6)",
    boxShadow: "0 0 16px rgba(124,58,237,0.4)",
    transform: "scale(1.05)"
  },
  actions: {
    display: "flex",
    gap: 8,
    marginTop: 12
  },
  actionBtn: {
    flex: 1,
    padding: "12px 16px",
    borderRadius: 8,
    background: "rgba(255,255,255,0.08)",
    border: "1px solid rgba(255,255,255,0.15)",
    color: "rgba(255,255,255,0.95)",
    fontSize: 13,
    fontWeight: 500,
    cursor: "pointer",
    transition: "all 0.2s ease"
  },
  confirmBtn: {
    background: "rgba(124,58,237,0.25)",
    borderColor: "rgba(124,58,237,0.4)"
  },
  hint: {
    marginTop: 12,
    fontSize: 11,
    color: "rgba(255,255,255,0.65)",
    textAlign: "center",
    padding: "8px 12px",
    background: "rgba(124,58,237,0.15)",
    borderRadius: 8
  }
};
