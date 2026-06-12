import { useEffect, useRef } from "react";
import type { Vec2 } from "../utils/math";

/**
 * Wirtualny kursor śledzący pozycję koniuszka palca wskazującego.
 * Pokazuje pozycję kursora na ekranie i obsługuje hover/select z timerem.
 */

type Props = {
  position: Vec2 | null; // znormalizowane współrzędne (0..1) z MediaPipe
  onHover?: (element: HTMLElement | null) => void;
  onSelect?: (element: HTMLElement | null) => void;
  selectDelayMs?: number; // opóźnienie przed select (domyślnie 2000ms = 2 sekundy)
};

export function VirtualCursor({ position, onHover, onSelect, selectDelayMs = 2000 }: Props) {
  const cursorRef = useRef<HTMLDivElement>(null);
  const hoverTimerRef = useRef<number | null>(null);
  const lastElementRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!position || !cursorRef.current) {
      // Ukryj kursor gdy nie ma pozycji
      cursorRef.current!.style.display = "none";
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }
      if (lastElementRef.current) {
        onHover?.(null);
        lastElementRef.current = null;
      }
      return;
    }

    // Pokaż kursor
    cursorRef.current.style.display = "block";

    // Konwertuj znormalizowane współrzędne (0..1) na piksele ekranu
    // MediaPipe: (0,0) = lewy górny róg, (1,1) = prawy dolny róg
    const x = position.x * window.innerWidth;
    const y = position.y * window.innerHeight;

    cursorRef.current.style.left = `${x}px`;
    cursorRef.current.style.top = `${y}px`;

    // Sprawdź czy kursor jest nad jakimś elementem interaktywnym
    const elementBelow = document.elementFromPoint(x, y) as HTMLElement | null;
    const interactiveElement = elementBelow?.closest("[data-gesture-target]") as HTMLElement | null;

    if (interactiveElement && interactiveElement !== lastElementRef.current) {
      // Nowy element - reset timera
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }

      lastElementRef.current = interactiveElement;
      onHover?.(interactiveElement);

      // Start timera dla select
      hoverTimerRef.current = window.setTimeout(() => {
        onSelect?.(interactiveElement);
        hoverTimerRef.current = null;
      }, selectDelayMs);
    } else if (!interactiveElement && lastElementRef.current) {
      // Kursor opuścił element
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }
      onHover?.(null);
      lastElementRef.current = null;
    }
  }, [position, onHover, onSelect, selectDelayMs]);

  // Cleanup przy unmount
  useEffect(() => {
    return () => {
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
      }
    };
  }, []);

  return (
    <div
      ref={cursorRef}
      style={styles.cursor}
      data-gesture-cursor
      aria-hidden="true"
    >
      <div style={styles.cursorDot} />
      <div style={styles.cursorRing} />
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  cursor: {
    position: "fixed",
    pointerEvents: "none",
    zIndex: 10000,
    transform: "translate(-50%, -50%)",
    display: "none"
  },
  cursorDot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: "rgba(124,58,237,0.95)",
    border: "2px solid rgba(255,255,255,0.9)",
    boxShadow: "0 0 12px rgba(124,58,237,0.6)"
  },
  cursorRing: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    width: 24,
    height: 24,
    borderRadius: "50%",
    border: "2px solid rgba(124,58,237,0.4)",
    animation: "pulse 1.5s ease-in-out infinite"
  }
};
