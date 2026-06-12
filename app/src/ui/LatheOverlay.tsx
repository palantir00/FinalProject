import { useEffect, useRef } from "react";

type Props = {
  /** Nagrane punkty gestu w znormalizowanych współrzędnych ekranu (0..1) */
  path: { x: number; y: number }[];
  active: boolean;
};

export function LatheOverlay({ path, active }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!active && path.length === 0) return;

    const W = canvas.width;
    const H = canvas.height;

    // Oś obrotu: pionowa linia po lewej stronie narysowanego profilu
    const axisX = path.length > 0
      ? Math.min(...path.map(p => p.x)) * W
      : W * 0.3;

    // Oś obrotu — przerywana linia
    ctx.save();
    ctx.strokeStyle = "rgba(34,197,94,0.6)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([8, 6]);
    ctx.beginPath();
    ctx.moveTo(axisX, 0);
    ctx.lineTo(axisX, H);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // Nagrany tor — punkty
    if (path.length > 1) {
      ctx.save();
      ctx.strokeStyle = "rgba(124,58,237,0.85)";
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(path[0].x * W, path[0].y * H);
      for (let i = 1; i < path.length; i++) {
        ctx.lineTo(path[i].x * W, path[i].y * H);
      }
      ctx.stroke();
      ctx.restore();

      // Punkty (małe kółka co 5 próbek)
      ctx.save();
      ctx.fillStyle = "rgba(167,139,250,0.9)";
      for (let i = 0; i < path.length; i += 5) {
        ctx.beginPath();
        ctx.arc(path[i].x * W, path[i].y * H, 3, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    // Etykieta osi obrotu
    ctx.save();
    ctx.fillStyle = "rgba(34,197,94,0.85)";
    ctx.font = "11px monospace";
    ctx.fillText("oś obrotu", axisX + 6, 18);
    ctx.restore();

  }, [path, active]);

  if (!active && path.length === 0) return null;

  return (
    <div style={styles.wrap}>
      <canvas ref={canvasRef} style={styles.canvas} />
      <div style={styles.badge}>
        {path.length < 8
          ? "Tryb szkicu aktywny — uzyj gestu Point aby rysowac profil"
          : `Narysowano ${path.length} pkt — gest V zatwierdza i generuje bryle`}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  wrap: {
    position: "absolute",
    inset: 0,
    pointerEvents: "none"
  },
  canvas: {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%"
  },
  badge: {
    position: "absolute",
    bottom: 60,
    left: "50%",
    transform: "translateX(-50%)",
    background: "rgba(20,24,34,0.82)",
    border: "1px solid rgba(124,58,237,0.5)",
    borderRadius: 10,
    padding: "8px 16px",
    fontSize: 13,
    color: "rgba(255,255,255,0.92)",
    backdropFilter: "blur(8px)",
    whiteSpace: "nowrap"
  }
};
