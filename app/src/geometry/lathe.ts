/**
 * Moduł matematyczny dla bryły obrotowej (LatheGeometry).
 * Pipeline: surowe punkty gestu → regresja wielomianowa → próbkowanie profilu → całki.
 */

// ─── Algebra liniowa ──────────────────────────────────────────────────────────

function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let k = 0; k < n; k++) {
    // Eliminacja Gaussa z częściowym przestawianiem wierszy (pivoting)
    let maxRow = k;
    for (let i = k + 1; i < n; i++) {
      if (Math.abs(M[i][k]) > Math.abs(M[maxRow][k])) maxRow = i;
    }
    [M[k], M[maxRow]] = [M[maxRow], M[k]];
    if (Math.abs(M[k][k]) < 1e-12) continue;

    for (let i = k + 1; i < n; i++) {
      const f = M[i][k] / M[k][k];
      for (let j = k; j <= n; j++) M[i][j] -= f * M[k][j];
    }
  }

  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = M[i][n];
    for (let j = i + 1; j < n; j++) x[i] -= M[i][j] * x[j];
    x[i] /= M[i][i] || 1;
  }
  return x;
}

// ─── Regresja wielomianowa (Least Squares) ────────────────────────────────────

/**
 * Dopasowuje wielomian stopnia `degree` do punktów (xs[i], ys[i]).
 * Zwraca współczynniki [a0, a1, ..., ad] takie że f(x) = a0 + a1*x + ... + ad*x^d.
 */
export function fitPolynomial(xs: number[], ys: number[], degree: number): number[] {
  const d = degree + 1;
  const XTX = Array.from({ length: d }, () => new Array(d).fill(0));
  const XTy = new Array(d).fill(0);

  for (let i = 0; i < xs.length; i++) {
    const xpow = Array.from({ length: d }, (_, j) => Math.pow(xs[i], j));
    for (let r = 0; r < d; r++) {
      XTy[r] += xpow[r] * ys[i];
      for (let c = 0; c < d; c++) XTX[r][c] += xpow[r] * xpow[c];
    }
  }

  return solveLinearSystem(XTX, XTy);
}

export function evalPoly(coeffs: number[], x: number): number {
  return coeffs.reduce((acc, c, i) => acc + c * Math.pow(x, i), 0);
}

export function evalPolyDerivative(coeffs: number[], x: number): number {
  return coeffs.slice(1).reduce((acc, c, i) => acc + c * (i + 1) * Math.pow(x, i), 0);
}

// ─── Całkowanie numeryczne — metoda Simpsona ──────────────────────────────────

function simpson(f: (x: number) => number, a: number, b: number, n = 200): number {
  if (n % 2 !== 0) n += 1;
  const h = (b - a) / n;
  let sum = f(a) + f(b);
  for (let i = 1; i < n; i++) {
    sum += f(a + i * h) * (i % 2 === 0 ? 2 : 4);
  }
  return (h / 3) * sum;
}

// ─── Pipeline: punkty gestu → dane dla LatheGeometry ─────────────────────────

export type LatheResult = {
  /** Punkty profilu dla THREE.LatheGeometry: x = promień ≥ 0, y = wysokość */
  lathePoints: { x: number; y: number }[];
  /** Współczynniki wielomianu r = f(h) (h ∈ [0,1]) */
  coeffs: number[];
  /** Wzór f(h) jako string do wyświetlenia */
  formula: string;
  /** Objętość bryły obrotowej [j.u.³] — całka Pappusa: V = π ∫ r(h)² dh */
  volume: number;
  /** Pole powierzchni bocznej [j.u.²] — A = 2π ∫ r(h) √(1 + r'(h)²) dh */
  lateralArea: number;
};

const HEIGHT_3D = 0.7;   // maksymalna wysokość bryły w jednostkach sceny Three.js
const RADIUS_3D = 0.35;  // maksymalny promień bryły
const PROFILE_SAMPLES = 40; // liczba punktów profilu dla LatheGeometry
const POLY_DEGREE = 3;   // stopień wielomianu dopasowania

/**
 * Główna funkcja pipeline:
 * Surowe punkty gestu (przestrzeń ekranu 0..1) → dane dla LatheGeometry + metryki.
 */
export function pathToLathe(screenPoints: { x: number; y: number }[]): LatheResult {
  if (screenPoints.length < 4) {
    throw new Error("Za mało punktów do dopasowania bryły");
  }

  // 1) Mapowanie z przestrzeni ekranu na (h, r):
  //    h = wysokość (oś pionowa, odwrócona: góra ekranu = maksymalna wysokość)
  //    r = promień (odległość pozioma od osi obrotu)
  const hs = screenPoints.map(p => 1 - p.y);   // h ∈ [0, 1] (0 = dół, 1 = góra)
  const rs = screenPoints.map(p => p.x);         // x jako surowy promień

  // 2) Normalizacja obu osi do [0, 1]
  const hMin = Math.min(...hs), hMax = Math.max(...hs);
  const rMin = Math.min(...rs);                   // leftmost point = oś obrotu (r = 0)
  const rMax = Math.max(...rs);

  const hRange = hMax - hMin || 1;
  const rRange = rMax - rMin || 1;

  const hNorm = hs.map(h => (h - hMin) / hRange);
  const rNorm = rs.map(r => (r - rMin) / rRange);  // promień ∈ [0, 1]

  // 3) Regresja wielomianowa: r_norm = f(h_norm)
  const coeffs = fitPolynomial(hNorm, rNorm, POLY_DEGREE);

  // 4) Próbkowanie profilu w 3D
  const lathePoints: { x: number; y: number }[] = [];
  for (let i = 0; i <= PROFILE_SAMPLES; i++) {
    const hN = i / PROFILE_SAMPLES;
    const rN = Math.max(0, evalPoly(coeffs, hN)); // promień nie może być ujemny
    lathePoints.push({
      x: rN * RADIUS_3D,
      y: hN * HEIGHT_3D
    });
  }

  // 5) Całkowanie — objętość (wzór Pappusa): V = π ∫₀¹ [r_3d(h)]² dh_3d
  //    = π * RADIUS_3D² * HEIGHT_3D * ∫₀¹ [f(h_n)]² dh_n
  const volume = Math.PI * RADIUS_3D * RADIUS_3D * HEIGHT_3D *
    simpson(h => Math.max(0, evalPoly(coeffs, h)) ** 2, 0, 1);

  // 6) Pole powierzchni bocznej: A = 2π * RADIUS_3D * HEIGHT_3D *
  //    ∫₀¹ f(h) √(1 + [f'(h) * RADIUS_3D/HEIGHT_3D]²) dh
  const k = RADIUS_3D / HEIGHT_3D;
  const lateralArea = 2 * Math.PI * RADIUS_3D * HEIGHT_3D *
    simpson(h => {
      const r = Math.max(0, evalPoly(coeffs, h));
      const dr = evalPolyDerivative(coeffs, h);
      return r * Math.sqrt(1 + (dr * k) ** 2);
    }, 0, 1);

  // 7) Wzór do wyświetlenia
  const formula = formatPolynomial(coeffs);

  return { lathePoints, coeffs, formula, volume, lateralArea };
}

function formatPolynomial(coeffs: number[]): string {
  const terms = coeffs.map((c, i) => {
    if (Math.abs(c) < 1e-4) return null;
    const val = c.toFixed(3);
    if (i === 0) return val;
    if (i === 1) return `${val}h`;
    return `${val}h²`.replace("h²", i === 2 ? "h²" : `h³`);
  }).filter(Boolean);

  return `r(h) ≈ ${terms.join(" + ").replace(/\+ -/g, "- ")}`;
}
