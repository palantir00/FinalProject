import type { NormalizedLandmarkList } from "@mediapipe/hands";
import { clamp, len3, sub3, cross, norm3 } from "../utils/math";

/**
 * Feature extraction: surowe landmarki -> cechy używane do detekcji gestów.
 * To odpowiada warstwie "mapping: surowe landmarki → stany gestów" ze specyfikacji.
 */

export function palmScale(lm: NormalizedLandmarkList) {
  // Skala dłoni do normalizacji progów: odległość między MCP index (5) a MCP pinky (17)
  const a = lm[5];
  const b = lm[17];
  return len3(sub3(a, b)) || 1e-6;
}

export function pinchStrength(lm: NormalizedLandmarkList, fs?: FingerState) {
  // WAŻNE: Jeśli ręka jest zaciśnięta w pięść (fist), to NIE może być pinch.
  // Sprawdzamy to najpierw, żeby uniknąć fałszywych wykryć.
  if (fs) {
    const isFist = !fs.indexExtended && !fs.middleExtended && !fs.ringExtended && !fs.pinkyExtended;
    if (isFist) {
      return 0; // Pięść = brak pinch
    }
  }
  
  // Pinch: thumb tip (4) + index tip (8)
  const d = len3(sub3(lm[4], lm[8]));
  const s = palmScale(lm);
  
  // Problem: gdy ręka jest daleko od kamery, palmScale może być zbyt małe lub zbyt duże.
  // Rozwiązanie: użyjemy hybrydowego podejścia - normalizacja względem skali dłoni + absolutny próg.
  
  // Normalizacja względem skali dłoni (działa dobrze gdy ręka jest blisko)
  const normalizedDist = d / s;
  
  // Ale też sprawdzamy absolutną odległość (działa lepiej gdy ręka jest daleko)
  // MediaPipe zwraca landmarki w zakresie 0-1, więc absolutna odległość też jest znormalizowana.
  // Próg absolutny: jeśli odległość < 0.08 (w przestrzeni znormalizowanej), to pinch jest silny.
  const absoluteThreshold = 0.08;
  
  // Używamy większej z dwóch wartości: normalizowanej (dla bliskich dłoni) i absolutnej (dla dalekich)
  const normalizedStrength = clamp(1 - normalizedDist / 0.6, 0, 1); // Próg zwiększony do 0.6
  const absoluteStrength = clamp(1 - d / absoluteThreshold, 0, 1);
  
  // Bierzemy maksimum - jeśli którykolwiek test wskazuje na pinch, uznajemy go za wykryty
  return Math.max(normalizedStrength, absoluteStrength);
}

export type FingerState = {
  thumbExtended: boolean;
  indexExtended: boolean;
  middleExtended: boolean;
  ringExtended: boolean;
  pinkyExtended: boolean;
};

function isExtended(lm: NormalizedLandmarkList, tipIdx: number, pipIdx: number, mcpIdx: number) {
  // Prosty heurystyczny test: tip "dalej" od nadgarstka niż PIP/MCP.
  // (W praktyce do strojenia; docelowo można przejść na testy kątów.)
  const wrist = lm[0];
  const tip = lm[tipIdx];
  const pip = lm[pipIdx];
  const mcp = lm[mcpIdx];
  const tipDist = len3(sub3(tip, wrist));
  const pipDist = len3(sub3(pip, wrist));
  const mcpDist = len3(sub3(mcp, wrist));
  return tipDist > pipDist && tipDist > mcpDist;
}

export function fingerState(lm: NormalizedLandmarkList): FingerState {
  return {
    thumbExtended: isExtended(lm, 4, 3, 2),
    indexExtended: isExtended(lm, 8, 6, 5),
    middleExtended: isExtended(lm, 12, 10, 9),
    ringExtended: isExtended(lm, 16, 14, 13),
    pinkyExtended: isExtended(lm, 20, 18, 17)
  };
}

export function isOpenPalm(fs: FingerState) {
  return fs.thumbExtended && fs.indexExtended && fs.middleExtended && fs.ringExtended && fs.pinkyExtended;
}

export function isFist(fs: FingerState) {
  return !fs.indexExtended && !fs.middleExtended && !fs.ringExtended && !fs.pinkyExtended;
}

export function isMeasureL(fs: FingerState, pinchStr: number) {
  // Spec 9): Measure: gest "L" (kciuk + wskazujący) → włącz/wyłącz etykiety.
  // Różnica vs Pinch: Measure wymaga wyraźnego "L" (palce wyprostowane, ale NIE zaciśnięte razem).
  // Jeśli pinchStrength jest wysokie (>0.25), to raczej Pinch niż Measure.
  const hasLShape = fs.thumbExtended && fs.indexExtended && !fs.middleExtended && !fs.ringExtended && !fs.pinkyExtended;
  // Measure tylko jeśli kształt L I pinch jest słaby (palce są wyprostowane, ale nie zaciśnięte).
  return hasLShape && pinchStr < 0.25;
}

export function isPointing(fs: FingerState) {
  // Point: tylko palec wskazujący wyprostowany, reszta zaciśnięta
  // Kciuk może być lekko wyprostowany (nie jest krytyczny dla gestu Point)
  // Używane do sterowania wirtualnym kursorem w menu i klawiaturze numerycznej
  // Warunek: index wyprostowany + reszta (middle, ring, pinky) zaciśnięta
  // Kciuk jest opcjonalny (może być wyprostowany lub zaciśnięty)
  return fs.indexExtended && !fs.middleExtended && !fs.ringExtended && !fs.pinkyExtended;
}

/** Znak V / nożyczki: index + middle wyprostowane, ring + pinky złożone. Wyzwala tryb Sketch (bryła obrotowa). */
export function isPeaceSign(fs: FingerState) {
  return fs.indexExtended && fs.middleExtended && !fs.ringExtended && !fs.pinkyExtended;
}

export function palmNormal(lm: NormalizedLandmarkList) {
  // Przybliżony normalny wektor płaszczyzny dłoni:
  // n = (indexMCP - wrist) x (pinkyMCP - wrist)
  const wrist = lm[0];
  const a = sub3(lm[5], wrist);
  const b = sub3(lm[17], wrist);
  return norm3(cross(a, b));
}

