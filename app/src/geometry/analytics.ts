/**
 * Moduł analityczny (spec: "Geometria: obliczenia pola/objętości").
 *
 * MVP (wymóg użytkownika): pole + objętość dla proceduralnych brył:
 * - sześcian
 * - kula
 * - walec
 * - stożek
 * - ostrosłup
 * - prostopadłościan
 *
 * W przyszłości: dla siatek trójkątowych (mesh) można dodać area/volume z triangulacji.
 */

export type SolidMetrics = {
  area: number; // pole powierzchni
  volume: number; // objętość
};

export function cubeMetrics(side: number): SolidMetrics {
  // Sześcian: A = 6a^2, V = a^3
  const a = side;
  return { area: 6 * a * a, volume: a * a * a };
}

export function sphereMetrics(radius: number): SolidMetrics {
  // Kula: A = 4πr^2, V = 4/3 π r^3
  const r = radius;
  return { area: 4 * Math.PI * r * r, volume: (4 / 3) * Math.PI * r * r * r };
}

export function cylinderMetrics(radius: number, height: number): SolidMetrics {
  // Walec: A = 2πr(h + r), V = πr^2 h
  const r = radius;
  const h = height;
  return { area: 2 * Math.PI * r * (h + r), volume: Math.PI * r * r * h };
}

export function coneMetrics(radius: number, height: number): SolidMetrics {
  // Stożek: A = πr(r + √(r² + h²)), V = (1/3)πr²h
  const r = radius;
  const h = height;
  const slant = Math.sqrt(r * r + h * h);
  return {
    area: Math.PI * r * (r + slant),
    volume: (1 / 3) * Math.PI * r * r * h
  };
}

export function pyramidMetrics(baseSide: number, height: number): SolidMetrics {
  // Ostrosłup czworokątny: A = a² + 2a√(a²/4 + h²), V = (1/3)a²h
  const a = baseSide;
  const h = height;
  const slant = Math.sqrt((a * a) / 4 + h * h);
  return {
    area: a * a + 2 * a * slant,
    volume: (1 / 3) * a * a * h
  };
}

export function boxMetrics(width: number, height: number, depth: number): SolidMetrics {
  // Prostopadłościan: A = 2(ab + ac + bc), V = abc
  const w = width;
  const h = height;
  const d = depth;
  return {
    area: 2 * (w * h + w * d + h * d),
    volume: w * h * d
  };
}
