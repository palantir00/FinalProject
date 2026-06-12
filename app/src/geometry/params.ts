import type { SolidType } from "./types";

/**
 * Parametry brył geometrycznych (używane do obliczeń area/volume).
 */

export type SolidParams = 
  | { type: "cube"; side: number }
  | { type: "sphere"; radius: number }
  | { type: "cylinder"; radius: number; height: number }
  | { type: "cone"; radius: number; height: number }
  | { type: "pyramid"; baseSide: number; height: number }
  | { type: "box"; width: number; height: number; depth: number };

export function getDefaultParams(type: SolidType | string): SolidParams {
  switch (type) {
    case "cube":
      return { type: "cube", side: 1.0 };
    case "sphere":
      return { type: "sphere", radius: 0.5 };
    case "cylinder":
      return { type: "cylinder", radius: 0.5, height: 1.0 };
    case "cone":
      return { type: "cone", radius: 0.5, height: 1.0 };
    case "pyramid":
      return { type: "pyramid", baseSide: 1.0, height: 1.0 };
    case "box":
      return { type: "box", width: 1.0, height: 1.0, depth: 1.0 };
    default:
      return { type: "cube", side: 1.0 };
  }
}
