import type { HandFrame } from "../cv/HandTracker";
import { OneEuroFilterVec3 } from "../cv/filters/oneEuro";
import type { GestureFrame, Mode, GestureName } from "./types";
import { fingerState, isFist, isMeasureL, isOpenPalm, isPeaceSign, isPointing, palmNormal, pinchStrength } from "./landmarkFeatures";
import { clamp, len3, sub3 } from "../utils/math";

/**
 * GestureEngine = "interpretacja (FSM + filtracja sygnału)" ze specyfikacji.
 * - wejście: HandFrame (MediaPipe)
 * - wyjście: GestureFrame (tryb + gest + parametry)
 *
 * Spec 9) Protokół gestów:
 * - Create: pinch (thumb-index) → menu brył
 * - Move: otwarta dłoń → translacja
 * - Rotate: zaciśnięta pięść → obrót
 * - Scale: dwie dłonie "rozciąganie" → skala jednolita
 * - Slice: dłoń poziomo → aktywuj płaszczyznę; ruch dłonią → położenie przekroju
 * - Measure: gest "L" → toggle etykiet
 */

type StableSwitch = {
  candidate: GestureName;
  frames: number;
};

export class GestureEngine {
  // Filtry per-ręka (w praktyce można mapować po "Left/Right").
  private filters = new Map<string, OneEuroFilterVec3[]>();
  private lastFrame: GestureFrame = { tMs: 0, mode: "Idle", gesture: "None", handsDetected: 0 };
  private stable: StableSwitch = { candidate: "None", frames: 0 };
  private measureEnabled = false;
  private lastHandCenter: { x: number; y: number } | null = null;
  private lastWrist: { x: number; y: number } | null = null;
  private lastTwoHandDist: number | null = null;
  private lastSliceState: boolean = false; // Pamięć poprzedniego stanu Slice dla lepszej stabilizacji

  // Stabilizacja stanów (histereza) – żeby gest nie "migał" i nie "wariował".
  // Doc wspomina o stabilności rozpoznania i filtracji.
  // Zwiększone do 4 klatek dla lepszej stabilności - gest musi być wykryty przez 4 klatki zanim się przełączy.
  // To zapobiega "wkradaniu się" innych gestów podczas wykonywania jednego gestu.
  private readonly stableFrames = 4;
  
  // Dodatkowa histereza: jeśli wychodzimy z gestu (np. Pinch → None), wymagamy więcej klatek
  // żeby uniknąć szybkiego "migania" między gestami.
  private readonly exitFrames = 2;

  /**
   * Produkuje kolejny stan gestów.
   * Uwaga: to jest MVP – progi i mapowania będą stroić się na podstawie spec/kalibracji.
   */
  update(input: HandFrame): GestureFrame {
    const { tMs, hands } = input;
    const handsDetected = hands.length;

    if (handsDetected === 0) {
      this.lastFrame = { tMs, mode: "Idle", gesture: "None", handsDetected: 0 };
      this.stable = { candidate: "None", frames: 0 };
      this.lastHandCenter = null;
      this.lastWrist = null;
      this.lastTwoHandDist = null;
      this.lastSliceState = false;
      return this.lastFrame;
    }

    // 1) Filtrowanie landmarków (One-Euro). W spec: "One-Euro + median + outlier reject".
    const filteredHands = hands.map((h) => ({
      ...h,
      landmarks: this.filterLandmarks(h.handedness, h.landmarks, tMs)
    }));

    // 2) Detekcja gestów (MVP heurystyki)
    const primary = filteredHands[0];
    const fs = fingerState(primary.landmarks);
    const openPalm = isOpenPalm(fs);
    const fist = isFist(fs);
    const n = palmNormal(primary.landmarks);

    // WAŻNE: Sprawdzamy fist PRZED pinch, bo fist jest bardziej specyficzny gest.
    // Jeśli ręka jest zaciśnięta w pięść, to NIE może być pinch (nawet jeśli palce są blisko siebie).
    const pinch = pinchStrength(primary.landmarks, fs); // Przekazujemy fs żeby pinchStrength mógł sprawdzić fist
    // Measure wymaga sprawdzenia pinchStrength - jeśli pinch jest silny, to nie Measure.
    const measureL = isMeasureL(fs, pinch);
    // Point: palec wskazujący wyprostowany, reszta zaciśnięta (bez kciuka) - dla wirtualnego kursora
    const pointing = isPointing(fs);

    // Slice: "dłoń poziomo" -> aktywuj.
    // MVP interpretacja: jeśli normalny wektor dłoni jest w przybliżeniu pionowy (|y| wysokie),
    // to dłoń jest "pozioma" (płaszczyzna dłoni ~ równoległa do podłoża).
    // Obniżony próg z 0.7 na 0.5 dla lepszej detekcji + sprawdzamy czy dłoń jest bardziej pozioma niż pionowa
    const palmVertical = Math.abs(n.y);
    const palmHorizontal = Math.abs(n.x) + Math.abs(n.z);
    const slicePalmHorizontal = palmVertical > 0.5 && palmVertical > palmHorizontal * 0.8;
    
    // Stabilizacja Slice: jeśli wcześniej był Slice i dłoń jest nadal w przybliżeniu pozioma, 
    // to kontynuuj Slice (histereza) - zapobiega miganiu między Slice a Move podczas ruchu
    const sliceStable = slicePalmHorizontal || (this.lastSliceState && palmVertical > 0.4);

    let gesture: GestureName = "None";
    let mode: Mode = "Idle";

    // Priorytety: Rotate (fist) > Point (palec wskazujący) > Create (pinch) > Measure (L bez pinch) > Scale (2 hands) > Slice > Move
    // WAŻNE: Fist ma najwyższy priorytet, bo jest najbardziej specyficzny (wszystkie palce zaciśnięte).
    // Point ma wyższy priorytet niż Pinch, bo jest bardziej specyficzny (tylko index extended, reszta zaciśnięta)
    // i jest używany do sterowania UI (menu, klawiatura) - musi być wykryty nawet gdy jest słaby pinch.
    if (fist) {
      // Rotate: zaciśnięta pięść - najwyższy priorytet, bo jest najbardziej specyficzny
      gesture = "Rotate";
      mode = "Transform";
    } else if (pointing) {
      // Point: palec wskazujący wyprostowany - wyższy priorytet niż Pinch
      // Używany do sterowania wirtualnym kursorem w menu/klawiaturze
      // Sprawdzamy PRZED Pinch, żeby Point był wykryty nawet gdy jest słaby pinch
      gesture = "Point";
      mode = "Idle"; // Point nie zmienia trybu, tylko steruje UI
    } else if (isPeaceSign(fs)) {
      // Sketch (znak V): index + middle wyprostowane → tryb rysowania profilu bryły obrotowej
      gesture = "Sketch";
      mode = "Lathe";
    } else if (pinch > 0.25) {
      // Pinch (Create) - tylko gdy NIE ma fist i NIE ma Point
      gesture = "Pinch";
      mode = "Create";
    } else if (measureL) {
      // Measure tylko jeśli pinch jest słaby (<0.25) - to rozróżnia Pinch od Measure
      gesture = "MeasureL";
      mode = "Measure";
    } else if (handsDetected >= 2) {
      // Dwie dłonie: sprawdzamy czy obie robią pinch (Create) czy rozciąganie (Scale)
      const fs1 = fingerState(filteredHands[0].landmarks);
      const fs2 = filteredHands.length > 1 ? fingerState(filteredHands[1].landmarks) : null;
      const pinch1 = pinchStrength(filteredHands[0].landmarks, fs1);
      const pinch2 = fs2 ? pinchStrength(filteredHands[1].landmarks, fs2) : 0;
      if (pinch1 > 0.25 || pinch2 > 0.25) {
        // Jeśli któraś dłoń robi pinch, to Create
        gesture = "Pinch";
        mode = "Create";
      } else {
        gesture = "Scale";
        mode = "Transform";
      }
    } else if (sliceStable) {
      // Slice: dłoń pozioma - wyższy priorytet niż Move, bo jest bardziej specyficzny
      // Używamy sliceStable (z histerezą) zamiast slicePalmHorizontal dla lepszej stabilności
      gesture = "Slice";
      mode = "Slice";
      this.lastSliceState = true; // Zapamiętujemy że Slice był aktywny
    } else if (openPalm) {
      gesture = "Move";
      mode = "Transform";
      this.lastSliceState = false; // Reset gdy przełączamy na Move
    } else {
      this.lastSliceState = false; // Reset gdy żaden gest nie jest aktywny
    }

    // 3) Stabilizacja przełączeń gestu (prosty debounce w klatkach)
    gesture = this.stabilize(gesture);

    // 4) Parametry gestów
    const out: GestureFrame = {
      tMs,
      mode,
      gesture,
      handsDetected,
      pinchStrength: pinch,
      // Debug: stan palców (dla debugowania Point)
      fingerState: {
        thumb: fs.thumbExtended,
        index: fs.indexExtended,
        middle: fs.middleExtended,
        ring: fs.ringExtended,
        pinky: fs.pinkyExtended
      }
    };

    // Point + Sketch: pozycja koniuszka palca wskazującego (landmark 8)
    if (gesture === "Point" || gesture === "Sketch") {
      const indexTip = primary.landmarks[8];
      out.pointPosition = { x: indexTip.x, y: indexTip.y };
    }

    if (gesture === "MeasureL") {
      // Toggle etykiet przy wejściu w gest (detekcja zbocza).
      if (this.lastFrame.gesture !== "MeasureL") {
        this.measureEnabled = !this.measureEnabled;
      }
    }

    if (gesture === "Move") {
      // Delta ruchu dłoni w przestrzeni ekranu (x,y) na bazie punktu 9 (MVP).
      const curr = primary.landmarks[9]; // przybliżony "środek" dłoni
      const prevHand = this.lastHandCenter;
      if (prevHand) out.moveDelta = { x: curr.x - prevHand.x, y: curr.y - prevHand.y };
      this.lastHandCenter = { x: curr.x, y: curr.y };
    }

    if (gesture === "Rotate") {
      const curr = primary.landmarks[0]; // wrist as anchor
      const prev = this.lastWrist;
      if (prev) {
        // Rotate: pełny swobodny obrót 360° wokół wszystkich osi
        // Ruch w osi X → obrót wokół Y (lewo/prawo)
        const deltaX = (curr.x - prev.x) * 3.5;
        // Ruch w osi Y → obrót wokół X (góra/dół)
        const deltaY = (curr.y - prev.y) * 3.5;
        
        // Clamp dla stabilności - jednakowe limity dla wszystkich kierunków
        out.rotateDelta = {
          x: clamp(deltaX, -0.15, 0.15), // Rotacja wokół Y (lewo/prawo)
          y: clamp(deltaY, -0.15, 0.15)  // Rotacja wokół X (góra/dół)
        };
      }
      this.lastWrist = { x: curr.x, y: curr.y };
    }

    if (gesture === "Scale" && handsDetected >= 2) {
      // Scale: dystans między środkami dłoni (landmark 9 = middle finger MCP, przybliżony środek)
      const a = filteredHands[0].landmarks[9];
      const b = filteredHands[1].landmarks[9];
      const dist = len3(sub3(a, b));
      const prevDist = this.lastTwoHandDist;
      if (prevDist && prevDist > 1e-6) {
        // Obliczamy współczynnik skali: jeśli dystans rośnie, obiekt się powiększa.
        // Clamp dla stabilności (zapobiega skokom).
        out.scaleFactor = clamp(dist / prevDist, 0.9, 1.1);
      }
      this.lastTwoHandDist = dist;
    } else if (gesture !== "Scale") {
      // Reset dystansu gdy wychodzimy z gestu Scale, żeby następne wejście było czyste.
      this.lastTwoHandDist = null;
    }

    if (gesture === "Slice") {
      // MVP: ruch dłoni w osi Y mapujemy na przesunięcie płaszczyzny przekroju.
      const curr = primary.landmarks[9];
      const prev = this.lastHandCenter;
      if (prev) out.sliceDelta = clamp((curr.y - prev.y) * 2.0, -0.2, 0.2);
      this.lastHandCenter = { x: curr.x, y: curr.y };
    }

    this.lastFrame = out;
    return out;
  }

  isMeasureEnabled() {
    return this.measureEnabled;
  }

  private stabilize(next: GestureName): GestureName {
    const current = this.lastFrame.gesture;
    
    // Jeśli gest się nie zmienił, zwracamy go natychmiast (bez opóźnienia)
    if (next === current) {
      this.stable = { candidate: next, frames: 0 };
      return next;
    }
    
    // Jeśli nowy kandydat jest inny niż poprzedni, resetujemy licznik
    if (this.stable.candidate !== next) {
      this.stable = { candidate: next, frames: 1 };
      // Zwracamy poprzedni gest - nowy musi być stabilny przez kilka klatek
      return current;
    }
    
    // Zwiększamy licznik stabilności nowego gestu
    this.stable.frames += 1;
    
    // Sprawdzamy czy nowy gest jest wystarczająco stabilny
    // Jeśli wychodzimy z gestu na None/Idle, wymagamy mniej klatek (szybsze wyjście)
    const requiredFrames = next === "None" ? this.exitFrames : this.stableFrames;
    
    if (this.stable.frames >= requiredFrames) {
      // Gest jest wystarczająco stabilny - przełączamy
      this.stable = { candidate: next, frames: 0 };
      return next;
    }
    
    // Gest jeszcze nie jest wystarczająco stabilny - zwracamy poprzedni
    return current;
  }

  private filterLandmarks(key: string, lm: Array<{ x: number; y: number; z: number }>, tMs: number) {
    let fs = this.filters.get(key);
    if (!fs) {
      // Parametry startowe: minCutoff=2.0Hz (jeszcze wyższy dla szybszej responsywności), beta=0.08 (bardziej adaptacyjny), dCutoff=1.5Hz.
      // Wyższy beta i minCutoff = bardziej reaguje na szybkie ruchy, co zmniejsza "zacinanie" podczas gestów.
      // dCutoff też zwiększony dla szybszej reakcji na zmiany prędkości.
      fs = new Array(21).fill(null).map(() => new OneEuroFilterVec3(2.0, 0.08, 1.5));
      this.filters.set(key, fs);
    }
    return lm.map((p, i) => fs![i].filter(p, tMs));
  }
}

