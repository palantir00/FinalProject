import { useState, useEffect, useRef } from "react";
import type { SolidType } from "../geometry/types";
import type { SolidParams } from "../geometry/params";
import { getDefaultParams } from "../geometry/params";
import type { Vec2 } from "../utils/math";
import { NumPad } from "./NumPad";

/**
 * Formularz parametrów bryły - pojawia się po wyborze bryły z menu.
 * Pozwala użytkownikowi wprowadzić parametry potrzebne do obliczenia area/volume.
 * Obsługuje gesty: palec wskazujący do wyboru pola, NumPad do wprowadzania wartości.
 */

type Props = {
  solidType: SolidType;
  visible: boolean;
  cursorPosition: Vec2 | null; // pozycja wirtualnego kursora (znormalizowane 0..1)
  onConfirm: (params: SolidParams) => void;
  onCancel: () => void;
};

export function SolidParamsForm({ solidType, visible, cursorPosition, onConfirm, onCancel }: Props) {
  const [params, setParams] = useState<SolidParams>(() => {
    return getDefaultParams(solidType);
  });
  const [activeInput, setActiveInput] = useState<string | null>(null);
  const [inputValue, setInputValue] = useState<string>("");
  const containerRef = useRef<HTMLDivElement>(null);
  const hoverTimerRef = useRef<number | null>(null);

  useEffect(() => {
    // Reset parametrów gdy zmienia się typ bryły
    setParams(getDefaultParams(solidType));
    setActiveInput(null);
    setInputValue("");
  }, [solidType]);

  // Obsługa gestów: wybór pola input
  useEffect(() => {
    if (!visible || !cursorPosition || !containerRef.current) {
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }
      return;
    }

    const cursorX = cursorPosition.x * window.innerWidth;
    const cursorY = cursorPosition.y * window.innerHeight;

    // Sprawdź wszystkie elementy interaktywne (inputy i przyciski) w całym kontenerze
    // Znajdź element najbliższy kursora (według pozycji Y), żeby uniknąć problemów z nakładającymi się elementami
    const containerRect = containerRef.current.getBoundingClientRect();
    const interactiveElements = containerRef.current.querySelectorAll<HTMLElement>("[data-input-field], [data-gesture-target]");
    let foundInput: string | null = null;
    let closestElement: HTMLElement | null = null;
    let closestDistance = Infinity;

    for (const element of interactiveElements) {
      const elementRect = element.getBoundingClientRect();
      if (
        cursorX >= elementRect.left &&
        cursorX <= elementRect.right &&
        cursorY >= elementRect.top &&
        cursorY <= elementRect.bottom
      ) {
        // Oblicz odległość od środka elementu do kursora
        const centerY = elementRect.top + elementRect.height / 2;
        const distance = Math.abs(cursorY - centerY);
        
        // Wybierz element najbliższy kursora
        if (distance < closestDistance) {
          closestDistance = distance;
          closestElement = element;
        }
      }
    }

    // Jeśli znaleziono najbliższy element, określ jego typ
    if (closestElement) {
      if (closestElement.dataset.inputField) {
        foundInput = closestElement.dataset.inputField || null;
      } else if (closestElement.tagName === "BUTTON") {
        // Przycisk Anuluj lub Utwórz
        if (closestElement.textContent?.includes("Anuluj")) {
          foundInput = "cancel";
        } else if (closestElement.textContent?.includes("Utwórz")) {
          foundInput = "confirm";
        }
      }
    }

    // Jeśli kursor jest poza kontenerem i nie ma aktywnego inputu, resetuj
    if (!foundInput && (cursorX < containerRect.left || cursorX > containerRect.right || cursorY < containerRect.top || cursorY > containerRect.bottom)) {
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }
      // Nie resetuj activeInput tutaj - pozwól użytkownikowi kontynuować edycję
      return;
    }

    // Aktualizuj activeInput dla wizualnej informacji zwrotnej (hover)
    // Ustawiamy activeInput od razu, żeby pokazać hover i otworzyć NumPad
    // Timer służy tylko do potwierdzenia wyboru (dla przycisków) lub do aktywacji inputa (dla pól)
    if (foundInput !== activeInput) {
      if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
      }

      if (foundInput) {
        // Dla przycisków ustawiamy activeInput od razu dla wizualnej informacji zwrotnej
        if (foundInput === "cancel" || foundInput === "confirm") {
          setActiveInput(foundInput);
          // Start timera dla akcji (2 sekundy)
          hoverTimerRef.current = window.setTimeout(() => {
            if (foundInput === "cancel") {
              onCancel();
            } else if (foundInput === "confirm") {
              onConfirm(params);
            }
            hoverTimerRef.current = null;
          }, 2000);
        } else {
          // Dla inputów ustawiamy activeInput od razu, żeby otworzyć NumPad
          setActiveInput(foundInput);
          // Ustaw wartość początkową z params
          const currentValue = getInputValue(foundInput, params);
          setInputValue(currentValue);
          // Timer nie jest potrzebny dla inputów - użytkownik może od razu używać NumPad
        }
      } else {
        // Reset activeInput gdy kursor opuścił elementy (tylko dla przycisków)
        // Dla inputów nie resetujemy, żeby użytkownik mógł kontynuować edycję
        if (activeInput === "cancel" || activeInput === "confirm") {
          setActiveInput(null);
        }
      }
    }
  }, [cursorPosition, visible, activeInput, params, onCancel, onConfirm]);

  const getInputValue = (field: string, p: SolidParams): string => {
    switch (field) {
      case "side": return p.type === "cube" ? String(p.side) : "";
      case "radius": return (p.type === "sphere" || p.type === "cylinder" || p.type === "cone") ? String(p.radius) : "";
      case "height": return (p.type === "cylinder" || p.type === "cone" || p.type === "pyramid") ? String(p.height) : "";
      case "baseSide": return p.type === "pyramid" ? String(p.baseSide) : "";
      case "width": return p.type === "box" ? String(p.width) : "";
      case "depth": return p.type === "box" ? String(p.depth) : "";
      default: return "";
    }
  };

  const updateParam = (field: string, value: number) => {
    const newParams = { ...params };
    switch (field) {
      case "side":
        if (newParams.type === "cube") newParams.side = value;
        break;
      case "radius":
        if (newParams.type === "sphere") newParams.radius = value;
        else if (newParams.type === "cylinder" || newParams.type === "cone") newParams.radius = value;
        break;
      case "height":
        if (newParams.type === "cylinder" || newParams.type === "cone" || newParams.type === "pyramid") newParams.height = value;
        else if (newParams.type === "box") newParams.height = value;
        break;
      case "baseSide":
        if (newParams.type === "pyramid") newParams.baseSide = value;
        break;
      case "width":
        if (newParams.type === "box") newParams.width = value;
        break;
      case "depth":
        if (newParams.type === "box") newParams.depth = value;
        break;
    }
    setParams(newParams);
  };

  const handleDigit = (digit: string) => {
    if (activeInput) {
      if (digit === ".") {
        if (!inputValue.includes(".")) {
          setInputValue(inputValue + ".");
        }
      } else {
        setInputValue(inputValue + digit);
        const numValue = parseFloat(inputValue + digit) || 0;
        updateParam(activeInput, numValue);
      }
    }
  };

  const handleBackspace = () => {
    if (activeInput && inputValue.length > 0) {
      const newValue = inputValue.slice(0, -1);
      setInputValue(newValue);
      const numValue = parseFloat(newValue) || 0;
      updateParam(activeInput, numValue);
    }
  };

  const handleClear = () => {
    if (activeInput) {
      setInputValue("");
      updateParam(activeInput, 0.1);
    }
  };

  if (!visible) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onConfirm(params);
  };

  const renderInputs = (): React.ReactNode => {
    switch (params.type) {
      case "cube":
        return (
          <div style={styles.inputGroup}>
            <label style={styles.label}>Bok (a):</label>
            <input
              type="text"
              data-input-field="side"
              data-gesture-target
              value={activeInput === "side" ? inputValue : String(params.side)}
              onChange={(e) => {
                const val = parseFloat(e.target.value) || 0;
                setParams({ ...params, side: val });
              }}
              onFocus={() => {
                setActiveInput("side");
                setInputValue(String(params.side));
              }}
              style={{
                ...styles.input,
                ...(activeInput === "side" ? styles.inputActive : {})
              }}
            />
          </div>
        );
      case "sphere":
        return (
          <div style={styles.inputGroup}>
            <label style={styles.label}>Promień (r):</label>
            <input
              type="text"
              data-input-field="radius"
              data-gesture-target
              value={activeInput === "radius" ? inputValue : String(params.radius)}
              onChange={(e) => {
                const val = parseFloat(e.target.value) || 0;
                setParams({ ...params, radius: val });
              }}
              onFocus={() => {
                setActiveInput("radius");
                setInputValue(String(params.radius));
              }}
              style={{
                ...styles.input,
                ...(activeInput === "radius" ? styles.inputActive : {})
              }}
            />
          </div>
        );
      case "cylinder":
        return (
          <>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Promień (r):</label>
              <input
                type="text"
                data-input-field="radius"
                data-gesture-target
                value={activeInput === "radius" ? inputValue : String(params.radius)}
                onChange={(e) => {
                  const val = parseFloat(e.target.value) || 0;
                  setParams({ ...params, radius: val });
                }}
                onFocus={() => {
                  setActiveInput("radius");
                  setInputValue(String(params.radius));
                }}
                style={{
                  ...styles.input,
                  ...(activeInput === "radius" ? styles.inputActive : {})
                }}
              />
            </div>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Wysokość (h):</label>
              <input
              type="text"
              data-input-field="height"
              data-gesture-target
              value={activeInput === "height" ? inputValue : String(params.height)}
              onChange={(e) => {
                const val = parseFloat(e.target.value) || 0;
                setParams({ ...params, height: val });
              }}
              onFocus={() => {
                setActiveInput("height");
                setInputValue(String(params.height));
              }}
              style={{
                ...styles.input,
                ...(activeInput === "height" ? styles.inputActive : {})
              }}
              />
            </div>
          </>
        );
      case "cone":
        return (
          <>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Promień podstawy (r):</label>
              <input
                type="text"
                data-input-field="radius"
                data-gesture-target
                value={activeInput === "radius" ? inputValue : String(params.radius)}
                onChange={(e) => {
                  const val = parseFloat(e.target.value) || 0;
                  setParams({ ...params, radius: val });
                }}
                onFocus={() => {
                  setActiveInput("radius");
                  setInputValue(String(params.radius));
                }}
                style={{
                  ...styles.input,
                  ...(activeInput === "radius" ? styles.inputActive : {})
                }}
              />
            </div>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Wysokość (h):</label>
              <input
              type="text"
              data-input-field="height"
              data-gesture-target
              value={activeInput === "height" ? inputValue : String(params.height)}
              onChange={(e) => {
                const val = parseFloat(e.target.value) || 0;
                setParams({ ...params, height: val });
              }}
              onFocus={() => {
                setActiveInput("height");
                setInputValue(String(params.height));
              }}
              style={{
                ...styles.input,
                ...(activeInput === "height" ? styles.inputActive : {})
              }}
              />
            </div>
          </>
        );
      case "pyramid":
        return (
          <>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Bok podstawy (a):</label>
              <input
              type="text"
              data-input-field="baseSide"
              data-gesture-target
              value={activeInput === "baseSide" ? inputValue : String(params.baseSide)}
              onChange={(e) => {
                const val = parseFloat(e.target.value) || 0;
                setParams({ ...params, baseSide: val });
              }}
              onFocus={() => {
                setActiveInput("baseSide");
                setInputValue(String(params.baseSide));
              }}
              style={{
                ...styles.input,
                ...(activeInput === "baseSide" ? styles.inputActive : {})
              }}
              />
            </div>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Wysokość (h):</label>
              <input
              type="text"
              data-input-field="height"
              data-gesture-target
              value={activeInput === "height" ? inputValue : String(params.height)}
              onChange={(e) => {
                const val = parseFloat(e.target.value) || 0;
                setParams({ ...params, height: val });
              }}
              onFocus={() => {
                setActiveInput("height");
                setInputValue(String(params.height));
              }}
              style={{
                ...styles.input,
                ...(activeInput === "height" ? styles.inputActive : {})
              }}
              />
            </div>
          </>
        );
      case "box":
        return (
          <>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Szerokość (a):</label>
              <input
              type="text"
              data-input-field="width"
              data-gesture-target
              value={activeInput === "width" ? inputValue : String(params.width)}
              onChange={(e) => {
                const val = parseFloat(e.target.value) || 0;
                setParams({ ...params, width: val });
              }}
              onFocus={() => {
                setActiveInput("width");
                setInputValue(String(params.width));
              }}
              style={{
                ...styles.input,
                ...(activeInput === "width" ? styles.inputActive : {})
              }}
              />
            </div>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Wysokość (b):</label>
              <input
              type="text"
              data-input-field="height"
              data-gesture-target
              value={activeInput === "height" ? inputValue : String(params.height)}
              onChange={(e) => {
                const val = parseFloat(e.target.value) || 0;
                setParams({ ...params, height: val });
              }}
              onFocus={() => {
                setActiveInput("height");
                setInputValue(String(params.height));
              }}
              style={{
                ...styles.input,
                ...(activeInput === "height" ? styles.inputActive : {})
              }}
              />
            </div>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Głębokość (c):</label>
              <input
              type="text"
              data-input-field="depth"
              data-gesture-target
              value={activeInput === "depth" ? inputValue : String(params.depth)}
              onChange={(e) => {
                const val = parseFloat(e.target.value) || 0;
                if (params.type === "box") setParams({ ...params, depth: val });
              }}
              onFocus={() => {
                setActiveInput("depth");
                setInputValue(String(params.depth));
              }}
              style={{
                ...styles.input,
                ...(activeInput === "depth" ? styles.inputActive : {})
              }}
              />
            </div>
          </>
        );
    }
  };

  return (
    <div style={styles.overlay}>
      <div style={styles.container}>
        <div ref={containerRef} style={styles.wrap}>
          <div style={styles.title}>Parametry bryły</div>
          <form onSubmit={handleSubmit}>
            {renderInputs()}
            <div style={styles.buttons}>
              <button
                type="button"
                data-gesture-target
                style={{
                  ...styles.cancelBtn,
                  ...(activeInput === "cancel" ? styles.buttonHovered : {})
                }}
                onClick={onCancel}
              >
                Anuluj
              </button>
              <button
                type="submit"
                data-gesture-target
                style={{
                  ...styles.confirmBtn,
                  ...(activeInput === "confirm" ? styles.buttonHovered : {})
                }}
              >
                Utwórz
              </button>
            </div>
          </form>
        </div>
        {activeInput && activeInput !== "cancel" && activeInput !== "confirm" && (
          <div style={styles.numpadContainer}>
            <NumPad
              visible={true}
              cursorPosition={cursorPosition}
              onDigit={handleDigit}
              onBackspace={handleBackspace}
              onClear={handleClear}
              onConfirm={() => setActiveInput(null)}
            />
          </div>
        )}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: "fixed",
    inset: 0,
    display: "grid",
    placeItems: "center",
    background: "rgba(0,0,0,0.6)",
    backdropFilter: "blur(4px)",
    zIndex: 1000,
    pointerEvents: "auto"
  },
  container: {
    display: "flex",
    gap: 20,
    alignItems: "flex-start"
  },
  wrap: {
    width: 360,
    borderRadius: 16,
    background: "rgba(20, 24, 34, 0.95)",
    border: "1px solid rgba(124,58,237,0.35)",
    backdropFilter: "blur(12px)",
    padding: 24,
    boxShadow: "0 8px 32px rgba(0,0,0,0.5)"
  },
  numpadContainer: {
    display: "flex",
    alignItems: "flex-start"
  },
  title: {
    fontWeight: 700,
    fontSize: 18,
    marginBottom: 20,
    textAlign: "center",
    color: "rgba(255,255,255,0.95)"
  },
  inputGroup: {
    marginBottom: 16
  },
  label: {
    display: "block",
    fontSize: 13,
    fontWeight: 500,
    color: "rgba(255,255,255,0.85)",
    marginBottom: 6
  },
  input: {
    width: "100%",
    padding: "10px 12px",
    borderRadius: 8,
    background: "rgba(255,255,255,0.08)",
    border: "1px solid rgba(255,255,255,0.15)",
    color: "rgba(255,255,255,0.95)",
    fontSize: 14,
    outline: "none",
    cursor: "pointer"
  },
  inputActive: {
    background: "rgba(124,58,237,0.25)",
    borderColor: "rgba(124,58,237,0.6)",
    boxShadow: "0 0 12px rgba(124,58,237,0.3)"
  },
  buttonHovered: {
    background: "rgba(124,58,237,0.35)",
    borderColor: "rgba(124,58,237,0.6)",
    boxShadow: "0 0 16px rgba(124,58,237,0.4)",
    transform: "scale(1.05)"
  },
  buttons: {
    display: "flex",
    gap: 12,
    marginTop: 24
  },
  cancelBtn: {
    flex: 1,
    padding: "12px 16px",
    borderRadius: 10,
    background: "rgba(255,255,255,0.08)",
    border: "1px solid rgba(255,255,255,0.15)",
    color: "rgba(255,255,255,0.90)",
    cursor: "pointer",
    fontWeight: 500
  },
  confirmBtn: {
    flex: 1,
    padding: "12px 16px",
    borderRadius: 10,
    background: "linear-gradient(135deg, rgba(124,58,237,0.85), rgba(99,102,241,0.65))",
    border: "1px solid rgba(124,58,237,0.35)",
    color: "white",
    cursor: "pointer",
    fontWeight: 600
  }
};
