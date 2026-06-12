import { useMemo, useState } from "react";

export function TutorialOverlay() {
  const [open, setOpen] = useState(true);

  const steps = useMemo(
    () => [
      {
        title: "Create",
        desc: "Pinch (kciuk + wskazujacy razem) otwiera menu bryl. Palcem wskazujacym (gest Point) najedz na bryle i przytrzymaj chwile aby wybrac."
      },
      {
        title: "Move",
        desc: "Otwarta dlon (wszystkie palce wyprostowane) przesuwa bryle w osi ekranu."
      },
      {
        title: "Rotate",
        desc: "Zacisnieta piesc obraca bryle we wszystkich osiach. Ruch dlonia = obrot."
      },
      {
        title: "Scale",
        desc: "Dwie dlonie otwarte, rozciaganie dystansu = zoom kamery."
      },
      {
        title: "Slice",
        desc: "Dlon ulozana poziomo (rownolegla do podlogi) aktywuje plaszczyzne przekroju. Ruch gora/dol przesuwa przekroj."
      },
      {
        title: "Measure",
        desc: "Gest L (kciuk + wskazujacy wyprostowane, reszta zlozona) wlacza lub wylacza etykiety z polem powierzchni i objetoscia bryly."
      },
      {
        title: "Sketch (bryly obrotowe)",
        desc: "Znak V (index + srodkowy wyprostowane) wlacza tryb szkicu. Nastepnie gestem Point (tylko palec wskazujacy) narysuj profil np. wazy. Ponowny gest V zatwierdza i generuje bryle obrotowa."
      },
      {
        title: "Eksport STL i OBJ",
        desc: "Po stworzeniu dowolnej bryly odblokowuja sie przyciski STL i OBJ w lewym dolnym rogu. Kliknij aby pobrac plik gotowy do drukarki 3D."
      }
    ],
    []
  );

  if (!open) {
    return (
      <button style={styles.fab} onClick={() => setOpen(true)}>
        Tutorial
      </button>
    );
  }

  return (
    <div style={styles.wrap}>
      <div style={styles.panel}>
        <div style={styles.head}>
          <div style={styles.title}>Tutorial gestow</div>
          <button style={styles.close} onClick={() => setOpen(false)}>
            Zamknij
          </button>
        </div>
        <div style={styles.list}>
          {steps.map((s) => (
            <div key={s.title} style={styles.item}>
              <div style={styles.itemTitle}>{s.title}</div>
              <div style={styles.itemDesc}>{s.desc}</div>
            </div>
          ))}
        </div>
        <div style={styles.footer}>
          Wskazowka: pierwsze uruchomienie wymaga zgody na kamere. Najlepiej dziala na localhost.
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  wrap: {
    position: "absolute",
    right: 12,
    top: 12,
    pointerEvents: "auto",
    maxHeight: "calc(100vh - 200px)",
    display: "flex",
    flexDirection: "column"
  },
  panel: {
    width: 360,
    borderRadius: 12,
    background: "rgba(20, 24, 34, 0.72)",
    border: "1px solid rgba(255,255,255,0.12)",
    backdropFilter: "blur(8px)",
    padding: 12,
    overflow: "hidden",
    display: "flex",
    flexDirection: "column"
  },
  head: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 },
  title: { fontWeight: 700 },
  close: {
    border: "1px solid rgba(255,255,255,0.16)",
    background: "rgba(255,255,255,0.06)",
    color: "rgba(255,255,255,0.86)",
    borderRadius: 10,
    padding: "6px 10px",
    cursor: "pointer"
  },
  list: { display: "grid", gap: 10, marginTop: 10, overflowY: "auto", flex: 1 },
  item: { padding: 10, borderRadius: 10, background: "rgba(0,0,0,0.15)", border: "1px solid rgba(255,255,255,0.06)" },
  itemTitle: { fontWeight: 700, marginBottom: 4 },
  itemDesc: { fontSize: 12, color: "rgba(255,255,255,0.72)", lineHeight: "18px" },
  footer: { marginTop: 10, fontSize: 11, color: "rgba(255,255,255,0.55)", lineHeight: "16px" },
  fab: {
    position: "absolute",
    right: 12,
    top: 12,
    pointerEvents: "auto",
    border: "1px solid rgba(255,255,255,0.16)",
    background: "rgba(255,255,255,0.06)",
    color: "rgba(255,255,255,0.90)",
    borderRadius: 999,
    padding: "10px 14px",
    cursor: "pointer"
  }
};
