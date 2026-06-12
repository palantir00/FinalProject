# Gesture-to-Geometry (Gesture-to-Geometry)

Projekt „Gesture-to-Geometry”: tworzenie i modyfikacja proceduralnych brył 3D gestami dłoni (kamera) + natychmiastowa analiza geometryczna (pole, objętość) oraz tryby Measure/Slice.

Repozytorium jest zainicjowane zgodnie ze specyfikacją z pracy inżynierskiej:
- **Struktura**: patrz `docs/SPEC_EXTRACTED.txt`, sekcja „5) Repozytorium – struktura”.
- **Protokół gestów**: patrz `docs/SPEC_EXTRACTED.txt`, sekcja „9) Protokół gestów”.

## Uruchomienie (frontend)

Frontend (Vite + React + TypeScript) znajduje się w folderze `app/`.

1) Zainstaluj Node.js (LTS) + npm  
2) W katalogu `app/`:

```bash
npm install
npm run dev
```

## Szybka nawigacja po kodzie

- `app/src/cv/HandTracker.ts`: pipeline wideo + MediaPipe Hands (landmarki)
- `app/src/cv/filters/oneEuro.ts`: One-Euro Filter (stabilizacja sygnału)
- `app/src/gestures/*`: mapping landmarków → gesty + FSM trybów
- `app/src/three/SceneManager.ts`: Three.js scena + pętla render
- `app/src/geometry/analytics.ts`: pole/objętość (sześcian, kula, walec)
- `app/src/ui/*`: HUD (autolabels) + tutorial overlay (MVP)
