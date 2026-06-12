# Gesture-to-Geometry — Roadmap urozmaiceń

Kolejność implementacji od najszybszego do najbardziej złożonego.

---

## 1. Eksport STL / OBJ / GLB
**Czas:** ~1 dzień  
**Cel:** Zamknięcie scenariusza demo z sekcji 14 spec: *"stworzenie walca → przekrój → eksport STL"*

- [x] Własny serializer STL binarny (ISO/ASTM 52900) — `app/src/export/exporters.ts`
- [x] Serializer OBJ ASCII — `app/src/export/exporters.ts`
- [x] Panel eksportu w prawym górnym rogu UI — `app/src/ui/ExportPanel.tsx`
- [x] Przyciski zablokowane dopóki nie ma bryły, feedback "Pobrano" po pobraniu
- [x] `SceneManager.getMesh()` — ekspozycja siatki do exportu

**Demo:** fizycznie przynosisz na obronę wydrukowany obiekt stworzony gestami.

---

## 2. Bryła obrotowa z toru dłoni + całki (ML curve fitting)
**Czas:** ~3–4 dni  
**Cel:** Unikalna funkcja której żaden podobny projekt nie ma — profil krzywej rysowany gestem w powietrzu staje się bryłą 3D.

### Gest i nagrywanie toru
- [x] Nowy tryb "Sketch" wyzwalany gestem V (dwa palce wyprostowane) — toggle
- [x] Nagrywanie sekwencji pozycji koniuszka palca wskazującego (landmark 8) gestem Point w trybie Sketch
- [x] Wizualizacja toru w czasie rzeczywistym (LatheOverlay — fioletowa linia + zielona oś obrotu)

### ML / curve fitting
- [x] Dopasowanie wielomianu (regresja wielomianowa, stopień 3) — `app/src/geometry/lathe.ts`
- [x] Wyświetlenie współczynników funkcji f(x) w HUD

### Bryła i całki
- [x] Generowanie `THREE.LatheGeometry` z dopasowanego profilu
- [x] Obliczenie objętości ze wzoru Pappusa: **V = π ∫ [f(x)]² dx** (metoda Simpsona)
- [x] Obliczenie pola powierzchni bocznej: **A = 2π ∫ f(x) √(1 + [f'(x)]²) dx**
- [x] Wyświetlenie wzoru z podstawionymi liczbami w HUD

**Demo:** rysujesz profil wazy w powietrzu → pojawia się realna bryła 3D z dokładnymi obliczeniami całkowymi.

---

## 3. CSG — operacje booleanowskie między bryłami
**Czas:** ~3–5 dni  
**Cel:** Union / difference / intersection dwóch brył kontrolowane gestami. Spektakularny efekt który komisja zapamiętuje.

### Wieloobiektowa scena (prerequisit)
- [x] Wiele obiektów w scenie (`Brush[]` — Brush extends THREE.Mesh, wymagane przez CSG evaluator)
- [x] Kliknięcie myszą na bryłę → raycasting → zmiana aktywnej bryły (highlight)
- [x] Wyróżnienie aktywnej bryły (emissive glow 0x4c1d95, intensywność 0.45)

### CSG
- [x] Zainstalować `three-mesh-bvh` + `three-bvh-csg`
- [x] Panel CSG (`app/src/ui/CSGPanel.tsx`) pojawia się gdy w scenie są ≥2 bryły
- [x] Operacje: Union (A∪B), Roznica (A−B), Czesc wspolna (A∩B) — `SceneManager.performCSG()`
- [x] Wynikowa siatka zastępuje oba źródłowe meshe, staje się jedynym obiektem w scenie
- [x] Eksport STL/OBJ działa na wynik CSG (getMesh() zwraca aktywny mesh)

---

## 4. Kalkulator całek oznaczonych (ML + gesty)
**Czas:** ~2 dni  
**Cel:** Osobny moduł — rysujesz palcem krzywą f(x) w powietrzu, program dopasowuje funkcję (ML) i oblicza całkę oznaczoną z wizualizacją obszaru.

### Nagrywanie i ML
- [ ] Nowy tryb "Integral" wyzwalany gestem (np. trzy palce wyprostowane)
- [ ] Nagrywanie toru koniuszka palca wskazującego (lewo → prawo, 1–3 sekundy)
- [ ] Regresja wielomianowa (least squares) do dopasowania f(x) do nagranych punktów
- [ ] Wybór stopnia wielomianu automatycznie (kryterium AIC/BIC — model selection)
- [ ] Wyświetlenie wzoru: `f(x) ≈ 0.3x² − 0.1x + 0.5`

### Całkowanie i wizualizacja
- [ ] Numeryczne całkowanie metodą Simpsona: **∫[a→b] f(x) dx**
- [ ] Granice a, b wyznaczane automatycznie z zakresu nagranego ruchu
- [ ] Nakładka 2D: narysowana krzywa + zacieniowany obszar pod krzywą
- [ ] Wynik liczbowy w HUD: `∫₋₁³ f(x) dx = 12.74`
- [ ] Opcjonalnie: wyświetlenie kroku obliczeń (podział Simpsona) dla celów edukacyjnych

**Demo:** rysujesz w powietrzu dowolną krzywą → program mówi jaka to funkcja i ile wynosi całka oznaczona.  
**Akademicko:** realna regresja (ML) + analiza numeryczna w jednym module.

---

## 5. Dynamiczne gesty czasowe (DTW / sekwencje)
**Czas:** ~2–3 dni  
**Cel:** Rozpoznawanie gestów które trwają w czasie (ruch, nie poza). Uczciwy komponent ML — klasyfikacja sekwencji landmarków.

- [ ] Nagrywanie okna czasowego (~1s) sekwencji pozycji dłoni
- [ ] Implementacja DTW (Dynamic Time Warping) jako miary podobieństwa sekwencji
- [ ] Zbiór wzorcowych gestów dynamicznych (templates):
  - [ ] Koło w powietrzu → stwórz torus / sferę
  - [ ] Swipe w prawo → undo
  - [ ] Swipe w lewo → redo
  - [ ] Machanie (wave) → reset sceny
- [ ] Wizualizacja rozpoznanego gestu w HUD (pewność dopasowania %)

---

## 6. Siatka rozwijalna z animacją (net unfolding)
**Czas:** ~4–5 dni  
**Cel:** Bryła "rozkłada się" w animacji na swoją siatkę 2D. Geometrycznie rygorystyczne i edukacyjne — serce idei "Gesture-to-Geometry".

- [ ] Algorytm rozwijania siatki dla: sześcianu, graniastosłupa, walca, stożka, ostrosłupa
- [ ] Animacja składania/rozkładania (GSAP lub własny interpolator kątów ścian)
- [ ] Gest wyzwalający: MeasureL przytrzymany na zaznaczonej bryle
- [ ] Wyświetlenie siatki 2D obok bryły 3D
- [ ] Opcjonalnie: eksport siatki jako SVG do wydruku i fizycznego złożenia

---

## Ogólne zadania towarzyszące

- [ ] Uzupełnić testy jednostkowe geometrii (`tests/unit/`) — Vitest
- [ ] Dodać benchmarki FPS/latency (`benchmarks/`) dla różnych kombinacji funkcji
- [ ] Przeprowadzić badanie użytkowników (SUS) — min. 5 osób
- [ ] Uzupełnić pracę pisemną o rozdziały dot. powyższych funkcji
