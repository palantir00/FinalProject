import * as THREE from "three";
import { ADDITION, SUBTRACTION, INTERSECTION, Evaluator, Brush } from "three-bvh-csg";
import type { GestureFrame } from "../gestures/types";
import { cubeMetrics, sphereMetrics, cylinderMetrics, coneMetrics, pyramidMetrics, boxMetrics } from "../geometry/analytics";
import type { SolidType } from "../geometry/types";
import type { SolidParams } from "../geometry/params";

export type SceneTelemetry = { fps: number; lastLatencyMs: number };
export type CSGOperation = "union" | "subtraction" | "intersection";

export class SceneManager {
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private root: THREE.Group;
  private slicePlane: THREE.Mesh;

  // Multi-mesh support — Brush extends THREE.Mesh and is required by the CSG evaluator
  private meshes: Brush[] = [];
  private solidParamsList: (SolidParams | null)[] = [];
  private activeIdx = -1;
  private raycaster = new THREE.Raycaster();
  private evaluator = new Evaluator();

  private lastT = performance.now();
  private fps = 0;
  private frameCount = 0;
  private fpsWindowStart = this.lastT;
  private lastLatencyMs = 0;
  private raf = 0;
  private pendingGesture: GestureFrame | null = null;
  private lastCvFrameT: number | null = null;
  private initialCameraZ = 2;

  private onMeshCountChange?: (count: number) => void;

  constructor(private canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.scene.background = null;

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
    this.camera.position.set(0, 0, this.initialCameraZ);
    this.camera.lookAt(0, 0, 0);

    this.root = new THREE.Group();
    this.scene.add(this.root);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(2, 3, 2);
    this.scene.add(dir);

    // Default starting solid (Brush extends THREE.Mesh — required for CSG evaluator)
    const geo = new THREE.BoxGeometry(0.35, 0.35, 0.35);
    const mat = new THREE.MeshStandardMaterial({ color: 0x7c3aed, roughness: 0.35, metalness: 0.15 });
    const cube = new Brush(geo, mat);
    this.root.add(cube);
    this.meshes.push(cube);
    this.solidParamsList.push(null);
    this.activeIdx = 0;

    // Slice plane
    const pGeo = new THREE.PlaneGeometry(1.2, 1.2);
    const pMat = new THREE.MeshBasicMaterial({ color: 0x22c55e, transparent: true, opacity: 0.15, side: THREE.DoubleSide });
    this.slicePlane = new THREE.Mesh(pGeo, pMat);
    this.slicePlane.rotation.x = Math.PI / 2;
    this.slicePlane.position.y = 0;
    this.slicePlane.visible = false;
    this.scene.add(this.slicePlane);

    // Click on canvas = select mesh under cursor
    this.canvas.addEventListener("click", this.handleCanvasClick);

    window.addEventListener("resize", this.handleResize);
    this.handleResize();
  }

  /** Register callback to be notified when the mesh count changes (for React state sync). */
  setOnMeshCountChange(cb: (count: number) => void) {
    this.onMeshCountChange = cb;
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    window.removeEventListener("resize", this.handleResize);
    this.canvas.removeEventListener("click", this.handleCanvasClick);
    for (const m of this.meshes) this.disposeMesh(m);
    this.slicePlane.geometry.dispose();
    (this.slicePlane.material as THREE.Material).dispose();
    this.renderer.dispose();
  }

  setGestureFrame(frame: GestureFrame) {
    this.pendingGesture = frame;
    this.lastCvFrameT = frame.tMs;
  }

  start() {
    this.lastT = performance.now();
    const loop = () => {
      this.raf = requestAnimationFrame(loop);
      const t = performance.now();
      const dt = (t - this.lastT) / 1000;
      this.lastT = t;
      if (this.pendingGesture) {
        this.applyGesture(this.pendingGesture, dt);
        if (this.lastCvFrameT != null) this.lastLatencyMs = t - this.lastCvFrameT;
        this.pendingGesture = null;
      }
      this.renderer.render(this.scene, this.camera);
      this.updateFps(t);
    };
    loop();
  }

  getTelemetry(): SceneTelemetry { return { fps: this.fps, lastLatencyMs: this.lastLatencyMs }; }

  getMesh() { return this.activeIdx >= 0 ? this.meshes[this.activeIdx] : null; }

  getMeshCount() { return this.meshes.length; }

  getActiveIdx() { return this.activeIdx; }

  getCubeMetrics() {
    const params = this.solidParamsList[this.activeIdx];
    if (!params) return cubeMetrics(1.0);
    switch (params.type) {
      case "cube":     return cubeMetrics(params.side);
      case "sphere":   return sphereMetrics(params.radius);
      case "cylinder": return cylinderMetrics(params.radius, params.height);
      case "cone":     return coneMetrics(params.radius, params.height);
      case "pyramid":  return pyramidMetrics(params.baseSide, params.height);
      case "box":      return boxMetrics(params.width, params.height, params.depth);
      default:         return cubeMetrics(1.0);
    }
  }

  /** Creates a new solid and adds it to the scene. The new solid becomes active. */
  createSolid(type: SolidType, params: SolidParams) {
    let geometry: THREE.BufferGeometry;
    switch (type) {
      case "cube":     geometry = new THREE.BoxGeometry(0.35, 0.35, 0.35); break;
      case "sphere":   geometry = new THREE.SphereGeometry(0.25, 32, 32); break;
      case "cylinder": geometry = new THREE.CylinderGeometry(0.2, 0.2, 0.4, 32); break;
      case "cone":     geometry = new THREE.ConeGeometry(0.2, 0.4, 32); break;
      case "pyramid":  geometry = new THREE.ConeGeometry(0.2, 0.4, 4); break;
      case "box":      geometry = new THREE.BoxGeometry(0.4, 0.3, 0.5); break;
      default:         geometry = new THREE.BoxGeometry(0.35, 0.35, 0.35);
    }
    const mat = new THREE.MeshStandardMaterial({ color: 0x7c3aed, roughness: 0.35, metalness: 0.15 });
    const mesh = new Brush(geometry, mat);
    this.root.add(mesh);
    this.meshes.push(mesh);
    this.solidParamsList.push(params);
    this.setActive(this.meshes.length - 1);
    this.resetActivePosition();
    this.camera.position.z = this.initialCameraZ;
    this.onMeshCountChange?.(this.meshes.length);
  }

  /** Creates a lathe solid from a profile and adds it to the scene. */
  createLathe(profilePoints: { x: number; y: number }[]) {
    const points = profilePoints.map(p => new THREE.Vector2(p.x, p.y));
    const geo = new THREE.LatheGeometry(points, 48);
    const mat = new THREE.MeshStandardMaterial({ color: 0x7c3aed, roughness: 0.35, metalness: 0.15, side: THREE.DoubleSide });
    const mesh = new Brush(geo, mat);
    this.root.add(mesh);
    this.meshes.push(mesh);
    this.solidParamsList.push(null);
    this.setActive(this.meshes.length - 1);
    this.resetActivePosition();
    this.camera.position.z = this.initialCameraZ;
    this.onMeshCountChange?.(this.meshes.length);
  }

  /**
   * Performs a CSG boolean operation on meshes[0] and meshes[1].
   * The result replaces both source meshes and becomes the single active mesh.
   */
  performCSG(op: CSGOperation) {
    if (this.meshes.length < 2) return;
    const a = this.meshes[0];
    const b = this.meshes[1];

    // Ensure world matrices are current
    a.updateMatrixWorld(true);
    b.updateMatrixWorld(true);

    const opConst = op === "union" ? ADDITION : op === "subtraction" ? SUBTRACTION : INTERSECTION;

    let result: Brush;
    try {
      result = this.evaluator.evaluate(a, b, opConst);
    } catch {
      return; // degenerate geometry — skip
    }

    // Replace existing meshes with CSG result
    for (const m of this.meshes) {
      this.root.remove(m);
      this.disposeMesh(m);
    }
    this.meshes = [];
    this.solidParamsList = [];

    // Style the result
    const mat = new THREE.MeshStandardMaterial({ color: 0x7c3aed, roughness: 0.35, metalness: 0.15 });
    result.material = mat;
    this.root.add(result);
    this.meshes.push(result);
    this.solidParamsList.push(null);
    this.activeIdx = 0;
    this.onMeshCountChange?.(this.meshes.length);
  }

  resetActivePosition() {
    const m = this.meshes[this.activeIdx];
    if (!m) return;
    m.position.set(0, 0, 0);
    m.rotation.set(0, 0, 0);
    m.scale.set(1, 1, 1);
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  private setActive(idx: number) {
    // Clear highlight on all meshes
    for (const m of this.meshes) this.applyHighlight(m, false);
    this.activeIdx = idx;
    // Highlight active only when there are multiple meshes
    if (this.meshes.length > 1 && idx >= 0 && idx < this.meshes.length) {
      this.applyHighlight(this.meshes[idx], true);
    }
  }

  private applyHighlight(mesh: THREE.Mesh, on: boolean) {
    const mat = mesh.material as THREE.MeshStandardMaterial;
    if (!mat.emissive) return;
    mat.emissive.set(on ? 0x4c1d95 : 0x000000);
    mat.emissiveIntensity = on ? 0.45 : 0;
  }

  private updateAllHighlights() {
    for (let i = 0; i < this.meshes.length; i++) {
      this.applyHighlight(this.meshes[i], this.meshes.length > 1 && i === this.activeIdx);
    }
  }

  private handleCanvasClick = (e: MouseEvent) => {
    if (this.meshes.length < 2) return; // nothing to select when only one mesh
    const rect = this.canvas.getBoundingClientRect();
    const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), this.camera);
    const hits = this.raycaster.intersectObjects(this.meshes, false);
    if (hits.length === 0) return;
    const idx = this.meshes.indexOf(hits[0].object as Brush);
    if (idx >= 0 && idx !== this.activeIdx) {
      this.setActive(idx);
    }
  };

  private disposeMesh(m: THREE.Mesh) {
    m.geometry.dispose();
    if (Array.isArray(m.material)) {
      m.material.forEach(mat => mat.dispose());
    } else {
      (m.material as THREE.Material).dispose();
    }
  }

  private applyGesture(g: GestureFrame, dt: number) {
    const mesh = this.activeIdx >= 0 ? this.meshes[this.activeIdx] : null;

    if (g.gesture === "Move" && g.moveDelta && mesh) {
      const speed = 1.2;
      mesh.position.x += g.moveDelta.x * speed;
      mesh.position.y -= g.moveDelta.y * speed;
    }

    if (g.gesture === "Rotate" && g.rotateDelta && mesh) {
      mesh.rotation.y += g.rotateDelta.x;
      mesh.rotation.x -= g.rotateDelta.y;
    }

    if (g.gesture === "Scale" && typeof g.scaleFactor === "number") {
      const newZ = THREE.MathUtils.clamp(this.camera.position.z / g.scaleFactor, 0.5, 5.0);
      this.camera.position.z = newZ;
    }

    if (g.gesture === "Slice") {
      this.slicePlane.visible = true;
      if (typeof g.sliceDelta === "number") {
        this.slicePlane.position.y = THREE.MathUtils.clamp(this.slicePlane.position.y - g.sliceDelta, -0.6, 0.6);
      }
    } else {
      this.slicePlane.visible = false;
    }

    void dt;
  }

  private updateFps(tMs: number) {
    this.frameCount += 1;
    const span = tMs - this.fpsWindowStart;
    if (span >= 500) {
      this.fps = Math.round((this.frameCount / span) * 1000);
      this.fpsWindowStart = tMs;
      this.frameCount = 0;
    }
  }

  private handleResize = () => {
    const { clientWidth: w, clientHeight: h } = this.canvas;
    if (w === 0 || h === 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  };
}
