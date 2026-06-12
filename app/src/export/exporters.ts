import type { Mesh } from "three";

/**
 * Binarny format STL (ISO/ASTM 52900).
 * 80B header + 4B count + N×50B triangles.
 */
export function exportSTL(mesh: Mesh, filename = "solid.stl") {
  const geo = mesh.geometry.clone();
  geo.applyMatrix4(mesh.matrixWorld);

  const pos = geo.attributes.position;
  const idx = geo.index;
  const triCount = idx ? idx.count / 3 : pos.count / 3;

  const buf = new ArrayBuffer(80 + 4 + triCount * 50);
  const view = new DataView(buf);

  const header = "Exported by Gesture-to-Geometry";
  for (let i = 0; i < 80; i++) {
    view.setUint8(i, i < header.length ? header.charCodeAt(i) : 0);
  }
  view.setUint32(80, triCount, true);

  let offset = 84;
  for (let t = 0; t < triCount; t++) {
    const a = idx ? idx.getX(t * 3) : t * 3;
    const b = idx ? idx.getX(t * 3 + 1) : t * 3 + 1;
    const c = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;

    // Normal (zerowy — programy do druku liczą go same z geometrii)
    view.setFloat32(offset, 0, true); offset += 4;
    view.setFloat32(offset, 0, true); offset += 4;
    view.setFloat32(offset, 0, true); offset += 4;

    for (const vi of [a, b, c]) {
      view.setFloat32(offset, pos.getX(vi), true); offset += 4;
      view.setFloat32(offset, pos.getY(vi), true); offset += 4;
      view.setFloat32(offset, pos.getZ(vi), true); offset += 4;
    }

    view.setUint16(offset, 0, true); offset += 2;
  }

  downloadBlob(new Blob([buf], { type: "application/octet-stream" }), filename);
}

/**
 * Format OBJ (Wavefront) — ASCII, powszechnie obsługiwany.
 */
export function exportOBJ(mesh: Mesh, filename = "solid.obj") {
  const geo = mesh.geometry.clone();
  geo.applyMatrix4(mesh.matrixWorld);

  const pos = geo.attributes.position;
  const idx = geo.index;

  const lines: string[] = ["# Exported by Gesture-to-Geometry", ""];

  for (let i = 0; i < pos.count; i++) {
    lines.push(`v ${pos.getX(i).toFixed(6)} ${pos.getY(i).toFixed(6)} ${pos.getZ(i).toFixed(6)}`);
  }
  lines.push("");

  const faceCount = idx ? idx.count / 3 : pos.count / 3;
  for (let t = 0; t < faceCount; t++) {
    const a = (idx ? idx.getX(t * 3) : t * 3) + 1;
    const b = (idx ? idx.getX(t * 3 + 1) : t * 3 + 1) + 1;
    const c = (idx ? idx.getX(t * 3 + 2) : t * 3 + 2) + 1;
    lines.push(`f ${a} ${b} ${c}`);
  }

  downloadBlob(new Blob([lines.join("\n")], { type: "text/plain" }), filename);
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
