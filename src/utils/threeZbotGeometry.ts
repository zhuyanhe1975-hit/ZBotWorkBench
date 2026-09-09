import * as THREE from 'three';

/**
 * Creates accurate procedural Three.js BufferGeometries for Zbot Semi-Modules A and B
 * Radius = 0.035m (diameter 70mm), Module Length = 0.106m (106mm)
 * Cut plane at z = 0.053 + y (inclined 45 degrees, normal [0, -1/sqrt2, 1/sqrt2])
 */
export function createZbotGeometries(radius = 0.035, segments = 32) {
  const L = 0.106;
  const H = 0.053;
  const s = 1 / Math.SQRT2;

  // Geometry A (Base half from z = 0 to cut plane)
  const geomA = new THREE.BufferGeometry();
  const posA: number[] = [];
  const normA: number[] = [];
  const uvsA: number[] = [];

  // Bottom cap (z = 0)
  for (let i = 0; i < segments; i++) {
    const a1 = (i / segments) * Math.PI * 2;
    const a2 = ((i + 1) / segments) * Math.PI * 2;
    // Triangles centered at (0, 0, 0)
    posA.push(0, 0, 0);
    posA.push(radius * Math.cos(a2), radius * Math.sin(a2), 0);
    posA.push(radius * Math.cos(a1), radius * Math.sin(a1), 0);
    for (let k = 0; k < 3; k++) normA.push(0, 0, -1);
    uvsA.push(0.5, 0.5, 0.5 + 0.5 * Math.cos(a2), 0.5 + 0.5 * Math.sin(a2), 0.5 + 0.5 * Math.cos(a1), 0.5 + 0.5 * Math.sin(a1));
  }

  // Cylindrical side for Part A
  for (let i = 0; i < segments; i++) {
    const a1 = (i / segments) * Math.PI * 2;
    const a2 = ((i + 1) / segments) * Math.PI * 2;
    const x1 = radius * Math.cos(a1);
    const y1 = radius * Math.sin(a1);
    const z1 = H + y1;

    const x2 = radius * Math.cos(a2);
    const y2 = radius * Math.sin(a2);
    const z2 = H + y2;

    // Bottom points at z=0, Top points at z=H+y
    // Quad (x1, y1, 0) -> (x2, y2, 0) -> (x2, y2, z2) -> (x1, y1, z1)
    posA.push(x1, y1, 0, x2, y2, 0, x2, y2, z2);
    normA.push(Math.cos(a1), Math.sin(a1), 0, Math.cos(a2), Math.sin(a2), 0, Math.cos(a2), Math.sin(a2), 0);
    uvsA.push(i / segments, 0, (i + 1) / segments, 0, (i + 1) / segments, 1);

    posA.push(x1, y1, 0, x2, y2, z2, x1, y1, z1);
    normA.push(Math.cos(a1), Math.sin(a1), 0, Math.cos(a2), Math.sin(a2), 0, Math.cos(a1), Math.sin(a1), 0);
    uvsA.push(i / segments, 0, (i + 1) / segments, 1, i / segments, 1);
  }

  // Angled Cut cap for Part A (normal [0, -s, s])
  for (let i = 0; i < segments; i++) {
    const a1 = (i / segments) * Math.PI * 2;
    const a2 = ((i + 1) / segments) * Math.PI * 2;
    const x1 = radius * Math.cos(a1);
    const y1 = radius * Math.sin(a1);
    const z1 = H + y1;

    const x2 = radius * Math.cos(a2);
    const y2 = radius * Math.sin(a2);
    const z2 = H + y2;

    posA.push(0, 0, H, x1, y1, z1, x2, y2, z2);
    for (let k = 0; k < 3; k++) normA.push(0, -s, s);
    uvsA.push(0.5, 0.5, 0.5 + 0.5 * Math.cos(a1), 0.5 + 0.5 * Math.sin(a1), 0.5 + 0.5 * Math.cos(a2), 0.5 + 0.5 * Math.sin(a2));
  }

  geomA.setAttribute('position', new THREE.Float32BufferAttribute(posA, 3));
  geomA.setAttribute('normal', new THREE.Float32BufferAttribute(normA, 3));
  geomA.setAttribute('uv', new THREE.Float32BufferAttribute(uvsA, 2));

  // Geometry B (Rotating half from cut plane to z = L)
  const geomB = new THREE.BufferGeometry();
  const posB: number[] = [];
  const normB: number[] = [];
  const uvsB: number[] = [];

  // Cut cap for Part B (normal [0, s, -s])
  for (let i = 0; i < segments; i++) {
    const a1 = (i / segments) * Math.PI * 2;
    const a2 = ((i + 1) / segments) * Math.PI * 2;
    const x1 = radius * Math.cos(a1);
    const y1 = radius * Math.sin(a1);
    const z1 = H + y1;

    const x2 = radius * Math.cos(a2);
    const y2 = radius * Math.sin(a2);
    const z2 = H + y2;

    posB.push(0, 0, H, x2, y2, z2, x1, y1, z1);
    for (let k = 0; k < 3; k++) normB.push(0, s, -s);
    uvsB.push(0.5, 0.5, 0.5 + 0.5 * Math.cos(a2), 0.5 + 0.5 * Math.sin(a2), 0.5 + 0.5 * Math.cos(a1), 0.5 + 0.5 * Math.sin(a1));
  }

  // Cylindrical side for Part B
  for (let i = 0; i < segments; i++) {
    const a1 = (i / segments) * Math.PI * 2;
    const a2 = ((i + 1) / segments) * Math.PI * 2;
    const x1 = radius * Math.cos(a1);
    const y1 = radius * Math.sin(a1);
    const z1 = H + y1;

    const x2 = radius * Math.cos(a2);
    const y2 = radius * Math.sin(a2);
    const z2 = H + y2;

    // Bottom at z1/z2, Top at z=L
    posB.push(x1, y1, z1, x2, y2, z2, x2, y2, L);
    normB.push(Math.cos(a1), Math.sin(a1), 0, Math.cos(a2), Math.sin(a2), 0, Math.cos(a2), Math.sin(a2), 0);
    uvsB.push(i / segments, 0, (i + 1) / segments, 0, (i + 1) / segments, 1);

    posB.push(x1, y1, z1, x2, y2, L, x1, y1, L);
    normB.push(Math.cos(a1), Math.sin(a1), 0, Math.cos(a2), Math.sin(a2), 0, Math.cos(a1), Math.sin(a1), 0);
    uvsB.push(i / segments, 0, (i + 1) / segments, 1, i / segments, 1);
  }

  // Top cap for Part B (z = L)
  for (let i = 0; i < segments; i++) {
    const a1 = (i / segments) * Math.PI * 2;
    const a2 = ((i + 1) / segments) * Math.PI * 2;
    posB.push(0, 0, L, radius * Math.cos(a1), radius * Math.sin(a1), L, radius * Math.cos(a2), radius * Math.sin(a2), L);
    for (let k = 0; k < 3; k++) normB.push(0, 0, 1);
    uvsB.push(0.5, 0.5, 0.5 + 0.5 * Math.cos(a1), 0.5 + 0.5 * Math.sin(a1), 0.5 + 0.5 * Math.cos(a2), 0.5 + 0.5 * Math.sin(a2));
  }

  geomB.setAttribute('position', new THREE.Float32BufferAttribute(posB, 3));
  geomB.setAttribute('normal', new THREE.Float32BufferAttribute(normB, 3));
  geomB.setAttribute('uv', new THREE.Float32BufferAttribute(uvsB, 2));

  return { geomA, geomB };
}
