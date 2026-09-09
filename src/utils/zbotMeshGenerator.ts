/**
 * Generates procedural OBJ strings for Zbot semi-modules A and B
 * Radius R = 0.035m, Length = 0.106m, Cut plane at z = 0.053 + y (axis 0 -1 1)
 */
export function generateZbotOBJ(part: 'ma' | 'mb', segments = 24, radius = 0.035): string {
  const lines: string[] = [];
  lines.push(`# Zbot semi-module ${part}`);
  lines.push(`o ${part}`);

  const vertices: [number, number, number][] = [];
  const normals: [number, number, number][] = [];

  const L = 0.106;
  const H = 0.053; // cut center height

  if (part === 'ma') {
    // Center of bottom disk (index 1)
    vertices.push([0, 0, 0]);
    normals.push([0, 0, -1]);

    // Center of cut ellipse (index 2)
    vertices.push([0, 0, H]);
    // Normal of plane: -y + z - H = 0 => normal is (0, -1/sqrt(2), 1/sqrt(2))
    const s = 1 / Math.SQRT2;
    normals.push([0, -s, s]);

    // Bottom ring vertices (indices 3 .. 2+segments)
    for (let i = 0; i < segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const x = radius * Math.cos(angle);
      const y = radius * Math.sin(angle);
      vertices.push([x, y, 0]);
      // side normal
      normals.push([Math.cos(angle), Math.sin(angle), 0]);
    }

    // Cut ring vertices (indices 3+segments .. 2+2*segments)
    for (let i = 0; i < segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const x = radius * Math.cos(angle);
      const y = radius * Math.sin(angle);
      const z = H + y;
      vertices.push([x, y, z]);
      normals.push([Math.cos(angle), Math.sin(angle), 0]);
    }

    // Emit vertices
    for (const v of vertices) {
      lines.push(`v ${v[0].toFixed(6)} ${v[1].toFixed(6)} ${v[2].toFixed(6)}`);
    }
    for (const n of normals) {
      lines.push(`vn ${n[0].toFixed(4)} ${n[1].toFixed(4)} ${n[2].toFixed(4)}`);
    }

    // Bottom disk faces
    for (let i = 0; i < segments; i++) {
      const next = (i + 1) % segments;
      const v1 = 1;
      const v2 = 3 + next;
      const v3 = 3 + i;
      lines.push(`f ${v1} ${v2} ${v3}`);
    }

    // Side cylinder faces
    for (let i = 0; i < segments; i++) {
      const next = (i + 1) % segments;
      const b1 = 3 + i;
      const b2 = 3 + next;
      const t1 = 3 + segments + i;
      const t2 = 3 + segments + next;
      lines.push(`f ${b1} ${b2} ${t2}`);
      lines.push(`f ${b1} ${t2} ${t1}`);
    }

    // Cut face (ellipse)
    for (let i = 0; i < segments; i++) {
      const next = (i + 1) % segments;
      const vCenter = 2;
      const v1 = 3 + segments + i;
      const v2 = 3 + segments + next;
      lines.push(`f ${vCenter} ${v1} ${v2}`);
    }

  } else {
    // Part mb: from cut face to top disk z = L
    // Center of cut ellipse (index 1)
    vertices.push([0, 0, H]);
    // Normal of cut plane pointing inward/downward for mb: (0, 1/sqrt(2), -1/sqrt(2))
    const s = 1 / Math.SQRT2;
    normals.push([0, s, -s]);

    // Center of top disk (index 2)
    vertices.push([0, 0, L]);
    normals.push([0, 0, 1]);

    // Cut ring vertices (indices 3 .. 2+segments)
    for (let i = 0; i < segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const x = radius * Math.cos(angle);
      const y = radius * Math.sin(angle);
      const z = H + y;
      vertices.push([x, y, z]);
      normals.push([Math.cos(angle), Math.sin(angle), 0]);
    }

    // Top ring vertices (indices 3+segments .. 2+2*segments)
    for (let i = 0; i < segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const x = radius * Math.cos(angle);
      const y = radius * Math.sin(angle);
      vertices.push([x, y, L]);
      normals.push([Math.cos(angle), Math.sin(angle), 0]);
    }

    // Emit vertices
    for (const v of vertices) {
      lines.push(`v ${v[0].toFixed(6)} ${v[1].toFixed(6)} ${v[2].toFixed(6)}`);
    }
    for (const n of normals) {
      lines.push(`vn ${n[0].toFixed(4)} ${n[1].toFixed(4)} ${n[2].toFixed(4)}`);
    }

    // Cut disk faces
    for (let i = 0; i < segments; i++) {
      const next = (i + 1) % segments;
      const vCenter = 1;
      const v1 = 3 + next;
      const v2 = 3 + i;
      lines.push(`f ${vCenter} ${v1} ${v2}`);
    }

    // Side cylinder faces
    for (let i = 0; i < segments; i++) {
      const next = (i + 1) % segments;
      const b1 = 3 + i;
      const b2 = 3 + next;
      const t1 = 3 + segments + i;
      const t2 = 3 + segments + next;
      lines.push(`f ${b1} ${b2} ${t2}`);
      lines.push(`f ${b1} ${t2} ${t1}`);
    }

    // Top disk faces
    for (let i = 0; i < segments; i++) {
      const next = (i + 1) % segments;
      const vCenter = 2;
      const v1 = 3 + segments + i;
      const v2 = 3 + segments + next;
      lines.push(`f ${vCenter} ${v1} ${v2}`);
    }
  }

  return lines.join('\n');
}
