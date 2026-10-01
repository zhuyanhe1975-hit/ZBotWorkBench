import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { generateZbotOBJ } from './zbotMeshGenerator';

/** Parse all OBJ objects and reject invalid input before replacing either mesh. */
export function parseObjGeometry(text: string): THREE.BufferGeometry {
  if (/<(?:!doctype|html|body)\b/i.test(text)) throw new Error('文件是 HTML 页面，不是 OBJ 网格');
  let vertices = 0;
  const faces: { indices: number[]; available: number }[] = [];
  for (const line of text.split(/\r?\n/)) {
    const fields = line.split('#')[0].trim().split(/\s+/);
    if (fields[0] === 'v') {
      if (fields.length < 4 || !fields.slice(1, 4).every(v => Number.isFinite(Number(v)))) throw new Error('OBJ 顶点坐标无效');
      vertices++;
    }
    if (fields[0] === 'f') {
      const indices = fields.slice(1).map(v => Number(v.split('/')[0]));
      if (indices.length < 3 || indices.some(v => !Number.isInteger(v) || v === 0)) throw new Error('OBJ 面索引无效');
      faces.push({ indices, available: vertices });
    }
  }
  if (vertices < 3 || !faces.length || faces.some(f => f.indices.some(v => v > vertices || v < -f.available))) throw new Error('OBJ 必须包含有效的顶点和三角面');
  const parsed = new OBJLoader().parse(text);
  const positions: number[] = [];
  parsed.updateMatrixWorld(true);
  parsed.traverse(child => {
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh) return;
    const geometry = mesh.geometry.clone().applyMatrix4(mesh.matrixWorld);
    const attribute = geometry.getAttribute('position');
    const index = geometry.getIndex();
    for (let i = 0; i < (index?.count ?? attribute.count); i++) {
      const j = index ? index.getX(i) : i;
      positions.push(attribute.getX(j), attribute.getY(j), attribute.getZ(j));
    }
    geometry.dispose();
    mesh.geometry.dispose();
    (Array.isArray(mesh.material) ? mesh.material : [mesh.material]).forEach(m => m.dispose());
  });
  if (!positions.length || positions.some(v => !Number.isFinite(v))) throw new Error('OBJ 网格解析失败');
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  return geometry;
}

export class MeshManager {
  private texts = { ma: generateZbotOBJ('ma'), mb: generateZbotOBJ('mb') };
  private cad = { ma: false, mb: false };
  private geometries: Partial<Record<'ma' | 'mb', THREE.BufferGeometry>> = {};
  private listeners = new Set<() => void>();

  public subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => { this.listeners.delete(fn); };
  }
  private replace(part: 'ma' | 'mb', text: string, cad: boolean): void {
    const geometry = parseObjGeometry(text);
    this.geometries[part]?.dispose();
    this.geometries[part] = geometry;
    this.texts[part] = text;
    this.cad[part] = cad;
  }
  public async init(): Promise<{ maText: string; mbText: string }> {
    await Promise.all((['ma', 'mb'] as const).map(async part => {
      try {
        const response = await fetch(`${import.meta.env?.BASE_URL ?? '/'}assets/${part}.obj`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        this.replace(part, await response.text(), true);
      } catch {
        this.replace(part, generateZbotOBJ(part), false);
      }
    }));
    this.listeners.forEach(fn => fn());
    return { maText: this.texts.ma, mbText: this.texts.mb };
  }
  public setCustomMesh(part: 'ma' | 'mb', text: string): void {
    this.replace(part, text, true);
    this.listeners.forEach(fn => fn());
  }
  public resetToDefault(part: 'ma' | 'mb'): void {
    this.replace(part, generateZbotOBJ(part), false);
    this.listeners.forEach(fn => fn());
  }
  public getMeshAText(): string { return this.texts.ma; }
  public getMeshBText(): string { return this.texts.mb; }
  public hasCadMeshA(): boolean { return this.cad.ma; }
  public hasCadMeshB(): boolean { return this.cad.mb; }
  public getGeometries(): { geomA: THREE.BufferGeometry; geomB: THREE.BufferGeometry } {
    this.geometries.ma ??= parseObjGeometry(this.texts.ma);
    this.geometries.mb ??= parseObjGeometry(this.texts.mb);
    return { geomA: this.geometries.ma, geomB: this.geometries.mb };
  }
}
export const meshManager = new MeshManager();
