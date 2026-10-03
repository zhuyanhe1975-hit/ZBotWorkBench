import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { MujocoEngine } from '../mujoco/MujocoEngine';
import { parseTrainingBundle } from '../training/bundle';
import type { TrainingLiveFrame, TrainingReplayBundle } from '../training/types';
import type { TrainingClient } from '../training/client';
import { compiledMeshGeometry } from '../utils/compiledMeshGeometry';

const MODEL_URL = `${import.meta.env?.BASE_URL ?? '/'}rl/training/walking-reference.json`;
const MAX_ENVIRONMENTS = 9;
export const trainingDisplayPosition = (pose: [number, number, number], origin: [number, number, number], tileX: number, tileY: number): [number, number, number] =>
  [pose[0] - origin[0] + tileX, pose[1] - origin[1] + tileY, pose[2] - origin[2]];

export function TrainingLiveView({ frame, jobId, client, native = false }: { frame?: TrainingLiveFrame; jobId: string; client: TrainingClient; native?: boolean }) {
  const container = useRef<HTMLDivElement>(null);
  const frameRef = useRef<TrainingLiveFrame | undefined>(frame);
  const updateRef = useRef<(next: TrainingLiveFrame) => void>(() => {});
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [displayFrame, setDisplayFrame] = useState(frame);
  frameRef.current = frame ?? frameRef.current;

  useEffect(() => {
    if (!container.current) return;
    const host = container.current;
    let disposed = false;
    const engine = new MujocoEngine();
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#0b1220');
    scene.fog = new THREE.FogExp2('#0b1220', .055);
    const camera = new THREE.PerspectiveCamera(42, 1, .01, 50);
    camera.up.set(0, 0, 1);
    camera.position.set(4, -5, 3.4);
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.shadowMap.autoUpdate = false;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.domElement.dataset.trainingLiveView = 'zbot-mesh-grid';
    renderer.domElement.style.cssText = 'display:block;width:100%;height:100%';
    // Keep React's loading/error overlay under React ownership.
    host.insertBefore(renderer.domElement, host.firstChild);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
    // Live poses are rebased around the environment origin and tiled around
    // x=0; targeting x=2 leaves the robot outside the initial view.
    controls.target.set(0, 0, .22);
    controls.minDistance = 1.2;
    controls.maxDistance = 8;
    controls.maxPolarAngle = Math.PI / 2 - .02;
    // Apply the initial target immediately; otherwise the camera keeps its
    // default -Z orientation until the user moves the orbit controls.
    controls.update();
    const render = () => { if (!disposed) renderer.render(scene, camera); };
    controls.addEventListener('change', render);

    scene.add(new THREE.HemisphereLight(0xdbeafe, 0x172033, 2.1));
    const sun = new THREE.DirectionalLight(0xffffff, 2.4);
    sun.position.set(3, -4, 7);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.left = sun.shadow.camera.bottom = -3;
    sun.shadow.camera.right = sun.shadow.camera.top = 3;
    sun.shadow.camera.far = 15;
    sun.shadow.bias = -.0003;
    scene.add(sun, sun.target);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ color: 0x172033, roughness: .82, metalness: .08 }));
    floor.receiveShadow = true;
    scene.add(floor);
    const grid = new THREE.GridHelper(40, 80, 0x0ea5e9, 0x334155);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = .001;
    scene.add(grid);

    const geometries = new Set<THREE.BufferGeometry>();
    const materials = new Set<THREE.Material>();
    const instances = new Map<number, THREE.InstancedMesh>();
    const bodyMarkerGeometry = new THREE.SphereGeometry(.055, 12, 8);
    const bodyMarkerMaterial = new THREE.MeshStandardMaterial({ color: 0x38bdf8, roughness: .45, metalness: .15 });
    const bodyMarkers = new THREE.InstancedMesh(bodyMarkerGeometry, bodyMarkerMaterial, MAX_ENVIRONMENTS * 32);
    bodyMarkers.frustumCulled = false;
    bodyMarkers.count = 0;
    scene.add(bodyMarkers);
    let model: any = null;
    const nameAt = (address: number) => {
      let value = '';
      for (let index = address; index >= 0 && index < model.names.length && model.names[index]; index++) value += String.fromCharCode(model.names[index]);
      return value;
    };
    const resize = () => {
      const width = Math.max(1, host.clientWidth), height = Math.max(1, host.clientHeight);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
      render();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    const applyFrame = (next: TrainingLiveFrame) => {
      if (!model || disposed) return;
      const count = Math.min(MAX_ENVIRONMENTS, next.environments.length);
      const bodyIndex = new Map(next.bodyNames.map((name, index) => [name, index]));
      const base = Math.max(0, next.bodyNames.findIndex(name => name === 'base' || name.endsWith('/base')));
      const columns = count <= 1 ? 1 : count <= 4 ? 2 : 3;
      const rows = Math.ceil(count / columns);
      const bodyMatrix = new THREE.Matrix4(), geomMatrix = new THREE.Matrix4(), worldMatrix = new THREE.Matrix4();
      const position = new THREE.Vector3(), rotation = new THREE.Quaternion(), one = new THREE.Vector3(1, 1, 1);
      for (const [geom, mesh] of instances) {
        const localPosition = new THREE.Vector3().fromArray(model.geom_pos, geom * 3);
        const localRotation = new THREE.Quaternion(model.geom_quat[geom * 4 + 1], model.geom_quat[geom * 4 + 2], model.geom_quat[geom * 4 + 3], model.geom_quat[geom * 4]);
        geomMatrix.compose(localPosition, localRotation, one);
        const fullBodyName = nameAt(model.name_bodyadr[model.geom_bodyid[geom]]);
        const shortBodyName = fullBodyName.split('/').at(-1) ?? fullBodyName;
        const index = bodyIndex.get(fullBodyName)
          ?? bodyIndex.get(fullBodyName.replace(/^robot\//, ''))
          ?? bodyIndex.get(shortBodyName)
          ?? next.bodyNames.findIndex(name => name.split('/').at(-1) === shortBodyName);
        mesh.count = index === undefined ? 0 : count;
        if (index === undefined) continue;
        for (let environmentIndex = 0; environmentIndex < count; environmentIndex++) {
          const environment = next.environments[environmentIndex];
          const pose = environment.bodyPositions[index], quaternion = environment.bodyQuaternions[index];
          // A fixed environment origin preserves real base translation. Using
          // the current base position here visually pins the robot in mid-air.
          const origin = environment.environmentOrigin ?? environment.bodyPositions[base];
          const column = environmentIndex % columns, row = Math.floor(environmentIndex / columns);
          const tileX = (column - (columns - 1) / 2) * .82;
          const tileY = ((rows - 1) / 2 - row) * .82;
          position.fromArray(trainingDisplayPosition(pose, origin, tileX, tileY));
          rotation.set(quaternion[1], quaternion[2], quaternion[3], quaternion[0]);
          bodyMatrix.compose(position, rotation, one);
          worldMatrix.multiplyMatrices(bodyMatrix, geomMatrix);
          mesh.setMatrixAt(environmentIndex, worldMatrix);
        }
        mesh.instanceMatrix.needsUpdate = true;
      }
      let markerCount = 0;
      for (let environmentIndex = 0; environmentIndex < count; environmentIndex++) {
        const environment = next.environments[environmentIndex];
        const origin = environment.environmentOrigin ?? environment.bodyPositions[base];
        const column = environmentIndex % columns, row = Math.floor(environmentIndex / columns);
        const tileX = (column - (columns - 1) / 2) * .82;
        const tileY = ((rows - 1) / 2 - row) * .82;
        for (let body = 0; body < environment.bodyPositions.length; body++) {
          const pose = environment.bodyPositions[body];
          if (!pose) continue;
          position.fromArray(trainingDisplayPosition(pose, origin, tileX, tileY));
          bodyMatrix.compose(position, new THREE.Quaternion(), new THREE.Vector3(.72, .72, .72));
          bodyMarkers.setMatrixAt(markerCount++, bodyMatrix);
        }
      }
      bodyMarkers.count = markerCount;
      bodyMarkers.instanceMatrix.needsUpdate = true;
      renderer.shadowMap.needsUpdate = true;
      render();
    };
    updateRef.current = applyFrame;

    void (async () => {
      try {
        const response = await fetch(MODEL_URL);
        if (!response.ok) throw new Error('ZBot训练模型资源不可用');
        const bundle = parseTrainingBundle(await response.json() as TrainingReplayBundle);
        await engine.init();
        if (disposed) return;
        engine.installTrainingAssets(bundle.assets);
        engine.loadModelFromXml(bundle.xml);
        model = engine.getModel();
        const geometryCache = new Map<number, THREE.BufferGeometry>();
        for (let geom = 0; geom < model.ngeom; geom++) {
          // Training assets can use either the standard visual group (1) or
          // the default group (0). Only skip collision geometry (group 4).
          if (model.geom_group[geom] === 4 || model.geom_type[geom] !== 7 || model.geom_dataid[geom] < 0) continue;
          const meshId = model.geom_dataid[geom];
          let geometry = geometryCache.get(meshId);
          if (!geometry) {
            geometry = compiledMeshGeometry(model, meshId);
            geometryCache.set(meshId, geometry);
            geometries.add(geometry);
          }
          const offset = geom * 4;
          const material = new THREE.MeshStandardMaterial({ color: new THREE.Color(model.geom_rgba[offset], model.geom_rgba[offset + 1], model.geom_rgba[offset + 2]), metalness: .35, roughness: .32 });
          materials.add(material);
          const mesh = new THREE.InstancedMesh(geometry, material, MAX_ENVIRONMENTS);
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          mesh.frustumCulled = false;
          scene.add(mesh);
          instances.set(geom, mesh);
        }
        setState('ready');
        if (frameRef.current) applyFrame(frameRef.current);
        resize();
      } catch (error) {
        if (!disposed) { console.error(error); setState('error'); }
      }
    })();

    return () => {
      disposed = true;
      updateRef.current = () => {};
      observer.disconnect();
      controls.removeEventListener('change', render);
      controls.dispose();
      instances.forEach(mesh => scene.remove(mesh));
      geometries.forEach(geometry => geometry.dispose());
      materials.forEach(material => material.dispose());
      bodyMarkerGeometry.dispose();
      bodyMarkerMaterial.dispose();
      floor.geometry.dispose();
      (floor.material as THREE.Material).dispose();
      renderer.dispose();
      renderer.domElement.remove();
      engine.destroy();
    };
  }, []);

  useEffect(() => { if (frame) { frameRef.current = frame; setDisplayFrame(frame); updateRef.current(frame); } }, [frame]);
  useEffect(() => (native ? client.nativeReplayLive(jobId, next => { if (next?.kind !== 'frame' || !Array.isArray(next.environments)) return; frameRef.current = next; setDisplayFrame(next); updateRef.current(next); }, () => {}) : client.live(jobId, next => {
    frameRef.current = next;
    setDisplayFrame(next);
    updateRef.current(next);
  }, () => {})), [client, jobId, native]);

  return <figure className="overflow-hidden rounded border border-slate-700 bg-[#0b1220]" aria-label={`mjlab ZBot实体连续训练画面${displayFrame ? `，第 ${displayFrame.iteration} 次迭代，显示 ${displayFrame.environments.length} 个环境` : ''}`}>
    <figcaption className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 px-3 py-2 text-xs">
      <span className="font-medium text-emerald-300"><span className="mr-2 inline-block h-2 w-2 rounded-full bg-emerald-400" />mjlab · ZBot实体训练画面</span>
      <span className="text-slate-400">{displayFrame ? `mjlab原始状态流 · 帧 ${displayFrame.iteration} · 实体 ${displayFrame.environments.length}/9` : '正在等待mjlab状态流…'}</span>
    </figcaption>
    <div ref={container} className="relative h-[390px] min-h-[300px] w-full">
      {state !== 'ready' && <div className="absolute inset-0 z-10 flex items-center justify-center bg-slate-950/80 text-xs text-slate-300">{state === 'error' ? 'ZBot实体模型加载失败，请检查静态资源。' : '正在加载真实 ZBot 网格…'}</div>}
    </div>
    <p className="border-t border-slate-800 px-3 py-2 text-[11px] text-slate-400">收到一帧就原样显示一帧；不移动相机，不做位置或四元数插值，环境重置也按mjlab原始状态显示。</p>
  </figure>;
}
