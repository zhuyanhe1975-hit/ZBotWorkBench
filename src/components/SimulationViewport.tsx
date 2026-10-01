import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { mujocoEngine, MujocoEngine } from '../mujoco/MujocoEngine';
import type { PhysxDiagnostics, PhysxRenderBody } from '../rl/physx';
import { ZbotConfiguration } from '../types/zbot';
import { meshManager } from '../utils/meshManager';
import { compiledMeshGeometry } from '../utils/compiledMeshGeometry';
import { EndEffectorTarget } from '../utils/inverseKinematics';
import { createArmTargetControl } from '../utils/armTargetControl';
import { createRenderScheduler } from '../utils/renderScheduler';
import { anchorGroundGrid, trackGroundShadow } from '../utils/shadowTracking';
import {
  Camera,
  Compass,
  Grid,
  Sparkles,
  Zap,
  ArrowUp,
  ArrowLeft,
  ArrowRight,
  Maximize2
} from 'lucide-react';

interface SimulationViewportProps {
  engine?: MujocoEngine;
  active?: boolean;
  revision?: number;
  physicsLabel?: string;
  config: ZbotConfiguration;
  simMetrics?: {
    time: number;
    speed: number;
    totalDistance: number;
    rootPos: [number, number, number];
  };
  armTarget?: EndEffectorTarget | null;
  armMode?: 'translate' | 'rotate';
  armValid?: boolean;
  onArmTargetChange?: (target: EndEffectorTarget) => void;
  onApplyImpulse?: (fx: number, fy: number, fz: number) => void;
  physicalEntity?: { getRenderBodies(): PhysxRenderBody[]; getDiagnostics(): PhysxDiagnostics } | null;
  showDisplayEntity?: boolean;
  showPhysicalEntity?: boolean;
  enablePhysicsDiagnostics?: boolean;
  onDiagnosticsChange?: (value: { contacts: boolean; forces: boolean; centerOfMass: boolean }) => void;
}

export const SimulationViewport: React.FC<SimulationViewportProps> = ({
  engine = mujocoEngine,
  active = true,
  revision = 0,
  physicsLabel = 'MuJoCo WASM 动力学仿真器',
  config,
  simMetrics,
  onApplyImpulse,
  physicalEntity = null,
  showDisplayEntity = true,
  showPhysicalEntity = false,
  enablePhysicsDiagnostics = false,
  onDiagnosticsChange,
  armTarget = null, armMode = 'translate', armValid = true, onArmTargetChange,
}) => {
  const armControlRef = useRef<ReturnType<typeof createArmTargetControl> | null>(null);
  const armChangeRef = useRef(onArmTargetChange);
  armChangeRef.current = onArmTargetChange;
  const diagnosticsChangeRef = useRef(onDiagnosticsChange);
  diagnosticsChangeRef.current = onDiagnosticsChange;
  const wrapperRef = useRef<HTMLDivElement>(null);
  const configRef = useRef(config);
  configRef.current = config;
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const invalidateRef = useRef<() => void>(() => {});
  const visibilityRef = useRef<() => void>(() => {});
  const activeRef = useRef(active);
  activeRef.current = active;
  const physicalEntityRef = useRef(physicalEntity);
  physicalEntityRef.current = physicalEntity;
  const entityVisibilityRef = useRef({ showDisplayEntity, showPhysicalEntity });
  entityVisibilityRef.current = { showDisplayEntity, showPhysicalEntity };
  const armTargetRef = useRef(armTarget);
  armTargetRef.current = armTarget;
  const poseDirtyRef = useRef(true);

  const [followRobot, setFollowRobot] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [showTrail, setShowTrail] = useState(false);
  const [showAxes, setShowAxes] = useState(false);
  const [showContactPoints, setShowContactPoints] = useState(false);
  const [showContactForces, setShowContactForces] = useState(false);
  const [showCenterOfMass, setShowCenterOfMass] = useState(false);
  const optionsRef = useRef({ followRobot, showGrid, showTrail, showAxes,
    showContactPoints, showContactForces, showCenterOfMass, enablePhysicsDiagnostics });
  optionsRef.current = { followRobot, showGrid, showTrail, showAxes,
    showContactPoints, showContactForces, showCenterOfMass, enablePhysicsDiagnostics };
  const jointAxesRef = useRef<THREE.Group | null>(null);
  const tipMarkerRef = useRef<THREE.Mesh | null>(null);
  const frameRobotRef = useRef<() => void>(() => {});
  useEffect(() => {
    needsRebuildRef.current = true;
    poseDirtyRef.current = true;
    invalidateRef.current();
  }, [config]);

  // Mesh map for MuJoCo geoms
  const geomMeshesRef = useRef<Map<number, THREE.Mesh>>(new Map());
  const trailPointsRef = useRef<THREE.Vector3[]>([]);
  const trailLineRef = useRef<THREE.Line | null>(null);
  const centerOfMassMarkerRef = useRef<THREE.Mesh | null>(null);
  const contactPointsGroupRef = useRef<THREE.Group | null>(null);
  const contactForcesGroupRef = useRef<THREE.Group | null>(null);
  const robotGroupRef = useRef<THREE.Group | null>(null);
  const physicalGroupRef = useRef<THREE.Group | null>(null);
  const physicalMeshesRef = useRef<Map<string, THREE.Points[]>>(new Map());
  const lastPhysicalEntityRef = useRef<typeof physicalEntity>(null);
  const gridHelperRef = useRef<THREE.Object3D | null>(null);
  const axesHelperRef = useRef<THREE.AxesHelper | null>(null);
  const lastModelRef = useRef<any>(null);
  const needsRebuildRef = useRef<boolean>(true);

  // Subscribe to mesh updates
  useEffect(() => {
    return meshManager.subscribe(() => {
      needsRebuildRef.current = true;
      poseDirtyRef.current = true;
      invalidateRef.current();
    });
  }, []);

  // Setup Three.js scene
  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 500;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#0f172a'); // slate-900
    scene.fog = new THREE.FogExp2('#0f172a', 0.035);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 100);
    camera.position.set(0.6, -1.0, 0.7);
    camera.up.set(0, 0, 1); // Z-up matching MuJoCo
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.shadowMap.autoUpdate = false;
    renderer.shadowMap.needsUpdate = true;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    renderer.domElement.style.display = 'block';
    container.replaceChildren(renderer.domElement);
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0.06);
    controls.maxPolarAngle = Math.PI / 2 + 0.02; // prevent dipping under floor
    controlsRef.current = controls;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.8);
    dirLight.position.set(2, -2, 4);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.near = 0.1;
    dirLight.shadow.camera.far = 10;
    dirLight.shadow.camera.left = -2;
    dirLight.shadow.camera.right = 2;
    dirLight.shadow.camera.top = 2;
    dirLight.shadow.camera.bottom = -2;
    dirLight.shadow.bias = -0.0005;
    scene.add(dirLight, dirLight.target);

    const fillLight = new THREE.DirectionalLight(0x94a3b8, 0.6);
    fillLight.position.set(-2, 2, 2);
    scene.add(fillLight);

    // Floor with shadow receiver
    const floorGeo = new THREE.PlaneGeometry(30, 30);
    const floorMat = new THREE.MeshStandardMaterial({
      color: '#1e293b', // slate-800
      roughness: 0.85,
      metalness: 0.1,
    });
    const floorMesh = new THREE.Mesh(floorGeo, floorMat);
    floorMesh.receiveShadow = true;
    scene.add(floorMesh);

    // Floor Grid
    const gridCanvas = document.createElement('canvas'); gridCanvas.width = gridCanvas.height = 128;
    const gridContext = gridCanvas.getContext('2d')!;
    gridContext.clearRect(0, 0, 128, 128);
    gridContext.strokeStyle = '#475569'; gridContext.lineWidth = 1;
    gridContext.beginPath(); gridContext.moveTo(.5, .5); gridContext.lineTo(127.5, .5); gridContext.moveTo(.5, .5); gridContext.lineTo(.5, 127.5); gridContext.stroke();
    const gridTexture = new THREE.CanvasTexture(gridCanvas);
    gridTexture.wrapS = gridTexture.wrapT = THREE.RepeatWrapping; gridTexture.repeat.set(40, 40); gridTexture.magFilter = THREE.LinearFilter;
    const grid = new THREE.Mesh(new THREE.PlaneGeometry(20, 20), new THREE.MeshBasicMaterial({ map: gridTexture, transparent: true, opacity: .35, depthTest: true, depthWrite: false, toneMapped: false, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 }));
    grid.position.z = 0.004;
    grid.renderOrder = 1;
    scene.add(grid);
    grid.visible = optionsRef.current.showGrid;
    gridHelperRef.current = grid;

    // Robot group
    const robotGroup = new THREE.Group();
    scene.add(robotGroup);
    robotGroupRef.current = robotGroup;

    const physicalGroup = new THREE.Group();
    physicalGroup.name = 'physx-collision-entity';
    scene.add(physicalGroup);
    physicalGroupRef.current = physicalGroup;

    // Axes helper
    const axes = new THREE.AxesHelper(0.2);
    axes.position.set(0, 0, 0.005);
    scene.add(axes);
    axesHelperRef.current = axes;
    axes.visible = false;

    const centerOfMassMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.014, 16, 16),
      new THREE.MeshBasicMaterial({ color: '#c084fc', depthTest: false }),
    );
    centerOfMassMarker.renderOrder = 12;
    centerOfMassMarker.visible = false;
    scene.add(centerOfMassMarker);
    centerOfMassMarkerRef.current = centerOfMassMarker;

    const contactPointsGroup = new THREE.Group();
    contactPointsGroup.name = 'mujoco-contact-points';
    contactPointsGroup.visible = false;
    scene.add(contactPointsGroup);
    contactPointsGroupRef.current = contactPointsGroup;

    const contactForcesGroup = new THREE.Group();
    contactForcesGroup.name = 'mujoco-contact-forces';
    contactForcesGroup.visible = false;
    scene.add(contactForcesGroup);
    contactForcesGroupRef.current = contactForcesGroup;

    const jointAxes = new THREE.Group();
    scene.add(jointAxes);
    jointAxesRef.current = jointAxes;
    const tipMarker = new THREE.Mesh(new THREE.SphereGeometry(0.009, 12, 12), new THREE.MeshBasicMaterial({ color: 0x34d399, depthTest: false }));
    tipMarker.renderOrder = 10;
    tipMarker.visible = false;
    scene.add(tipMarker);
    tipMarkerRef.current = tipMarker;

    // Trajectory Line
    const trailGeo = new THREE.BufferGeometry();
    const trailMat = new THREE.LineBasicMaterial({
      color: '#38bdf8',
      linewidth: 2,
      transparent: true,
      opacity: 0.7,
    });
    const trailLine = new THREE.Line(trailGeo, trailMat);
    scene.add(trailLine);
    trailLineRef.current = trailLine;

    // Resize joins the same demand-render queue; there is no second RAF loop.
    let pendingSize: { width: number; height: number } | null = null;
    let currentWidth = width;
    let currentHeight = height;
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: newW, height: newH } = entry.contentRect;
        if (newW > 0 && newH > 0 && (newW !== currentWidth || newH !== currentHeight)) {
          pendingSize = { width: newW, height: newH };
          invalidateRef.current();
        }
      }
    });
    resizeObserver.observe(container);

    const armControl = createArmTargetControl(scene, camera, renderer.domElement, controls, pose => armChangeRef.current?.(pose));
    armControlRef.current = armControl;

    // Draw only on changes, continuing just long enough for orbit damping.
    let lastTrailTime = -Infinity;
    let previousSimTime = -Infinity;
    let needsFrame = true;
    const frameRobot = () => {
      robotGroup.updateMatrixWorld(true);
      const bounds = new THREE.Box3().setFromObject(robotGroup);
      if (bounds.isEmpty()) return;
      const center = bounds.getCenter(new THREE.Vector3());
      const radius = Math.max(0.15, bounds.getSize(new THREE.Vector3()).length() / 2);
      const angle = Math.min(camera.fov * Math.PI / 360, Math.atan(Math.tan(camera.fov * Math.PI / 360) * camera.aspect));
      const distance = radius / Math.sin(angle) * 1.25;
      controls.target.copy(center);
      camera.position.copy(center).add(new THREE.Vector3(0.6, -0.9, 0.6).normalize().multiplyScalar(distance));
      controls.update();
      invalidateRef.current();
    };
    frameRobotRef.current = frameRobot;

    let lastCameraFrame = performance.now();
    const render = () => {
      const now = performance.now();
      // Wall-clock smoothing stays consistent across simulation speeds and FPS.
      // Cap gaps after hidden tabs so returning to the scene cannot snap the camera.
      const cameraDt = Math.min(.05, Math.max(0, (now - lastCameraFrame) / 1000));
      lastCameraFrame = now;
      let followSettling = false;
      if (pendingSize) {
        currentWidth = pendingSize.width;
        currentHeight = pendingSize.height;
        pendingSize = null;
        camera.aspect = currentWidth / currentHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(currentWidth, currentHeight, false);
        needsFrame = true;
      }
      // Camera moves alone do not invalidate the cached shadow map.
      if (poseDirtyRef.current || lastModelRef.current !== engine.getModel() || needsRebuildRef.current
        || (engine.isReady() && engine.getData()?.time !== previousSimTime)) {
        renderer.shadowMap.needsUpdate = true;
      }
      poseDirtyRef.current = false;

      // Sync MuJoCo physics to Three.js meshes
      if (engine.isReady()) {
        const model = engine.getModel();
        const data = engine.getData();

        if (model && data && robotGroupRef.current) {
          robotGroupRef.current.visible = entityVisibilityRef.current.showDisplayEntity;
          const ngeom = model.ngeom;
          const geomPos = data.geom_xpos;
          const geomMat = data.geom_xmat;

          // Rebuild meshes if model or meshes changed
          if (geomMeshesRef.current.size === 0 || lastModelRef.current !== model || needsRebuildRef.current) {
            if (lastModelRef.current !== model) {
              clearTrail();
              lastTrailTime = -Infinity;
              needsFrame = true;
            }
            lastModelRef.current = model;
            needsRebuildRef.current = false;
            rebuildGeomMeshes(model, configRef.current);
          }

          // Update each geom transform (skip floor geom 0)
          for (let g = 1; g < ngeom; g++) {
            const mesh = geomMeshesRef.current.get(g);
            if (mesh) {
              const px = geomPos[g * 3];
              const py = geomPos[g * 3 + 1];
              const pz = geomPos[g * 3 + 2];

              // Row-major 3x3 rotation matrix in MuJoCo
              const m00 = geomMat[g * 9];
              const m01 = geomMat[g * 9 + 1];
              const m02 = geomMat[g * 9 + 2];

              const m10 = geomMat[g * 9 + 3];
              const m11 = geomMat[g * 9 + 4];
              const m12 = geomMat[g * 9 + 5];

              const m20 = geomMat[g * 9 + 6];
              const m21 = geomMat[g * 9 + 7];
              const m22 = geomMat[g * 9 + 8];

              mesh.matrix.set(
                m00, m01, m02, px,
                m10, m11, m12, py,
                m20, m21, m22, pz,
                0, 0, 0, 1
              );
            }
          }

          // A simulation reset clears time-dependent overlays but preserves the
          // user's camera position, orientation and zoom.
          if (data.time < previousSimTime) { clearTrail(); lastTrailTime = -Infinity; }
          previousSimTime = data.time;
          if (needsFrame) { frameRobot(); needsFrame = false; }
          jointAxes.visible = optionsRef.current.showAxes;
          axes.visible = optionsRef.current.showAxes;
          jointAxes.children.forEach((child) => {
            const j = child.userData.joint;
            child.position.fromArray(data.xanchor, j * 3);
            child.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3().fromArray(data.xaxis, j * 3).normalize());
          });
          // The model exposes its terminal endpoint as the named tip site.
          const tipId = tipMarker.userData.site;
          tipMarker.visible = Number.isInteger(tipId) && tipId >= 0;
          if (tipMarker.visible) tipMarker.position.fromArray(data.site_xpos, tipId * 3);

          // Root-body origin
          const rootOffset = engine.getRootBodyId() * 3;
          const rootX = data.xpos[rootOffset] || 0;
          const rootY = data.xpos[rootOffset + 1] || 0;
          const rootZ = data.xpos[rootOffset + 2] ?? 0.05;

          // Keep sunlight and its bounded shadow camera over the moving robot,
          // independently of whether the viewing camera follows it.
          if (trackGroundShadow(dirLight, floorMesh, rootX, rootY, [grid])) renderer.shadowMap.needsUpdate = true;
          anchorGroundGrid(gridTexture, rootX, rootY);

          const diagnosticsEnabled = optionsRef.current.enablePhysicsDiagnostics;
          const physicalDiagnostics = physicalEntityRef.current?.getDiagnostics();
          const showPoints = diagnosticsEnabled && optionsRef.current.showContactPoints;
          const showForces = diagnosticsEnabled && optionsRef.current.showContactForces;
          contactPointsGroup.visible = showPoints;
          contactForcesGroup.visible = showForces;
          if (showPoints || showForces) {
            const contacts = (physicalDiagnostics?.contacts ?? engine.getContactDiagnostics())
              .filter(contact => contact.magnitude > .05)
              .sort((a, b) => b.magnitude - a.magnitude)
              .slice(0, 24);
            const forceLines = physicalDiagnostics?.groundResultant ? [physicalDiagnostics.groundResultant] : [];
            const forceContacts = physicalDiagnostics ? forceLines.map(line => {
              const force = line.end.map((value, axis) => value - line.start[axis]) as [number, number, number];
              return { position: line.start, force, magnitude: Math.hypot(...force), kind: line.kind };
            }) : contacts;
            while (contactPointsGroup.children.length < contacts.length) {
              const marker = new THREE.Mesh(
                new THREE.SphereGeometry(0.008, 12, 12),
                new THREE.MeshBasicMaterial({ color: 0xfb7185, depthTest: false }),
              );
              marker.renderOrder = 13;
              contactPointsGroup.add(marker);
            }
            while (contactForcesGroup.children.length < forceContacts.length) {
              const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), .05, 0x22d3ee, .018, .01);
              (arrow.line.material as THREE.Material).depthTest = false;
              (arrow.cone.material as THREE.Material).depthTest = false;
              arrow.line.renderOrder = 13;
              arrow.cone.renderOrder = 13;
              contactForcesGroup.add(arrow);
            }
            contactPointsGroup.children.forEach((marker, index) => {
              marker.visible = showPoints && index < contacts.length;
              if (index < contacts.length) marker.position.fromArray(contacts[index].position);
            });
            contactForcesGroup.children.forEach((object, index) => {
              const arrow = object as THREE.ArrowHelper;
              const contact = forceContacts[index];
              arrow.visible = showForces && !!contact && contact.magnitude > 1e-6;
              if (arrow.visible) {
                arrow.position.fromArray(contact.position);
                arrow.setDirection(new THREE.Vector3(...contact.force).normalize());
                const length = physicalDiagnostics ? contact.magnitude : Math.min(.3, .025 + Math.log1p(contact.magnitude) * .025);
                arrow.setColor(new THREE.Color('kind' in contact && contact.kind === 'friction' ? 0xf59e0b : 0x22d3ee));
                arrow.setLength(length, Math.min(.018, length * .35), Math.min(.01, length * .2));
              }
            });
          } else {
            contactPointsGroup.children.forEach(child => { child.visible = false; });
            contactForcesGroup.children.forEach(child => { child.visible = false; });
          }

          centerOfMassMarker.visible = diagnosticsEnabled && optionsRef.current.showCenterOfMass;
          if (centerOfMassMarker.visible) {
            centerOfMassMarker.position.fromArray(physicalDiagnostics?.centerOfMass ?? data.subtree_com, physicalDiagnostics ? 0 : rootOffset);
          }

          // subtree_com is mass-weighted over the entire robot, even when the
          // kinematic tree starts at a foot. Keep the orbit centered on that COM.
          if (optionsRef.current.followRobot) {
            const center = new THREE.Vector3().fromArray(data.subtree_com, rootOffset);
            if (center.toArray().every(Number.isFinite)) {
              const delta = center.clone().sub(controls.target);
              // Stronger vertical filtering suppresses gait-induced bobbing.
              const horizontalAlpha = -Math.expm1(-cameraDt / .22);
              const verticalAlpha = -Math.expm1(-cameraDt / .45);
              followSettling = delta.lengthSq() > 1e-8;
              if (followSettling) {
                delta.multiply(new THREE.Vector3(horizontalAlpha, horizontalAlpha, verticalAlpha));
              }
              camera.position.add(delta);
              controls.target.add(delta);
            }
          }

          // Trail points
          if (optionsRef.current.showTrail && data.time - lastTrailTime > 0.08) {
            lastTrailTime = data.time;
            const pts = trailPointsRef.current;
            pts.push(new THREE.Vector3(rootX, rootY, rootZ));
            if (pts.length > 250) pts.shift();

            if (trailLineRef.current) {
              trailLineRef.current.geometry.dispose();
              trailLineRef.current.geometry = new THREE.BufferGeometry().setFromPoints(pts);
            }
          }
        }
      } else {
        if (robotGroupRef.current) robotGroupRef.current.visible = false;
        centerOfMassMarker.visible = false;
        contactPointsGroup.visible = false;
        contactForcesGroup.visible = false;
        jointAxes.visible = false;
        tipMarker.visible = false;
        if (lastModelRef.current) {
          clearTrail(); lastTrailTime = -Infinity;
          lastModelRef.current = null;
        }
      }


      const physicalEntity = physicalEntityRef.current;
      const physicalGroup = physicalGroupRef.current;
      if (physicalGroup) {
        const shouldShowPhysical = !!physicalEntity && entityVisibilityRef.current.showPhysicalEntity;
        physicalGroup.visible = shouldShowPhysical;
        if (!physicalEntity && lastPhysicalEntityRef.current) {
          physicalMeshesRef.current.clear();
          while (physicalGroup.children.length) {
            const mesh = physicalGroup.children[0] as THREE.Points;
            physicalGroup.remove(mesh);
            mesh.geometry.dispose();
            (mesh.material as THREE.Material).dispose();
          }
          lastPhysicalEntityRef.current = null;
        }
        // Use the exact cooked-collider vertices as a diagnostic point cloud.
        // Reconstructing convex faces with QuickHull blocked the browser for
        // high-resolution archived assets and made policy loading appear stuck.
        if (shouldShowPhysical) {
          const renderBodies = physicalEntity.getRenderBodies();
          if (lastPhysicalEntityRef.current !== physicalEntity) {
            physicalMeshesRef.current.clear();
            while (physicalGroup.children.length) {
              const mesh = physicalGroup.children[0] as THREE.Points;
              physicalGroup.remove(mesh);
              mesh.geometry.dispose();
              (mesh.material as THREE.Material).dispose();
            }
            for (const body of renderBodies) {
              const meshes = body.hulls.map((vertices, hullIndex) => {
                const geometry = new THREE.BufferGeometry().setFromPoints(vertices.map(point => new THREE.Vector3(...point)));
                const material = new THREE.PointsMaterial({ color: 0xf43f5e, size: .004,
                  sizeAttenuation: true, transparent: true, opacity: .9, depthTest: true, depthWrite: false });
                const mesh = new THREE.Points(geometry, material);
                mesh.name = `physx-${body.name}-${hullIndex}`;
                mesh.renderOrder = 5;
                mesh.matrixAutoUpdate = false;
                physicalGroup.add(mesh);
                return mesh;
              });
              physicalMeshesRef.current.set(body.name, meshes);
            }
            lastPhysicalEntityRef.current = physicalEntity;
          }
          for (const body of renderBodies) {
            const matrix = new THREE.Matrix4().compose(
              new THREE.Vector3(...body.position),
              new THREE.Quaternion(body.quaternion[1], body.quaternion[2], body.quaternion[3], body.quaternion[0]),
              new THREE.Vector3(1, 1, 1),
            );
            physicalMeshesRef.current.get(body.name)?.forEach(mesh => mesh.matrix.copy(matrix));
          }
        }
      }

      const changed = controls.update();
      renderer.render(scene, camera);
      return changed || followSettling;
    };

    const scheduler = createRenderScheduler(render);
    invalidateRef.current = scheduler.invalidate;
    controls.addEventListener('change', scheduler.invalidate);
    // TransformControls changes hover highlighting without a React state update.
    // Listen above its canvas capture handlers, which may stop propagation.
    const pointerFeedback = () => { if (armTargetRef.current) scheduler.invalidate(); };
    const pointerEvents = ['pointermove', 'pointerdown', 'pointerup', 'pointercancel', 'pointerleave'] as const;
    pointerEvents.forEach(event => container.addEventListener(event, pointerFeedback, true));
    let intersecting = typeof IntersectionObserver === 'undefined';
    const updateVisibility = () => scheduler.setVisible(activeRef.current && !document.hidden && intersecting);
    visibilityRef.current = updateVisibility;
    const intersectionObserver = typeof IntersectionObserver === 'undefined' ? null : new IntersectionObserver(entries => {
      intersecting = entries.some(entry => entry.isIntersecting);
      updateVisibility();
    });
    intersectionObserver?.observe(container);
    document.addEventListener('visibilitychange', updateVisibility);
    updateVisibility();
    scheduler.invalidate();

    return () => {
      scheduler.dispose();
      invalidateRef.current = () => {};
      visibilityRef.current = () => {};
      controls.removeEventListener('change', scheduler.invalidate);
      pointerEvents.forEach(event => container.removeEventListener(event, pointerFeedback, true));
      document.removeEventListener('visibilitychange', updateVisibility);
      intersectionObserver?.disconnect();
      resizeObserver.disconnect();
      armControl.dispose();
      armControlRef.current = null;
      controls.dispose();
      scene.traverse(object => {
        const renderable = object as THREE.Mesh;
        renderable.geometry?.dispose();
        if (renderable.material) (Array.isArray(renderable.material) ? renderable.material : [renderable.material]).forEach(material => material.dispose());
      });
      dirLight.shadow.dispose();
      gridTexture.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      lastModelRef.current = null;
      trailPointsRef.current = [];
      geomMeshesRef.current.clear();
      physicalMeshesRef.current.clear();
      physicalGroupRef.current = null;
      centerOfMassMarkerRef.current = null;
      contactPointsGroupRef.current = null;
      contactForcesGroupRef.current = null;
      lastPhysicalEntityRef.current = null;
      frameRobotRef.current = () => {};
    };
  }, [engine]);

  useEffect(() => { visibilityRef.current(); }, [active]);

  // Parent pose edits and async model readiness may leave simulation time at zero.
  useEffect(() => {
    poseDirtyRef.current = true;
    invalidateRef.current();
  }, [simMetrics, revision]);

  useEffect(() => {
    if (robotGroupRef.current) robotGroupRef.current.visible = showDisplayEntity;
    if (physicalGroupRef.current) physicalGroupRef.current.visible = !!physicalEntity && showPhysicalEntity;
    poseDirtyRef.current = true;
    invalidateRef.current();
  }, [physicalEntity, showDisplayEntity, showPhysicalEntity]);

  useEffect(() => {
    if (gridHelperRef.current) gridHelperRef.current.visible = showGrid;
    invalidateRef.current();
  }, [followRobot, showGrid, showTrail, showAxes, showContactPoints, showContactForces, showCenterOfMass, enablePhysicsDiagnostics]);

  useEffect(() => {
    if (!enablePhysicsDiagnostics) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.repeat || event.altKey || event.ctrlKey || event.metaKey) return;
      const target = event.target as HTMLElement | null;
      const textInput = target instanceof HTMLInputElement
        && !['button', 'checkbox', 'file', 'radio', 'range', 'reset', 'submit'].includes(target.type);
      if (target?.isContentEditable || target instanceof HTMLTextAreaElement || textInput) return;
      switch (event.key.toLowerCase()) {
        case 'c': setShowContactPoints(value => !value); break;
        case 'f': setShowContactForces(value => !value); break;
        case 'm': setShowCenterOfMass(value => !value); break;
        default: return;
      }
      event.preventDefault();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [enablePhysicsDiagnostics]);

  useEffect(() => {
    diagnosticsChangeRef.current?.({ contacts: showContactPoints, forces: showContactForces, centerOfMass: showCenterOfMass });
  }, [showContactPoints, showContactForces, showCenterOfMass]);

  useEffect(() => {
    armControlRef.current?.setTarget(armTarget, armMode, armValid);
    invalidateRef.current();
  }, [armTarget, armMode, armValid, engine]);

  // Rebuild Three.js geometries for MuJoCo model
  const rebuildGeomMeshes = (model: any, currentConfig: ZbotConfiguration) => {
    if (!sceneRef.current || !robotGroupRef.current) return;
    const group = robotGroupRef.current;

    // Clear old meshes
    while (group.children.length > 0) {
      const child = group.children[0] as THREE.Mesh;
      group.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) (Array.isArray(child.material) ? child.material : [child.material]).forEach(m => m.dispose());
    }
    geomMeshesRef.current.clear();

    const axesGroup = jointAxesRef.current!;
    axesGroup.children.forEach(child => {
      const mesh = child as THREE.Mesh;
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    });
    axesGroup.clear();
    for (let j = 0; j < model.njnt; j++) {
      if (model.jnt_type[j] === 0) continue; // Free joints have no single rotation axis.
      const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.002, 0.002, 0.13, 8), new THREE.MeshBasicMaterial({ color: 0xfbbf24, depthTest: false }));
      rod.renderOrder = 9;
      rod.userData.joint = j;
      axesGroup.add(rod);
    }
    let tipId = -1;
    for (let i = 0; i < model.nsite; i++) {
      let name = '';
      for (let k = model.name_siteadr[i]; model.names[k]; k++) name += String.fromCharCode(model.names[k]);
      if (name === 'tip' || /^tip_\d+$/.test(name)) tipId = i;
    }
    if (tipMarkerRef.current) tipMarkerRef.current.userData.site = tipId;
    const { geomA, geomB } = meshManager.getGeometries();

    // Palettes
    const moduleColors = currentConfig.modules.map((m) => ({
      colorA: m.colorA || '#2563eb',
      colorB: m.colorB || '#38bdf8',
    }));

    const ngeom = model.ngeom;
    const geomGroups = model.geom_group;

    let visualGeomIndex = 0;
    for (let g = 1; g < ngeom; g++) {
      // Only render visual geoms (group === 1)
      if (geomGroups && geomGroups[g] !== 1) {
        continue;
      }

      // Check if Part A or Part B:
      // In canonical Zbot MJCF structure, visual geoms are strictly paired:
      // visualGeomIndex 0: Module 0 Part A (ma)
      // visualGeomIndex 1: Module 0 Part B (mb)
      // visualGeomIndex 2: Module 1 Part A (ma)
      // visualGeomIndex 3: Module 1 Part B (mb), etc.
      let geomName = '';
      for (let k = model.name_geomadr[g]; model.names[k]; k++) geomName += String.fromCharCode(model.names[k]);
      const namedIndex = /^visual_[ab]_(\d+)(?:_\d+)?$/.exec(geomName);
      const isPartA = namedIndex ? geomName.startsWith('visual_a_') : visualGeomIndex % 2 === 0;
      const modIndex = namedIndex ? Number(namedIndex[1]) : Math.floor(visualGeomIndex / 2);
      if (geomName !== 'visual_cube' && geomName !== 'visual_tetrahedron') visualGeomIndex++;
      const palette = moduleColors[modIndex % moduleColors.length] || { colorA: '#2563eb', colorB: '#38bdf8' };

      const baseColor = (geomName === 'visual_cube' || geomName === 'visual_tetrahedron') ? '#d9dee6' : isPartA ? palette.colorA : palette.colorB;
      const mat = new THREE.MeshStandardMaterial({
        color: baseColor,
        flatShading: geomName === 'visual_tetrahedron',
        metalness: 0.4,
        roughness: 0.35,
        envMapIntensity: 1.0,
      });

      const meshId = model.geom_dataid[g];
      let meshName = '';
      if (meshId >= 0) for (let k = model.name_meshadr[meshId]; k >= 0 && model.names[k]; k++) meshName += String.fromCharCode(model.names[k]);
      const compiledMesh = meshId >= 0 && meshName !== 'ma' && meshName !== 'mb';
      const geometry = model.geom_type[g] === 6
        ? new THREE.BoxGeometry(2 * model.geom_size[g*3], 2 * model.geom_size[g*3+1], 2 * model.geom_size[g*3+2])
        : compiledMesh ? compiledMeshGeometry(model, meshId) : model.geom_type[g] === 5
        ? new THREE.CylinderGeometry(model.geom_size[g*3], model.geom_size[g*3], 2*model.geom_size[g*3+1], 32).rotateX(Math.PI/2)
        : isPartA ? geomA.clone() : geomB.clone();
      // Compiler centers and aligns source meshes, composing that offset into geom_xpose.
      // Undo mesh_pos/mesh_quat on raw OBJ vertices before applying the compiled geom pose.
      // https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-mesh
      const p = model.mesh_pos;
      const q = model.mesh_quat;
      if (meshId >= 0 && !compiledMesh && p && q) {
        const scale = model.mesh_scale;
        if (scale) geometry.scale(scale[meshId * 3], scale[meshId * 3 + 1], scale[meshId * 3 + 2]);
        const transform = new THREE.Matrix4().compose(
          new THREE.Vector3().fromArray(p, meshId * 3),
          new THREE.Quaternion(q[meshId * 4 + 1], q[meshId * 4 + 2], q[meshId * 4 + 3], q[meshId * 4]),
          new THREE.Vector3(1, 1, 1),
        ).invert();
        geometry.applyMatrix4(transform);
      }
      const mesh = new THREE.Mesh(geometry, mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.renderOrder = 2;
      mesh.matrixAutoUpdate = false;

      group.add(mesh);
      geomMeshesRef.current.set(g, mesh);
    }
  };

  // View presets
  const setCameraView = (view: 'iso' | 'top' | 'front' | 'side') => {
    if (!cameraRef.current || !controlsRef.current) return;
    if (view === 'iso') { frameRobotRef.current(); return; }
    const target = controlsRef.current.target;
    const distance = cameraRef.current.position.distanceTo(target);
    switch (view) {
      case 'top':
        cameraRef.current.position.set(target.x, target.y - 0.001, target.z + distance);
        break;
      case 'front':
        cameraRef.current.position.set(target.x, target.y - distance, target.z + 0.15);
        break;
      case 'side':
        cameraRef.current.position.set(target.x + distance, target.y, target.z + 0.15);
        break;
    }
    controlsRef.current.update();
    invalidateRef.current();
  };

  const clearTrail = () => {
    trailPointsRef.current = [];
    if (trailLineRef.current) {
      trailLineRef.current.geometry.dispose();
      trailLineRef.current.geometry = new THREE.BufferGeometry();
    }
  };

  const handleFullscreen = () => {
    if (!wrapperRef.current) return;
    if (!document.fullscreenElement) {
      void wrapperRef.current.requestFullscreen?.().catch(console.warn);
    } else {
      void document.exitFullscreen?.().catch(console.warn);
    }
  };

  return (
    <div ref={wrapperRef} className="relative w-full h-full min-h-[420px] bg-slate-950 rounded-xl overflow-hidden shadow-inner flex flex-col border border-slate-800">
      {/* 3D WebGL Canvas */}
      <div ref={containerRef} className="w-full flex-1 touch-none" />

      {/* Top Floating Overlay Controls */}
      <div className="absolute top-3 left-3 right-3 flex flex-wrap items-center gap-2 z-10">
        <div className="bg-slate-900/85 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-700/60 shadow-lg flex items-center gap-2 text-xs text-slate-200">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
          <span className="font-medium">{physicsLabel}</span>
          <span className="text-slate-500">|</span>
          <span className="text-slate-400 font-mono">
            {simMetrics ? `${simMetrics.speed.toFixed(2)} m/s` : '0.00 m/s'}
          </span>
        </div>

        {/* Camera Views dropdown / buttons */}
        <div className="bg-slate-900/85 backdrop-blur-md p-1 rounded-lg border border-slate-700/60 shadow-lg flex items-center gap-1">
          <button
            onClick={() => setCameraView('iso')}
            className="px-2 py-1 text-xs text-slate-300 hover:text-white hover:bg-slate-800 rounded transition"
            title="3D透视"
          >
            3D轴测
          </button>
          <button
            onClick={() => setCameraView('top')}
            className="px-2 py-1 text-xs text-slate-300 hover:text-white hover:bg-slate-800 rounded transition"
            title="俯视视角"
          >
            俯视
          </button>
          <button
            onClick={() => setCameraView('front')}
            className="px-2 py-1 text-xs text-slate-300 hover:text-white hover:bg-slate-800 rounded transition"
            title="前视视角"
          >
            前视
          </button>
          <button
            onClick={() => setCameraView('side')}
            className="px-2 py-1 text-xs text-slate-300 hover:text-white hover:bg-slate-800 rounded transition"
            title="侧视视角"
          >
            侧视
          </button>
        </div>
      </div>

      {/* Top Right Quick Actions */}
      <div className="absolute top-24 sm:top-14 right-3 flex items-center gap-1.5 z-10">
        <button
          onClick={() => setFollowRobot(!followRobot)}
          className={`px-2.5 py-1.5 rounded-lg border text-xs font-medium flex items-center gap-1.5 shadow-lg transition backdrop-blur-md ${
            followRobot
              ? 'bg-blue-600/90 border-blue-500 text-white'
              : 'bg-slate-900/80 border-slate-700 text-slate-300 hover:bg-slate-800'
          }`}
          title="平滑跟随机器人整体质心，抑制上下抖动"
        >
          <Camera className="w-3.5 h-3.5" />
          {followRobot ? '跟随锁定' : '自由视角'}
        </button>

        <button
          onClick={() => {
            const next = !showGrid;
            setShowGrid(next);
            if (gridHelperRef.current) gridHelperRef.current.visible = next;
          }}
          className={`p-1.5 rounded-lg border text-xs shadow-lg transition backdrop-blur-md ${
            showGrid ? 'bg-slate-800 border-slate-600 text-sky-400' : 'bg-slate-900/80 border-slate-700 text-slate-400'
          }`}
          title="切换参考网格"
        >
          <Grid className="w-4 h-4" />
        </button>

        <button
          onClick={() => {
            const next = !showTrail;
            setShowTrail(next);
            if (!next) clearTrail();
          }}
          className={`p-1.5 rounded-lg border text-xs shadow-lg transition backdrop-blur-md ${
            showTrail ? 'bg-slate-800 border-slate-600 text-cyan-400' : 'bg-slate-900/80 border-slate-700 text-slate-400'
          }`}
          title="运动轨迹追踪线"
        >
          <Sparkles className="w-4 h-4" />
        </button>

        <button
          onClick={handleFullscreen}
          className="p-1.5 rounded-lg border border-slate-700 bg-slate-900/80 text-slate-300 hover:bg-slate-800 text-xs shadow-lg transition backdrop-blur-md"
          title="全屏画布"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

      <button onClick={() => setShowAxes(!showAxes)} aria-pressed={showAxes} className={`absolute right-3 top-36 sm:top-24 z-10 rounded-lg border px-3 py-1.5 text-xs ${showAxes ? 'bg-amber-900/90 border-amber-500 text-amber-200' : 'bg-slate-900/90 border-slate-700 text-slate-300'}`} title="显示实时关节旋转轴与世界坐标轴">
        <Compass className="inline w-3.5 h-3.5 mr-1" />关节轴 {showAxes ? '开' : '关'}
      </button>
      {enablePhysicsDiagnostics && <div aria-label="物理可视化快捷键" className="absolute bottom-20 left-3 z-10 flex flex-wrap gap-1.5 rounded-lg border border-slate-700/70 bg-slate-900/90 p-1.5 text-xs shadow-lg backdrop-blur-md">
        {([
          ['C', '接触点', showContactPoints, () => setShowContactPoints(value => !value)],
          ['F', physicalEntity ? '脚底合力' : '接触力', showContactForces, () => setShowContactForces(value => !value)],
          ['M', '质心', showCenterOfMass, () => setShowCenterOfMass(value => !value)],
        ] as const).map(([key, label, pressed, toggle]) => <button key={key} aria-pressed={pressed} onClick={toggle}
          className={`rounded border px-2 py-1 transition ${pressed ? 'border-cyan-400 bg-cyan-900/80 text-cyan-100' : 'border-slate-700 bg-slate-800 text-slate-300 hover:bg-slate-700'}`}
          title={`快捷键 ${key}：显示${label}`}><kbd className="mr-1 font-mono text-[10px] text-slate-400">{key}</kbd>{label}</button>)}
      </div>}
      {/* Bottom Interactive Perturbation Bar (Force Nudge) */}
      <div className="absolute bottom-9 left-3 right-3 hidden sm:flex flex-wrap items-center gap-2 z-10">
        {onApplyImpulse && <div className="bg-slate-900/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-700/70 shadow-lg flex flex-wrap items-center gap-2 text-xs text-slate-300">
          <Zap className="w-3.5 h-3.5 text-amber-400" />
          <span className="font-medium text-slate-200">速度扰动 Δv:</span>
          <button
            disabled={config.baseMode === 'fixed'}
            onClick={() => onApplyImpulse?.(0, 0.4, 0.1)}
            className="px-2 py-0.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded border border-slate-700 flex items-center gap-1 active:scale-95 transition"
            title="自由基座速度增加 [0, 0.4, 0.1] m/s"
          >
            <ArrowUp className="w-3 h-3 text-sky-400" />
            推力
          </button>
          <button
            disabled={config.baseMode === 'fixed'}
            onClick={() => onApplyImpulse?.(-0.4, 0, 0.1)}
            className="px-2 py-0.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded border border-slate-700 flex items-center gap-1 active:scale-95 transition"
            title="自由基座速度增加 [-0.4, 0, 0.1] m/s"
          >
            <ArrowLeft className="w-3 h-3 text-emerald-400" />
            左侧撞击
          </button>
          <button
            disabled={config.baseMode === 'fixed'}
            onClick={() => onApplyImpulse?.(0.4, 0, 0.1)}
            className="px-2 py-0.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded border border-slate-700 flex items-center gap-1 active:scale-95 transition"
            title="自由基座速度增加 [0.4, 0, 0.1] m/s"
          >
            <ArrowRight className="w-3 h-3 text-orange-400" />
            右侧撞击
          </button>
          <button
            disabled={config.baseMode === 'fixed'}
            onClick={() => onApplyImpulse?.(0, 0, 0.6)}
            className="px-2 py-0.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded border border-slate-700 active:scale-95 transition"
            title="向上颠簸"
          >
            颠簸
          </button>
        </div>}

        {showTrail && (
          <button
            onClick={() => { clearTrail(); invalidateRef.current(); }}
            className="bg-slate-900/90 backdrop-blur-md px-2.5 py-1.5 rounded-lg border border-slate-700/70 text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition"
          >
            清除轨迹
          </button>
        )}
      </div>

      {/* Bottom Right Coordinates / Scale Indicator */}
      <div className="absolute bottom-3 right-3 bg-slate-900/80 backdrop-blur-md px-2.5 py-1 rounded border border-slate-800 text-[11px] font-mono text-slate-400 pointer-events-none">
        绿色：末端{showCenterOfMass && enablePhysicsDiagnostics ? ' · 紫色：质心' : ''}{showContactPoints && enablePhysicsDiagnostics ? ' · 红点：接触' : ''}{showContactForces && enablePhysicsDiagnostics ? ` · ${physicalEntity ? '青色：脚底合力' : '青色：接触力'}` : ''}{showPhysicalEntity && physicalEntity ? ' · 红点：物理碰撞体' : ''} | Z↑ · 网格 0.5m
      </div>
    </div>
  );
};
