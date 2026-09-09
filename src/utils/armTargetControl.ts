import * as THREE from 'three';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EndEffectorTarget } from './inverseKinematics';

/** A world-space pose target, separate from the physics-owned endpoint marker. */
export function createArmTargetControl(scene: THREE.Scene, camera: THREE.Camera, canvas: HTMLCanvasElement,
  orbit: OrbitControls, onChange: (target: EndEffectorTarget) => void) {
  const target = new THREE.Mesh(new THREE.SphereGeometry(.016, 24, 16), new THREE.MeshBasicMaterial({ color: '#22d3ee', transparent: true, opacity: .8, depthTest: false }));
  target.name = 'arm-end-effector-target'; target.renderOrder = 20;
  target.add(new THREE.AxesHelper(.065));
  scene.add(target);
  const controls = new TransformControls(camera, canvas);
  controls.setSpace('world'); controls.setSize(.8);
  const helper = controls.getHelper(); scene.add(helper);
  let ballDrag = false;
  const plane = new THREE.Plane();
  const ray = new THREE.Raycaster();
  const offset = new THREE.Vector3();
  let pointerId = -1;
  const setRay = (event: PointerEvent) => {
    const rect = canvas.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1), camera);
  };
  const changed = () => onChange({ position: target.position.toArray() as [number,number,number], quaternion: target.quaternion.toArray() as [number,number,number,number] });
  const dragging = (event: { value: unknown }) => { orbit.enabled = !event.value; };
  controls.addEventListener('objectChange', changed);
  controls.addEventListener('dragging-changed', dragging);
  // The colored sphere itself supports view-plane translation in addition to the axis gizmo.
  const down = (event: PointerEvent) => {
    if (!target.visible || controls.mode !== 'translate' || event.button !== 0 || controls.dragging) return;
    setRay(event);
    if (!ray.intersectObject(target, false).length || (controls.axis && controls.axis !== 'XYZ')) return;
    plane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()), target.position);
    const hit = ray.ray.intersectPlane(plane, new THREE.Vector3());
    if (!hit) return;
    offset.copy(target.position).sub(hit);
    ballDrag = true; pointerId = event.pointerId; orbit.enabled = false;
    canvas.setPointerCapture(pointerId); event.stopImmediatePropagation(); event.preventDefault();
  };
  const move = (event: PointerEvent) => {
    if (!ballDrag || event.pointerId !== pointerId) return;
    setRay(event); const hit = ray.ray.intersectPlane(plane, new THREE.Vector3());
    if (hit) { target.position.copy(hit).add(offset); changed(); }
    event.stopImmediatePropagation(); event.preventDefault();
  };
  const up = (event: PointerEvent) => {
    if (!ballDrag || event.pointerId !== pointerId) return;
    ballDrag = false; orbit.enabled = true;
    if (canvas.hasPointerCapture(pointerId)) canvas.releasePointerCapture(pointerId);
    pointerId = -1; event.stopImmediatePropagation();
  };
  canvas.addEventListener('pointerdown', down, true);
  canvas.addEventListener('pointermove', move, true);
  canvas.addEventListener('pointerup', up, true);
  canvas.addEventListener('pointercancel', up, true);
  target.visible = false;
  return {
    setTarget(pose: EndEffectorTarget | null, mode: 'translate' | 'rotate', valid: boolean) {
      controls.setMode(mode);
      target.visible = !!pose;
      if (pose) {
        target.position.fromArray(pose.position); target.quaternion.fromArray(pose.quaternion);
        target.material.color.set(valid ? '#22d3ee' : '#fb7185');
        if (controls.object !== target) controls.attach(target);
      } else {
        controls.detach();
        ballDrag = false; orbit.enabled = true;
        if (pointerId >= 0 && canvas.hasPointerCapture(pointerId)) canvas.releasePointerCapture(pointerId);
        pointerId = -1;
      }
    },
    dispose() {
      canvas.removeEventListener('pointerdown', down, true);
      canvas.removeEventListener('pointermove', move, true);
      canvas.removeEventListener('pointerup', up, true);
      canvas.removeEventListener('pointercancel', up, true);
      controls.removeEventListener('objectChange', changed);
      controls.removeEventListener('dragging-changed', dragging);
      controls.detach(); controls.dispose(); orbit.enabled = true;
      scene.remove(helper, target);
      target.geometry.dispose(); target.material.dispose();
      (target.children[0] as THREE.AxesHelper).dispose();
    },
  };
}
