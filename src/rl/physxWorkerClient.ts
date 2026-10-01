import type { MujocoEngine } from '../mujoco/MujocoEngine';
import type { PolicyState, ReplayDynamics } from './replay';
import type { PhysxDiagnostics, PhysxModel, PhysxRenderBody } from './physx';
import { logRlDebug } from './debugLog';

interface Snapshot { state: PolicyState; bodies: PhysxRenderBody[]; diagnostics: PhysxDiagnostics; basePosition: number[] }

export class PhysxWorkerSimulation implements ReplayDynamics {
  private worker: Worker;
  private sequence = 0;
  private controlDt = 1 / 30;
  private pending = new Map<number, { resolve: (value: any) => void; reject: (error: Error) => void; timer: number }>();
  private snapshot!: Snapshot;
  private constructor(private model: PhysxModel, private display: MujocoEngine) {
    this.worker = new Worker(new URL('./physxWorker.ts', import.meta.url), { type: 'module', name: 'zbot-physx' });
    this.worker.onmessage = event => {
      if (event.data.log) { logRlDebug(event.data.log, event.data.details); return; }
      const request = this.pending.get(event.data.id);
      if (!request) return;
      clearTimeout(request.timer); this.pending.delete(event.data.id);
      event.data.error ? request.reject(new Error(event.data.error)) : request.resolve(event.data.result);
    };
    this.worker.onerror = event => this.failAll(new Error(event.message || 'PhysX Worker 异常'));
  }
  static async create(model: PhysxModel, display: MujocoEngine, timing: Record<string, unknown>, timeoutMs = 20_000): Promise<PhysxWorkerSimulation> {
    const result = new PhysxWorkerSimulation(model, display);
    result.controlDt = typeof timing.controlDt === 'number' ? timing.controlDt : (model.metadata?.controlDt as number | undefined) ?? 1 / 30;
    logRlDebug('physx.worker.create');
    result.snapshot = await result.request('init', { model, timing }, timeoutMs);
    result.synchronizeDisplay(model.defaultQ, 0);
    logRlDebug('physx.worker.init.done');
    return result;
  }
  private request(command: string, payload: unknown, timeoutMs: number): Promise<any> {
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      const timer = window.setTimeout(() => {
        this.pending.delete(id); this.worker.terminate();
        const error = new Error(`PhysX ${command} 超过 ${timeoutMs / 1000} 秒，Worker 已终止`);
        logRlDebug('physx.worker.timeout', { command, timeoutMs }); reject(error); this.failAll(error);
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      this.worker.postMessage({ id, command, payload });
    });
  }
  private failAll(error: Error) {
    for (const request of this.pending.values()) { clearTimeout(request.timer); request.reject(error); }
    this.pending.clear();
  }
  private synchronizeDisplay(targets: ArrayLike<number>, time: number) {
    const data = this.display.getData(), display = this.display.getModel();
    const root = this.snapshot.bodies.find(body => body.name === this.model.rootBody)!;
    const q = this.model.displayRootQuaternionOffset
      ? multiply(root.quaternion, this.model.displayRootQuaternionOffset) : root.quaternion;
    const free = Array.from(display.jnt_type as Int32Array).findIndex(type => type === 0);
    const rootQpos = display.jnt_qposadr[free];
    data.qpos.set([...root.position, ...q], rootQpos);
    const names = this.model.displayJointNames ?? this.model.jointNames;
    const signs = this.model.displayJointSigns ?? names.map(() => 1);
    names.forEach((name, i) => {
      const joint = findName(display, display.name_jntadr, name);
      const actuator = Array.from({ length: display.nu }, (_, a) => a).find(a => display.actuator_trnid[2 * a] === joint)!;
      data.qpos[display.jnt_qposadr[joint]] = signs[i] * this.snapshot.state.positions[i];
      data.qvel[display.jnt_dofadr[joint]] = signs[i] * this.snapshot.state.velocities[i];
      data.ctrl[actuator] = signs[i] * targets[i];
    });
    data.time = time; this.display.forward();
  }
  reset(): void { /* Initial worker snapshot is already reset before PolicyReplay construction. */ }
  step(): void { throw new Error('PhysX Worker 必须异步步进'); }
  state(): PolicyState { return this.snapshot.state; }
  async stepAsync(targets: ArrayLike<number>): Promise<void> {
    this.snapshot = await this.request('step', { targets: Array.from(targets) }, 2_000);
    this.synchronizeDisplay(targets, this.display.getTime() + this.controlDt);
  }
  async resetAsync(elapsedTime = 0, targets: ArrayLike<number> = this.model.defaultQ): Promise<void> {
    this.snapshot = await this.request('reset', { elapsedTime, targets: Array.from(targets) }, 5_000);
    this.synchronizeDisplay(targets, elapsedTime);
  }
  getBasePosition(): number[] { return [...this.snapshot.basePosition]; }
  getRenderBodies(): PhysxRenderBody[] { return this.snapshot.bodies; }
  getDiagnostics(): PhysxDiagnostics { return this.snapshot.diagnostics; }
  setDebugVisualization(showForces: boolean): void {
    this.worker.postMessage({ id: 0, command: 'visualization', payload: { showForces } });
  }
  dispose(): void {
    this.failAll(new Error('PhysX Worker 已释放'));
    this.worker.postMessage({ id: ++this.sequence, command: 'dispose' });
    this.worker.terminate();
  }
}

function findName(model: any, addresses: Int32Array, expected: string): number {
  return Array.from(addresses).findIndex(address => {
    let value = ''; for (let i = address; i >= 0 && model.names[i]; i++) value += String.fromCharCode(model.names[i]);
    return value === expected;
  });
}
function multiply(a: number[], b: number[]): number[] {
  return [a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3], a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
    a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1], a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]];
}
