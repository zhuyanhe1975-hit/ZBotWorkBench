import loadMujoco from '@mujoco/mujoco';
import { meshManager } from '../utils/meshManager';
import { GaitConfig, SimMetrics, ZbotConfiguration } from '../types/zbot';

export interface BodyTransform {
  name: string;
  pos: [number, number, number];
  quat: [number, number, number, number]; // [x, y, z, w] for Three.js
}

export interface GeomTransform {
  name: string;
  type: number;
  meshName: string;
  pos: [number, number, number];
  mat: number[]; // 3x3 rotation matrix
}

export class MujocoEngine {
  private mujoco: any = null;
  private model: any = null;
  private data: any = null;
  private isLoaded = false;
  private currentXml = '';
  private initialAngles: Record<string, number> = {};
  private joints: { name: string; actuator: number; qpos: number; dof: number; min: number; max: number }[] = [];
  private initialKeyframe = -1;
  private tipSite = -1;
  private freeDof = -1;
  private freeQpos = -1;
  private initPromise: Promise<void> | null = null;

  public async init(): Promise<void> {
    if (this.initPromise) return this.initPromise;
    if (this.mujoco) return;

    this.initPromise = (async () => {
      this.mujoco = await loadMujoco({
        locateFile: (file: string) => `/${file}`,
      });
      // Ensure meshes are loaded from meshManager
      await meshManager.init();
      this.updateVfsMeshes();
    })();

    try {
      await this.initPromise;
    } catch (error) {
      this.mujoco = null;
      this.initPromise = null;
      throw error;
    }
  }

  public updateVfsMeshes(customA?: string, customB?: string): void {
    if (!this.mujoco) return;
    if (customA !== undefined) meshManager.setCustomMesh('ma', customA);
    if (customB !== undefined) meshManager.setCustomMesh('mb', customB);

    const ma = meshManager.getMeshAText();
    const mb = meshManager.getMeshBText();

    this.mujoco.FS.writeFile('ma.obj', ma);
    this.mujoco.FS.writeFile('mb.obj', mb);
  }

  public loadModelFromXml(xml: string): boolean {
    if (!this.mujoco) {
      throw new Error('MuJoCo WASM not initialized yet');
    }

    let nextModel: any = null;
    let nextData: any = null;
    try {
      nextModel = this.mujoco.MjModel.from_xml_string(xml);
      for (let a = 0; a < nextModel.nu; a++) {
        const j = nextModel.actuator_trnid[2 * a];
        if (nextModel.actuator_trntype[a] !== 0 || j < 0 || nextModel.jnt_type[j] !== 3
          || nextModel.actuator_dyntype[a] !== 0 || nextModel.actuator_gaintype[a] !== 0
          || nextModel.actuator_biastype[a] !== 1 || nextModel.actuator_gainprm[10 * a] <= 0
          || nextModel.actuator_biasprm[10 * a + 1] !== -nextModel.actuator_gainprm[10 * a]
          || nextModel.actuator_gear[6 * a] !== 1) {
          throw new Error('仅支持单位传动比的铰链位置执行器；无法使用当前步态控制此 XML 的执行器。');
        }
      }
      nextData = new this.mujoco.MjData(nextModel);
      this.mujoco.mj_forward(nextModel, nextData);
    } catch (err) {
      nextData?.delete();
      nextModel?.delete();
      throw err;
    }
    this.cleanupModel();
    this.model = nextModel;
    this.data = nextData;
    this.currentXml = xml;
    this.initialAngles = {};
    this.joints = [];
    this.freeDof = -1;
    this.freeQpos = -1;
    this.tipSite = -1;
    const nameAt = (start: number): string => {
      let name = '';
      for (let k = start; k >= 0 && k < this.model.names.length && this.model.names[k]; k++) name += String.fromCharCode(this.model.names[k]);
      return name;
    };
    this.initialKeyframe = -1;
    for (let key = 0; key < this.model.nkey; key++) {
      if (nameAt(this.model.name_keyadr[key]) === 'initial') this.initialKeyframe = key;
    }
    let highestTip = -1;
    for (let site = 0; site < this.model.nsite; site++) {
      const name = nameAt(this.model.name_siteadr[site]);
      const index = /^tip_\d+$/.test(name) ? Number(name.slice(4)) : -1;
      if (name === 'tip') { this.tipSite = site; break; }
      if (index > highestTip) { this.tipSite = site; highestTip = index; }
    }
    for (let j = 0; j < this.model.njnt; j++) {
      if (this.model.jnt_type[j] === 0 && this.model.jnt_bodyid[j] === 1) {
        this.freeDof = this.model.jnt_dofadr[j];
        this.freeQpos = this.model.jnt_qposadr[j];
      }
    }
    for (let a = 0; a < this.model.nu; a++) {
      const j = this.model.actuator_trnid[2 * a];
      const name = nameAt(this.model.name_jntadr[j]);
      this.joints.push({ name, actuator: a, qpos: this.model.jnt_qposadr[j], dof: this.model.jnt_dofadr[j],
        min: this.model.actuator_ctrlrange[2 * a] < this.model.actuator_ctrlrange[2 * a + 1] ? this.model.actuator_ctrlrange[2 * a] : -Math.PI, max: this.model.actuator_ctrlrange[2 * a] < this.model.actuator_ctrlrange[2 * a + 1] ? this.model.actuator_ctrlrange[2 * a + 1] : Math.PI });
    }
    this.isLoaded = true;
    return true;
  }

  public cleanupModel(): void {
    if (this.data) {
      this.data.delete();
      this.data = null;
    }
    if (this.model) {
      this.model.delete();
      this.model = null;
    }
    this.isLoaded = false;
  }

  public step(substeps = 4, gait?: GaitConfig): void {
    if (!this.isLoaded || !this.model || !this.data) return;
    for (let i = 0; i < substeps; i++) {
      if (gait) this.applyGaitControl(gait, this.data.time);
      this.mujoco.mj_step(this.model, this.data);
    }
  }

  public applyGaitControl(gait: GaitConfig, simTime: number): void {
    if (!this.isLoaded || !this.model || !this.data) return;

    const nu = this.model.nu; // number of actuators
    if (nu === 0) return;

    const f = gait.frequency * (gait.speed ?? 1.0);
    const omega = 2 * Math.PI * f;
    const amp = gait.amplitude;
    const phaseRad = ((gait.phaseLag ?? 60) * Math.PI) / 180;
    const steering = gait.steering ?? 0;

    for (const joint of this.joints) {
      const parsedIndex = Number(joint.name.replace('joint_', ''));
      const i = Number.isFinite(parsedIndex) ? parsedIndex : joint.actuator;
      let targetDeg = 0;

      if (gait.type === 'manual') {
        const key = joint.name;
        targetDeg = gait.manualAngles[key] ?? 0;
      } else if (gait.type === 'serpentine') {
        // Serpentine traveling wave: theta_i = A * sin(omega*t + i*phase) + steering
        targetDeg = amp * Math.sin(omega * simTime - i * phaseRad) + steering;
      } else if (gait.type === 'inchworm') {
        // Inchworm / caterpillar arching
        // Symmetric peristaltic wave
        const wave = Math.sin(omega * simTime - i * phaseRad);
        targetDeg = amp * wave;
      } else if (gait.type === 'rolling') {
        // Rolling loop gait: synchronized rotating offset
        targetDeg = amp * Math.sin(omega * simTime + i * phaseRad);
      } else if (gait.type === 'sidewind') {
        // Orthogonal 2-axis sinusoidal wave
        const isPitch = i % 2 === 0;
        const phaseShift = isPitch ? 0 : Math.PI / 2;
        targetDeg = amp * Math.sin(omega * simTime - (i / 2) * phaseRad + phaseShift);
      } else if (gait.type === 'trot') {
        // Quadruped trotting
        const legPhase = (i % 4) < 2 ? 0 : Math.PI;
        targetDeg = amp * Math.sin(omega * simTime + legPhase);
      }

      if (gait.type !== 'manual') targetDeg += gait.manualAngles[joint.name] ?? 0;
      this.data.ctrl[joint.actuator] = Math.max(joint.min, Math.min(joint.max, targetDeg * Math.PI / 180));
    }
  }

  public resetSimulation(config?: ZbotConfiguration): void {
    if (!this.isLoaded || !this.model || !this.data) return;
    if (config) this.initialAngles = Object.fromEntries(config.modules.map((m, i) => [`joint_${i}`, m.initialAngle ?? config.defaultGait.manualAngles[`joint_${i}`] ?? 0]));
    if (!Object.keys(this.initialAngles).length && this.initialKeyframe >= 0) {
      this.mujoco.mj_resetDataKeyframe(this.model, this.data, this.initialKeyframe);
    } else {
      this.mujoco.mj_resetData(this.model, this.data);
    }
    this.applyPose(this.initialAngles);
    // Lift a free root out of the floor using the compiled mesh bounds after posing.
    if (this.freeQpos >= 0) {
      let lowest = Infinity;
      for (let g = 0; g < this.model.ngeom; g++) {
        if (this.model.geom_bodyid[g] === 0 || !this.model.geom_contype[g]) continue;
        const mesh = this.model.geom_dataid[g];
        if (mesh < 0) continue;
        const start = this.model.mesh_vertadr[mesh], count = this.model.mesh_vertnum[mesh];
        for (let v = start; v < start + count; v++) {
          const z = this.data.geom_xpos[g * 3 + 2] + this.data.geom_xmat[g * 9 + 6] * this.model.mesh_vert[v * 3]
            + this.data.geom_xmat[g * 9 + 7] * this.model.mesh_vert[v * 3 + 1] + this.data.geom_xmat[g * 9 + 8] * this.model.mesh_vert[v * 3 + 2];
          lowest = Math.min(lowest, z);
        }
      }
      if (lowest < 0.003) this.data.qpos[this.freeQpos + 2] += 0.003 - lowest;
    }
    this.mujoco.mj_forward(this.model, this.data);
  }

  public applyPose(angles: Record<string, number>): void {
    if (!this.isLoaded) return;
    for (const joint of this.joints) {
      const value = angles[joint.name];
      if (!Number.isFinite(value)) continue;
      const radians = Math.max(joint.min, Math.min(joint.max, value * Math.PI / 180));
      this.data.qpos[joint.qpos] = radians;
      this.data.qvel[joint.dof] = 0;
      this.data.ctrl[joint.actuator] = radians;
    }
    this.mujoco.mj_forward(this.model, this.data);
  }

  public applyImpulse(fx: number, fy: number, fz: number): void {
    if (!this.isLoaded || !this.model || !this.data) return;
    if (this.freeDof >= 0) {
      this.data.qvel[this.freeDof + 0] += fx;
      this.data.qvel[this.freeDof + 1] += fy;
      this.data.qvel[this.freeDof + 2] += fz;
    }
  }

  public getTime(): number {
    return this.data ? this.data.time : 0;
  }

  public getActuatorCount(): number {
    return this.model ? this.model.nu : 0;
  }

  public getBodyTransforms(): BodyTransform[] {
    if (!this.isLoaded || !this.model || !this.data) return [];
    const nbody = this.model.nbody;
    const xpos = this.data.xpos;
    const xquat = this.data.xquat;

    const transforms: BodyTransform[] = [];
    for (let i = 0; i < nbody; i++) {
      // xpos: 3 per body
      const px = xpos[i * 3];
      const py = xpos[i * 3 + 1];
      const pz = xpos[i * 3 + 2];

      // xquat in mujoco: [w, x, y, z]
      const qw = xquat[i * 4];
      const qx = xquat[i * 4 + 1];
      const qy = xquat[i * 4 + 2];
      const qz = xquat[i * 4 + 3];

      transforms.push({
        name: `body_${i}`,
        pos: [px, py, pz],
        quat: [qx, qy, qz, qw], // Three.js format [x, y, z, w]
      });
    }

    return transforms;
  }

  public getMetrics(prevMetrics?: SimMetrics): SimMetrics {
    if (!this.isLoaded || !this.model || !this.data) {
      return {
        time: 0,
        fps: 60,
        rootPos: [0, 0, 0],
        rootVelocity: [0, 0, 0],
        speed: 0,
        totalDistance: 0,
        jointAngles: {},
        jointTorques: {},
      };
    }

    const t = this.data.time;
    const rootPos: [number, number, number] = [
      this.data.xpos[3] || 0,
      this.data.xpos[4] || 0,
      this.data.xpos[5] || 0,
    ];

    let vx = 0;
    let vy = 0;
    let vz = 0;
    if (this.freeDof >= 0) {
      vx = this.data.qvel[this.freeDof + 0];
      vy = this.data.qvel[this.freeDof + 1];
      vz = this.data.qvel[this.freeDof + 2];
    }
    const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);

    let totalDist = prevMetrics && t >= prevMetrics.time ? prevMetrics.totalDistance : 0;
    if (prevMetrics && t > prevMetrics.time && prevMetrics.rootPos) {
      const dx = rootPos[0] - prevMetrics.rootPos[0];
      const dy = rootPos[1] - prevMetrics.rootPos[1];
      const dz = rootPos[2] - prevMetrics.rootPos[2];
      const stepDist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (stepDist < 1.0) {
        totalDist += stepDist;
      }
    }

    const jointAngles: Record<string, number> = {};
    const jointTorques: Record<string, number> = {};
    for (const joint of this.joints) {
      jointAngles[joint.name] = this.data.qpos[joint.qpos] * 180 / Math.PI;
      jointTorques[joint.name] = this.data.qfrc_actuator[joint.dof] ?? 0;
    }
    const tip = this.tipSite >= 0 ? Array.from(this.data.site_xpos.slice(this.tipSite * 3, (this.tipSite + 1) * 3)) as [number, number, number] : undefined;

    return {
      time: t,
      fps: 60,
      rootPos,
      rootVelocity: [vx, vy, vz],
      speed,
      totalDistance: totalDist,
      jointAngles,
      jointTorques,
      endEffectorPos: tip,
      contactCount: this.data.ncon,
      centerOfMass: Array.from(this.data.subtree_com.slice(3, 6)) as [number, number, number],
    };
  }

  public getModel(): any {
    return this.model;
  }

  public getData(): any {
    return this.data;
  }

  public getCurrentXml(): string {
    return this.currentXml;
  }

  public isReady(): boolean {
    return this.isLoaded;
  }

  public destroy(): void {
    this.cleanupModel();
  }
}

// Singleton instance
export const mujocoEngine = new MujocoEngine();
