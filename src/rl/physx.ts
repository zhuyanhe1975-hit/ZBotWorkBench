import type { MujocoEngine } from '../mujoco/MujocoEngine';
import type { PolicyState } from './replay';

export interface PhysxModel {
  simulation?: {
    staticFriction?: number; dynamicFriction?: number; restitution?: number;
    stiffness?: number; damping?: number; maxForce?: number; maxVelocity?: number;
    contactObservations?: boolean;
    legacyObservations?: boolean;
    periodicObservations?: boolean;
    disableSelfCollision?: boolean;
    contactOffset?: number; restOffset?: number;
    bounceThresholdVelocity?: number; frictionOffsetThreshold?: number; frictionCorrelationDistance?: number;
    maxDepenetrationVelocity?: number; positionIterations?: number; velocityIterations?: number;
    footNames?: [string, string];
  };
  metadata?: { physicsDt?: number; controlDt?: number; [key: string]: unknown };
  key: string;
  rootBody: string;
  baseBody: string;
  jointNames: string[];
  displayJointNames?: string[];
  displayJointSigns?: number[];
  displayRootQuaternionOffset?: number[];
  defaultQ: number[];
  bodies: {
    name: string;
    mass: number;
    inertia: number[];
    com: number[];
    axes: number[];
    pos: number[];
    quat: number[];
  }[];
  joints: {
    name: string;
    a: string;
    b: string;
    p0: number[];
    p1: number[];
    q0: number[];
    q1: number[];
    hinge: boolean;
    axis?: string;
  }[];
  legacyBodyGroups?: string[][];
  hulls: { colliders: { bodyName: string; hulls: { vertices: number[][] }[] }[] };
}

export interface PhysxRenderBody {
  name: string;
  position: [number, number, number];
  quaternion: [number, number, number, number]; // wxyz
  hulls: number[][][];
}

export interface PhysxContactDiagnostic {
  position: [number, number, number];
  force: [number, number, number];
  magnitude: number;
}

export interface PhysxDebugForceLine {
  start: [number, number, number];
  end: [number, number, number];
  kind: 'normal' | 'friction' | 'resultant';
}

export interface PhysxDiagnostics {
  contacts: PhysxContactDiagnostic[];
  forceLines: PhysxDebugForceLine[];
  groundResultant?: PhysxDebugForceLine;
  centerOfMass: [number, number, number];
}

// PhysX permits only one foundation per WASM instance. Keep one SDK context for
// that instance; individual simulations own and release their scenes and actors.
const contexts = new WeakMap<object, { physics: any; scale: any; dispatcher: any; foundation: any; allocator: any; errors: any }>();

function context(P: any) {
  let result = contexts.get(P);
  if (!result) {
    const allocator = new P.PxDefaultAllocator();
    const errors = new P.PxDefaultErrorCallback();
    const foundation = P.CreateFoundation(P.PHYSICS_VERSION, allocator, errors);
    const scale = new P.PxTolerancesScale();
    const physics = P.CreatePhysics(P.PHYSICS_VERSION, foundation, scale);
    P.InitExtensions(physics);
    result = { allocator, errors, foundation, scale, physics, dispatcher: P.DefaultCpuDispatcherCreate(0) };
    contexts.set(P, result);
  }
  return result;
}

function multiply(a: number[], b: number[]): number[] {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

function rotate(q: number[], v: number[]): number[] {
  return multiply(multiply(q, [0, ...v]), [q[0], -q[1], -q[2], -q[3]]).slice(1);
}

function vector(v: any): number[] { return [v.x, v.y, v.z]; }
function quaternion(q: any): number[] { return [q.w, q.x, q.y, q.z]; }

/** Original 60 Hz PhysX dynamics; MuJoCo only supplies display geometry. */
export class PhysxSimulation {
  private scene: any = null;
  private articulation: any = null;
  private ground: any = null;
  private material: any = null;
  private links = new Map<string, any>();
  private hinges: any[] = [];
  private time = 0;
  private contactCallback: any = null;
  private contactFilter: any = null;
  private contactPoints: any = null;
  private robotActors = new Set<number>();
  private footActors = new Map<number, number>();
  private footForces = [[0, 0, 0], [0, 0, 0]];
  private contactDiagnostics: (PhysxContactDiagnostic & { external: boolean })[] = [];
  private footForceHistory = [Array(5).fill(0), Array(5).fill(0)];
  private footAirTimes = [0, 0];
  private debugForces = false;
  private disposed = false;
  private displayJoints: { qpos: number; dof: number; actuator: number; sign: number }[] = [];
  private rootQpos = -1;
  private rootDof = -1;
  private readonly physicsDt: number;
  private readonly controlDt: number;
  private readonly substeps: number;
  private readonly initialRootPosition?: number[];
  private readonly initialRootQuaternion?: number[];

  constructor(private P: any, private model: PhysxModel, private displayEngine: MujocoEngine | null,
    timing: { physicsDt?: number; controlDt?: number; rootPosition?: number[]; rootQuaternion?: number[] } = {}) {
    this.physicsDt = timing.physicsDt ?? model.metadata?.physicsDt ?? 1 / 60;
    this.controlDt = timing.controlDt ?? model.metadata?.controlDt ?? 1 / 30;
    const ratio = this.controlDt / this.physicsDt;
    if (!Number.isFinite(ratio) || this.physicsDt <= 0 || ratio < 1 || ratio > 1000 || Math.abs(ratio - Math.round(ratio)) > 1e-6) {
      throw new Error('PhysX 控制周期必须是物理周期的整数倍');
    }
    this.substeps = Math.round(ratio);
    const finite = (values: number[] | undefined, length: number) => values === undefined || values.length === length && values.every(Number.isFinite);
    if (!finite(timing.rootPosition, 3) || !finite(timing.rootQuaternion, 4)
      || timing.rootQuaternion && Math.abs(Math.hypot(...timing.rootQuaternion) - 1) > 1e-5) throw new Error('PhysX 初始根位姿无效');
    if (!finite(model.displayRootQuaternionOffset, 4)
      || model.displayRootQuaternionOffset && Math.abs(Math.hypot(...model.displayRootQuaternionOffset) - 1) > 1e-5) {
      throw new Error('PhysX 显示根坐标变换无效');
    }
    this.initialRootPosition = timing.rootPosition;
    this.initialRootQuaternion = timing.rootQuaternion;
    this.validateModel();
    const display = displayEngine?.getModel();
    if (displayEngine && (!display || !displayEngine.getData())) throw new Error('PhysX 回放缺少显示模型');
    if (display) {
    const nameAt = (address: number) => {
      let name = '';
      for (let i = address; i >= 0 && i < display.names.length && display.names[i]; i++) name += String.fromCharCode(display.names[i]);
      return name;
    };
    const root = Array.from(display.jnt_type as Int32Array).findIndex(type => type === 0);
    if (root < 0) throw new Error('PhysX 回放显示模型缺少自由根关节');
    this.rootQpos = display.jnt_qposadr[root];
    this.rootDof = display.jnt_dofadr[root];
    const displayNames = model.displayJointNames ?? model.jointNames;
    const displaySigns = model.displayJointSigns ?? model.jointNames.map(() => 1);
    if (displayNames.length !== model.jointNames.length || displaySigns.length !== model.jointNames.length
      || displaySigns.some(sign => sign !== 1 && sign !== -1)) throw new Error('PhysX 显示关节映射无效');
    this.displayJoints = displayNames.map((name, index) => {
      const joint = Array.from(display.name_jntadr as Int32Array).findIndex(address => nameAt(address) === name);
      const actuator = Array.from({ length: display.nu }, (_, i) => i).find(i => display.actuator_trnid[2 * i] === joint);
      if (joint < 0 || actuator === undefined) throw new Error(`显示模型缺少 PhysX 关节 ${name}`);
      return { qpos: display.jnt_qposadr[joint], dof: display.jnt_dofadr[joint], actuator, sign: displaySigns[index] };
    });
    }
    this.reset();
  }

  private validateModel(): void {
    const { model } = this;
    const finite = (v: number[], length: number) => Array.isArray(v) && v.length === length && v.every(Number.isFinite);
    if (!model || !Array.isArray(model.bodies) || !Array.isArray(model.joints) || !Array.isArray(model.jointNames)
      || !finite(model.defaultQ, model.jointNames.length) || !model.hulls?.colliders) throw new Error('PhysX 模型数据无效');
    for (const [name, value] of Object.entries(model.simulation ?? {})) {
      if (name === 'footNames') {
        if (!Array.isArray(value) || value.length !== 2 || value[0] === value[1]
          || value.some(foot => typeof foot !== 'string' || !model.bodies.some(body => body.name === foot))) {
          throw new Error('PhysX 足部传感器顺序无效');
        }
      } else if (name === 'contactObservations' || name === 'legacyObservations'
        || name === 'periodicObservations' || name === 'disableSelfCollision') {
        if (typeof value !== 'boolean') throw new Error('PhysX 接触观测配置无效');
      } else if (typeof value !== 'number' || !Number.isFinite(value) || value < 0
        || (name === 'maxVelocity' && value === 0) || (name === 'restitution' && value > 1)) {
        throw new Error('PhysX 仿真参数无效');
      }
    }
    const seen = new Set<string>();
    for (const body of model.bodies) {
      if (seen.has(body.name) || !Number.isFinite(body.mass) || body.mass <= 0 || !finite(body.inertia, 3)
        || body.inertia.some(v => v <= 0) || !finite(body.com, 3) || !finite(body.pos, 3)
        || !finite(body.axes, 4) || !finite(body.quat, 4)) throw new Error('PhysX 刚体数据无效');
      const incoming = model.joints.filter(j => j.b === body.name);
      if (body.name === model.rootBody ? incoming.length !== 0 : incoming.length !== 1 || !seen.has(incoming[0].a)) {
        throw new Error('PhysX 刚体必须按运动链顺序排列');
      }
      const colliders = model.hulls.colliders.filter(c => c.bodyName === body.name);
      if (colliders.length !== 1 || !colliders[0].hulls.length || colliders[0].hulls.some(h =>
        h.vertices.length < 4 || h.vertices.some(v => !finite(v, 3)))) throw new Error('PhysX 碰撞几何无效');
      seen.add(body.name);
    }
    if (!seen.has(model.rootBody) || !seen.has(model.baseBody) || new Set(model.jointNames).size !== model.jointNames.length
      || model.joints.filter(j => j.hinge).length !== model.jointNames.length) throw new Error('PhysX 关节映射无效');
    for (const joint of model.joints) {
      if (!seen.has(joint.a) || !seen.has(joint.b) || !finite(joint.p0, 3) || !finite(joint.p1, 3)
        || !finite(joint.q0, 4) || !finite(joint.q1, 4) || (joint.hinge && (!model.jointNames.includes(joint.name)
          || (joint.axis !== undefined && joint.axis !== 'Z')))) throw new Error('PhysX 关节数据无效');
    }
  }

  public reset(elapsedTime = 0, targets?: ArrayLike<number>): void {
    if (this.disposed) throw new Error('PhysX 仿真已释放');
    if (!Number.isFinite(elapsedTime) || elapsedTime < 0) throw new Error('PhysX 重置时间无效');
    if (targets && (targets.length !== this.model.jointNames.length || !Array.from(targets).every(Number.isFinite))) {
      throw new Error('PhysX 重置目标无效');
    }
    this.releaseScene();
    const P = this.P;
    const sdk = context(P);
    const temporary: any[] = [];
    const own = (object: any) => { temporary.push(object); return object; };
    const vec = (v: number[]) => own(new P.PxVec3(...v));
    const pose = (p: number[], q: number[] = [1, 0, 0, 0]) => own(new P.PxTransform(vec(p), own(new P.PxQuat(q[1], q[2], q[3], q[0]))));
    try {
      const desc = own(new P.PxSceneDesc(sdk.scale));
      desc.gravity = vec([0, 0, -9.81]);
      desc.cpuDispatcher = sdk.dispatcher;
      desc.filterShader = P.DefaultFilterShader();
      desc.solverType = P.PxSolverTypeEnum.eTGS;
      desc.flags.raise(P.PxSceneFlagEnum.eENABLE_PCM);
      desc.bounceThresholdVelocity = this.model.simulation?.bounceThresholdVelocity ?? 0.5;
      desc.frictionOffsetThreshold = this.model.simulation?.frictionOffsetThreshold ?? 0.04;
      desc.frictionCorrelationDistance = this.model.simulation?.frictionCorrelationDistance ?? 0.025;
      this.configureContacts(desc);
      this.scene = sdk.physics.createScene(desc);
      const settings = this.model.simulation;
      this.material = sdk.physics.createMaterial(settings?.staticFriction ?? 1, settings?.dynamicFriction ?? 1, settings?.restitution ?? 0);
      this.material.setFrictionCombineMode(P.PxCombineModeEnum.eMULTIPLY);
      this.material.setRestitutionCombineMode(P.PxCombineModeEnum.eMULTIPLY);
      const flags = own(new P.PxShapeFlags(P.PxShapeFlagEnum.eSCENE_QUERY_SHAPE | P.PxShapeFlagEnum.eSIMULATION_SHAPE));
      const filter = own(new P.PxFilterData(1, 1, 0, 0));
      const attachShape = (actor: any, geometry: any) => {
        const shape = sdk.physics.createShape(geometry, this.material, true, flags);
        try {
          shape.setSimulationFilterData(filter);
          if (settings?.contactOffset !== undefined) shape.setContactOffset(settings.contactOffset);
          if (settings?.restOffset !== undefined) shape.setRestOffset(settings.restOffset);
          if (!actor.attachShape(shape)) throw new Error('无法添加 PhysX 碰撞形状');
        } finally {
          shape.release(); // The actor retains its own shape reference.
        }
      };
      this.ground = sdk.physics.createRigidStatic(pose([0, 0, 0], [Math.SQRT1_2, 0, -Math.SQRT1_2, 0]));
      this.ground.setActorFlag(P.PxActorFlagEnum.eVISUALIZATION, true);
      attachShape(this.ground, own(new P.PxPlaneGeometry()));
      this.scene.addActor(this.ground);
      this.articulation = sdk.physics.createArticulationReducedCoordinate();
      this.articulation.setSolverIterationCounts(settings?.positionIterations ?? 4, settings?.velocityIterations ?? 0);
      this.articulation.setArticulationFlag(P.PxArticulationFlagEnum.eDISABLE_SELF_COLLISION, settings?.disableSelfCollision ?? false);
      this.articulation.setArticulationFlag(P.PxArticulationFlagEnum.eDRIVE_LIMITS_ARE_FORCES, true);
      const axis = P.PxArticulationAxisEnum.eTWIST;
      const cooking = own(new P.PxCookingParams(sdk.scale));
      const hinges = new Map<string, any>();
      const sourceRoot = this.model.bodies.find(body => body.name === this.model.rootBody)!;
      const targetRootPosition = this.initialRootPosition ?? sourceRoot.pos;
      const targetRootQuaternion = this.initialRootQuaternion ?? sourceRoot.quat;
      const frame = multiply(targetRootQuaternion, [sourceRoot.quat[0], -sourceRoot.quat[1], -sourceRoot.quat[2], -sourceRoot.quat[3]]);
      const initialPose = (body: PhysxModel['bodies'][number]) => {
        const offset = rotate(frame, body.pos.map((value, i) => value - sourceRoot.pos[i]));
        return pose(targetRootPosition.map((value, i) => value + offset[i]), multiply(frame, body.quat));
      };
      for (const body of this.model.bodies) {
        const incoming = this.model.joints.find(j => j.b === body.name);
        const parent = incoming ? this.links.get(incoming.a) : P.wrapPointer(0, P.PxArticulationLink);
        const link = this.articulation.createLink(parent, initialPose(body));
        link.setActorFlag(P.PxActorFlagEnum.eVISUALIZATION, true);
        this.links.set(body.name, link);
        this.robotActors.add(P.getPointer(link));
        const footIndex = (this.model.simulation?.footNames ?? ['foot_0', 'foot_1']).indexOf(body.name);
        if (footIndex >= 0) this.footActors.set(P.getPointer(link), footIndex);
        link.setMass(body.mass);
        link.setCMassLocalPose(pose(body.com, body.axes));
        link.setMassSpaceInertiaTensor(vec(body.inertia));
        link.setLinearDamping(0);
        link.setAngularDamping(0);
        link.setMaxDepenetrationVelocity(settings?.maxDepenetrationVelocity ?? 1);
        link.setMaxLinearVelocity(1000);
        link.setMaxAngularVelocity(1000);
        for (const hull of this.model.hulls.colliders.find(c => c.bodyName === body.name)!.hulls) {
          const vertices = new P.Vector_PxVec3();
          const meshDesc = new P.PxConvexMeshDesc();
          const convexFlags = new P.PxConvexFlags(P.PxConvexFlagEnum.eCOMPUTE_CONVEX);
          let mesh: any;
          let geometry: any;
          try {
            for (const point of hull.vertices) {
              const vertex = new P.PxVec3(...point);
              try { vertices.push_back(vertex); } finally { P.destroy(vertex); }
            }
            meshDesc.points.count = vertices.size();
            meshDesc.points.stride = 12;
            meshDesc.points.data = vertices.data();
            meshDesc.flags = convexFlags;
            mesh = P.CreateConvexMesh(cooking, meshDesc);
            if (!P.getPointer(mesh)) throw new Error('无法生成 PhysX 凸碰撞体');
            geometry = new P.PxConvexMeshGeometry(mesh);
            attachShape(link, geometry);
          } finally {
            if (geometry) P.destroy(geometry);
            if (mesh && P.getPointer(mesh)) mesh.release(); // Shapes retain the mesh.
            P.destroy(convexFlags);
            P.destroy(meshDesc);
            P.destroy(vertices);
          }
        }
        if (incoming) {
          const joint = link.getInboundJoint();
          joint.setJointType(incoming.hinge ? P.PxArticulationJointTypeEnum.eREVOLUTE : P.PxArticulationJointTypeEnum.eFIX);
          // PhysX reduced-coordinate hinges rotate about joint-frame X; USD uses Z.
          const frame = incoming.hinge ? [Math.SQRT1_2, 0, -Math.SQRT1_2, 0] : [1, 0, 0, 0];
          joint.setParentPose(pose(incoming.p0, multiply(incoming.q0, frame)));
          joint.setChildPose(pose(incoming.p1, multiply(incoming.q1, frame)));
          if (incoming.hinge) {
            joint.setMotion(axis, P.PxArticulationMotionEnum.eFREE);
            joint.setDriveParams(axis, own(new P.PxArticulationDrive(settings?.stiffness ?? 50, settings?.damping ?? 5, settings?.maxForce ?? 2000, P.PxArticulationDriveTypeEnum.eFORCE)));
            joint.setMaxJointVelocity(settings?.maxVelocity ?? 1000);
            joint.setFrictionCoefficient(0);
            joint.setArmature(axis, 0);
            const initial = this.model.defaultQ[this.model.jointNames.indexOf(incoming.name)];
            const target = targets?.[this.model.jointNames.indexOf(incoming.name)] ?? initial;
            joint.setJointPosition(axis, initial);
            joint.setDriveTarget(axis, target);
            hinges.set(incoming.name, joint);
          }
        }
      }
      this.hinges = this.model.jointNames.map(name => hinges.get(name));
      this.scene.addArticulation(this.articulation);
      this.applyDebugVisualization();
      this.articulation.updateKinematic(own(new P.PxArticulationKinematicFlags(
        P.PxArticulationKinematicFlagEnum.ePOSITION | P.PxArticulationKinematicFlagEnum.eVELOCITY)));
      this.time = elapsedTime;
      this.footForces = [[0, 0, 0], [0, 0, 0]];
      this.contactDiagnostics = [];
      this.footForceHistory = [Array(5).fill(0), Array(5).fill(0)];
      this.footAirTimes = [0, 0];
      this.displayEngine?.resetPolicyPose();
      this.synchronizeDisplay(targets ?? this.model.defaultQ);
    } catch (error) {
      this.releaseScene();
      throw error;
    } finally {
      for (const object of temporary.reverse()) P.destroy(object);
    }
  }

  public step(targets: ArrayLike<number>): void {
    this.requireScene();
    if (targets.length !== this.hinges.length || !Array.from(targets).every(Number.isFinite)) throw new Error('PhysX 关节目标无效');
    const axis = this.P.PxArticulationAxisEnum.eTWIST;
    this.hinges.forEach((joint, i) => joint.setDriveTarget(axis, targets[i]));
    this.contactDiagnostics = [];
    for (let i = 0; i < this.substeps; i++) {
      this.footForces[0].fill(0); this.footForces[1].fill(0);
      this.scene.simulate(this.physicsDt);
      if (!this.scene.fetchResults(true)) throw new Error('PhysX 仿真步进失败');
      if (this.model.simulation?.contactObservations) {
        for (let foot = 0; foot < 2; foot++) {
          this.footForceHistory[foot].shift();
          this.footForceHistory[foot].push(this.footForces[foot][2]);
          this.footAirTimes[foot] = Math.hypot(...this.footForces[foot]) > 1 ? 0 : this.footAirTimes[foot] + this.physicsDt;
        }
      }
    }
    this.time += this.controlDt;
    this.synchronizeDisplay(targets);
  }

  public state(): PolicyState {
    this.requireScene();
    const base = this.links.get(this.model.baseBody);
    // These WebIDL getters return shared borrowed value buffers, verified against
    // the runtime pointers. Copy immediately; never destroy them or their members.
    const q = quaternion(base.getGlobalPose().q);
    const angularVelocity = vector(base.getAngularVelocity());
    const linearVelocity = vector(base.getLinearVelocity());
    const basePosition = vector(base.getGlobalPose().p);
    const rootQuaternion = quaternion(this.links.get(this.model.rootBody).getGlobalPose().q);
    const axis = this.P.PxArticulationAxisEnum.eTWIST;
    const result: PolicyState = { quaternion: q, angularVelocity, linearVelocity, rootQuaternion, basePosition,
      footContactForces: this.footForceHistory.map(history => history.reduce((sum, force) => sum + force, 0) / 5),
      footAirTimes: [...this.footAirTimes], positions: this.hinges.map(j => j.getJointPosition(axis)),
      velocities: this.hinges.map(j => j.getJointVelocity(axis)) };
    if (!this.model.simulation?.legacyObservations && !this.model.simulation?.periodicObservations) return result;
    const bodyComPositions = this.model.bodies.map(body => {
      const pose = this.links.get(body.name).getGlobalPose();
      const position = vector(pose.p), offset = rotate(quaternion(pose.q), body.com);
      return position.map((value, i) => value + offset[i]);
    });
    const bodyOriginPositions = this.model.bodies.map(body => vector(this.links.get(body.name).getGlobalPose().p));
    const extremityNames = this.model.simulation?.footNames ?? [this.model.rootBody, this.model.bodies.at(-1)!.name];
    if (this.model.simulation?.periodicObservations) {
      const footIndices = extremityNames.map(name => this.model.bodies.findIndex(body => body.name === name));
      if (footIndices.some(index => index < 0)) throw new Error('周期行走回放缺少足端刚体');
      return { ...result, bodyComPositions, bodyMasses: this.model.bodies.map(body => body.mass),
        extremityPositions: footIndices.map(index => bodyOriginPositions[index]) as [number[], number[]],
        extremityContactForces: this.footForces.map(force => [...force]) as [number[], number[]] };
    }
    const exactBodyFrames = this.model.legacyBodyGroups !== undefined;
    const legacyMassPositions = exactBodyFrames ? bodyOriginPositions : bodyComPositions;
    const extremityPositions = extremityNames.map(name => legacyMassPositions[this.model.bodies.findIndex(body => body.name === name)]) as [number[], number[]];
    const bodyGroups: number[][] = this.model.legacyBodyGroups
      ? this.model.legacyBodyGroups.map(group => group.map(name => this.model.bodies.findIndex(body => body.name === name)))
      : [[0]];
    if (!this.model.legacyBodyGroups) {
      for (let i = 1; i < this.model.bodies.length - 1; i += 2) bodyGroups.push([i, i + 1]);
      bodyGroups.push([this.model.bodies.length - 1]);
    }
    if (bodyGroups.some(group => !group.length || group.some(index => index < 0))) throw new Error('旧 Isaac Gym 刚体分组无效');
    // Isaac Gym's rigid-body state tensor reports the actor-frame pose and the
    // PhysX linear velocity (at the COM). Dedicated legacy mappings need the
    // actor origin for position; fallback split-link models retain COM aggregation.
    const legacyBodies = bodyGroups.map(indices => {
      const mass = indices.reduce((sum, i) => sum + this.model.bodies[i].mass, 0);
      const weighted = (values: (i: number) => number[]) => [0, 1, 2].map(axis => indices.reduce((sum, i) => sum + values(i)[axis] * this.model.bodies[i].mass, 0) / mass);
      const link = this.links.get(this.model.bodies[indices[0]].name);
      return { mass, position: weighted(i => exactBodyFrames ? bodyOriginPositions[i] : bodyComPositions[i]), quaternion: quaternion(link.getGlobalPose().q),
        linearVelocity: weighted(i => vector(this.links.get(this.model.bodies[i].name).getLinearVelocity())),
        angularVelocity: weighted(i => vector(this.links.get(this.model.bodies[i].name).getAngularVelocity())) };
    });
    return { ...result, bodyComPositions: legacyMassPositions,
      bodyMasses: this.model.bodies.map(body => body.mass), extremityPositions,
      extremityContactForces: this.footForces.map(force => [...force]) as [number[], number[]], legacyBodies };
  }

  public getBasePosition(): number[] {
    this.requireScene();
    return vector(this.links.get(this.model.baseBody).getGlobalPose().p);
  }

  /** Read-only world poses and collision hulls for diagnostic rendering. */
  public getRenderBodies(): PhysxRenderBody[] {
    if (this.disposed) return [];
    this.requireScene();
    return this.model.bodies.map(body => {
      const transform = this.links.get(body.name).getGlobalPose();
      return {
        name: body.name,
        position: vector(transform.p) as [number, number, number],
        quaternion: quaternion(transform.q) as [number, number, number, number],
        hulls: this.model.hulls.colliders.find(collider => collider.bodyName === body.name)!.hulls
          .map(hull => hull.vertices),
      };
    });
  }

  /** Read-only PhysX contacts and mass-weighted world center for viewer diagnostics. */
  public getDiagnostics(): PhysxDiagnostics {
    if (this.disposed) return { contacts: [], forceLines: [], centerOfMass: [0, 0, 0] };
    this.requireScene();
    const totalMass = this.model.bodies.reduce((sum, body) => sum + body.mass, 0);
    const centerOfMass = [0, 1, 2].map(axis => this.model.bodies.reduce((sum, body) => {
      const pose = this.links.get(body.name).getGlobalPose();
      const offset = rotate(quaternion(pose.q), body.com);
      return sum + body.mass * (vector(pose.p)[axis] + offset[axis]);
    }, 0) / totalMass) as [number, number, number];
    const buffer = this.scene.getRenderBuffer();
    const helpers = this.P.NativeArrayHelpers.prototype;
    const forceLines: PhysxDebugForceLine[] = [];
    for (let index = 0; index < buffer.getNbLines(); index++) {
      const line = helpers.getDebugLineAt(buffer.getLines(), index);
      const start = vector(line.pos0) as [number, number, number];
      const end = vector(line.pos1) as [number, number, number];
      if (![...start, ...end].every(Number.isFinite)
        || Math.hypot(...end.map((value, axis) => value - start[axis])) < 1e-7) continue;
      // PhysX debug colors: contact impulse is bright red (0xffff0000),
      // friction impulse is dark red (0xff880000). We recolor in the viewer.
      forceLines.push({ start, end, kind: (line.color0 >>> 0) === 0xff880000 ? 'friction' : 'normal' });
    }
    const groundLines = forceLines.filter(line => Math.abs(line.start[2]) < .02);
    const groundNormals = groundLines.filter(line => line.kind === 'normal');
    const resultantVector = [0, 1, 2].map(axis => groundLines.reduce((sum, line) =>
      sum + line.end[axis] - line.start[axis], 0)) as [number, number, number];
    const normalWeights = groundNormals.map(line => Math.max(0, line.end[2] - line.start[2]));
    const totalNormalWeight = normalWeights.reduce((sum, value) => sum + value, 0);
    const centerOfPressure = totalNormalWeight > 0 ? [0, 1, 2].map(axis => groundNormals.reduce((sum, line, index) =>
      sum + line.start[axis] * normalWeights[index], 0) / totalNormalWeight) as [number, number, number] : undefined;
    const groundResultant = centerOfPressure && Math.hypot(...resultantVector) > 1e-7 ? {
      start: centerOfPressure,
      end: centerOfPressure.map((value, axis) => value + resultantVector[axis]) as [number, number, number],
      kind: 'resultant' as const,
    } : undefined;
    return {
      contacts: this.contactDiagnostics.map(contact => ({
        position: [...contact.position], force: [...contact.force], magnitude: contact.magnitude,
      })),
      forceLines,
      groundResultant,
      centerOfMass,
    };
  }

  /** Enable native PhysX contact/friction impulse debug lines only while requested. */
  public setDebugVisualization(showForces: boolean): void {
    this.debugForces = showForces;
    if (this.scene) this.applyDebugVisualization();
  }

  private applyDebugVisualization(): void {
    if (!this.scene) return;
    // PhysX 5.6.1 enum values: 9=eCONTACT_IMPULSE (legacy eCONTACT_FORCE alias),
    // 12=eFRICTION_IMPULSE. physx-js-webidl 2.7.3 omits the newer names only.
    this.scene.setVisualizationParameter(0, this.debugForces ? 1 : 0);
    this.scene.setVisualizationParameter(9, this.debugForces ? .2 : 0);
    this.scene.setVisualizationParameter(12, this.debugForces ? .2 : 0);
  }

  private synchronizeDisplay(targets: ArrayLike<number>): void {
    if (!this.displayEngine) return;
    const root = this.links.get(this.model.rootBody);
    const pose = root.getGlobalPose();
    const position = vector(pose.p);
    const q = quaternion(pose.q);
    const displayQ = this.model.displayRootQuaternionOffset ? multiply(q, this.model.displayRootQuaternionOffset) : q;
    const velocity = vector(root.getLinearVelocity());
    const omega = vector(root.getAngularVelocity());
    const com = rotate(q, this.model.bodies.find(b => b.name === this.model.rootBody)!.com);
    // PhysX reports COM linear velocity; MuJoCo's free joint uses body origin.
    const cross = [omega[1] * com[2] - omega[2] * com[1], omega[2] * com[0] - omega[0] * com[2], omega[0] * com[1] - omega[1] * com[0]];
    const localOmega = rotate([displayQ[0], -displayQ[1], -displayQ[2], -displayQ[3]], omega);
    const state = this.state();
    const values = [...position, ...q, ...velocity, ...omega, ...state.positions, ...state.velocities];
    if (!values.every(Number.isFinite)) throw new Error('PhysX 仿真产生无效状态');
    const data = this.displayEngine.getData();
    data.qpos.set([...position, ...displayQ], this.rootQpos);
    data.qvel.set([...velocity.map((v, i) => v - cross[i]), ...localOmega], this.rootDof);
    this.displayJoints.forEach((joint, i) => {
      data.qpos[joint.qpos] = joint.sign * state.positions[i];
      data.qvel[joint.dof] = joint.sign * state.velocities[i];
      data.ctrl[joint.actuator] = joint.sign * targets[i];
    });
    data.time = this.time;
    this.displayEngine.forward();
  }

  /** Contact impulses point from actor 1 to actor 0; divide by dt for world forces.
   * https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/_api_build/structPxContactPair.html
   */
  private configureContacts(desc: any): void {
    const P = this.P;
    this.contactPoints = new P.Vector_PxContactPairPoint(256);
    this.contactCallback = new P.PxSimulationEventCallbackImpl();
    for (const event of ['onConstraintBreak', 'onWake', 'onSleep', 'onTrigger']) this.contactCallback[event] = () => {};
    this.contactCallback.onContact = (headerPointer: number, pairsPointer: number, count: number) => {
      const header = P.wrapPointer(headerPointer, P.PxContactPairHeader);
      const actorPointers = [0, 1].map(index => P.getPointer(header.get_actors(index)));
      const feet = actorPointers.map(pointer => this.footActors.get(pointer));
      const robotPair = actorPointers.map(pointer => this.robotActors.has(pointer));
      const external = robotPair[0] !== robotPair[1];
      for (let pairIndex = 0; pairIndex < count; pairIndex++) {
        const pair = P.NativeArrayHelpers.prototype.getContactPairAt(pairsPointer, pairIndex);
        const n = pair.extractContacts(this.contactPoints.data(), this.contactPoints.size());
        for (let contact = 0; contact < n; contact++) {
          const point = this.contactPoints.at(contact);
          const impulse = vector(point.impulse);
          const normal = vector(point.normal);
          const direction = external && robotPair[1] ? -1 : 1;
          const force = impulse.map(value => direction * value / this.physicsDt) as [number, number, number];
          const position = vector(point.position) as [number, number, number];
          const magnitude = Math.hypot(...force);
          if (position.every(Number.isFinite) && force.every(Number.isFinite) && Number.isFinite(magnitude)) {
            const previous = this.contactDiagnostics.find(item => item.external === external
              && item.position.reduce((sum, value, axis) => sum + (value - position[axis]) ** 2, 0) < 1e-4);
            if (previous) {
              if (magnitude > previous.magnitude) Object.assign(previous, { position, force, magnitude });
            } else if (this.contactDiagnostics.length < 64) {
              this.contactDiagnostics.push({ position, force, magnitude, external });
            }
          }
          // Isaac ContactSensor reports net normal forces, excluding friction.
          const normalImpulse = impulse.reduce((sum, value, axis) => sum + value * normal[axis], 0);
          if (!this.model.simulation?.contactObservations) continue;
          for (let actor = 0; actor < 2; actor++) {
            const foot = feet[actor];
            if (foot === undefined) continue;
            for (let axis = 0; axis < 3; axis++) this.footForces[foot][axis] += normal[axis] * normalImpulse / this.physicsDt * (actor === 0 ? 1 : -1);
          }
        }
      }
    };
    desc.simulationEventCallback = this.contactCallback;
    this.contactFilter = new P.PassThroughFilterShaderImpl();
    this.contactFilter.filterShader = () => {
      this.contactFilter.outputPairFlags = P.PxPairFlagEnum.eCONTACT_DEFAULT | P.PxPairFlagEnum.eNOTIFY_TOUCH_FOUND
        | P.PxPairFlagEnum.eNOTIFY_TOUCH_PERSISTS | P.PxPairFlagEnum.eNOTIFY_CONTACT_POINTS;
      return P.PxFilterFlagEnum.eDEFAULT;
    };
    P.PxTopLevelFunctions.prototype.setupPassThroughFilterShader(desc, this.contactFilter);
  }

  private requireScene(): void {
    if (this.disposed || !this.scene) throw new Error('PhysX 仿真未就绪');
  }

  private releaseScene(): void {
    if (this.articulation) { this.articulation.release(); this.articulation = null; }
    if (this.ground) { this.ground.release(); this.ground = null; }
    if (this.scene) { this.scene.release(); this.scene = null; }
    if (this.material) { this.material.release(); this.material = null; }
    if (this.contactCallback) { this.P.destroy(this.contactCallback); this.contactCallback = null; }
    if (this.contactFilter) { this.P.destroy(this.contactFilter); this.contactFilter = null; }
    if (this.contactPoints) { this.P.destroy(this.contactPoints); this.contactPoints = null; }
    this.footActors.clear();
    this.robotActors.clear();
    this.links.clear();
    this.hinges = [];
  }

  public dispose(): void {
    this.releaseScene();
    this.disposed = true;
  }
}
