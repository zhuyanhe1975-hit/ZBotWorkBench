import type { LegacyObservation, ReplayProfile } from './profiles';
import { RUN_SH_BOOTSTRAP } from './runShBootstrap';

interface LegacyCheckpoint {
  file: string;
  input: number;
  output: number;
  sourceTask?: string;
}

const checkpoints: LegacyCheckpoint[] = [
  { file: 'last_ZBotFootDown_ep_2500_rew_970.54364.pth', input: 17, output: 24, sourceTask: 'ZBotFootDown.py' },
  { file: 'zbot_bipedal_forward.pth', input: 25, output: 18 },
  { file: 'zbot_bipedal_running.pth', input: 31, output: 18 },
  { file: 'ZBotBipedalWalking_0.2pi.pth', input: 34, output: 6 },
  { file: 'ZBotBipedalWalking_0.45pi.pth', input: 32, output: 24 },
  { file: 'ZBotBipedalWalking_20250514.pth', input: 36, output: 6 },
  { file: 'ZBotBipedalWalking_20250527.pth', input: 36, output: 6, sourceTask: 'ZBotBipedalWalking 20250527.py' },
  { file: 'ZBotBipedalWalking.pth', input: 36, output: 6 },
  { file: 'ZBotBipedalWalking_BigFoot.pth', input: 38, output: 6, sourceTask: 'ZBotBipedalWalking_BigFoot.py' },
  { file: 'ZBotBipedalWalking_BigFoot_8DOF.pth', input: 42, output: 8, sourceTask: 'ZBotBipedalWalking_BigFoot_8DOF.py' },
  { file: 'ZBotBipedalWalking_BigFoot_8DOF_Best.pth', input: 42, output: 8, sourceTask: 'ZBotBipedalWalking_BigFoot_8DOF_Best.py' },
  { file: 'ZBotBipedalWalking_BigFoot_8DOF_R1.pth', input: 42, output: 32, sourceTask: 'ZBotBipedalWalking_BigFoot_8DOF_R1.py' },
  { file: 'ZBotBipedalWalking_BigFoot_Best.pth', input: 38, output: 24 },
  { file: 'ZBotBipedalWalking_BigFoot_R1.pth', input: 38, output: 24, sourceTask: 'ZBotBipedalWalking_BigFoot_R1.py' },
  { file: 'ZBotBipedalWalking_BigFoot_R2.pth', input: 38, output: 24, sourceTask: 'ZBotBipedalWalking_BigFoot_R2.py' },
  { file: 'ZBotBipedalWalking_BigFoot_R3.pth', input: 38, output: 24 },
  { file: 'ZBotBipedalWalking_BigFoot_Rotate.pth', input: 38, output: 24 },
  { file: 'ZBotBipedalWalking_BigFoot_VerySlow.pth', input: 32, output: 24 },
  { file: 'ZBotBipedalWalking_ChangeSpeed.pth', input: 32, output: 24, sourceTask: 'ZBotBipedalWalking_ChangeSpeed.py' },
  { file: 'ZBotBipedalWalking_ChangeSpeedLimit.pth', input: 32, output: 24, sourceTask: 'ZBotBipedalWalking_ChangeSpeedLimit.py' },
  { file: 'ZBotBipedalWalking_ChangeSpeedLimit_v1.pth', input: 32, output: 24, sourceTask: 'ZBotBipedalWalking_ChangeSpeedLimit_v1.py' },
  { file: 'ZBotBipedalWalking_ChangeSpeedLimit_v2.pth', input: 32, output: 24, sourceTask: 'ZBotBipedalWalking_ChangeSpeedLimit_v2.py' },
  { file: 'ZBotBipedalWalking_ChangeSpeedLimit_v2_new.pth', input: 32, output: 24, sourceTask: 'ZBotBipedalWalking_ChangeSpeedLimit_v2.py' },
  { file: 'ZBotBipedalWalking_Side.pth', input: 31, output: 18, sourceTask: 'ZBotBipedalWalking_Side.py' },
  { file: 'ZBotBipedalWalking_Side_BigFoot_Best.pth', input: 40, output: 24 },
  { file: 'ZBotBipedalWalking_Side_Slow.pth', input: 31, output: 18, sourceTask: 'ZBotBipedalWalking_Side_Slow.py' },
  { file: 'ZBotFootDown.pth', input: 17, output: 24, sourceTask: 'ZBotFootDown.py' },
  { file: 'ZBotFootDownBack.pth', input: 17, output: 24 },
  { file: 'ZBotFootDownBack1.pth', input: 17, output: 24 },
  { file: 'ZBotFootDown_8DOF.pth', input: 21, output: 32, sourceTask: 'ZBotFootDown_8DOF.py' },
  { file: 'ZBotSingleLeg.pth', input: 26, output: 6, sourceTask: 'ZBotSingleLeg.py' },
  { file: 'ZBotSnakeAndBipedal.pth', input: 29, output: 6, sourceTask: 'ZBotSnakeAndBipedal.py' },
  { file: 'ZBotSnakeAndBipedal_R1.pth', input: 32, output: 24 },
  { file: 'ZBotSnakeAndBipedal_cpg.pth', input: 29, output: 6 },
  { file: 'ZBotSnakeAndBipedal_pd.pth', input: 29, output: 6, sourceTask: 'ZBotSnakeAndBipedal_pd.py' },
  { file: 'ZBotStandUp.pth', input: 31, output: 24, sourceTask: 'ZBotStandUp.py' },
  { file: 'ZBotStandUpAndWalking.pth', input: 31, output: 24 },
  { file: 'zbot_bipedal_walking_R2.pth', input: 31, output: 18 },
  { file: 'zbot_forward_moving.pth', input: 25, output: 18, sourceTask: 'ZBotForwardMoving.py' },
  { file: 'zbot_rolling.pth', input: 25, output: 18, sourceTask: 'ZBotRolling.py' },
  { file: 'zbot_side_moving.pth', input: 25, output: 18, sourceTask: 'ZBotSideMoving.py' },
  { file: 'zbot_standup.pth', input: 28, output: 18 },
  { file: 'zbot_standup_R1.pth', input: 28, output: 18 },
  { file: 'zbot_standup_R2.pth', input: 25, output: 18 },
  { file: 'zbot_standup_back.pth', input: 25, output: 18 },
];

const observationBySize: Record<number, LegacyObservation> = {
  17: 'footdown', 21: 'footdown', 25: 'body-joints', 26: 'speed-body-joints',
  28: 'body-com-joints', 29: 'command-body-com-joints', 31: 'posture', 32: 'scalar-posture',
  34: 'body-com-joints-contact', 36: 'kinematic-contact', 38: 'scalar-posture-contact',
  40: 'scalar-posture-contact-step', 42: 'scalar-posture-contact',
};
const bipedalBootstrapVariants: Record<string, number[][]> = {
  'ZBotBipedalWalking_20250514.pth': [
    [.5, -.00013866114022675902, -.00021439138799905777, .31801432371139526, .006104129366576672,
      -.00040135206654667854, .9999812841415405, -.0008831900777295232, -.008312852121889591,
      -.0009878737619146705, -.01270148903131485, .3528191149234772, .12268897145986557,
      -.0001011933054542169, -.00021213656873442233, .3179720938205719, .00035896810004487634,
      .00012575559958349913, -.0993126928806305, -.26642537117004395, .6429859399795532,
      -.6429859399795532, .26642537117004395, .0993126928806305, 0, 0, 0, 0, 0, 0,
      3.7742061067547183e-6, 0, 28.09049415588379, 0, 0, 0],
    [.5, -.0009449461358599365, .025105079635977745, .25417211651802063, -.9999637603759766,
      .0004398226737976074, .008512556552886963, -.10024887323379517, .04497389867901802,
      .022124145179986954, -.7421886920928955, .46226751804351807, -.21298445761203766,
      -.0025768636260181665, .007312051951885223, .14581042528152466, .05228210613131523,
      .05947434902191162, -.0993126928806305, -.26642537117004395, .6429859399795532,
      -.6429859399795532, .26642537117004395, .0993126928806305, 0, 0, 0, 0, 0, 0,
      4.58807608083589e-6, 0, 34.147926330566406, 3.2782363490468924e-9, 0, .024399111047387123],
  ],
  'ZBotBipedalWalking_20250527.pth': [
    [.5, -9.639775817049667e-5, -.0001835450530052185, .3180183470249176, .00283109862357378,
      -.000629343674518168, .999995768070221, -.006991817150264978, -.0052185580134391785,
      -.0005104056326672435, .004497903399169445, .10611940175294876, -.5653536319732666,
      -3.005560029123444e-5, -.0001837143354350701, .31797587871551514, .00025277971872128546,
      .0005032102926634252, -.0993126928806305, -.26642537117004395, .6429859399795532,
      -.6429859399795532, .26642537117004395, .0993126928806305, 0, 0, 0, 0, 0, 0,
      3.7886766222072765e-6, 0, 28.198196411132812, 0, 0, 0],
    [.5, -.0010407294612377882, .02524500899016857, .2544492781162262, -.9999362826347351,
      .001988828182220459, .011112749576568604, -.1079174354672432, .06710528582334518,
      .05669810622930527, -1.2460414171218872, .7432849407196045, -.3395459055900574,
      -.002952233189716935, .007351835258305073, .1460653692483902, .05156194418668747,
      .05916301906108856, -.0993126928806305, -.26642537117004395, .6429859399795532,
      -.6429859399795532, .26642537117004395, .0993126928806305, 0, 0, 0, 0, 0, 0,
      5.393028914113529e-6, 0, 40.13899230957031, 0, 0, 0],
  ],
};
const sixJoints = Array.from({ length: 6 }, (_, i) => `joint${i + 1}`);
const eightJoints = Array.from({ length: 8 }, (_, i) => `joint${i}`);
const sixWalking = [.312, .837, -2.02, 2.02, -.837, -.312];
const eightWalking = [0, ...sixWalking, 0];
const sixReference = [0, Math.PI / 4, -Math.PI / 2, Math.PI / 2, -Math.PI / 4, 0];
const eightBigFoot = [-.520365, -.283715, 1.34606, -1.23715, 1.23219, -1.37363, .270398, .54471];
const sixNativeWalking = [-.312, -.837, 2.02, -2.02, .837, .312];
const sixNativeBent = [0, -Math.PI / 4, Math.PI / 2, -Math.PI / 2, Math.PI / 4, 0];
const eightNativeBigFoot = [.520365, .283715, -1.34606, 1.23715, -1.23219, 1.37363, -.270398, -.54471];
const safeName = (file: string) => file.replace(/\.pth$/, '').replace(/[^A-Za-z0-9_.-]+/g, '-');

function defaultAngles(checkpoint: LegacyCheckpoint, joints: number): number[] {
  if (/FootDown|StandUp|SnakeAndBipedal/i.test(checkpoint.file)) return Array(joints).fill(0);
  if (/SingleLeg/i.test(checkpoint.file) || /^zbot_/.test(checkpoint.file)) return Array(joints).fill(0);
  if (/BigFoot_8DOF/.test(checkpoint.file)) return eightBigFoot;
  if (/BigFoot|ChangeSpeed|_Side|0\.[24]5?pi/.test(checkpoint.file)) return sixReference;
  return joints === 8 ? eightWalking : sixWalking;
}

function rootPose(checkpoint: LegacyCheckpoint): { position: [number, number, number]; quaternion: [number, number, number, number] } {
  if (/^zbot_bipedal_/.test(checkpoint.file)) {
    return { position: [.1, 0, 0], quaternion: [1, 0, 0, 0] };
  }
  if (/^zbot_standup/.test(checkpoint.file)) {
    return { position: [-.318, 0, .053], quaternion: [Math.SQRT1_2, 0, Math.SQRT1_2, 0] };
  }
  if (/^zbot_(?:forward_|side_|rolling)/.test(checkpoint.file)) {
    return { position: [.35, 0, .053], quaternion: [.5, .5, -.5, -.5] };
  }
  if (/StandUp|SnakeAndBipedal/i.test(checkpoint.file)) {
    return { position: [-.318, 0, .053], quaternion: [Math.SQRT1_2, 0, Math.SQRT1_2, 0] };
  }
  if (/FootDown|SingleLeg/i.test(checkpoint.file)) return { position: [0, 0, 0], quaternion: [1, 0, 0, 0] };
  if (checkpoint.file === 'ZBotBipedalWalking_BigFoot_R3.pth') {
    return { position: [-.05, 0, .05], quaternion: [Math.cos(Math.PI / 8), 0, 0, Math.sin(Math.PI / 8)] };
  }
  if (/BigFoot_8DOF/.test(checkpoint.file)) return { position: [-.1, 0, .02], quaternion: [1, 0, 0, 0] };
  if (/BigFoot/.test(checkpoint.file)) return { position: [.1, 0, .02], quaternion: [1, 0, 0, 0] };
  return { position: [checkpoint.input === 36 ? .06 : .1, 0, 0], quaternion: [1, 0, 0, 0] };
}

/**
 * The first bulk import used display-model sign heuristics for every checkpoint.
 * That makes the network load, but it does not reproduce the reset pose: the
 * archived R1/BigFoot assets use their native joint coordinates. Keep the
 * source-backed families on the exported native models and native coordinates.
 */
function applyNativeInitialPose(result: ReplayProfile, checkpoint: LegacyCheckpoint): void {
  const file = checkpoint.file;
  if (/^zbot_bipedal_/.test(file)) {
    result.model = 'legacy_zbot_6dof_bipedal_walking';
    result.jointNames = Array.from({ length: 6 }, (_, index) => `joint_${index}`);
    result.jointSigns = Array(6).fill(1);
    result.defaultAngles = sixNativeBent;
    result.physxSimulation = { ...result.physxSimulation, contactObservations: false,
      footNames: ['body_0', 'body_6'] };
    return;
  }
  if (/^zbot_standup/.test(file)) {
    // The retained Python 3.7 bytecode identifies ZBot_R1.xml for the 28-D
    // stand-up pair. The later 25-D variants use the older ZBot.xml family.
    result.model = checkpoint.input === 28 ? 'legacy_zbot_6dof_bipedal_walking' : 'legacy_zbot_rolling';
    result.jointNames = Array.from({ length: 6 }, (_, index) => `joint_${index}`);
    result.jointSigns = Array(6).fill(1);
    result.defaultAngles = Array(6).fill(0);
    result.physxSimulation = { disableSelfCollision: false, legacyObservations: true, contactObservations: false };
    return;
  }
  if (!file.startsWith('ZBot') && !file.startsWith('last_ZBot')) return;

  const joints = result.jointNames.length;
  if (/BigFoot_8DOF/.test(file)) {
    result.model = 'legacy_zbot_bigfoot_8dof';
    result.jointNames = Array.from({ length: 8 }, (_, index) => `joint_${index}`);
    result.jointSigns = Array(8).fill(1);
    result.defaultAngles = eightNativeBigFoot;
    result.physxSimulation = { ...result.physxSimulation, footNames: ['body_0', 'body_8'] };
    return;
  }
  if (/BigFoot/.test(file)) {
    result.model = 'legacy_zbot_bigfoot';
    result.jointNames = Array.from({ length: 6 }, (_, index) => `joint_${index}`);
    result.jointSigns = Array(6).fill(1);
    result.defaultAngles = file === 'ZBotBipedalWalking_BigFoot_R3.pth'
      ? [.4, .8, -2, 2, -.8, -.4]
      : sixNativeBent;
    result.physxSimulation = { ...result.physxSimulation, footNames: ['body_0', 'body_6'] };
    return;
  }
  // ZBot_8dof.xml is not the BigFoot 8DOF asset. Keep its existing display
  // model until a native capture/export exists, but its reset joints are zero.
  if (joints === 8) {
    result.defaultAngles = Array(8).fill(0);
    return;
  }

  result.model = 'legacy_zbot_6dof_bipedal_walking';
  result.jointNames = Array.from({ length: 6 }, (_, index) => `joint_${index}`);
  result.jointSigns = Array(6).fill(1);
  result.physxSimulation = { ...result.physxSimulation, footNames: ['body_0', 'body_6'] };
  if (/FootDown|StandUp|SnakeAndBipedal|SingleLeg/.test(file)) result.defaultAngles = Array(6).fill(0);
  else if (/^ZBotBipedalWalking(?:_20250514|_20250527)?\.pth$/.test(file)) result.defaultAngles = sixNativeWalking;
  else result.defaultAngles = sixNativeBent;
}

function controller(checkpoint: LegacyCheckpoint, joints: number) {
  const factor = checkpoint.output / joints;
  const usesJointSpeed = ['speed-body-joints', 'scalar-posture', 'scalar-posture-contact',
    'scalar-posture-contact-step', 'kinematic-contact'].includes(observationBySize[checkpoint.input]);
  if (factor === 1) {
    const scale = /SingleLeg/.test(checkpoint.file) ? 4 * Math.PI
      : /SnakeAndBipedal_pd/.test(checkpoint.file) ? 5 * Math.PI
        : /BigFoot_8DOF_Best/.test(checkpoint.file) ? 3 * Math.PI : 2 * Math.PI;
    return { kind: 'direct' as const, velocityScale: scale, usesJointSpeed };
  }
  if (factor === 3) return { kind: 'cpg3' as const, velocityScale: 2 * Math.PI, omegaScale: 2 * Math.PI, usesJointSpeed };
  const scale = /FootDown_8DOF/.test(checkpoint.file) ? .2 * Math.PI
    : /FootDown/.test(checkpoint.file) ? .5 * Math.PI
      : /BigFoot/.test(checkpoint.file) ? 3 * Math.PI
        : /StandUp/.test(checkpoint.file) ? 5 * Math.PI : 2 * Math.PI;
  return { kind: 'cpg4' as const, velocityScale: scale, omegaScale: Math.PI, omegaBias: 1, usesJointSpeed };
}

function deltaLimit(checkpoint: LegacyCheckpoint): number {
  if (/FootDown/i.test(checkpoint.file)) return .75 * Math.PI;
  if (/SingleLeg|StandUp/i.test(checkpoint.file)) return .5 * Math.PI;
  if (/SnakeAndBipedal/i.test(checkpoint.file) && !/_pd/i.test(checkpoint.file)) return .5 * Math.PI;
  if (/^zbot_(?:bipedal_|forward_|side_|rolling)/.test(checkpoint.file)) return .5 * Math.PI;
  return Math.PI;
}

function profile(checkpoint: LegacyCheckpoint): ReplayProfile {
  const joints = checkpoint.output % 8 === 0 && (checkpoint.output === 8 || checkpoint.output === 32) ? 8 : 6;
  const name = safeName(checkpoint.file);
  const exact = checkpoint.sourceTask !== undefined;
  const initialRoot = rootPose(checkpoint);
  const result: ReplayProfile = {
    id: `IsaacGym-${name}`,
    label: `旧 Isaac Gym · ${checkpoint.file.replace(/\.pth$/, '').replaceAll('_', ' ')}`,
    model: joints === 8 ? 'zbot_8s_human' : 'zbot_6s_new',
    checkpoint: `${name}.pt`, observation: 'isaacgym', legacyObservation: observationBySize[checkpoint.input],
    inputSize: checkpoint.input, policyOutputSize: checkpoint.output,
    jointNames: joints === 8 ? eightJoints : sixJoints, defaultAngles: defaultAngles(checkpoint, joints),
    jointSigns: Array(joints).fill(-1), actionTransform: 'clamp', deltaLimit: deltaLimit(checkpoint),
    legacyController: controller(checkpoint, joints), legacyCommand: 1,
    initialRootPosition: initialRoot.position, initialRootQuaternion: initialRoot.quaternion,
    controlDt: 1 / 60, physicsDt: 1 / 60, jointSpeedLimit: .5, mujocoCompatible: false,
    physxSimulation: { stiffness: 10, damping: 1, contactObservations: true, legacyObservations: true, footNames: ['foot_0', 'foot_1'] },
    origin: 'isaacgym', motion: 'walking',
    note: `旧 Isaac Gym / rl_games 迁移回放。${exact ? `观测结构依据归档任务 ${checkpoint.sourceTask}` : '归档中缺少唯一同名任务源码，观测按同维度任务族恢复'}；网络推理保持原归一化与动作裁剪，当前 PhysX WASM 仅用于查看执行效果，不代表复现原运动质量。`,
  };
  applyNativeInitialPose(result, checkpoint);
  if (/^ZBotBipedalWalking(?:_20250514|_20250527)?\.pth$/.test(checkpoint.file)) {
    result.model = 'legacy_zbot_6dof_bipedal_walking';
    result.jointNames = Array.from({ length: 6 }, (_, index) => `joint_${index}`);
    result.defaultAngles = [-.312, -.837, 2.02, -2.02, .837, .312];
    result.jointSigns = Array(6).fill(1);
    result.controlDt = result.physicsDt = .0166;
    result.jointSpeedLimit = .5;
    result.legacyController = { kind: 'direct', velocityScale: 2 * Math.PI, usesJointSpeed: true };
    result.initialRootPosition = undefined;
    result.initialRootQuaternion = undefined;
    result.zeroInitialObservation = true;
    result.bootstrapResetTargets = 'defaults';
    result.legacyAutoReset = { fallingBaseHeight: .2, minimumExtremityDistance: .1 };
    result.bootstrapResetObservations = [
      [.5, -.00037846327177248895, -.0001953430473804474, .3180133104324341, -.0006006656913086772,
        -.0005215419223532081, .9999997019767761, -.027610192075371742, -.006576187908649445,
        -.0009731517639011145, -.004395860247313976, -.06447679549455643, -.5166895985603333,
        -7.864942017477006e-5, -.00019853602861985564, .31797102093696594, .00016593237523920834,
        .0005138057167641819, -.0993126928806305, -.26642537117004395, .6429859399795532,
        -.6429859399795532, .26642537117004395, .0993126928806305, 0, 0, 0, 0, 0, 0,
        3.776902303798124e-6, 0, 28.110559463500977, 0, 0, 0],
      [.5, -.0009303158149123192, .025200936943292618, .2543160319328308, -.9999505281448364,
        .0007517337799072266, .009906649589538574, -.09973633289337158, .05972474440932274,
        .034611910581588745, -1.3764921426773071, .567193865776062, -.2075999528169632,
        -.002701787743717432, .007321567740291357, .14589636027812958, .051450215280056,
        .058917734771966934, -.0993126928806305, -.26642537117004395, .6429859399795532,
        -.6429859399795532, .26642537117004395, .0993126928806305, 0, 0, 0, 0, 0, 0,
        4.763342076330446e-6, 0, 35.452388763427734, 1.4015870242189976e-8, 0, .10431669652462006],
    ];
    if (bipedalBootstrapVariants[checkpoint.file]) result.bootstrapResetObservations = bipedalBootstrapVariants[checkpoint.file];
    result.physxSimulation = undefined;
    result.note = '旧 Isaac Gym 6DOF BipedalWalking 专用对齐模型：ZBot_R1 七刚体、原生质量/COM/惯量、36 维观测、0.0166 s 直连速度控制及双启动重置时序；使用原生 golden trace 验证。';
  }
  if (RUN_SH_BOOTSTRAP[checkpoint.file]) {
    result.jointNames = Array.from({ length: joints }, (_, index) => `joint_${index}`);
    result.jointSigns = Array(joints).fill(1);
    result.controlDt = result.physicsDt = .0166;
    result.initialRootPosition = undefined;
    result.initialRootQuaternion = undefined;
    result.zeroInitialObservation = true;
    result.bootstrapResetTargets = 'defaults';
    result.bootstrapResetObservations = RUN_SH_BOOTSTRAP[checkpoint.file];
    result.physxSimulation = { disableSelfCollision: false };
    result.note = `旧 Isaac Gym run.sh 原生回放专项对齐：${checkpoint.sourceTask ?? checkpoint.file}，使用归档 MJCF、原生物理属性、控制器与启动 reset trace。`;
    switch (checkpoint.file) {
      case 'zbot_side_moving.pth':
      case 'zbot_forward_moving.pth':
        result.model = 'legacy_zbot_rolling';
        result.defaultAngles = Array(6).fill(0);
        result.legacyBodyIndex = 4;
        result.legacyController = { kind: 'cpg3', velocityScale: 2 * Math.PI, omegaScale: 2 * Math.PI, timeOffsetSteps: 1 };
        result.legacyAutoReset = { maximumAbsFirstExtremityY: .4 };
        break;
      case 'ZBotStandUp.pth':
        result.model = 'legacy_zbot_6dof_bipedal_walking';
        result.defaultAngles = Array(6).fill(0);
        result.legacyBodyIndex = 4;
        result.legacyController = { kind: 'cpg4', velocityScale: 5 * Math.PI, omegaScale: Math.PI, omegaBias: 1.1,
          timeOffsetSteps: 1, loopDecayRate: .0002, loopDecayMinimum: .15 };
        result.legacyForcedResetSteps = [2];
        result.initialRootPosition = [-.3179999887943268, -2.2572465240955353e-5, .05029675364494324];
        result.initialRootQuaternion = [.7071067690849304, -9.425575875354752e-10, .7071067690849304, 7.617819264282844e-9];
        result.legacyAutoReset = { fallingBaseHeight: .2 };
        break;
      case 'last_ZBotFootDown_ep_2500_rew_970.54364.pth':
        result.model = 'legacy_zbot_6dof_bipedal_walking';
        result.defaultAngles = Array(6).fill(0);
        result.legacyController = { kind: 'cpg4', velocityScale: .5 * Math.PI, omegaScale: Math.PI, omegaBias: 1.1, timeOffsetSteps: 1 };
        result.initialRootPosition = [-9.411918290425092e-5, -1.7239712178707123e-5, 9.704381227493286e-6];
        result.initialRootQuaternion = [.9999994039535522, .00011714117863448337, -.00022578664356842637, -.0011052112095057964];
        break;
      case 'ZBotSingleLeg.pth':
        result.model = 'legacy_zbot_single_leg';
        result.displayModel = 'zbot_6s_new';
        result.displayJointNames = Array.from({ length: 6 }, (_, index) => `joint${index + 1}`);
        result.displayJointSigns = Array(6).fill(-1);
        result.displayRootQuaternionOffset = [Math.SQRT1_2, 0, 0, Math.SQRT1_2];
        // PhysX 5 reduced-coordinate force drives are substantially softer than
        // the archived GPU PhysX positional drive for this vertical chain.
        result.physxSimulation = { disableSelfCollision: false, stiffness: 130, damping: 4 };
        result.defaultAngles = Array(6).fill(0);
        result.jointSpeedLimit = 1;
        result.legacyController = { kind: 'direct', velocityScale: 4 * Math.PI, usesJointSpeed: true };
        result.legacyAutoReset = { minimumLastExtremityHeight: .4 };
        break;
      case 'ZBotSnakeAndBipedal.pth':
        result.model = 'legacy_zbot_6dof_bipedal_walking';
        result.defaultAngles = [0, .25 * Math.PI, -.5 * Math.PI, .5 * Math.PI, -.25 * Math.PI, 0];
        result.legacyController = { kind: 'direct', velocityScale: 2 * Math.PI };
        result.initialRootPosition = [-.10005798190832138, -2.596527338027954e-6, 1.1801719665527344e-5];
        result.initialRootQuaternion = [1, .00024943926837295294, .00011962498683715239, -.00020808471890632063];
        result.legacyAutoReset = { fallingBaseHeight: .2 };
        break;
      case 'ZBotBipedalWalking_BigFoot.pth':
        result.model = 'legacy_zbot_bigfoot';
        result.defaultAngles = [0, -.25 * Math.PI, .5 * Math.PI, -.5 * Math.PI, .25 * Math.PI, 0];
        result.legacyBodyIndex = 3;
        result.jointSpeedLimit = .1;
        result.legacyController = { kind: 'direct', velocityScale: 2 * Math.PI, usesJointSpeed: true };
        result.legacyAutoReset = { minimumBaseHeight: .2, maximumBaseHeight: .35 };
        break;
      case 'ZBotBipedalWalking_BigFoot_8DOF.pth':
        result.model = 'legacy_zbot_bigfoot_8dof';
        result.defaultAngles = [.520365, .283715, -1.34606, 1.23715, -1.23219, 1.37363, -.270398, -.54471];
        result.legacyBodyIndex = 4;
        result.jointSpeedLimit = .5;
        result.legacyController = { kind: 'direct', velocityScale: 2 * Math.PI, usesJointSpeed: true };
        result.legacyAutoReset = { minimumBaseHeight: .2, maximumBaseHeight: .45, minimumExtremityDistance: .12, maximumAbsBaseX: .2 };
        break;
    }
  }
  if (checkpoint.file === 'zbot_rolling.pth') {
    result.model = 'legacy_zbot_rolling';
    result.jointNames = Array.from({ length: 6 }, (_, index) => `joint_${index}`);
    result.jointSigns = Array(6).fill(1);
    result.controlDt = result.physicsDt = .0166;
    result.legacyController = { kind: 'cpg3', velocityScale: 2 * Math.PI, omegaScale: 2 * Math.PI, timeOffsetSteps: 1 };
    result.initialRootPosition = undefined;
    result.initialRootQuaternion = undefined;
    result.zeroInitialObservation = true;
    // The original VecTask starts with reset_buf=1. Its first env.step simulates,
    // reset_idx then restores root/DOF state, while compute_observations still
    // exposes the rigid-body tensor captured for that effective player step.
    result.bootstrapResetObservations = [[-0.00242511834949255, -0.3190598487854004, 0.05654872581362724,
      -0.026693087071180344, 0.7123156785964966, -0.7010008096694946, -0.022173209115862846,
      0.03197012096643448, 0.06948001682758331, -0.18689045310020447, -0.833968997001648,
      3.9913313388824463, 0.5128341913223267, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
    result.physxSimulation = { disableSelfCollision: false };
    result.note = '旧 Isaac Gym ZBotRolling 专用对齐模型：7 刚体、原生质量/COM/惯量、MJCF 关节与碰撞、0.0166 s CPG 时序；使用原生 golden trace 验证。';
  }
  return result;
}

export const LEGACY_ISAACGYM_PROFILES = checkpoints.map(profile);
