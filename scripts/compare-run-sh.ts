import { readFileSync } from 'node:fs';
import loadMujoco from '@mujoco/mujoco';
import loadPhysx from 'physx-js-webidl';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';
import { loadCheckpoint } from '../src/rl/checkpoint';
import { PhysxSimulation, type PhysxModel } from '../src/rl/physx';
import { REPLAY_PROFILES } from '../src/rl/profiles';
import { PolicyReplay } from '../src/rl/replay';

const root = new URL('../', import.meta.url);
const bytes = (path: string) => readFileSync(new URL(path, root));
const json = <T>(path: string): T => JSON.parse(bytes(path).toString());
const tasks = [
  ['IsaacGym-zbot_side_moving', 'side-moving.json'], ['IsaacGym-zbot_forward_moving', 'forward-moving.json'],
  ['IsaacGym-ZBotStandUp', 'stand-up.json'], ['IsaacGym-last_ZBotFootDown_ep_2500_rew_970.54364', 'foot-down.json'],
  ['IsaacGym-ZBotSingleLeg', 'single-leg.json'], ['IsaacGym-ZBotSnakeAndBipedal', 'snake-and-bipedal.json'],
  ['IsaacGym-ZBotBipedalWalking_BigFoot', 'bipedal-bigfoot.json'],
  ['IsaacGym-ZBotBipedalWalking_BigFoot_8DOF', 'bipedal-bigfoot-8dof.json'],
] as const;
const selected = process.argv[2];
const difference = (actual: ArrayLike<number>, expected: ArrayLike<number>) => {
  const values = Array.from(actual, (value, i) => value - expected[i]);
  const largest = values.map((value, index) => ({ index, value, actual: actual[index], expected: expected[index] }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 3);
  return { max: Math.max(...values.map(Math.abs)), rms: Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length), largest };
};

const mujoco = await loadMujoco();
mujoco.FS.writeFile('ma.obj', bytes('public/assets/ma.obj'));
mujoco.FS.writeFile('mb.obj', bytes('public/assets/mb.obj'));
const wasmBinary = bytes('node_modules/physx-js-webidl/physx-js-webidl.wasm');
const originalProcess = globalThis.process;
(globalThis as any).process = undefined; (globalThis as any).window = {};
const physx = await (loadPhysx as any)({ wasmBinary });
globalThis.process = originalProcess; delete (globalThis as any).window;

for (const [id, fixture] of tasks) {
  if (selected && selected !== id && selected !== fixture.replace(/\.json$/, '')) continue;
  const native = json<any>(`results/run-sh/${fixture}`);
  const profile = REPLAY_PROFILES.find(value => value.id === id)!;
  const display = new MujocoEngine();
  (display as any).mujoco = mujoco;
  display.loadModelFromXml(bytes(`public/rl/models/${profile.displayModel ?? profile.model}.xml`).toString());
  const model = json<PhysxModel>(`public/rl/physx/${profile.model}.json`);
  model.defaultQ = [...profile.defaultAngles];
  model.simulation = { ...model.simulation, ...profile.physxSimulation };
  model.displayJointNames = profile.displayJointNames;
  model.displayJointSigns = profile.displayJointSigns;
  model.displayRootQuaternionOffset = profile.displayRootQuaternionOffset;
  const dynamics = new PhysxSimulation(physx, model, display, { physicsDt: profile.physicsDt, controlDt: profile.controlDt,
    rootPosition: profile.initialRootPosition, rootQuaternion: profile.initialRootQuaternion });
  const checkpoint = bytes(`public/rl/checkpoints/${profile.id}/${profile.checkpoint}`);
  const policy = loadCheckpoint(checkpoint.buffer.slice(checkpoint.byteOffset, checkpoint.byteOffset + checkpoint.byteLength));
  const replay = new PolicyReplay(display, profile, policy, dynamics);
  const displayModel = display.getModel();
  const displayData = display.getData();
  const nameAt = (address: number) => {
    let name = '';
    for (let i = address; i >= 0 && i < displayModel.names.length && displayModel.names[i]; i++) name += String.fromCharCode(displayModel.names[i]);
    return name;
  };
  const visualError = () => {
    if (profile.displayModel) return null;
    const links = (dynamics as any).links as Map<string, any>;
    return Math.max(...model.bodies.map(body => {
      const bodyIndex = Array.from(displayModel.name_bodyadr as Int32Array).findIndex(address => nameAt(address) === body.name);
      const p = links.get(body.name).getGlobalPose().p;
      return Math.hypot(displayData.xpos[3 * bodyIndex] - p.x, displayData.xpos[3 * bodyIndex + 1] - p.y,
        displayData.xpos[3 * bodyIndex + 2] - p.z);
    }));
  };
  const physicalBodyPositions = () => native.body_names.flatMap((name: string) => {
    const p = ((dynamics as any).links as Map<string, any>).get(name).getGlobalPose().p;
    return [p.x, p.y, p.z];
  });
  const physicalBodyQuaternions = () => native.body_names.map((name: string) => {
    const q = ((dynamics as any).links as Map<string, any>).get(name).getGlobalPose().q;
    return [q.w, q.x, q.y, q.z];
  });
  const quaternionAngle = (actual: number[], expectedXyzw: number[]) => {
    const expected = [expectedXyzw[3], expectedXyzw[0], expectedXyzw[1], expectedXyzw[2]];
    const dot = Math.min(1, Math.abs(actual.reduce((sum, value, i) => sum + value * expected[i], 0)));
    return 2 * Math.acos(dot);
  };
  const baseName = native.body_names[Math.floor(native.body_names.length / 2)];
  const initialState = dynamics.state();
  const initialBodyPositions = initialState.legacyBodies?.flatMap(body => body.position) ?? [];
  const expectedBodyPositions = native.body_names.flatMap((name: string) => native.clean_reset.body_poses[name].p);
  const result: any = { id, initialBase: difference(dynamics.getBasePosition(), native.clean_reset.body_poses[baseName].p),
    initialBodies: difference(initialBodyPositions, expectedBodyPositions), steps: [] };
  if (selected) result.displayInitialBodies = Array.from(displayModel.name_bodyadr as Int32Array).map((address, index) => ({
    name: nameAt(address), position: Array.from(displayData.xpos.slice(3 * index, 3 * index + 3)),
    quaternion: Array.from(displayData.xquat.slice(4 * index, 4 * index + 4)),
  }));
  for (const expected of native.effective_steps) {
    replay.step();
    const actualObservation = replay.observe();
    const currentPositions = physicalBodyPositions();
    const currentQuaternions = physicalBodyQuaternions();
    const nativePositions = expected.body_states?.flatMap((body: number[]) => body.slice(0, 3));
    const separation = (values: number[]) => Math.hypot(values[3] - values[0], values[4] - values[1], values[5] - values[2]);
    result.steps.push({ step: expected.step, observation: difference(actualObservation, expected.obs_after),
      action: difference(replay.lastActions, expected.actions), target: difference(display.getData().ctrl, expected.pos_d),
      base: dynamics.getBasePosition(), visualPositionMax: visualError(),
      physicalBodies: nativePositions ? difference(currentPositions, nativePositions) : undefined,
      physicalQuaternionMax: expected.body_states ? Math.max(...currentQuaternions.map((q: number[], i: number) =>
        quaternionAngle(q, expected.body_states[i].slice(3, 7)))) : undefined,
      rootQuaternionAngle: expected.body_states ? quaternionAngle(currentQuaternions[0], expected.body_states[0].slice(3, 7)) : undefined,
      rootChildSeparation: nativePositions ? { actual: separation(currentPositions), expected: separation(nativePositions) } : undefined });
    if (selected) result.steps.at(-1).values = { actualObservation: Array.from(actualObservation), expectedObservation: expected.obs_after };
  }
  console.log(JSON.stringify(result));
  dynamics.dispose(); display.destroy();
}
