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
const native = json<any>('tests/fixtures/rollingNative.json');
const profile = REPLAY_PROFILES.find(value => value.id === 'IsaacGym-zbot_rolling')!;

const mujoco = await loadMujoco();
mujoco.FS.writeFile('ma.obj', bytes('public/assets/ma.obj'));
mujoco.FS.writeFile('mb.obj', bytes('public/assets/mb.obj'));
const display = new MujocoEngine();
(display as any).mujoco = mujoco;
display.loadModelFromXml(bytes(`public/rl/models/${profile.model}.xml`).toString());

const wasmBinary = bytes('node_modules/physx-js-webidl/physx-js-webidl.wasm');
const originalProcess = globalThis.process;
(globalThis as any).process = undefined;
(globalThis as any).window = {};
const physx = await (loadPhysx as any)({ wasmBinary });
globalThis.process = originalProcess;
delete (globalThis as any).window;

const model = json<PhysxModel>(`public/rl/physx/${profile.model}.json`);
const dynamics = new PhysxSimulation(physx, model, display,
  { physicsDt: profile.physicsDt, controlDt: profile.controlDt });
const checkpoint = bytes(`public/rl/checkpoints/${profile.id}/${profile.checkpoint}`);
const policy = loadCheckpoint(checkpoint.buffer.slice(checkpoint.byteOffset, checkpoint.byteOffset + checkpoint.byteLength));
const replay = new PolicyReplay(display, profile, policy, dynamics);

const error = (actual: ArrayLike<number>, expected: ArrayLike<number>) => {
  const values = Array.from(actual, (value, i) => value - expected[i]);
  return { max: Math.max(...values.map(Math.abs)), rms: Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length) };
};
console.log(JSON.stringify({ initialBase: dynamics.getBasePosition(), expected: native.clean_reset_reference.body_state_4_13.slice(0, 3) }));
for (const expected of native.effective_deterministic_steps) {
  replay.step();
  const observation = replay.observe();
  console.log(JSON.stringify({ step: expected.step, observation: error(observation, expected.obs_returned_after_env_step_25),
    action: error(replay.lastActions, expected.clipped_action_18), target: error(display.getData().ctrl, expected.pos_d_after_env_step_6),
    base: Array.from(observation.slice(0, 13)), joints: dynamics.state().positions }));
}
dynamics.dispose();
display.destroy();
