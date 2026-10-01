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
const variant = process.argv[2] ?? 'ZBotBipedalWalking';
const fixture = variant === 'ZBotBipedalWalking' ? 'native.json' : `${variant}.json`;
const native = json<any>(`results/legacy-bipedal-walking/${fixture}`);
const profile = REPLAY_PROFILES.find(value => value.id === `IsaacGym-${variant}`)!;
if (!profile || profile.model !== 'legacy_zbot_6dof_bipedal_walking') throw new Error(`未知 6DOF 双足策略 ${variant}`);

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
const difference = (actual: ArrayLike<number>, expected: ArrayLike<number>) => {
  const values = Array.from(actual, (value, i) => value - expected[i]);
  return { max: Math.max(...values.map(Math.abs)), rms: Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length) };
};

console.log(JSON.stringify({ initialBase: dynamics.getBasePosition(), expected: native.clean_reset.body_poses.body_3.p }));
for (const expected of native.effective_steps) {
  replay.step();
  console.log(JSON.stringify({ step: expected.step,
    observation: difference(replay.observe(), expected.obs_after), action: difference(replay.lastActions, expected.actions),
    target: difference(display.getData().ctrl, expected.pos_d), base: dynamics.getBasePosition(), joints: dynamics.state().positions }));
}
dynamics.dispose();
display.destroy();
