import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import loadMujoco from '@mujoco/mujoco';
import { MujocoEngine } from '../src/mujoco/MujocoEngine';
import { loadCheckpoint } from '../src/rl/checkpoint';
import { REPLAY_PROFILES } from '../src/rl/profiles';
import type { PhysxModel } from '../src/rl/physx';

const expectedTasks = ['8dof-bipedal-v0', '8dof-bipedal-v1', '8dof-bipedal-v2', '8dof-bipedal-v3',
  '8dof-bipedal-run-v0', '8dof-bipedal-velocity-v0', '8dof-snake-v0', '8dof-bird-v0', '8dof-wheel-v0',
  '6dof-bipedal-v0', '6dof-bipedal-quat-v0', '6dof-bipedal-to-snake-v0', '6dof-bipedal-to-snake-v1',
  '6dof-bipedal-velocity-v0', '6dof-bipedal-velocity-imu-v0'].map(task => `Zbot-Direct-${task}`);
const asset = (path: string) => readFileSync(new URL(`../public/rl/${path}`, import.meta.url));

test('catalog includes all 15 saved policies with matching packaged dimensions and training poses', async () => {
  assert.deepEqual(REPLAY_PROFILES.map(p => p.id).sort(), expectedTasks.sort());
  assert.equal(new Set(REPLAY_PROFILES.map(p => p.id)).size, 15);
  const mujoco = await loadMujoco();
  for (const profile of REPLAY_PROFILES) {
    const bytes = asset(`checkpoints/${profile.id}/${profile.checkpoint}`);
    const policy = loadCheckpoint(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
    assert.equal(policy.inputSize, profile.inputSize, profile.id);
    assert.equal(policy.outputSize, profile.jointNames.length, profile.id);
    assert.equal(!!policy.normalization, profile.observation === 'run', profile.id);
    const dynamics = JSON.parse(asset(`physx/${profile.model}.json`).toString()) as PhysxModel;
    assert.deepEqual(dynamics.jointNames, profile.jointNames, profile.id);
    assert.equal(dynamics.defaultQ.length, profile.defaultAngles.length);
    dynamics.defaultQ.forEach((q, i) => assert.ok(Math.abs(q - profile.defaultAngles[i]) < 1e-6, `${profile.id}: PhysX default joint ${i}`));
    const display = new MujocoEngine();
    try {
      (display as any).mujoco = mujoco;
      display.loadModelFromXml(asset(`models/${profile.model}.xml`).toString());
      display.resetPolicyPose();
      const model = display.getModel(), data = display.getData();
      const nameAt = (address: number) => {
        let name = '';
        for (let i = address; i >= 0 && i < model.names.length && model.names[i]; i++) name += String.fromCharCode(model.names[i]);
        return name;
      };
      for (let i = 0; i < profile.jointNames.length; i++) {
        const joint = Array.from(model.name_jntadr as Int32Array).findIndex(a => nameAt(a) === profile.jointNames[i]);
        assert.ok(joint >= 0, profile.id);
        assert.ok(Math.abs(data.qpos[model.jnt_qposadr[joint]] - profile.defaultAngles[i]) < 1e-6, `${profile.id}: displayed default joint ${i}`);
      }
      assert.equal(model.nu, profile.jointNames.length);
    } finally { display.destroy(); }
  }
});

test('three experimental policies retain native-instability disclosure', () => {
  const expected = ['Zbot-Direct-8dof-bipedal-v3', 'Zbot-Direct-6dof-bipedal-velocity-v0', 'Zbot-Direct-8dof-bipedal-velocity-v0'];
  const experimental = REPLAY_PROFILES.filter(p => p.note);
  assert.deepEqual(experimental.map(p => p.id).sort(), expected.sort());
  experimental.forEach(p => assert.match(p.note!, /原生/));
});
