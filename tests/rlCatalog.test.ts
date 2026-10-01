import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
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
  '6dof-bipedal-velocity-v0', '6dof-bipedal-velocity-imu-v0'].map(task => `Zbot-Direct-${task}`)
  .concat('ZbotRlIsaaclab-6DOF-Periodic-Walking');
const asset = (path: string) => readFileSync(new URL(`../public/rl/${path}`, import.meta.url));

test('catalog includes all saved policies with matching packaged dimensions and training poses', async () => {
  const current = REPLAY_PROFILES.filter(p => p.origin !== 'isaacgym');
  const legacy = REPLAY_PROFILES.filter(p => p.origin === 'isaacgym');
  assert.deepEqual(current.map(p => p.id).sort(), expectedTasks.sort());
  assert.equal(legacy.length, 45);
  assert.equal(new Set(REPLAY_PROFILES.map(p => p.id)).size, expectedTasks.length + 45);
  const mujoco = await loadMujoco();
  mujoco.FS.writeFile('ma.obj', readFileSync(new URL('../public/assets/ma.obj', import.meta.url)));
  mujoco.FS.writeFile('mb.obj', readFileSync(new URL('../public/assets/mb.obj', import.meta.url)));
  for (const profile of REPLAY_PROFILES) {
    const bytes = asset(`checkpoints/${profile.id}/${profile.checkpoint}`);
    const policy = loadCheckpoint(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
    assert.equal(policy.inputSize, profile.inputSize, profile.id);
    assert.equal(policy.outputSize, profile.policyOutputSize ?? profile.jointNames.length, profile.id);
    assert.equal(!!policy.normalization, profile.observation === 'run' || profile.origin === 'isaacgym', profile.id);
    if (profile.origin === 'isaacgym') {
      const provenance = JSON.parse(asset(`checkpoints/${profile.id}/provenance.json`).toString());
      assert.equal(provenance.exportedSha256, createHash('sha256').update(bytes).digest('hex'), profile.id);
    }
    if (profile.origin === 'isaaclab') {
      const provenance = JSON.parse(asset(`checkpoints/${profile.id}/provenance.json`).toString());
      assert.equal(provenance.checkpointSha256, createHash('sha256').update(bytes).digest('hex'), profile.id);
      assert.equal(provenance.network.inputSize, profile.inputSize);
    }
    const dynamics = JSON.parse(asset(`physx/${profile.model}.json`).toString()) as PhysxModel;
    assert.deepEqual(dynamics.jointNames, profile.jointNames, profile.id);
    assert.equal(dynamics.defaultQ.length, profile.defaultAngles.length);
    if (profile.origin !== 'isaacgym') dynamics.defaultQ.forEach((q, i) => assert.ok(Math.abs(q - profile.defaultAngles[i]) < 1e-6, `${profile.id}: PhysX default joint ${i}`));
    const display = new MujocoEngine();
    try {
      (display as any).mujoco = mujoco;
      display.loadModelFromXml(asset(`models/${profile.displayModel ?? profile.model}.xml`).toString());
      display.resetPolicyPose();
      const model = display.getModel(), data = display.getData();
      const nameAt = (address: number) => {
        let name = '';
        for (let i = address; i >= 0 && i < model.names.length && model.names[i]; i++) name += String.fromCharCode(model.names[i]);
        return name;
      };
      const displayJointNames = profile.displayJointNames ?? profile.jointNames;
      for (let i = 0; i < displayJointNames.length; i++) {
        const joint = Array.from(model.name_jntadr as Int32Array).findIndex(a => nameAt(a) === displayJointNames[i]);
        assert.ok(joint >= 0, profile.id);
        if (profile.origin !== 'isaacgym') assert.ok(Math.abs(data.qpos[model.jnt_qposadr[joint]] - profile.defaultAngles[i]) < 1e-6, `${profile.id}: displayed default joint ${i}`);
      }
      assert.equal(model.nu, profile.jointNames.length);
    } finally { display.destroy(); }
  }
});

test('experimental and all migrated policies retain limitation disclosures', () => {
  const expected = ['Zbot-Direct-8dof-bipedal-v3', 'Zbot-Direct-6dof-bipedal-velocity-v0',
    'Zbot-Direct-8dof-bipedal-velocity-v0', 'ZbotRlIsaaclab-6DOF-Periodic-Walking'];
  const experimental = REPLAY_PROFILES.filter(p => p.origin !== 'isaacgym' && p.note);
  assert.deepEqual(experimental.map(p => p.id).sort(), expected.sort());
  REPLAY_PROFILES.filter(p => p.origin === 'isaacgym').forEach(p => assert.match(p.note!, /旧 Isaac Gym/));
});

test('every unique run.sh replay command has a dedicated native alignment contract', () => {
  const ids = ['IsaacGym-zbot_rolling', 'IsaacGym-zbot_side_moving', 'IsaacGym-zbot_forward_moving',
    'IsaacGym-ZBotStandUp', 'IsaacGym-last_ZBotFootDown_ep_2500_rew_970.54364', 'IsaacGym-ZBotSingleLeg',
    'IsaacGym-ZBotSnakeAndBipedal', 'IsaacGym-ZBotBipedalWalking', 'IsaacGym-ZBotBipedalWalking_BigFoot',
    'IsaacGym-ZBotBipedalWalking_BigFoot_8DOF'];
  for (const id of ids) {
    const profile = REPLAY_PROFILES.find(value => value.id === id)!;
    assert.ok(profile.model.startsWith('legacy_zbot_'), id);
    assert.equal(profile.zeroInitialObservation, true, id);
    assert.ok(profile.bootstrapResetObservations?.length, id);
    assert.equal(profile.controlDt, .0166, id);
    assert.equal(profile.physicsDt, .0166, id);
    assert.deepEqual(profile.jointSigns, Array(profile.jointNames.length).fill(1), id);
  }
});

test('all source-backed ZBot checkpoints use native reset coordinate families', () => {
  const legacy = REPLAY_PROFILES.filter(p => p.origin === 'isaacgym');
  const byId = (id: string) => legacy.find(p => p.id === id)!;
  for (const profile of legacy.filter(p => /^IsaacGym-(?:last_)?ZBot/.test(p.id))) {
    if (profile.id.includes('BigFoot_8DOF')) {
      assert.equal(profile.model, 'legacy_zbot_bigfoot_8dof', profile.id);
      assert.deepEqual(profile.jointSigns, Array(8).fill(1), profile.id);
    } else if (profile.id.includes('BigFoot')) {
      assert.equal(profile.model, 'legacy_zbot_bigfoot', profile.id);
      assert.deepEqual(profile.jointSigns, Array(6).fill(1), profile.id);
    } else if (!profile.id.includes('FootDown_8DOF') && !profile.id.includes('SingleLeg')) {
      assert.equal(profile.model, 'legacy_zbot_6dof_bipedal_walking', profile.id);
      assert.deepEqual(profile.jointSigns, Array(6).fill(1), profile.id);
    }
  }
  const r3 = byId('IsaacGym-ZBotBipedalWalking_BigFoot_R3');
  assert.deepEqual(r3.defaultAngles, [.4, .8, -2, 2, -.8, -.4]);
  assert.deepEqual(r3.initialRootPosition, [-.05, 0, .05]);
  const yaw = r3.initialRootQuaternion!;
  assert.ok(Math.abs(yaw[0] - Math.cos(Math.PI / 8)) < 1e-12);
  assert.ok(Math.abs(yaw[3] - Math.sin(Math.PI / 8)) < 1e-12);
  assert.deepEqual(byId('IsaacGym-ZBotStandUpAndWalking').defaultAngles, Array(6).fill(0));
  assert.deepEqual(byId('IsaacGym-ZBotFootDownBack').defaultAngles, Array(6).fill(0));
  for (const id of ['IsaacGym-zbot_bipedal_forward', 'IsaacGym-zbot_bipedal_running',
    'IsaacGym-zbot_bipedal_walking_R2']) {
    const profile = byId(id);
    assert.equal(profile.model, 'legacy_zbot_6dof_bipedal_walking', id);
    assert.deepEqual(profile.defaultAngles, [0, -Math.PI / 4, Math.PI / 2, -Math.PI / 2, Math.PI / 4, 0], id);
    assert.deepEqual(profile.initialRootQuaternion, [1, 0, 0, 0], id);
  }
  for (const id of ['IsaacGym-zbot_standup', 'IsaacGym-zbot_standup_R1',
    'IsaacGym-zbot_standup_R2', 'IsaacGym-zbot_standup_back']) {
    const profile = byId(id);
    assert.deepEqual(profile.defaultAngles, Array(6).fill(0), id);
    assert.deepEqual(profile.initialRootPosition, [-.318, 0, .053], id);
    assert.deepEqual(profile.initialRootQuaternion, [Math.SQRT1_2, 0, Math.SQRT1_2, 0], id);
  }
});

test('run.sh display models omit duplicate collision-group geometry', async () => {
  const mujoco = await loadMujoco();
  mujoco.FS.writeFile('ma.obj', readFileSync(new URL('../public/assets/ma.obj', import.meta.url)));
  mujoco.FS.writeFile('mb.obj', readFileSync(new URL('../public/assets/mb.obj', import.meta.url)));
  const models = ['legacy_zbot_rolling', 'legacy_zbot_single_leg', 'legacy_zbot_6dof_bipedal_walking',
    'legacy_zbot_bigfoot', 'legacy_zbot_bigfoot_8dof'];
  for (const name of models) {
    const display = new MujocoEngine();
    try {
      (display as any).mujoco = mujoco;
      display.loadModelFromXml(asset(`models/${name}.xml`).toString());
      const model = display.getModel();
      assert.ok(Array.from(model.geom_group as Int32Array).every(group => group !== 4), `${name} renders collision geometry`);
    } finally { display.destroy(); }
  }
});

test('SingleLeg reuses the standard 6DOF display with explicit joint mapping', () => {
  const profile = REPLAY_PROFILES.find(value => value.id === 'IsaacGym-ZBotSingleLeg')!;
  assert.equal(profile.displayModel, 'zbot_6s_new');
  assert.deepEqual(profile.displayJointNames, ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']);
  assert.deepEqual(profile.displayJointSigns, [-1, -1, -1, -1, -1, -1]);
  assert.deepEqual(profile.displayRootQuaternionOffset, [Math.SQRT1_2, 0, 0, Math.SQRT1_2]);
});
