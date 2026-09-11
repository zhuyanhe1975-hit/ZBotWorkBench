import assert from 'node:assert/strict';
import { test } from 'node:test';
import { QUASISTATIC_TASK, TRAINING_TASKS, WALKING_FINETUNE_TASK, taskDefaults } from '../src/training/tasks';
import type { TrainingResources } from '../src/training/types';

const resources = { cpu: { availableThreads: 2 }, runtime: { available: true, cudaBuild: true }, gpus: [{ id: 3 }] } as TrainingResources;
test('walking presets select the detected GPU and verified task-specific scale', () => {
  assert.equal(TRAINING_TASKS[0].id, WALKING_FINETUNE_TASK);
  const preset = taskDefaults(WALKING_FINETUNE_TASK, resources);
  assert.equal(preset.device, 'cuda:3');
  assert.equal(preset.cpuThreads, 2);
  assert.equal(preset.numEnvs, 256);
  assert.equal(preset.iterations, 100);
  assert.equal(preset.resumeJobId, undefined);
  const scratch = taskDefaults('Mjlab-Zbot-6dof-Bipedal-Walking', resources);
  assert.equal(scratch.device, 'cuda:3');
  assert.equal(scratch.numEnvs, 4096);
  assert.equal(scratch.iterations, 300);
  assert.equal(scratch.saveInterval, 50);
  const slow = taskDefaults(QUASISTATIC_TASK, resources);
  assert.equal(slow.numEnvs, 1024); assert.equal(slow.iterations, 1000);
  assert.equal(slow.taskCard!.stage2Rewards.slow_speed_tracking, 2);
  assert.equal(slow.taskCard!.stage2Rewards.double_flight, -10);
});
test('walking preset supports CPU-only runtime and explicit device changes', () => {
  for (const fallback of [{ ...resources, gpus: [] }, { ...resources, runtime: { ...resources.runtime, cudaBuild: false } }]) {
    const config = taskDefaults(WALKING_FINETUNE_TASK, fallback);
    assert.equal(config.device, 'cpu'); assert.equal(config.numEnvs, 4);
  }
  assert.equal(taskDefaults(WALKING_FINETUNE_TASK, resources, 'cpu').numEnvs, 4);
});
