import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { get as httpGet } from 'node:http';
import { spawn, type ChildProcess } from 'node:child_process';
import { createTrainingApp } from '../server/trainingServer';
import { createTrainingService, sanitizeLiveFrame } from '../server/trainingService';
import type { TrainingConfig, TrainingResources } from '../src/training/types';

const config: TrainingConfig = { taskId: 'Mjlab-Zbot-6dof-Bipedal-Walking', device: 'cpu', cpuThreads: 2, numEnvs: 4, iterations: 2, saveInterval: 1, seed: 42, maxSeconds: 30 };
const resource: TrainingResources = { hostname: 'test', platform: 'linux', cpu: { model: 'CPU', logicalCores: 4, availableThreads: 4 }, memory: { totalBytes: 1000, availableBytes: 500 }, gpus: [{ id: 0, name: 'GPU', uuid: 'gpu', totalMemoryMiB: 1000, freeMemoryMiB: 900, utilization: 0, driver: 'test' }], runtime: { available: true, python: 'test-python' }, tasks: [], scannedRoots: [], checkpoints: [] };
class FakeWorker extends EventEmitter {
  stdout = new PassThrough(); stderr = new PassThrough(); pid = 123456;
  event(value: object) { this.stdout.write(JSON.stringify(value) + '\n'); }
  finish(code = 0) { this.emit('close', code, null); }
}
async function until(check: () => boolean | Promise<boolean>) {
  for (let i = 0; i < 200; i++) { if (await check()) return; await new Promise(resolve => setTimeout(resolve, 5)); }
  assert.fail('Timed out waiting for worker lifecycle');
}
async function fixture() {
  const root = await mkdtemp(path.join(os.tmpdir(), 'zbot-training-test-'));
  const workers: { child: FakeWorker; args: string[]; jobDir: string }[] = [];
  const signals: NodeJS.Signals[] = [];
  const scanned = path.join(root, 'source'); await mkdir(scanned);
  const options = { workDir: path.join(root, 'state'), scannedRoots: [scanned], resources: async () => structuredClone(resource), stopGraceMs: 25,
    launch: (_cmd: string, args: string[]) => { const child = new FakeWorker(); workers.push({ child, args, jobDir: args.at(-1)! }); return child as unknown as ChildProcess; },
    killProcess: (_child: ChildProcess, signal: NodeJS.Signals) => { signals.push(signal); } };
  const instance = await createTrainingApp(options);
  const server = instance.app.listen(0, '127.0.0.1');
  await new Promise<void>(resolve => server.once('listening', resolve));
  const address = server.address(); assert.ok(address && typeof address !== 'string');
  const base = `http://127.0.0.1:${address.port}/api/training`;
  const token = (await (await fetch(base + '/session')).json()).token;
  const request = (suffix: string, body?: unknown, extra: Record<string, string> = {}) => fetch(base + suffix, { method: body === undefined ? 'GET' : 'POST', headers: { ...(body === undefined ? {} : { 'Content-Type': 'application/json', 'X-Training-Token': token }), ...extra }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  const dispose = async () => { const closing = instance.close(); for (const w of workers) w.child.finish(1); await closing; await new Promise(resolve => setTimeout(resolve, 20)); await new Promise<void>(resolve => server.close(() => resolve())); await rm(root, { recursive: true, force: true }); };
  return { ...instance, root, scanned, workers, signals, options, request, base, token, dispose };
}
async function saveCheckpoint(jobDir: string, name = 'model_0.pt') {
  await writeFile(path.join(jobDir, 'checkpoints', name), Buffer.from([80, 75, 3, 4]));
}

test('walking finetune is discoverable and launches only its fixed task identity', async () => {
  const f = await fixture();
  try {
    const resources = await (await f.request('/resources')).json();
    assert.equal(resources.tasks[0].id, 'Mjlab-Zbot-6dof-Walking-Finetune');
    const response = await f.request('/jobs', { ...config, taskId: resources.tasks[0].id, resumePath: '/untrusted/model.pt' });
    assert.equal(response.status, 201);
    await until(() => f.workers.length === 1);
    const request = JSON.parse(await readFile(path.join(f.workers[0].jobDir, 'request.json'), 'utf8'));
    assert.equal(request.taskId, resources.tasks[0].id);
    assert.equal(request.resumePath, undefined);
    assert.equal(request.taskCard.stage2Rewards.feet_slide, -10);
  } finally { await f.dispose(); }
});

test('task card reward weights update atomically while a job is active', async () => {
  const f = await fixture();
  try {
    const job = await f.service.start(config); await until(() => f.workers.length === 1);
    const current = job.taskCard!;
    const changed = { ...current, stage1Rewards: { ...current.stage1Rewards, feet_force_diff: 3.5 }, stage2Rewards: { ...current.stage2Rewards, feet_slide: -7 }, terminatedRewardPenalty: 25 };
    assert.equal((await f.request(`/jobs/${job.id}/task-card`, { ...changed, stage1Rewards: { missing: 1 } })).status, 400);
    const response = await f.request(`/jobs/${job.id}/task-card`, changed);
    assert.equal(response.status, 200);
    const updated = await response.json();
    assert.equal(updated.taskCardRevision, 1);
    assert.equal(updated.taskCard.stage2Rewards.feet_slide, -7);
    assert.deepEqual(JSON.parse(await readFile(path.join(f.workers[0].jobDir, 'TASK_CARD.json'), 'utf8')), { revision: 1, taskCard: changed });
    await f.service.stop(job.id); f.workers[0].child.finish(143); await until(() => job.status === 'stopped');
    assert.equal((await f.request(`/jobs/${job.id}/task-card`, changed)).status, 409);
  } finally { await f.dispose(); }
});

test('live training frames are strictly bounded and copied into job state', async () => {
  const pose = { bodyNames: ['base', 'foot'], environments: [{ environmentOrigin: [1, 2, 0], bodyPositions: [[1, 2, .25], [1, 2, 0]], bodyQuaternions: [[1, 0, 0, 0], [1, 0, 0, 0]] }] };
  assert.equal(sanitizeLiveFrame({ ...pose, environments: Array(10).fill(pose.environments[0]) }, 1), undefined);
  assert.equal(sanitizeLiveFrame({ ...pose, environments: [{ ...pose.environments[0], bodyPositions: [[Infinity, 0, 0], [0, 0, 0]] }] }, 1), undefined);
  const f = await fixture();
  try {
    const job = await f.service.start(config); await until(() => f.workers.length === 1);
    assert.equal(job.livePreviewEnabled, false);
    assert.equal((await f.request(`/jobs/${job.id}/preview`, { enabled: 'yes' })).status, 400);
    const enabled = await (await f.request(`/jobs/${job.id}/preview`, { enabled: true })).json();
    assert.equal(enabled.livePreviewEnabled, true);
    assert.equal(await readFile(path.join(f.workers[0].jobDir, 'LIVE_PREVIEW'), 'utf8'), 'enabled\n');
    const streamed: unknown[] = [];
    const unsubscribe = f.service.subscribeLive(job.id, frame => streamed.push(frame));
    f.workers[0].child.event({ event: 'ready' });
    f.workers[0].child.event({ event: 'liveFrame', liveFrame: { ...pose, iteration: 1 } });
    await until(() => f.service.get(job.id).liveFrame?.iteration === 1);
    assert.equal((streamed[0] as any).iteration, 1);
    assert.equal(unsubscribe(), 0);
    f.workers[0].child.event({ event: 'progress', iteration: 1, totalIterations: 2 });
    await until(() => f.service.get(job.id).liveFrame?.iteration === 1);
    assert.deepEqual(f.service.get(job.id).liveFrame?.environments[0].bodyPositions[0], [1, 2, .25]);
    assert.deepEqual(f.service.get(job.id).liveFrame?.environments[0].environmentOrigin, [1, 2, 0]);
    assert.doesNotMatch(f.service.get(job.id).logTail, /bodyPositions/);
    pose.environments[0].bodyPositions[0][0] = 999;
    assert.equal(f.service.get(job.id).liveFrame?.environments[0].bodyPositions[0][0], 1);
    const disabled = await (await f.request(`/jobs/${job.id}/preview`, { enabled: false })).json();
    assert.equal(disabled.livePreviewEnabled, false);
    await assert.rejects(readFile(path.join(f.workers[0].jobDir, 'LIVE_PREVIEW')), { code: 'ENOENT' });
  } finally { await f.dispose(); }
});

test('training API requires local host/origin and session token, validates resource bounds and JSON size', async () => {
  const f = await fixture();
  try {
    assert.equal((await f.request('/jobs', config, { 'X-Training-Token': '' })).status, 403);
    assert.equal((await f.request('/jobs', config, { Origin: 'https://evil.example' })).status, 403);
    assert.equal(await new Promise<number>(resolve => { httpGet(f.base + '/session', { headers: { Host: 'evil.example' } }, res => { res.resume(); resolve(res.statusCode!); }); }), 403);
    assert.equal((await f.request('/session', undefined, { 'Sec-Fetch-Site': 'cross-site' })).status, 403);
    assert.equal((await f.request('/jobs', config, { 'Content-Type': 'text/plain' })).status, 415);
    assert.equal((await f.request('/jobs', { ...config, padding: 'x'.repeat(70 * 1024) })).status, 413);
    for (const bad of [{ taskId: 'unknown' }, { device: 'cuda:9' }, { cpuThreads: 5 }, { numEnvs: 65 }, { iterations: 0 }, { maxSeconds: 29 }, { seed: 1.2 }]) {
      assert.equal((await f.request('/jobs', { ...config, ...bad })).status, 400);
    }
    assert.equal(f.workers.length, 0);
  } finally { await f.dispose(); }
});

test('resource scanner excludes exported and symlink files; queue progresses only after real completion', async () => {
  const f = await fixture();
  try {
    await writeFile(path.join(f.scanned, 'external.pt'), 'model');
    await mkdir(path.join(f.scanned, 'exported')); await writeFile(path.join(f.scanned, 'exported', 'policy.pt'), 'ignored');
    await symlink(path.join(f.scanned, 'external.pt'), path.join(f.scanned, 'alias.pt'));
    let resources = await (await f.request('/resources')).json();
    assert.deepEqual(resources.checkpoints.map((c: any) => [c.name, c.resumable]), [['external.pt', false]]);
    const first = await (await f.request('/jobs', config)).json();
    await until(() => f.workers.length === 1);
    const second = await (await f.request('/jobs', config)).json();
    assert.equal(second.status, 'queued'); assert.equal(f.workers.length, 1);
    f.workers[0].child.event({ event: 'ready' });
    f.workers[0].child.event({ event: 'progress', iteration: 1, totalIterations: 2, reward: 4.2, fps: 15 });
    await until(() => f.service.get(first.id).metrics.iteration === 1);
    await saveCheckpoint(f.workers[0].jobDir);
    await writeFile(path.join(f.workers[0].jobDir, 'bundle.json'), JSON.stringify({ schemaVersion: 1 }));
    f.workers[0].child.finish();
    await until(() => f.workers.length === 2);
    assert.equal(f.service.get(first.id).status, 'completed'); assert.equal(f.service.get(first.id).bundleReady, true);
    assert.equal((await f.request(`/jobs/${first.id}/bundle`)).status, 200);
    const download = await f.request(`/jobs/${first.id}/checkpoints/model_0.pt`); assert.equal(download.status, 200); assert.equal((await download.arrayBuffer()).byteLength, 4);
    assert.equal((await f.request(`/jobs/${first.id}/checkpoints/${encodeURIComponent('../request.json')}`)).status, 404);
    resources = await (await f.request('/resources')).json();
    assert.equal(resources.checkpoints.find((c: any) => c.name === 'model_0.pt').resumable, true);
    f.workers[1].child.finish();
    await until(() => f.service.get(second.id).status === 'failed');
    assert.match(f.service.get(second.id).error!, /未生成检查点/);
  } finally { await f.dispose(); }
});

test('stop writes cooperative marker, escalates only its worker, and resume resolves only owned checkpoints', async () => {
  const f = await fixture();
  try {
    const first = await f.service.start(config); await until(() => f.workers.length === 1);
    await saveCheckpoint(f.workers[0].jobDir);
    const queued = await f.service.start(config); await f.service.stop(queued.id);
    assert.equal(queued.status, 'stopped'); assert.equal(f.workers.length, 1);
    await f.service.stop(first.id); assert.equal(first.status, 'stopping');
    assert.equal(await readFile(path.join(f.workers[0].jobDir, 'STOP'), 'utf8'), 'stop\n');
    assert.deepEqual(f.signals, ['SIGTERM']);
    await until(() => f.signals.includes('SIGKILL'));
    f.workers[0].child.finish(143); await until(() => first.status === 'stopped');
    const resume = await f.service.start({ ...config, resumeJobId: first.id, resumeCheckpoint: 'model_0.pt', resumePath: '/etc/passwd' });
    await until(() => f.workers.length === 2);
    const request = JSON.parse(await readFile(path.join(f.workers[1].jobDir, 'request.json'), 'utf8'));
    assert.equal(request.resumePath, path.join(f.workers[0].jobDir, 'checkpoints', 'model_0.pt'));
    assert.equal(resume.config.resumeCheckpoint, 'model_0.pt');
    assert.equal((await f.request('/jobs', { ...config, resumeJobId: first.id, resumeCheckpoint: '../../etc/passwd' })).status, 404);
    await symlink('/etc/passwd', path.join(f.workers[0].jobDir, 'checkpoints', 'model_bad.pt'));
    assert.equal((await f.request(`/jobs/${first.id}/checkpoints/model_bad.pt`)).status, 404);
    f.workers[1].child.finish(2); await until(() => resume.status === 'failed');
  } finally { await f.dispose(); }
});

test('worker logs are bounded and persisted; service restart marks interrupted jobs failed without launch', async () => {
  const f = await fixture();
  try {
    const job = await f.service.start(config); await until(() => f.workers.length === 1);
    for (let i = 0; i < 400; i++) f.workers[0].child.stdout.write(`${i}: ${'x'.repeat(1000)}\n`);
    assert.ok(job.logTail.length <= 100 * 1024); assert.ok(job.logTail.split('\n').length <= 201);
    await until(async () => { try { const saved = JSON.parse(await readFile(path.join(f.workers[0].jobDir, 'job.json'), 'utf8')); return saved.logTail.includes('399:'); } catch { return false; } });
    const record = path.join(f.workers[0].jobDir, 'job.json');
    const original = await readFile(record, 'utf8');
    await assert.rejects(createTrainingService(f.options), /已有训练服务/);
    assert.equal(await readFile(record, 'utf8'), original, 'duplicate startup must not rewrite active status');
    const closing = f.close(); f.workers[0].child.finish(1); await closing;
    await writeFile(record, original); // Simulate the persisted record of an interrupted process.
    const restarted = await createTrainingService({ ...f.options, launch: () => { throw new Error('Must not auto-respawn'); } });
    assert.equal(restarted.get(job.id).status, 'failed'); assert.match(restarted.get(job.id).error!, /重启/);
    await restarted.close();
  } finally { await f.dispose(); }
});

test('unavailable Python runtime blocks submissions without launching anything', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'zbot-training-unavailable-'));
  const service = await createTrainingService({ workDir: root, scannedRoots: [], resources: async () => ({ ...resource, runtime: { available: false, python: 'missing', error: 'not installed' } }), launch: () => { throw new Error('Must not launch'); } });
  try { await assert.rejects(service.start(config), /not installed/); }
  finally { await service.close(); await rm(root, { recursive: true, force: true }); }
});


test('close waits for its worker process to exit and persists stopped queued jobs', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'zbot-training-close-'));
  let child: ChildProcess;
  let ready = false;
  const service = await createTrainingService({ workDir: root, scannedRoots: [], resources: async () => structuredClone(resource), stopGraceMs: 25,
    launch: () => {
      child = spawn(process.execPath, ['-e', "process.on('SIGTERM',()=>{});console.log('ready');setInterval(()=>{},1000)"], { detached: true, stdio: ['ignore', 'pipe', 'pipe'] });
      child.stdout!.on('data', () => { ready = true; });
      return child;
    } });
  try {
    const running = await service.start(config); await until(() => ready);
    const queued = await service.start(config);
    const pid = child!.pid!;
    await service.close();
    assert.throws(() => process.kill(pid, 0), (error: NodeJS.ErrnoException) => error.code === 'ESRCH');
    assert.equal(running.status, 'stopped'); assert.equal(queued.status, 'stopped');
    for (const job of [running, queued]) {
      const saved = JSON.parse(await readFile(path.join(root, 'jobs', job.id, 'job.json'), 'utf8'));
      assert.equal(saved.status, 'stopped'); assert.ok(saved.finishedAt);
    }
    await service.close();
  } finally { await service.close(); await rm(root, { recursive: true, force: true }); }
});

test('concurrent resource requests share one Python runtime probe', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'zbot-training-probe-'));
  const probe = path.join(root, 'probe.cjs');
  const counter = path.join(root, 'calls.txt');
  await writeFile(probe, `require('node:fs').appendFileSync(${JSON.stringify(counter)}, 'probe\\n');setTimeout(()=>console.log(JSON.stringify({available:true})),75)`);
  const service = await createTrainingService({ workDir: path.join(root, 'state'), scannedRoots: [], python: process.execPath, probe });
  try {
    const results = await Promise.all(Array.from({ length: 8 }, () => service.resources()));
    assert.ok(results.every(result => result.runtime.available));
    await service.resources();
    assert.equal((await readFile(counter, 'utf8')).match(/probe/g)!.length, 1);
  } finally { await service.close(); await rm(root, { recursive: true, force: true }); }
});


test('progress history is finite, unique per iteration and bounded to 200 samples', async () => {
  const f = await fixture();
  try {
    const job = await f.service.start(config); await until(() => f.workers.length === 1);
    for (let iteration = 0; iteration < 250; iteration++) f.workers[0].child.event({ event: 'progress', iteration, reward: iteration, loss: 1 / (iteration + 1), rewardTerms: { feet_downward: -iteration / 10, unknown: 999 } });
    f.workers[0].child.event({ event: 'progress', iteration: 249, reward: 999 });
    f.workers[0].child.event({ event: 'progress', iteration: 0, reward: 999 });
    assert.equal(job.history!.length, 200);
    assert.equal(job.history![0].iteration, 50);
    assert.equal(job.history!.at(-1)!.reward, 249);
    assert.deepEqual(job.history!.at(-1)!.rewardTerms, { feet_downward: -24.9 });
    assert.deepEqual(job.metrics.rewardTerms, { feet_downward: -24.9 });
    await until(async () => JSON.parse(await readFile(path.join(f.workers[0].jobDir, 'job.json'), 'utf8')).history?.length === 200);
  } finally { await f.dispose(); }
});

test('CPU-only runtime rejects detected GPU and the queue has a fixed capacity', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'zbot-training-limits-'));
  const worker = new FakeWorker();
  const service = await createTrainingService({ workDir: root, scannedRoots: [], resources: async () => ({ ...resource, runtime: { ...resource.runtime, cudaBuild: false } }),
    launch: () => worker as unknown as ChildProcess, killProcess: () => { setImmediate(() => worker.finish(1)); } });
  try {
    await assert.rejects(service.start({ ...config, device: 'cuda:0' }), /CUDA/);
    await service.start(config);
    await until(() => service.list().some(job => job.status === 'starting'));
    for (let i = 0; i < 32; i++) await service.start(config);
    await assert.rejects(service.start(config), /队列已满/);
  } finally { await service.close(); await rm(root, { recursive: true, force: true }); }
});

test('a dead process lock is recoverable and released on close', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'zbot-training-lock-'));
  await writeFile(path.join(root, 'service.lock'), JSON.stringify({ pid: 2147483647, token: 'dead' }));
  const service = await createTrainingService({ workDir: root, scannedRoots: [], resources: async () => resource });
  await service.close();
  await assert.rejects(readFile(path.join(root, 'service.lock')), /ENOENT/);
  await rm(root, { recursive: true, force: true });
});

test('operator scan roots accept a bounded JSON list without browser-provided paths', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'zbot-training-scan-env-'));
  const original = process.env.ZBOT_TRAINING_SCAN_ROOTS;
  try {
    const scanned = path.join(root, 'models'); await mkdir(scanned); await writeFile(path.join(scanned, 'model_7.pt'), 'checkpoint');
    process.env.ZBOT_TRAINING_SCAN_ROOTS = JSON.stringify([scanned]);
    const service = await createTrainingService({ workDir: path.join(root, 'state'), resources: async () => resource });
    try {
      const result = await service.resources();
      assert.deepEqual(result.scannedRoots, [path.join(root, 'state', 'jobs'), scanned]);
      assert.deepEqual(result.checkpoints.map(checkpoint => checkpoint.name), ['model_7.pt']);
    } finally { await service.close(); }
    process.env.ZBOT_TRAINING_SCAN_ROOTS = JSON.stringify(Array(11).fill(scanned));
    await assert.rejects(createTrainingService({ workDir: path.join(root, 'state') }), /最多 10/);
  } finally {
    if (original === undefined) delete process.env.ZBOT_TRAINING_SCAN_ROOTS;
    else process.env.ZBOT_TRAINING_SCAN_ROOTS = original;
    await rm(root, { recursive: true, force: true });
  }
});
