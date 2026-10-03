import { spawn, execFile, type ChildProcess, type SpawnOptions } from 'node:child_process';
import { existsSync } from 'node:fs';
import { appendFile, mkdir, open, readFile, readdir, realpath, rename, stat, unlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { promisify } from 'node:util';
import type { TrainingConfig, TrainingJob, TrainingLiveFrame, TrainingResources, TrainingTaskCardSettings } from '../src/training/types';
import { copyTaskCardSettings, TRAINING_TASKS } from '../src/training/tasks';

const execute = promisify(execFile);
const activeStatuses = new Set(['starting', 'running', 'stopping']);
const idPattern = /^[0-9a-f]{8}-[0-9a-f-]{27}$/;
const checkpointPattern = /^model_[A-Za-z0-9_-]+\.(pt|pth)$/;
const within = (root: string, file: string) => file === root || file.startsWith(root + path.sep);
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);
export class TrainingError extends Error { constructor(public status: number, message: string) { super(message); } }
function validateTaskCard(value: unknown, taskId: string): TrainingTaskCardSettings {
  const defaults = TRAINING_TASKS.find(task => task.id === taskId)?.taskCard;
  if (!defaults) throw new TrainingError(400, '任务卡不存在');
  if (value === undefined) return copyTaskCardSettings(defaults);
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TrainingError(400, '任务卡必须为JSON对象');
  const input = value as Record<string, unknown>;
  const rewards = (name: 'stage1Rewards' | 'stage2Rewards') => {
    if (taskId === 'Mjlab-Zbot-6dof-InPlace-Stepping' && name === 'stage1Rewards') return {};
    const record = input[name];
    const expected = Object.keys(defaults[name]);
    const supplied = record && typeof record === 'object' && !Array.isArray(record) ? Object.keys(record) : [];
    const optional = name === 'stage2Rewards' ? new Set(['energy_consumption', 'small_step', 'body_shake']) : new Set<string>();
    if (!record || typeof record !== 'object' || Array.isArray(record) || supplied.some(key => !expected.includes(key))
      || expected.some(key => !optional.has(key) && !supplied.includes(key))) throw new TrainingError(400, `${name}奖励项不完整`);
    const result: Record<string, number> = {};
    for (const key of expected) {
      const number = (record as Record<string, unknown>)[key] ?? defaults[name][key];
      if (typeof number !== 'number' || !Number.isFinite(number) || number < -100 || number > 100) throw new TrainingError(400, `${name}.${key}权重必须在-100到100之间`);
      result[key] = number;
    }
    return result;
  };
  const penalty = input.terminatedRewardPenalty;
  if (typeof penalty !== 'number' || !Number.isFinite(penalty) || penalty < 0 || penalty > 1000) throw new TrainingError(400, '终止惩罚必须在0到1000之间');
  return { stage1Rewards: rewards('stage1Rewards'), stage2Rewards: rewards('stage2Rewards'), terminatedRewardPenalty: penalty };
}

export function sanitizeLiveFrame(value: unknown, iteration: number): TrainingLiveFrame | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const raw = value as Record<string, unknown>;
  if (!Array.isArray(raw.bodyNames) || raw.bodyNames.length < 1 || raw.bodyNames.length > 32
    || raw.bodyNames.some(name => typeof name !== 'string' || !name || name.length > 80)) return undefined;
  if (!Array.isArray(raw.environments) || raw.environments.length < 1 || raw.environments.length > 9) return undefined;
  const vector = (item: unknown, size: number, bound: number): number[] | undefined => Array.isArray(item) && item.length === size
    && item.every(number => typeof number === 'number' && Number.isFinite(number) && Math.abs(number) <= bound) ? [...item] : undefined;
  const environments: TrainingLiveFrame['environments'] = [];
  for (const item of raw.environments) {
    if (!item || typeof item !== 'object') return undefined;
    const environment = item as Record<string, unknown>;
    if (!Array.isArray(environment.bodyPositions) || !Array.isArray(environment.bodyQuaternions)
      || environment.bodyPositions.length !== raw.bodyNames.length || environment.bodyQuaternions.length !== raw.bodyNames.length) return undefined;
    const bodyPositions = environment.bodyPositions.map(item => vector(item, 3, 10_000));
    const bodyQuaternions = environment.bodyQuaternions.map(item => vector(item, 4, 2));
    const environmentOrigin = environment.environmentOrigin === undefined ? undefined : vector(environment.environmentOrigin, 3, 10_000);
    if (bodyPositions.some(item => !item) || bodyQuaternions.some(item => !item) || environment.environmentOrigin !== undefined && !environmentOrigin) return undefined;
    environments.push({ ...(environmentOrigin ? { environmentOrigin: environmentOrigin as [number, number, number] } : {}), bodyPositions: bodyPositions as [number, number, number][], bodyQuaternions: bodyQuaternions as [number, number, number, number][] });
  }
  return { iteration, capturedAt: new Date().toISOString(), bodyNames: [...raw.bodyNames] as string[], environments };
}
export interface TrainingServiceOptions {
  workDir?: string;
  python?: string;
  worker?: string;
  probe?: string;
  scannedRoots?: string[];
  launch?: (command: string, args: string[], options: SpawnOptions) => ChildProcess;
  killProcess?: (child: ChildProcess, signal: NodeJS.Signals) => void;
  resources?: () => Promise<TrainingResources>;
  stopGraceMs?: number;
}

async function processMarker(pid: number) {
  return readFile(`/proc/${pid}/stat`, 'utf8').then(text => text.slice(text.lastIndexOf(')') + 2).trim().split(/\s+/)[19]).catch(() => undefined);
}
async function lockWorkDir(workDir: string) {
  const file = path.join(workDir, 'service.lock');
  const token = randomUUID();
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const handle = await open(file, 'wx');
      try { await handle.writeFile(JSON.stringify({ pid: process.pid, marker: await processMarker(process.pid), token })); }
      finally { await handle.close(); }
      return async () => {
        const current = await readFile(file, 'utf8').then(JSON.parse).catch(() => undefined);
        if (current?.token === token) await unlink(file);
      };
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
      const text = await readFile(file, 'utf8').catch(() => undefined);
      if (text === undefined) continue;
      let owner: { pid?: number; marker?: string };
      try { owner = JSON.parse(text); } catch { throw new TrainingError(409, '训练服务锁正在初始化，请稍后重试'); }
      if (!Number.isInteger(owner?.pid) || owner.pid! < 1) throw new TrainingError(409, '训练服务锁无效，请检查 service.lock');
      let alive = true;
      try { process.kill(owner.pid!, 0); } catch (failure) { alive = (failure as NodeJS.ErrnoException).code !== 'ESRCH'; }
      if (alive && (!owner.marker || owner.marker === await processMarker(owner.pid!))) throw new TrainingError(409, '已有训练服务正在使用此工作目录');
      // Recover a dead owner only while the recorded lock is still the same.
      if (await readFile(file, 'utf8').catch(() => undefined) === text) await unlink(file).catch(error => { if (error.code !== 'ENOENT') throw error; });
    }
  }
  throw new TrainingError(409, '训练工作目录正在被其他进程使用');
}

export async function createTrainingService(options: TrainingServiceOptions = {}) {
  const workDir = path.resolve(options.workDir ?? '.training');
  const jobsDir = path.join(workDir, 'jobs');
  const repoRoot = process.cwd();
  const pythonRelativePath = os.platform() === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python';
  const pythonCandidates = [path.join(repoRoot, 'training', pythonRelativePath)];
  const python = options.python ?? process.env.ZBOT_TRAINING_PYTHON ?? pythonCandidates.find(existsSync) ?? (os.platform() === 'win32' ? 'python' : path.join(os.homedir(), 'mjlab', '.venv', 'bin', 'python'));
  const worker = path.resolve(options.worker ?? 'training/worker.py');
  const probe = path.resolve(options.probe ?? 'training/probe.py');
  let operatorRoots: string[] | undefined;
  if (process.env.ZBOT_TRAINING_SCAN_ROOTS !== undefined) {
    try {
      const value: unknown = JSON.parse(process.env.ZBOT_TRAINING_SCAN_ROOTS);
      if (!Array.isArray(value) || value.length > 10 || value.some(root => typeof root !== 'string' || !root.trim())) throw new Error();
      operatorRoots = value.map(root => path.resolve(repoRoot, root));
    } catch { throw new TrainingError(400, 'ZBOT_TRAINING_SCAN_ROOTS 必须是最多 10 个目录字符串组成的 JSON 数组'); }
  }
  const scannedRoots = options.scannedRoots ?? operatorRoots ?? [path.join(repoRoot, 'training', 'vendor', 'zbot_rl_mjlab', 'logs')];
  const jobs = new Map<string, TrainingJob>();
  let active: { job: TrainingJob; child?: ChildProcess; stopRequested: boolean; deadline?: ReturnType<typeof setTimeout>; killTimer?: ReturnType<typeof setTimeout>; logBytes: number; done: Promise<void>; resolveDone: () => void } | undefined;
  let closed = false, pumping = false;
  let closePromise: Promise<void> | undefined;
  let hardwarePending: Promise<TrainingResources> | undefined;
  const submissions = new Set<Promise<TrainingJob>>();
  let persistence = Promise.resolve();
  const scheduledWrites = new Map<string, Promise<void>>();
  const dirtyWrites = new Set<string>();
  const liveListeners = new Map<string, Set<(frame: TrainingLiveFrame) => void>>();
  let gpuCache: { at: number; value: TrainingResources['gpus'] } | undefined;
  let runtimeCache: { at: number; value: TrainingResources['runtime'] } | undefined;
  const dir = (id: string) => path.join(jobsDir, id);
  const kill = options.killProcess ?? ((child: ChildProcess, signal: NodeJS.Signals) => {
    if (!child.pid) return;
    if (os.platform() === 'win32') {
      // The STOP marker requests a cooperative exit. Windows has no POSIX
      // process groups, and SIGTERM would terminate Python immediately.
      if (signal !== 'SIGTERM') child.kill(signal);
      return;
    }
    try { process.kill(-child.pid, signal); } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ESRCH') throw error; }
  });
  function persist(job: TrainingJob) {
    dirtyWrites.add(job.id);
    const pending = scheduledWrites.get(job.id);
    if (pending) return pending;
    const target = path.join(dir(job.id), 'job.json');
    persistence = persistence.catch(() => {}).then(async () => {
      do {
        dirtyWrites.delete(job.id);
        await writeFile(target + '.tmp', JSON.stringify(job));
        await rename(target + '.tmp', target);
      } while (dirtyWrites.has(job.id));
    });
    const write = persistence.finally(() => { scheduledWrites.delete(job.id); });
    scheduledWrites.set(job.id, write);
    return write;
  }
  async function walk(root: string, depth = 0, budget = { remaining: 1200 }): Promise<string[]> {
    if (depth > 6 || budget.remaining <= 0) return [];
    const result: string[] = [];
    const entries = await readdir(root, { withFileTypes: true }).catch(() => []);
    for (const entry of entries) {
      if (--budget.remaining < 0) break;
      if (entry.isSymbolicLink() || entry.name.toLowerCase() === 'exported') continue;
      const file = path.join(root, entry.name);
      if (entry.isDirectory()) result.push(...await walk(file, depth + 1, budget));
      else if (entry.isFile() && /\.(pt|pth)$/i.test(entry.name)) result.push(file);
    }
    return result;
  }
  async function refresh(job: TrainingJob) {
    const root = path.join(dir(job.id), 'checkpoints');
    const files = (await walk(root)).filter(file => checkpointPattern.test(path.basename(file)));
    const checkpoints = await Promise.all(files.map(async file => {
      const info = await stat(file).catch(() => undefined);
      return info && { name: path.relative(root, file).split(path.sep).join('/'), bytes: info.size, modifiedAt: info.mtime.toISOString() };
    }));
    job.checkpoints = checkpoints.filter((checkpoint): checkpoint is NonNullable<typeof checkpoint> => Boolean(checkpoint && checkpoint.bytes > 0)).sort((a, b) => a.modifiedAt.localeCompare(b.modifiedAt));
    job.bundleReady = await stat(path.join(dir(job.id), 'bundle.json')).then(info => info.isFile() && info.size > 0).catch(() => false);
  }
  await mkdir(jobsDir, { recursive: true });
  const unlock = await lockWorkDir(workDir);
  for (const entry of await readdir(jobsDir, { withFileTypes: true })) {
    if (!entry.isDirectory() || !idPattern.test(entry.name)) continue;
    try {
      const job = JSON.parse(await readFile(path.join(dir(entry.name), 'job.json'), 'utf8')) as TrainingJob;
      if (job.id !== entry.name || !job.config || typeof job.createdAt !== 'string') continue;
      job.livePreviewEnabled = false;
      job.taskCard = validateTaskCard(job.taskCard, job.config.taskId);
      job.config.taskCard = validateTaskCard(job.config.taskCard, job.config.taskId);
      job.taskCardRevision ??= 0;
      await unlink(path.join(dir(entry.name), 'LIVE_PREVIEW')).catch(error => { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; });
      if (activeStatuses.has(job.status) || job.status === 'queued') {
        job.status = 'failed'; job.error = '训练服务已重启，原任务被中断；可从已保存的检查点恢复。'; job.finishedAt = new Date().toISOString();
      }
      await refresh(job); jobs.set(job.id, job); await persist(job);
    } catch { /* Ignore incomplete job records left by a failed disk write. */ }
  }

  async function hardware(): Promise<TrainingResources> {
    if (options.resources) return options.resources();
    if (hardwarePending) return hardwarePending;
    hardwarePending = inspectHardware();
    try { return await hardwarePending; } finally { hardwarePending = undefined; }
  }
  async function inspectHardware(): Promise<TrainingResources> {
    const now = Date.now();
    if (!gpuCache || now - gpuCache.at > 3000) {
      let gpus: TrainingResources['gpus'] = [];
      try {
        const { stdout } = await execute('nvidia-smi', ['--query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu,driver_version', '--format=csv,noheader,nounits'], { timeout: 2500, maxBuffer: 64 * 1024 });
        gpus = stdout.trim().split('\n').filter(Boolean).map(line => {
          const [id, name, uuid, total, free, usage, driver] = line.split(',').map(value => value.trim());
          return { id: Number(id), name, uuid, totalMemoryMiB: Number(total), freeMemoryMiB: Number(free), utilization: Number(usage), driver };
        }).filter(gpu => Number.isInteger(gpu.id) && Number.isFinite(gpu.totalMemoryMiB));
      } catch { /* CPU-only hosts are supported. */ }
      gpuCache = { at: now, value: gpus };
    }
    if (!runtimeCache || now - runtimeCache.at > 60000) {
      let runtime: TrainingResources['runtime'];
      try {
        const { stdout } = await execute(python, [probe], { timeout: 20000, maxBuffer: 128 * 1024 });
        const value = JSON.parse(stdout.trim().split('\n').filter(Boolean).at(-1)!);
        runtime = { ...value, available: value.available === true, python };
      } catch (error) { runtime = { available: false, python, error: errorText(error) }; }
      runtimeCache = { at: now, value: runtime };
    }
    const cpus = os.cpus();
    let availableBytes = os.freemem();
    if (os.platform() === 'linux') {
      const meminfo = await readFile('/proc/meminfo', 'utf8').catch(() => '');
      const available = /^MemAvailable:\s+(\d+)\s+kB$/m.exec(meminfo);
      if (available) availableBytes = Number(available[1]) * 1024;
    }
    return { hostname: os.hostname(), platform: os.platform(), cpu: { model: cpus[0]?.model ?? 'CPU', logicalCores: cpus.length, availableThreads: os.availableParallelism() }, memory: { totalBytes: os.totalmem(), availableBytes }, gpus: gpuCache.value, runtime: runtimeCache.value, tasks: TRAINING_TASKS, scannedRoots: [], checkpoints: [] };
  }
  async function resources(): Promise<TrainingResources> {
    const result = await hardware();
    const roots = [jobsDir, ...scannedRoots];
    const budget = { remaining: 1500 };
    const files: string[] = [];
    for (const root of roots) files.push(...await walk(root, 0, budget));
    const owned = new Set<string>();
    for (const job of jobs.values()) {
      await refresh(job);
      for (const checkpoint of job.checkpoints) owned.add(path.join(dir(job.id), 'checkpoints', checkpoint.name));
    }
    const checkpoints = await Promise.all(files.slice(0, 500).map(async file => {
      const info = await stat(file).catch(() => undefined);
      return info && { name: path.basename(file), path: file, bytes: info.size, modifiedAt: info.mtime.toISOString(), resumable: owned.has(file) };
    }));
    return { ...result, tasks: TRAINING_TASKS, scannedRoots: roots, checkpoints: checkpoints.filter((checkpoint): checkpoint is NonNullable<typeof checkpoint> => Boolean(checkpoint)) };
  }
  function get(id: string) {
    if (!idPattern.test(id) || !jobs.has(id)) throw new TrainingError(404, '训练任务不存在');
    return jobs.get(id)!;
  }
  async function checkpointPath(id: string, name: string) {
    const job = get(id); await refresh(job);
    if (!job.checkpoints.some(checkpoint => checkpoint.name === name)) throw new TrainingError(404, '检查点不存在');
    const root = await realpath(path.join(dir(id), 'checkpoints'));
    const file = await realpath(path.join(root, name));
    if (!within(root, file) || !(await stat(file)).isFile()) throw new TrainingError(400, '无效检查点路径');
    return file;
  }
  async function validate(input: unknown): Promise<TrainingConfig> {
    if (!input || typeof input !== 'object' || Array.isArray(input)) throw new TrainingError(400, '训练参数必须为 JSON 对象');
    const value = input as TrainingConfig;
    const resource = await hardware();
    if (!resource.runtime.available) throw new TrainingError(503, resource.runtime.error ?? '本机训练环境尚未安装');
    if (!TRAINING_TASKS.some(task => task.id === value.taskId)) throw new TrainingError(400, '未知训练任务');
    if (value.device !== 'cpu' && resource.runtime.cudaBuild === false) throw new TrainingError(400, '当前 PyTorch 未安装 CUDA 支持，请选择 CPU');
    if (value.device !== 'cpu' && !resource.gpus.some(gpu => value.device === `cuda:${gpu.id}`)) throw new TrainingError(400, '请选择当前可用的 CPU 或 GPU');
    const integer = (name: keyof TrainingConfig, min: number, max: number) => {
      const v = value[name];
      if (typeof v !== 'number' || !Number.isInteger(v) || v < min || v > max) throw new TrainingError(400, `${name} 必须为 ${min}–${max} 的整数`);
      return v;
    };
    const config: TrainingConfig = { taskId: value.taskId, device: value.device, cpuThreads: integer('cpuThreads', 1, resource.cpu.availableThreads), numEnvs: integer('numEnvs', 1, value.device === 'cpu' ? 64 : 4096), iterations: integer('iterations', 1, 20000), saveInterval: integer('saveInterval', 1, 1000), seed: integer('seed', 0, 2147483647), maxSeconds: integer('maxSeconds', 30, 86400), taskCard: validateTaskCard(value.taskCard, value.taskId) };
    if (value.resumeJobId !== undefined || value.resumeCheckpoint !== undefined) {
      if (typeof value.resumeJobId !== 'string' || typeof value.resumeCheckpoint !== 'string') throw new TrainingError(400, '恢复训练需要任务 ID 和检查点名称');
      const previous = get(value.resumeJobId);
      if (previous.config.taskId !== config.taskId) throw new TrainingError(400, '检查点训练任务不匹配');
      await checkpointPath(value.resumeJobId, value.resumeCheckpoint);
      config.resumeJobId = value.resumeJobId; config.resumeCheckpoint = value.resumeCheckpoint;
    }
    return config;
  }
  function log(job: TrainingJob, line: string) {
    job.logTail = (job.logTail + line + '\n').slice(-100 * 1024).split('\n').slice(-201).join('\n');
    const running = active;
    if (running?.job === job && running.logBytes < 5 * 1024 * 1024) {
      const text = (line.slice(0, 64 * 1024) + '\n').slice(0, 5 * 1024 * 1024 - running.logBytes); running.logBytes += Buffer.byteLength(text);
      void appendFile(path.join(dir(job.id), 'worker.log'), text).catch(() => {});
    }
  }
  async function finish(job: TrainingJob, code: number | null, signal: NodeJS.Signals | null, error?: Error) {
    if (active?.job !== job) return;
    const running = active;
    clearTimeout(running.deadline); clearTimeout(running.killTimer);
    await refresh(job);
    job.status = running.stopRequested ? 'stopped' : code === 0 && job.checkpoints.length && !error && !job.error ? 'completed' : 'failed';
    if (job.status === 'failed') job.error = error?.message ?? job.error ?? (code === 0 ? '训练结束但未生成检查点' : `训练进程退出（${signal ?? code}），请查看日志`);
    job.finishedAt = new Date().toISOString();
    try { await persist(job); } finally {
      active = undefined;
      running.resolveDone();
    }
    void pump();
  }
  async function pump() {
    if (closed || active || pumping) return;
    const job = [...jobs.values()].find(item => item.status === 'queued');
    if (!job) return;
    pumping = true;
    let resolveDone!: () => void;
    const done = new Promise<void>(resolve => { resolveDone = resolve; });
    active = { job, stopRequested: false, logBytes: 0, done, resolveDone };
    try {
      job.status = 'starting'; job.startedAt = new Date().toISOString(); await persist(job);
      const running = active;
      if (!running || closed || running.stopRequested) { await finish(job, null, null); return; }
      const child = (options.launch ?? spawn)(python, [worker, '--request', path.join(dir(job.id), 'request.json'), '--job-dir', dir(job.id)], { cwd: path.dirname(path.dirname(worker)), detached: true, shell: false, stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, PYTHONUNBUFFERED: '1', OMP_NUM_THREADS: String(job.config.cpuThreads), MKL_NUM_THREADS: String(job.config.cpuThreads) } });
      running.child = child;
      running.deadline = setTimeout(() => { log(job, '已达到训练时间上限，正在保存并停止。'); void stop(job.id); }, job.config.maxSeconds * 1000);
      running.deadline.unref();
      let pending = '';
      const onLine = (line: string) => {
        line = line.slice(0, 64 * 1024);
        let event: any;
        try { event = JSON.parse(line); }
        catch { log(job, line); void persist(job).catch(() => {}); return; }
        const frameCount = Array.isArray(event.liveFrame?.environments) ? event.liveFrame.environments.length : 0;
        if (event.event !== 'liveFrame') log(job, frameCount ? JSON.stringify({ ...event, liveFrame: `${frameCount} environments` }) : line);
        try {
          if (event.event === 'ready' && job.status === 'starting') job.status = 'running';
          if (event.event === 'progress') {
            for (const key of ['iteration', 'totalIterations', 'reward', 'episodeLength', 'fps', 'loss'] as const) if (typeof event[key] === 'number' && Number.isFinite(event[key])) job.metrics[key] = event[key];
            if (event.rewardTerms && typeof event.rewardTerms === 'object' && !Array.isArray(event.rewardTerms)) {
              const allowed = new Set([...Object.keys(job.taskCard?.stage1Rewards ?? {}), ...Object.keys(job.taskCard?.stage2Rewards ?? {})]);
              const rewardTerms: Record<string, number> = {};
              for (const [key, value] of Object.entries(event.rewardTerms)) if (allowed.has(key) && typeof value === 'number' && Number.isFinite(value) && Math.abs(value) <= 1_000_000) rewardTerms[key] = value;
              if (Object.keys(rewardTerms).length) job.metrics.rewardTerms = rewardTerms;
            }
            if (Number.isInteger(event.iteration) && event.iteration >= 0) {
              const liveFrame = sanitizeLiveFrame(event.liveFrame, event.iteration);
              if (liveFrame) job.liveFrame = liveFrame;
              const history = job.history ??= [];
              let point = history.at(-1);
              if (!point || event.iteration > point.iteration) {
                point = { iteration: event.iteration };
                history.push(point);
                if (history.length > 200) history.splice(0, history.length - 200);
              }
              // mjlab prints the iteration header before its reward metrics.
              // Update the same point as later metric lines arrive.
              if (point.iteration === event.iteration) {
                if (typeof event.reward === "number" && Number.isFinite(event.reward)) point.reward = event.reward;
                if (typeof event.loss === "number" && Number.isFinite(event.loss)) point.loss = event.loss;
                if (job.metrics.rewardTerms) point.rewardTerms = { ...job.metrics.rewardTerms };
              }
            }
          }
          if (event.event === 'liveFrame' && Number.isInteger(event.liveFrame?.iteration) && event.liveFrame.iteration >= 0) {
            const liveFrame = sanitizeLiveFrame(event.liveFrame, event.liveFrame.iteration);
            if (liveFrame) {
              job.liveFrame = liveFrame;
              for (const listener of liveListeners.get(job.id) ?? []) listener(liveFrame);
            }
          }
          if (event.event === 'finished' && event.stopped === true && active?.job === job) active.stopRequested = true;
          if (event.event === 'error') job.error = String(event.message ?? '训练出错').slice(0, 4000);
          if (event.event === 'checkpoint' || event.event === 'bundle') void refresh(job).then(() => persist(job)).catch(() => {});
        } catch { /* Malformed protocol fields are ignored after the bounded log copy. */ }
        if (event.event !== 'liveFrame') void persist(job).catch(() => {});
      };
      child.stdout?.on('data', data => {
        pending += data.toString();
        let index: number;
        while ((index = pending.indexOf('\n')) >= 0) { onLine(pending.slice(0, index)); pending = pending.slice(index + 1); }
        if (pending.length > 64 * 1024) { onLine(pending.slice(0, 64 * 1024)); pending = ''; }
      });
      child.stderr?.on('data', data => { for (const line of data.toString().split('\n').filter(Boolean)) log(job, line); void persist(job).catch(() => {}); });
      let finalized = false;
      const finalize = (code: number | null, signal: NodeJS.Signals | null, error?: Error) => { if (finalized) return; finalized = true; if (pending) onLine(pending); void finish(job, code, signal, error); };
      child.once('error', error => finalize(null, null, error));
      child.once('close', (code, signal) => finalize(code, signal));
    } catch (error) { await finish(job, null, null, error instanceof Error ? error : new Error(String(error))); }
    finally { pumping = false; if (!active) void pump(); }
  }
  function start(input: unknown) {
    const submission = submit(input);
    submissions.add(submission);
    void submission.finally(() => submissions.delete(submission)).catch(() => {});
    return submission;
  }
  async function submit(input: unknown) {
    if (closed) throw new TrainingError(503, '训练服务已关闭');
    if ([...jobs.values()].filter(job => job.status === 'queued').length + submissions.size >= 32) throw new TrainingError(429, '训练队列已满，请等待当前任务完成');
    const config = await validate(input);
    if (closed) throw new TrainingError(503, '训练服务已关闭');
    const id = randomUUID();
    const job: TrainingJob = { id, status: 'queued', config, createdAt: new Date().toISOString(), metrics: { iteration: 0, totalIterations: config.iterations }, checkpoints: [], logTail: '', bundleReady: false, livePreviewEnabled: false, taskCard: copyTaskCardSettings(config.taskCard!), taskCardRevision: 0 };
    await mkdir(path.join(dir(id), 'checkpoints'), { recursive: true });
    const resumePath = config.resumeJobId ? await checkpointPath(config.resumeJobId, config.resumeCheckpoint!) : undefined;
    await writeFile(path.join(dir(id), 'request.json'), JSON.stringify({ ...config, ...(resumePath ? { resumePath } : {}) }), { flag: 'wx' });
    jobs.set(id, job); await persist(job); void pump(); return job;
  }
  async function stop(id: string) {
    const job = get(id);
    if (job.status === 'queued') { job.status = 'stopped'; job.finishedAt = new Date().toISOString(); await persist(job); return job; }
    if (active?.job !== job || active.stopRequested) return job;
    const running = active; running.stopRequested = true; job.status = 'stopping';
    await writeFile(path.join(dir(id), 'STOP'), 'stop\n'); await persist(job);
    if (running.child) {
      kill(running.child, 'SIGTERM');
      running.killTimer = setTimeout(() => { if (active === running && running.child) kill(running.child, 'SIGKILL'); }, options.stopGraceMs ?? 20000);
      running.killTimer.unref();
    }
    return job;
  }
  async function setPreview(id: string, enabled: unknown) {
    if (typeof enabled !== 'boolean') throw new TrainingError(400, '训练画面开关必须是布尔值');
    const job = get(id);
    const marker = path.join(dir(id), 'LIVE_PREVIEW');
    if (enabled) await writeFile(marker, 'enabled\n');
    else await unlink(marker).catch(error => { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; });
    job.livePreviewEnabled = enabled;
    await persist(job);
    return job;
  }
  async function setTaskCard(id: string, value: unknown) {
    const job = get(id);
    if (!['queued', 'starting', 'running'].includes(job.status)) throw new TrainingError(409, '只能修改等待或运行中的任务卡');
    const taskCard = validateTaskCard(value, job.config.taskId);
    const revision = (job.taskCardRevision ?? 0) + 1;
    const file = path.join(dir(id), 'TASK_CARD.json');
    await writeFile(file + '.tmp', JSON.stringify({ revision, taskCard }));
    await rename(file + '.tmp', file);
    job.taskCard = taskCard; job.taskCardRevision = revision;
    await persist(job);
    return job;
  }
  async function bundle(id: string) {
    const job = get(id); await refresh(job);
    if (!job.bundleReady) throw new TrainingError(404, '回放包尚未生成');
    const file = path.join(dir(id), 'bundle.json');
    const resolved = await realpath(file);
    if (!within(await realpath(dir(id)), resolved)) throw new TrainingError(400, '无效回放包路径');
    return resolved;
  }
  function subscribeLive(id: string, listener: (frame: TrainingLiveFrame) => void) {
    const job = get(id);
    const listeners = liveListeners.get(id) ?? new Set<(frame: TrainingLiveFrame) => void>();
    listeners.add(listener); liveListeners.set(id, listeners);
    if (job.liveFrame) listener(job.liveFrame);
    return () => {
      listeners.delete(listener);
      if (!listeners.size) liveListeners.delete(id);
      return listeners.size;
    };
  }
  return {
    resources, start, stop, setPreview, setTaskCard, subscribeLive, get, checkpointPath, bundle,
    list: () => [...jobs.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt)),
    close() {
      if (!closePromise) {
        closed = true;
        closePromise = (async () => {
          await Promise.allSettled([...submissions]);
          // Queued work must never start while the active process is shutting down.
          for (const job of jobs.values()) if (job.status === 'queued') await stop(job.id);
          const running = active;
          if (running) {
            await stop(running.job.id);
            await running.done;
          }
          await persistence;
          await unlock();
        })();
      }
      return closePromise;
    },
  };
}
export type TrainingService = Awaited<ReturnType<typeof createTrainingService>>;
