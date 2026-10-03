import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Pause, Play, RotateCcw, SkipForward, X } from 'lucide-react';
import { MujocoEngine } from '../mujoco/MujocoEngine';
import { ZbotConfiguration } from '../types/zbot';
import { SimulationViewport } from './SimulationViewport';
import { loadCheckpoint } from '../rl/checkpoint';
import { JOINT_SPEED_LIMIT, REPLAY_PROFILES } from '../rl/profiles';
import { PolicyReplay } from '../rl/replay';
import type { PhysxModel } from '../rl/physx';
import { PhysxWorkerSimulation } from '../rl/physxWorkerClient';
import type { TrainingReplayBundle, TrainingLiveFrame } from '../training/types';
import { TrainingClient } from '../training/client';
import { TrainingLiveView } from './TrainingLiveView';
import { loadTrainingBundle, trainingBundleProfile } from '../training/bundle';
import bundledPhysxModels from 'virtual:zbot-physx-models';
import { getRlDebugRecords, logRlDebug, subscribeRlDebug } from '../rl/debugLog';

const MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024;
const asset = (path: string) => `${import.meta.env.BASE_URL}rl/${path}`;

async function fetchWeights(url: string, signal: AbortSignal): Promise<ArrayBuffer> {
  const parsed = new URL(url, window.location.href);
  if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('权重网址必须使用 HTTP 或 HTTPS');
  const response = await fetch(parsed, { signal, credentials: 'omit' });
  if (!response.ok) throw new Error(`权重下载失败（HTTP ${response.status}）`);
  if (Number(response.headers.get('content-length')) > MAX_CHECKPOINT_BYTES) throw new Error('权重文件超过 128 MiB 限制');
  if (!response.body) throw new Error('权重响应没有内容');
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      const result = await reader.read();
      if (result.done) break;
      length += result.value.byteLength;
      if (length > MAX_CHECKPOINT_BYTES) throw new Error('权重文件超过 128 MiB 限制');
      chunks.push(result.value);
    }
  } catch (error) { await reader.cancel(); throw error; }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes.buffer;
}

export function RlReplayPanel({ onClose, initialBundle }: { onClose: () => void; initialBundle?: TrainingReplayBundle }) {
  const [engine] = useState(() => new MujocoEngine());
  const [ready, setReady] = useState(false);
  const [profileId, setProfileId] = useState(REPLAY_PROFILES[0].id);
  const [backend, setBackend] = useState<'physx' | 'mujoco' | 'mjwarp'>('physx');
  const [nativeReplayId, setNativeReplayId] = useState('');
  const [nativeFrame, setNativeFrame] = useState<TrainingLiveFrame>();
  const nativeClient = useMemo(() => new TrainingClient(), []);
  const [importedBundle, setImportedBundle] = useState<TrainingReplayBundle | null>(null);
  const importedProfile = useMemo(() => importedBundle ? trainingBundleProfile(importedBundle) : null, [importedBundle]);
  const profile = importedProfile ?? REPLAY_PROFILES.find(p => p.id === profileId)!;
  const [source, setSource] = useState<'example' | 'file' | 'url' | 'bundle'>('example');
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState('');
  const [running, setRunning] = useState(false);
  const [loadedName, setLoadedName] = useState('');
  const [error, setError] = useState('');
  const [speed, setSpeed] = useState(1);
  const [commands, setCommands] = useState<[number, number, number]>([0, 0, 0]);
  const [jointSpeedLimit, setJointSpeedLimit] = useState(JOINT_SPEED_LIMIT);
  const [periodicFrequency, setPeriodicFrequency] = useState(1);
  const [metrics, setMetrics] = useState(() => engine.getMetrics());
  const [steps, setSteps] = useState(0);
  const [actionPeak, setActionPeak] = useState(0);
  const [debugRecords, setDebugRecords] = useState(() => [...getRlDebugRecords()]);
  const [showPhysicalEntity, setShowPhysicalEntity] = useState(false);
  const [showDisplayEntity, setShowDisplayEntity] = useState(true);
  const diagnosticsRef = useRef({ contacts: false, forces: false, centerOfMass: false });
  const runner = useRef<PolicyReplay | null>(null);
  const physics = useRef<PhysxWorkerSimulation | null>(null);
  const runningRef = useRef(false);
  const speedRef = useRef(speed);
  speedRef.current = speed;
  const requestRef = useRef<AbortController | null>(null);
  const metricsRef = useRef(metrics);
  const generation = useRef(0);
  const displayConfig = useMemo<ZbotConfiguration>(() => ({
    id: profile.id, name: profile.label, category: 'custom', baseMode: 'free', description: '强化学习策略回放',
    rootPos: [0, 0, 0], rootEuler: [0, 0, 0],
    modules: profile.jointNames.map((name, i) => ({ id: name, name, parentId: i ? profile.jointNames[i - 1] : null,
      dockAngle: 0, jointAxis: [0, -1, 1], jointRange: [-180, 180], colorA: '#0284c7', colorB: '#7dd3fc' })),
    defaultGait: { type: 'manual', frequency: 1, amplitude: 0, phaseLag: 0, steering: 0, speed: 1, manualAngles: {} },
  }), [profile]);

  const pause = () => { runningRef.current = false; setRunning(false); };
  const update = () => {
    const next = engine.getMetrics(metricsRef.current);
    metricsRef.current = next; setMetrics(next);
    setSteps(runner.current?.controlSteps ?? 0);
    setActionPeak(runner.current ? Math.max(0, ...Array.from(runner.current.lastActions, Math.abs)) : 0);
  };

  useEffect(() => {
    return subscribeRlDebug(() => setDebugRecords([...getRlDebugRecords()]));
  }, []);

  useEffect(() => {
    let active = true;
    logRlDebug('mujoco.initialize.start');
    engine.init().then(() => {
      logRlDebug('mujoco.initialize.done');
      if (active) setReady(true);
    }).catch(err => {
      logRlDebug('mujoco.initialize.error', { message: String(err) });
      if (active) setError(String(err));
    });
    return () => {
      active = false; generation.current++; requestRef.current?.abort(); runningRef.current = false;
      runner.current = null; physics.current?.dispose(); physics.current = null; engine.destroy();
    };
  }, [engine]);

  useEffect(() => {
    if (!running) return;
    let frame = 0, last = performance.now(), accumulated = 0;
    let stepping = false;
    const tick = async (now: number) => {
      if (document.hidden || !runningRef.current) return;
      if (stepping) { frame = requestAnimationFrame(value => void tick(value)); return; }
      stepping = true;
      const elapsed = Math.min((now - last) / 1000, .1); last = now;
      if (runningRef.current && runner.current) {
        accumulated += elapsed * speedRef.current;
        try {
          // Bound per-frame work, keeping the UI responsive on CPU-only machines.
          let count = 0;
          while (accumulated >= runner.current.controlDt && count++ < 6) {
            await runner.current.stepAsync(); accumulated -= runner.current.controlDt;
          }
          if (count > 0) update();
        } catch (err) { runningRef.current = false; setRunning(false); setError((err as Error).message); }
      } else accumulated = 0;
      stepping = false;
      if (runningRef.current) frame = requestAnimationFrame(tick);
    };
    const visibilityChanged = () => {
      cancelAnimationFrame(frame);
      last = performance.now(); accumulated = 0;
      if (!document.hidden && runningRef.current) frame = requestAnimationFrame(value => void tick(value));
    };
    document.addEventListener('visibilitychange', visibilityChanged);
    visibilityChanged();
    return () => { cancelAnimationFrame(frame); document.removeEventListener('visibilitychange', visibilityChanged); };
  }, [engine, running]);

  const clearSelection = () => {
    pause(); generation.current++; requestRef.current?.abort(); setLoading(false); setLoadingStage('');
    runner.current = null; physics.current?.dispose(); physics.current = null;
    engine.cleanupModel(); setLoadedName(''); setError(''); update();
  };
  const cancelLoad = () => {
    generation.current++;
    requestRef.current?.abort(); requestRef.current = null;
    setLoading(false); setLoadingStage(''); setError('已取消加载。');
  };
  const applyBundle = (value: unknown) => {
    pause(); runner.current = null; setLoadedName('');
    physics.current?.dispose(); physics.current = null;
    const loaded = loadTrainingBundle(engine, value);
    setImportedBundle(loaded.bundle); setSource('bundle'); setBackend('mujoco');
    runner.current = loaded.replay;
    setJointSpeedLimit(loaded.replay.getJointSpeedLimit());
    setLoadedName(loaded.bundle.name); setError(''); update();
  };
  useEffect(() => {
    if (!ready || !initialBundle) return;
    try { applyBundle(initialBundle); } catch (err) { setError((err as Error).message); }
  }, [ready, initialBundle]);
  const load = async () => {
    let stage = '读取权重';
    const loadStarted = performance.now();
    const updateStage = (value: string) => {
      stage = value; setLoadingStage(value);
      logRlDebug('replay.load.stage', { profile: profile.id, stage, elapsedMs: Math.round(performance.now() - loadStarted) });
    };
    logRlDebug('replay.load.start', { profile: profile.id, backend, source });
    pause(); setError(''); setLoading(true); updateStage(stage);
    if (backend === 'mjwarp') {
      try {
        const native = await nativeClient.nativeReplay({ frequency: periodicFrequency, device: 'cuda:0', steps: 600 });
        setNativeReplayId(native.id); setNativeFrame(undefined); setLoadedName('MJLab native model_599 / MJWarp'); setRunning(true);
      } catch (err) { setError((err as Error).message); }
      finally { setLoading(false); setLoadingStage(''); }
      return;
    }
    const token = ++generation.current;
    requestRef.current?.abort();
    const controller = new AbortController(); requestRef.current = controller;
    let timedOut = false;
    const timeout = window.setTimeout(() => { timedOut = true; controller.abort(); }, 30_000);
    try {
      if (source === 'example' && profile.bundlePath) {
        const response = await fetch(asset(profile.bundlePath), { signal: controller.signal });
        if (!response.ok) throw new Error("回放资产包加载失败");
        if (token === generation.current) applyBundle(await response.json());
        return;
      }
      if (source === 'bundle') {
        if (file) {
          if (file.size > MAX_CHECKPOINT_BYTES) throw new Error('训练结果包过大');
          const value = JSON.parse(await file.text());
          if (token === generation.current) applyBundle(value);
        } else if (importedBundle) applyBundle(importedBundle);
        else throw new Error('请选择训练结果 JSON 包');
        return;
      }
      let bytes: ArrayBuffer;
      let name: string;
      if (source === 'file') {
        if (!file) throw new Error('请先选择 .pt 或 .pth 权重文件');
        if (file.size > MAX_CHECKPOINT_BYTES) throw new Error('权重文件超过 128 MiB 限制');
        bytes = await file.arrayBuffer(); name = file.name;
      } else {
        name = source === 'example' ? profile.checkpoint : url.trim();
        if (!name) throw new Error('请输入权重下载网址');
        bytes = await fetchWeights(source === 'example' ? asset(`checkpoints/${profile.id}/${profile.checkpoint}`) : name, controller.signal);
      }
      logRlDebug('replay.weights.done', { profile: profile.id, bytes: bytes.byteLength,
        elapsedMs: Math.round(performance.now() - loadStarted) });
      if (token !== generation.current) return;
      updateStage('解析权重');
      const policy = loadCheckpoint(bytes, 'elu');
      const expectedOutput = profile.policyOutputSize ?? profile.jointNames.length;
      if (policy.inputSize !== profile.inputSize || policy.outputSize !== expectedOutput) {
        throw new Error(`网络为 ${policy.inputSize}→${policy.outputSize}，当前任务需要 ${profile.inputSize}→${expectedOutput}。请选择匹配任务。`);
      }
      updateStage('读取显示模型');
      const response = await fetch(asset(`models/${profile.displayModel ?? profile.model}.xml`), { signal: controller.signal });
      if (!response.ok) throw new Error(`机器人模型加载失败（HTTP ${response.status}）`);
      const xml = await response.text();
      logRlDebug('replay.display-model.done', { profile: profile.id, bytes: xml.length,
        elapsedMs: Math.round(performance.now() - loadStarted) });
      if (token !== generation.current) return;
      let physicalModel: PhysxModel | undefined;
      if (backend === 'physx') {
        const packagedModel = bundledPhysxModels[profile.model];
        if (!packagedModel) throw new Error(`缺少 PhysX 模型 ${profile.model}`);
        physicalModel = { ...packagedModel, simulation: { ...packagedModel.simulation } };
        logRlDebug('replay.physical-model.ready', { profile: profile.id, model: profile.model,
          bodies: physicalModel.bodies.length, elapsedMs: Math.round(performance.now() - loadStarted) });
        if (profile.physxSimulation) physicalModel = {
          ...physicalModel,
          simulation: { ...physicalModel.simulation, ...profile.physxSimulation },
        };
        if (profile.origin === 'isaacgym') physicalModel = { ...physicalModel, defaultQ: [...profile.defaultAngles] };
        if (profile.displayJointNames) physicalModel = { ...physicalModel, displayJointNames: [...profile.displayJointNames] };
        if (profile.displayJointSigns) physicalModel = { ...physicalModel, displayJointSigns: [...profile.displayJointSigns] };
        if (profile.displayRootQuaternionOffset) physicalModel = {
          ...physicalModel, displayRootQuaternionOffset: [...profile.displayRootQuaternionOffset],
        };
      }
      if (token !== generation.current) return;
      // Drop the previous controller before replacing its model; failures stay paused.
      runner.current = null; setLoadedName('');
      physics.current?.dispose(); physics.current = null;
      engine.loadModelFromXml(xml);
      updateStage('启动 PhysX Worker');
      if (physicalModel) physics.current = await PhysxWorkerSimulation.create(physicalModel, engine,
        { physicsDt: profile.physicsDt, controlDt: profile.controlDt,
          rootPosition: profile.initialRootPosition, rootQuaternion: profile.initialRootQuaternion });
      physics.current?.setDebugVisualization(diagnosticsRef.current.forces);
      if (token !== generation.current) { physics.current?.dispose(); physics.current = null; return; }
      runner.current = new PolicyReplay(engine, profile, policy, physics.current ?? undefined);
      runner.current.setJointSpeedLimit(jointSpeedLimit);
      if (profile.observation === 'isaaclab-periodic' || profile.observation === 'mjlab-zbot') runner.current.setPeriodicFrequency(periodicFrequency);
      if (profile.commands) runner.current.setCommands(commands);
      setLoadedName(name); update();
      logRlDebug('replay.load.done', { profile: profile.id, elapsedMs: Math.round(performance.now() - loadStarted) });
    } catch (err) {
      logRlDebug('replay.load.error', { profile: profile.id, stage,
        elapsedMs: Math.round(performance.now() - loadStarted), message: err instanceof Error ? err.message : String(err) });
      if (token === generation.current && (!controller.signal.aborted || timedOut)) setError(`${timedOut ? `读取权重或模型超过 30 秒，已停止。当前阶段：${stage}。` : (err as Error).message}${source === 'url' ? '（跨站网址需允许 CORS，也可下载后选择本地文件。）' : ''}`);
    } finally {
      window.clearTimeout(timeout);
      if (token === generation.current) {
        requestRef.current = null;
        setLoading(false); setLoadingStage('');
      }
    }
  };
  const reset = async () => { pause(); setError(''); try { await runner.current?.resetAsync(); update(); } catch (err) { setError((err as Error).message); } };
  const play = () => {
    if (!runner.current) return;
    if (runningRef.current) { pause(); return; }
    runningRef.current = true; setRunning(true);
  };
  const singleStep = async () => { pause(); try { await runner.current?.stepAsync(); update(); } catch (err) { setError((err as Error).message); } };
  const changeProfile = (id: string) => {
    if (id === importedProfile?.id) return;
    clearSelection();
    setImportedBundle(null); setSource('example'); setFile(null);
    const selected = REPLAY_PROFILES.find(p => p.id === id)!;
    setProfileId(id); if (selected.origin === 'mjlab') setBackend('mujoco'); setCommands([...(selected.commands ?? [0, 0, 0])]);
    setJointSpeedLimit(selected.jointSpeedLimit ?? JOINT_SPEED_LIMIT);
    setPeriodicFrequency(selected.stepFrequency ?? selected.phaseFrequency ?? (selected.origin === 'mjlab' ? .5 : 1));
    if (selected.mujocoCompatible === false) setBackend('physx');
  };
  const changeCommand = (index: number, value: number) => {
    const next = [...commands] as [number, number, number]; next[index] = value;
    setCommands(next);
    if (runner.current) runner.current.setCommands(next);
  };
  const hasJointSpeedObservation = profile.origin === 'isaacgym'
    ? profile.legacyController?.usesJointSpeed === true
    : profile.jointNames.length === 6 && profile.observation !== 'velocity'
      && profile.observation !== 'imu' && profile.observation !== 'isaaclab-periodic' && profile.observation !== 'mjlab-zbot';
  const hasStepFrequency = profile.observation === 'mjlab-zbot';
  const changePeriodicFrequency = (value: number) => {
    const min = hasStepFrequency ? .2 : .5;
    const max = hasStepFrequency ? 1 : 2;
    if (!Number.isFinite(value) || value < min || value > max) return;
    setPeriodicFrequency(value);
    runner.current?.setPeriodicFrequency(value);
  };
  const changeJointSpeedLimit = (value: number) => {
    if (!Number.isFinite(value) || value < .1 || value > 5) return;
    setJointSpeedLimit(value);
    runner.current?.setJointSpeedLimit(value);
  };

  return <div className="fixed inset-0 z-40 bg-slate-950 flex flex-col" role="dialog" aria-modal="true" aria-label="强化学习回放">
    <header className="flex items-center justify-between gap-4 px-5 py-4 border-b border-slate-800">
      <div><h2 className="font-semibold">强化学习回放</h2><p className="text-xs text-slate-400 mt-1">浏览器 CPU 策略推理 + {backend === 'physx' ? 'PhysX' : 'MuJoCo'} WASM · 无需 Python、GPU 或回放服务器</p></div>
      <button onClick={onClose} aria-label="返回实验台" className="p-2 rounded bg-slate-800 hover:bg-slate-700"><X size={20} /></button>
    </header>
    <div className="flex-1 min-h-0 overflow-y-auto lg:flex">
      <section className="p-5 space-y-4 lg:w-[360px] shrink-0 border-r border-slate-800">
<label className="block text-sm">Physics engine<select aria-label="physics" value={backend} onChange={e => { clearSelection(); setBackend(e.target.value as typeof backend); }} className="mt-2 w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs"><option value="physx" disabled={profile.origin === "mjlab"}>PhysX WASM</option><option value="mujoco" disabled={profile.mujocoCompatible === false}>MuJoCo WASM</option><option value="mjwarp" disabled={profile.origin !== "mjlab"}>MJWarp - local MJLab</option></select></label>
        <label className="block text-sm">训练任务<select aria-label="训练任务" value={importedProfile?.id ?? profileId} onChange={e => changeProfile(e.target.value)} className="mt-2 w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs">
          {importedProfile && <option value={importedProfile.id}>{importedProfile.label}</option>}
          <optgroup label="ZBot RL / Isaac Lab 训练结果">{REPLAY_PROFILES.filter(p => p.origin !== 'isaacgym' && p.origin !== 'mjlab').map(p => <option key={p.id} value={p.id}>{p.label}</option>)}</optgroup>
          <optgroup label="旧版 Isaac Gym 训练结果">{REPLAY_PROFILES.filter(p => p.origin === 'isaacgym').map(p => <option key={p.id} value={p.id}>{p.label}</option>)}</optgroup>
          <optgroup label="MJLab 训练结果">{REPLAY_PROFILES.filter(p => p.origin === 'mjlab').map(p => <option key={p.id} value={p.id}>{p.label}</option>)}</optgroup>
        </select></label>
        <label className="block text-sm">权重来源<select aria-label="权重来源" value={source} onChange={e => { clearSelection(); setSource(e.target.value as typeof source); setFile(null); if (e.target.value !== 'bundle') { setImportedBundle(null); if (importedBundle) setBackend('physx'); } }} className="mt-2 w-full bg-slate-900 border border-slate-700 rounded p-2 text-xs"><option value="example">内置训练权重</option><option value="file">本地 .pt / .pth 文件</option><option value="url">网络权重网址</option><option value="bundle">mjlab 训练结果包 (.json)</option></select></label>
        {source === 'example' && <p className="text-xs text-slate-400 break-all">{profile.checkpoint} · 已随页面打包</p>}
        {source === 'file' && <label className="block text-xs text-slate-300">选择权重文件<input aria-label="选择权重文件" type="file" accept=".pt,.pth" onChange={e => { clearSelection(); setFile(e.target.files?.[0] ?? null); }} className="block mt-2 w-full text-xs file:bg-slate-800 file:text-slate-200 file:rounded file:border-0 file:px-2 file:py-2" /><span className="block mt-2 text-slate-400">文件仅在浏览器读取，不上传。请选择与任务相同观测定义的训练权重。</span></label>}
        {source === 'url' && <label className="block text-xs">权重下载网址<input aria-label="权重下载网址" type="url" value={url} onChange={e => { clearSelection(); setUrl(e.target.value); }} placeholder="https://…/model.pt" className="mt-2 w-full bg-slate-900 border border-slate-700 rounded p-2" /></label>}
        {source === 'bundle' && <label className="block text-xs">训练结果 JSON 包<input aria-label="训练结果 JSON 包" type="file" accept=".json" onChange={e => { clearSelection(); setFile(e.target.files?.[0] ?? null); }} className="block w-full mt-2" /><span className="block text-slate-400 mt-2">包含模型、权重和训练参数，下载后可独立回放。{importedBundle && `当前：${importedBundle.name}`}</span></label>}
        <button disabled={!loading && (!ready || source === 'file' && !file || source === 'bundle' && !file && !importedBundle || source === 'url' && !url.trim())}
          onClick={() => loading ? cancelLoad() : void load()}
          className={`w-full rounded px-3 py-2 text-sm disabled:opacity-40 ${loading ? 'bg-rose-700 hover:bg-rose-600' : 'bg-blue-600 hover:bg-blue-500'}`}>
          {loading ? `${loadingStage || '加载中'}…（点击取消）` : ready ? '加载策略' : '正在初始化 WASM…'}
        </button>
        {profile.commands && <fieldset className="space-y-2 border border-slate-700 rounded p-3 text-xs">
          <legend className="px-1 text-sky-300">速度命令</legend>
          {['前进速度', '侧移速度', '转向速度'].map((label, i) => <label key={label} className="block">{label} <span className="text-slate-400">{commands[i].toFixed(2)} {i === 2 ? 'rad/s' : 'm/s'}</span>
            <input aria-label={label} type="range" min={i === 2 ? -1 : -.4} max={i === 2 ? 1 : .4} step={.05} value={commands[i]} onChange={e => changeCommand(i, Number(e.target.value))} className="block w-full mt-1" />
          </label>)}
          <button onClick={() => { setCommands([0, 0, 0]); runner.current?.setCommands([0, 0, 0]); }} className="px-2 py-1 rounded bg-slate-800">速度归零</button>
        </fieldset>}
        {(profile.observation === 'isaaclab-periodic' || hasStepFrequency) && <label className="block rounded border border-sky-900 bg-sky-950/20 p-3 text-xs">
          {hasStepFrequency ? '踏步频率' : '周期频率'} <span className="ml-2 font-mono text-sky-300">{periodicFrequency.toFixed(2)} Hz</span>
          <input aria-label={hasStepFrequency ? '踏步频率' : '周期频率'} type="range" min={hasStepFrequency ? '.2' : '.5'} max={hasStepFrequency ? '1' : '2'} step={hasStepFrequency ? '.05' : '.1'} value={periodicFrequency}
            onChange={event => changePeriodicFrequency(Number(event.target.value))} className="block w-full mt-2" />
          {hasStepFrequency && <span className="block mt-1 text-slate-400">写入策略观测的 frequency 项，训练范围 0.2–1.0 Hz。</span>}
        </label>}
        {hasJointSpeedObservation && <label className="block rounded border border-sky-900 bg-sky-950/20 p-3 text-xs">
          策略关节速度参数
          <span className="ml-2 font-mono text-sky-300">{jointSpeedLimit.toFixed(2)}</span>
          <span className="block mt-1 text-slate-400">写入策略观测，并用于动作积分；{profile.origin === 'isaacgym' ? '旧任务训练范围为 0.1–1.0。' : '训练默认值为 2。'}</span>
          <input aria-label="策略关节速度参数" type="number" min="0.1" max={profile.origin === 'isaacgym' ? 1 : 5} step="0.1" value={jointSpeedLimit}
            onChange={e => changeJointSpeedLimit(e.target.valueAsNumber)} className="mt-2 w-full rounded border border-slate-700 bg-slate-900 p-2" />
        </label>}
        <div className="grid grid-cols-3 gap-2">
          <button disabled={!loadedName || loading || !!error} onClick={play} className="flex items-center justify-center gap-1 rounded bg-emerald-700 px-2 py-2 text-xs disabled:opacity-40">{running ? <Pause size={14} /> : <Play size={14} />}{running ? '暂停' : 'Play'}</button>
          <button disabled={!loadedName || loading || !!error || running} onClick={() => void singleStep()} className="flex items-center justify-center gap-1 rounded bg-slate-800 p-2 text-xs disabled:opacity-40"><SkipForward size={14} />单步</button>
          <button disabled={!loadedName || loading} onClick={() => void reset()} className="flex items-center justify-center gap-1 rounded bg-slate-800 p-2 text-xs disabled:opacity-40"><RotateCcw size={14} />重置</button>
        </div>
        <label className="block text-xs">播放速度<select aria-label="播放速度" value={speed} onChange={e => setSpeed(Number(e.target.value))} className="ml-3 bg-slate-900 border border-slate-700 rounded p-1">{[.25, .5, 1, 2].map(n => <option key={n} value={n}>{n}×</option>)}</select></label>
        <fieldset className="rounded border border-slate-700 p-3 text-xs">
          <legend className="px-1 text-sky-300">实体显示</legend>
          <div className="grid grid-cols-2 gap-2">
            <label className={`flex items-center gap-2 rounded border px-2 py-2 ${backend === 'physx' ? 'border-rose-900/70 bg-rose-950/20' : 'border-slate-800 text-slate-500'}`} title={backend === 'physx' ? 'PhysX 实际参与碰撞的凸包实体' : 'MuJoCo 后端没有独立的 PhysX 实体'}>
              <input type="checkbox" aria-label="显示物理实体" checked={backend === 'physx' && showPhysicalEntity} disabled={backend !== 'physx'} onChange={event => setShowPhysicalEntity(event.target.checked)} />
              <span><span className="text-rose-300">物理实体</span><span className="block text-[10px] text-slate-500">红色碰撞点</span></span>
            </label>
            <label className="flex items-center gap-2 rounded border border-sky-900/70 bg-sky-950/20 px-2 py-2">
              <input type="checkbox" aria-label="显示实体" checked={showDisplayEntity} onChange={event => setShowDisplayEntity(event.target.checked)} />
              <span><span className="text-sky-300">显示实体</span><span className="block text-[10px] text-slate-500">外观模型</span></span>
            </label>
          </div>
        </fieldset>
        {error && <p role="alert" className="rounded border border-rose-900 bg-rose-950/40 p-3 text-xs text-rose-200 break-words">{error}</p>}
        <details className="rounded border border-slate-800 bg-slate-950/60 p-2 text-[10px] text-slate-400">
          <summary className="cursor-pointer select-none text-slate-300">加载诊断日志（{debugRecords.length}）</summary>
          <pre className="mt-2 max-h-44 overflow-auto whitespace-pre-wrap break-all font-mono">{debugRecords.slice(-40).map(record =>
            `${record.elapsedMs.toString().padStart(6)} ms  ${record.event}${record.details ? ` ${JSON.stringify(record.details)}` : ''}`).join('\n') || '暂无日志'}</pre>
        </details>
        <div className="border-t border-slate-800 pt-4 text-xs text-slate-400 space-y-2 leading-5">
          {profile.origin === 'mjlab' ? <p>此结果来自 mjlab，按结果包保存的 MuJoCo 模型、物理步长和积分速度回放。短训练仅证明流程可运行，运动质量仍需评估。</p> : <><p>已收录训练目录的 {REPLAY_PROFILES.length} 组权重，按所选任务加载机器人及观测／动作定义；当前策略控制频率为 {(1 / (profile.controlDt ?? 1 / 30)).toFixed(0)} Hz。</p>
          {backend === 'physx' ? <p>PhysX 使用原训练的 TGS 求解器、{(1 / (profile.physicsDt ?? 1 / 60)).toFixed(0)} Hz 步长和隐式电机驱动。各权重按自身任务回放，实验权重保留原生表现；具体轨迹仍可能不同。</p> : <p className="text-amber-300">MuJoCo 使用 600 Hz 子步，仅用于迁移对照。接触与驱动响应尚不等价，原策略可能跌倒；默认回放推荐使用 PhysX。</p>}</>}
          <p>策略会持续推理，直到手动暂停、重置、切换任务或关闭回放；不复现训练环境的跌倒自动重置。</p>
          {profile.note && <p className="text-amber-200">{profile.note}</p>}
          {profile.commands && <p>速度命令可在播放时调整。回放使用固定 1.2 Hz 步频和初始相位，以便重复比较。</p>}
        </div>
      </section>
      <section className="flex-1 min-w-0 flex flex-col min-h-[540px]">
        <div className="px-4 py-3 text-xs text-slate-400 flex flex-wrap gap-4 border-b border-slate-800" role="status">
          <span className="text-sky-300">{loading ? '加载中' : running ? '策略持续运行中' : loadedName ? '策略已就绪 · 已暂停' : '等待加载策略'}</span>
          <span>时间 {metrics.time.toFixed(2)} s</span><span>控制步 {steps}</span><span>动作峰值 {actionPeak.toFixed(3)}</span>
        </div>
        <div className="flex-1 min-h-[440px] relative">{backend === 'mjwarp' && nativeReplayId ? <TrainingLiveView frame={nativeFrame} jobId={nativeReplayId} client={nativeClient} native /> : <SimulationViewport engine={engine} config={displayConfig} visualStyle={profile.origin === 'mjlab' ? 'mjlab' : 'default'} physicsLabel={backend === 'physx' ? 'PhysX WASM' : backend === 'mjwarp' ? 'MJLab MJWarp' : 'MuJoCo WASM'} physicalEntity={physics.current} showPhysicalEntity={showPhysicalEntity} showDisplayEntity={showDisplayEntity} />}</div>
        <p className="px-4 py-2 text-[11px] text-slate-500 break-all">{loadedName || '权重不会离开浏览器；内置示例可直接加载。'}</p>
      </section>
    </div>
  </div>;
}
