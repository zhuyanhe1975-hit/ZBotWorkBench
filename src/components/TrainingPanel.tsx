import React, { useEffect, useRef, useState } from 'react';
import { Download, Eye, EyeOff, Play, Square, X } from 'lucide-react';
import { DIRECT_WALKING_TASK, taskDefaults } from '../training/tasks';
import { TrainingClient, trainingCheckpointUrl, trainingBundleUrl } from '../training/client';
import type { TrainingConfig, TrainingJob, TrainingReplayBundle, TrainingResources, TrainingTask, TrainingTaskCardSettings } from '../training/types';
import { TrainingLiveView } from './TrainingLiveView';

const activeStatuses = new Set<TrainingJob['status']>(['queued', 'starting', 'running', 'stopping']);
const statusLabel: Record<TrainingJob['status'], string> = { queued: '等待启动', starting: '正在启动', running: '训练中', stopping: '正在停止并保存', stopped: '已停止', completed: '已完成', failed: '失败' };
const initialConfig: TrainingConfig = { taskId: '', device: 'cpu', cpuThreads: 4, numEnvs: 4, iterations: 100, saveInterval: 10, seed: 42, maxSeconds: 3600 };
const memory = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
const inputClass = 'mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-2 text-sm disabled:opacity-40';
const rewardLabels: Record<string, string> = { feet_downward: '足部朝下', feet_forward: '足部朝前', base_heading_x: '机身航向', feet_force_diff: '左右受力交替', feet_force_sum: '累计受力差', base_vel_forward: '前进速度', slow_speed_tracking: '慢速跟踪', similar_to_default: '默认姿态', base_heading_x_sum: '累计航向', support_stability: '支撑稳定', base_tilt: '机身倾斜', step_length: '步长', small_step: '小碎步', step_cadence: '慢步频', airtime_balance: '腾空时间差', airtime_sum: '腾空总时长', double_flight: '双脚腾空', action_rate: '动作变化', body_shake: '机身抖动', joint_velocity: '关节速度', joint_acceleration: '关节加速度', torques: '力矩平方', energy_consumption: '总机械能耗', feet_slide: '滑脚', base_pos_y_err: '横向偏移', support_phase: '支撑相位跟踪', com_centering: '重心居中', com_phase: '重心相位跟踪', lateral_drift: '横向漂移', heading: '方向朝向', joint_pose: '初始构型偏离', body_velocity: '机身速度', upright: '机身直立', fall_protection: '保持不摔倒', single_support: '单脚支撑' };

function TaskCardEditor({ task, initial, running, revision, disabled, onCommit }: { key?: React.Key; task: TrainingTask; initial: TrainingTaskCardSettings; running: boolean; revision?: number; disabled: boolean; onCommit: (value: TrainingTaskCardSettings) => void }) {
  const [draft, setDraft] = useState(() => ({ stage1Rewards: { ...initial.stage1Rewards }, stage2Rewards: { ...initial.stage2Rewards }, terminatedRewardPenalty: initial.terminatedRewardPenalty }));
  const weight = (stage: 'stage1Rewards' | 'stage2Rewards', key: string, value: number) => setDraft(current => ({ ...current, [stage]: { ...current[stage], [key]: value } }));
  const group = (stage: 'stage1Rewards' | 'stage2Rewards', title: string) => <fieldset className="rounded border border-slate-800 p-3"><legend className="px-1 text-xs text-sky-300">{title}</legend><div className="grid grid-cols-2 gap-2">{Object.entries(draft[stage]).map(([key, value]) => <label key={key} className="text-[11px] text-slate-400">{rewardLabels[key] ?? key}<input aria-label={`${title}-${key}`} type="number" min={-100} max={100} step="0.1" value={value} disabled={disabled} onChange={event => weight(stage, key, Number(event.target.value))} className={inputClass} /></label>)}</div></fieldset>;
  return <details className="rounded border border-slate-700 bg-slate-950/40 p-3 text-xs"><summary className="cursor-pointer font-medium text-slate-200">任务卡 · {running ? `运行中修改${revision ? `（版本 ${revision}）` : ''}` : '下一次训练'}</summary>
    <div className="mt-3 space-y-3"><div className="grid grid-cols-2 gap-2 rounded bg-slate-900 p-2 text-[11px] text-slate-400"><span>物理 {task.taskCard.physicsHz} Hz</span><span>控制 {task.taskCard.controlHz} Hz</span><span>接触历史 {task.taskCard.contactHistory} 步</span><span>初始阶段 {task.taskCard.initialStage}</span><span className="col-span-2">积分速度 {task.taskCard.jointSpeedRange[0]}–{task.taskCard.jointSpeedRange[1]}</span></div>
      {task.id !== DIRECT_WALKING_TASK && group('stage1Rewards', '第一阶段奖励权重')}{group('stage2Rewards', '奖励权重')}
      <label className="block text-xs">提前终止惩罚<input aria-label="提前终止惩罚" type="number" min={0} max={1000} step="1" value={draft.terminatedRewardPenalty} disabled={disabled} onChange={event => setDraft(current => ({ ...current, terminatedRewardPenalty: Number(event.target.value) }))} className={inputClass} /></label>
      <button type="button" disabled={disabled} onClick={() => onCommit({ stage1Rewards: { ...draft.stage1Rewards }, stage2Rewards: { ...draft.stage2Rewards }, terminatedRewardPenalty: draft.terminatedRewardPenalty })} className="w-full rounded bg-violet-700 px-3 py-2 text-xs hover:bg-violet-600 disabled:opacity-40">{running ? '应用到运行任务（下一次PPO更新生效）' : '应用到下一次训练'}</button>
      {running && <p className="text-[11px] text-amber-300">权重更新不会回溯修改已经采集的rollout；下一次PPO更新统一使用新任务卡。</p>}
    </div></details>;
}

function RewardHistory({ history }: { history: NonNullable<TrainingJob['history']> }) {
  const samples = history.slice(-200).filter(point => Number.isFinite(point.iteration) && typeof point.reward === 'number' && Number.isFinite(point.reward)).sort((a, b) => a.iteration - b.iteration);
  if (!samples.length) return <div className="text-xs text-slate-500">奖励曲线 · 最近 200 次更新：尚无奖励数据。</div>;
  const rewards = samples.map(point => point.reward!);
  const minimum = Math.min(...rewards), maximum = Math.max(...rewards);
  const padding = maximum === minimum ? Math.max(1, Math.abs(minimum) * .05) : (maximum - minimum) * .1;
  const lower = minimum - padding, upper = maximum + padding;
  const first = samples[0].iteration, last = samples.at(-1)!.iteration;
  const coordinates = samples.map(point => ({
    x: first === last ? 220 : 50 + (point.iteration - first) / (last - first) * 340,
    y: 90 - (point.reward! - lower) / (upper - lower) * 78,
  }));
  const latest = coordinates.at(-1)!;
  return <figure className="rounded border border-slate-800 bg-slate-950/50 px-3 py-2">
    <figcaption className="text-xs text-slate-400">奖励曲线 · 最近 200 次更新</figcaption>
    <svg viewBox="0 0 400 118" className="mt-1 w-full text-sky-400" role="img" aria-label={`奖励曲线：${samples.length} 个采样点，奖励范围 ${minimum.toFixed(3)} 至 ${maximum.toFixed(3)}`}>
      <line x1="50" y1="12" x2="50" y2="90" stroke="#334155" /><line x1="50" y1="90" x2="390" y2="90" stroke="#334155" />
      <text x="45" y="16" textAnchor="end" fontSize="10" fill="#94a3b8">{upper.toFixed(2)}</text><text x="45" y="93" textAnchor="end" fontSize="10" fill="#94a3b8">{lower.toFixed(2)}</text>
      {coordinates.length > 1 && <polyline points={coordinates.map(point => `${point.x},${point.y}`).join(' ')} fill="none" stroke="currentColor" strokeWidth="2" />}
      <circle cx={latest.x} cy={latest.y} r="3" fill="currentColor" />
      <text x={first === last ? 220 : 50} y="110" textAnchor={first === last ? 'middle' : 'start'} fontSize="10" fill="#94a3b8">迭代 {first}</text>{last !== first && <text x="390" y="110" textAnchor="end" fontSize="10" fill="#94a3b8">{last}</text>}
    </svg>
  </figure>;
}

function RewardTermBars({ current, history, card, defaults, disabled, onCommit }: { current?: Record<string, number>; history?: TrainingJob['history']; card?: TrainingTaskCardSettings; defaults?: TrainingTaskCardSettings; disabled?: boolean; onCommit?: (value: TrainingTaskCardSettings) => void }) {
  const [draft, setDraft] = useState(card ?? defaults);
  const cardSignature = JSON.stringify(card ?? defaults);
  useEffect(() => setDraft(card ?? defaults), [cardSignature]);
  if (!card) return <div className="rounded border border-dashed border-slate-800 p-4 text-xs text-slate-500">奖励分项贡献将在选择训练任务后显示。</div>;
  const keys = [...new Set([...Object.keys(card.stage1Rewards), ...Object.keys(card.stage2Rewards)])];
  const curriculumStage = current?.curriculum_stage;
  const maximum = Math.max(1e-9, ...keys.map(key => Math.abs(current?.[key] ?? 0)));
  const samples = (history ?? []).slice(-20);
  if (!draft) return null;
  const update = (key: string, value: number) => setDraft(previous => ({ ...previous, stage1Rewards: key in previous.stage1Rewards ? { ...previous.stage1Rewards, [key]: value } : previous.stage1Rewards, stage2Rewards: key in previous.stage2Rewards ? { ...previous.stage2Rewards, [key]: value } : previous.stage2Rewards }));
  const reset = () => defaults && setDraft({ stage1Rewards: { ...defaults.stage1Rewards }, stage2Rewards: { ...defaults.stage2Rewards }, terminatedRewardPenalty: defaults.terminatedRewardPenalty });
  return <figure className="rounded border border-slate-800 bg-slate-950/50 p-3">
    <figcaption className="mb-3 flex flex-wrap items-center gap-2"><span className="text-sm text-slate-200">奖励分项贡献</span><span className="text-[11px] text-slate-500">当前值 · 权重可随时修改</span>{curriculumStage !== undefined && <span className="rounded bg-slate-800 px-2 py-1 text-[11px] text-sky-300">课程阶段：{curriculumStage.toFixed(0)}</span>}<span className="ml-auto flex gap-2"><button type="button" disabled={disabled || !onCommit} onClick={reset} className="rounded bg-slate-800 px-2 py-1 text-[11px] hover:bg-slate-700 disabled:opacity-40">恢复默认权重</button><button type="button" disabled={disabled || !onCommit} onClick={() => onCommit?.(draft)} className="rounded bg-violet-700 px-2 py-1 text-[11px] hover:bg-violet-600 disabled:opacity-40">应用权重</button></span></figcaption>
    <div className="space-y-2">{keys.map(key => {
      const value = current?.[key] ?? 0, weight = key in draft.stage1Rewards ? draft.stage1Rewards[key] : draft.stage2Rewards[key];
      return <div key={key} className="grid grid-cols-[7.5rem_minmax(6rem,1fr)_4.5rem_5rem] items-center gap-2 text-[11px]">
        <span className="truncate text-slate-300" title={key}>{rewardLabels[key] ?? key}</span>
        <div className="relative h-4 overflow-hidden rounded bg-slate-900" title={`${key}: ${value.toFixed(5)}`}><span className="absolute inset-y-0 left-1/2 w-px bg-slate-600" /><span className={`absolute top-0.5 h-3 rounded-sm ${value >= 0 ? 'left-1/2 bg-emerald-500' : 'right-1/2 bg-rose-500'}`} style={{ width: `${Math.abs(value) / maximum * 50}%` }} /></div>
        <span className={value >= 0 ? 'text-right text-emerald-300' : 'text-right text-rose-300'}>{value.toFixed(3)}</span>
        <input aria-label={`${rewardLabels[key] ?? key}权重`} type="number" min={-100} max={100} step="0.1" value={weight} disabled={disabled || !onCommit} onChange={event => update(key, Number(event.target.value))} className="w-full rounded border border-slate-700 bg-slate-900 px-1 py-1 text-left text-[11px]" />
      </div>;
    })}</div>
    <div className="mt-3 flex gap-4 text-[10px] text-slate-500"><span><i className="mr-1 inline-block h-2 w-2 bg-emerald-500" />正贡献</span><span><i className="mr-1 inline-block h-2 w-2 bg-rose-500" />负贡献</span></div>
  </figure>;
}

export function TrainingPanel({ onClose, onReplay }: { onClose: () => void; onReplay: (bundle: TrainingReplayBundle) => void }) {
  const [client] = useState(() => new TrainingClient());
  const [resources, setResources] = useState<TrainingResources | null>(null);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [config, setConfig] = useState<TrainingConfig>(initialConfig);
  const [selectedId, setSelectedId] = useState('');
  const [connectionError, setConnectionError] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [checking, setChecking] = useState(true);
  const [retry, setRetry] = useState(0);
  const action = useRef<AbortController | null>(null);
  const mounted = useRef(false);
  const initialized = useRef(false);
  const closeButton = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const previewJob = useRef<string | null>(null);
  const selected = jobs.find(job => job.id === selectedId) ?? (selectedId ? jobs[0] : undefined);
  const activeJob = jobs.find(job => activeStatuses.has(job.status));
  const selectedTask = resources?.tasks.find(task => task.id === config.taskId);
  const cardJob = selected && ['queued', 'starting', 'running'].includes(selected.status) ? selected : undefined;
  const cardTask = resources?.tasks.find(task => task.id === (selected?.config.taskId ?? config.taskId));
  const rawCardSettings = selected?.taskCard ?? config.taskCard ?? cardTask?.taskCard;
  const cardSettings = rawCardSettings;
  const resumeJobs = jobs.filter(job => ['completed', 'stopped'].includes(job.status) && job.checkpoints.length);
  const deviceAvailable = !!resources?.runtime.available && (config.device === 'cpu' || resources.runtime.cudaBuild !== false && resources.gpus.some(gpu => config.device === `cuda:${gpu.id}`));
  previewJob.current = selected?.livePreviewEnabled && activeStatuses.has(selected.status) ? selected.id : null;

  useEffect(() => {
    mounted.current = true;
    const previous = document.activeElement as HTMLElement | null;
    closeButton.current?.focus();
    return () => {
      mounted.current = false; action.current?.abort(); previous?.focus();
      if (previewJob.current) void client.preview(previewJob.current, false).catch(() => {});
    };
  }, [client]);

  useEffect(() => {
    let cancelled = false, pending = false, resourceTime = 0;
    let resourceFailure: unknown = null;
    let timer: ReturnType<typeof setTimeout>;
    let controller: AbortController | null = null;
    const poll = async () => {
      if (cancelled || pending || document.hidden) return;
      pending = true;
      controller = new AbortController();
      const signal = controller.signal;
      try {
        const loadResources = Date.now() - resourceTime >= 10_000;
        if (loadResources) resourceTime = Date.now();
        const [jobResult, resourceResult] = await Promise.allSettled([
          client.jobs(signal), loadResources ? client.resources(signal) : Promise.resolve(null),
        ]);
        if (cancelled || signal.aborted) return;
        if (jobResult.status === 'fulfilled') setJobs(jobResult.value);
        if (resourceResult.status === 'fulfilled' && resourceResult.value) {
          const next = resourceResult.value;
          setResources(next); resourceFailure = null;
          if (!initialized.current) {
            initialized.current = true;
            setConfig(taskDefaults(next.tasks[0]?.id ?? '', next));
          }
        }
        if (resourceResult.status === 'rejected') resourceFailure = resourceResult.reason;
        const failure = jobResult.status === 'rejected' ? jobResult.reason : resourceFailure;
        setConnectionError(failure ? failure instanceof Error ? failure.message : String(failure) : '');
      } finally {
        pending = false;
        if (!cancelled) { setChecking(false); timer = setTimeout(() => void poll(), 2000); }
      }
    };
    const visibility = () => {
      clearTimeout(timer);
      if (document.hidden) controller?.abort();
      else if (!pending) void poll();
    };
    document.addEventListener('visibilitychange', visibility);
    void poll();
    return () => { cancelled = true; clearTimeout(timer); controller?.abort(); document.removeEventListener('visibilitychange', visibility); };
  }, [client, retry]);

  const act = async (task: (signal: AbortSignal) => Promise<void>) => {
    if (busy) return;
    setError(''); setBusy(true);
    const controller = new AbortController(); action.current = controller;
    try { await task(controller.signal); }
    catch (caught) { if (mounted.current && !controller.signal.aborted) setError(caught instanceof Error ? caught.message : String(caught)); }
    finally { if (mounted.current && !controller.signal.aborted) setBusy(false); }
  };
  const acceptJob = (job: TrainingJob) => {
    if (!mounted.current) return;
    setJobs(current => [job, ...current.filter(item => item.id !== job.id)]); setSelectedId(job.id);
  };
  const start = () => void act(async signal => {
    for (const key of ['cpuThreads', 'numEnvs', 'iterations', 'saveInterval', 'maxSeconds'] as const) {
      if (!Number.isSafeInteger(config[key]) || config[key] < 1) throw new Error('线程、环境、迭代、保存间隔与时间限制必须是正整数。');
    }
    if (!Number.isSafeInteger(config.seed) || config.seed < 0) throw new Error('随机种子必须是非负整数。');
    if (!deviceAvailable || !config.taskId) throw new Error('请检查运行环境、设备及训练任务。');
    acceptJob(await client.start(config, signal));
  });
  const resume = (value: string) => {
    setError('');
    if (!value) { setConfig(current => { const { resumeJobId, resumeCheckpoint, ...rest } = current; return rest; }); return; }
    const [jobId, checkpoint] = JSON.parse(value) as [string, string];
    const job = resumeJobs.find(item => item.id === jobId)!;
    const available = job.config.device === 'cpu' || resources?.runtime.cudaBuild !== false && resources?.gpus.some(gpu => job.config.device === `cuda:${gpu.id}`);
    setConfig({ ...job.config, device: available ? job.config.device : 'cpu', cpuThreads: Math.max(1, Math.min(job.config.cpuThreads, resources?.cpu.availableThreads ?? 4)), numEnvs: available ? job.config.numEnvs : 4,
      resumeJobId: jobId, resumeCheckpoint: checkpoint, iterations: 100 });
  };
  const numberField = (key: 'cpuThreads' | 'numEnvs' | 'iterations' | 'saveInterval' | 'seed' | 'maxSeconds', label: string, min = 1, max?: number) => <label className="block text-xs" htmlFor={`training-${key}`}>{label}<input id={`training-${key}`} type="number" min={min} max={max} step={1} value={config[key]} onChange={event => setConfig(current => ({ ...current, [key]: Number(event.target.value) }))} className={inputClass} disabled={busy} /></label>;
  const trapFocus = (event: React.KeyboardEvent) => {
    if (event.key === 'Escape') { event.stopPropagation(); onClose(); }
    if (event.key !== 'Tab') return;
    const nodes = Array.from(panel.current?.querySelectorAll<HTMLElement>('button:not(:disabled), a[href], input:not(:disabled), select:not(:disabled), summary') ?? []) as HTMLElement[];
    const visible = nodes.filter(node => node.getClientRects().length);
    const first = visible[0], last = visible.at(-1);
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
  };

  return <div ref={panel} onKeyDown={trapFocus} className="fixed inset-0 z-50 flex flex-col bg-slate-950 text-slate-200" role="dialog" aria-modal="true" aria-labelledby="training-title">
    <header className="flex items-center justify-between gap-4 border-b border-slate-800 px-5 py-4">
      <div><h2 id="training-title" className="font-semibold">强化学习训练</h2><p className="mt-1 text-xs text-slate-400">本地计算服务训练 · 浏览器管理任务与回放</p></div>
      <button ref={closeButton} onClick={onClose} aria-label="关闭训练面板" className="rounded bg-slate-800 p-2 hover:bg-slate-700"><X size={20} /></button>
    </header>
    <div className="min-h-0 flex-1 overflow-y-auto p-5 lg:grid lg:grid-cols-[360px_minmax(0,1fr)] lg:gap-6">
      <section className="space-y-4">
        {(connectionError || !resources) && <div className="space-y-2 rounded border border-amber-900 bg-amber-950/20 p-3 text-xs leading-5" role="status">
          <p>{checking ? '正在检测训练服务…' : connectionError || '训练服务尚未就绪。'}</p>
          {!checking && <><p>在工程目录执行 <code>npm run build</code>，然后执行 <code>npm run training</code>。</p><p>打开 <a className="text-sky-300 underline" href="http://127.0.0.1:8767/?training=1">本地训练页面</a>。普通静态页面仍可独立使用回放功能。</p><button onClick={() => { setChecking(true); setRetry(value => value + 1); }} className="rounded bg-slate-800 px-3 py-1">重新检测</button></>}
        </div>}
        {resources && <div className="space-y-2 rounded border border-slate-800 p-3 text-xs leading-5">
          <p className="font-medium text-sky-300">{resources.hostname} · {resources.platform}</p>
          <p>{resources.cpu.model}<br />{resources.cpu.logicalCores} 逻辑核心 · 可用 {resources.cpu.availableThreads} 线程</p>
          <p>内存可用 {memory(resources.memory.availableBytes)} / {memory(resources.memory.totalBytes)}</p>
          {resources.gpus.length ? resources.gpus.map(gpu => <p key={gpu.id}>GPU {gpu.id} · {gpu.name}<br />显存空闲 {(gpu.freeMemoryMiB / 1024).toFixed(1)} / {(gpu.totalMemoryMiB / 1024).toFixed(1)} GiB · 当前利用率 {gpu.utilization}%</p>) : <p>未检测到可用 GPU</p>}
          {resources.runtime.cudaBuild === false && <p className="text-amber-300">当前 PyTorch 为 CPU 构建；即使检测到 GPU，也只能选择 CPU 训练。</p>}
          <p className={resources.runtime.available ? 'text-emerald-300' : 'text-amber-300'}>{resources.runtime.available ? `运行环境就绪 · mjlab ${resources.runtime.mjlabVersion ?? '已检测'} · PyTorch ${resources.runtime.torchVersion ?? '已检测'}` : `运行环境不可用：${resources.runtime.error ?? '请检查本地训练依赖'}`}</p>
        </div>}
        <label htmlFor="training-resume" className="block text-xs">新训练 / 继续本服务保存的训练<select id="training-resume" value={config.resumeJobId ? JSON.stringify([config.resumeJobId, config.resumeCheckpoint]) : ''} onChange={event => resume(event.target.value)} disabled={busy} className={inputClass}><option value="">新训练</option>{resumeJobs.flatMap(job => job.checkpoints.map(checkpoint => <option key={`${job.id}/${checkpoint.name}`} value={JSON.stringify([job.id, checkpoint.name])}>{job.id} · {checkpoint.name}</option>))}</select></label>
        <label htmlFor="training-task" className="block text-xs">训练任务<select id="training-task" value={config.taskId} onChange={event => { if (!resources) return; setSelectedId(''); setConfig(taskDefaults(event.target.value, resources)); }} disabled={busy || !!config.resumeJobId} className={inputClass}><option value="" disabled>选择任务</option>{resources?.tasks.map(task => <option key={task.id} value={task.id}>{task.label}</option>)}</select></label>
        <p className="text-xs leading-5 text-slate-400">{selectedTask?.description}</p>
        {selectedTask?.referenceBundleUrl && <button disabled={busy} className="rounded bg-emerald-800 px-3 py-2 text-xs hover:bg-emerald-700 disabled:opacity-40" onClick={() => void act(async signal => {
          const response = await fetch(selectedTask.referenceBundleUrl!, { signal });
          if (!response.ok) throw new Error('参考回放包不可用，请重新构建应用。');
          onReplay(await response.json());
        })}>回放参考策略（model_801）</button>}
        <label htmlFor="training-device" className="block text-xs">计算设备<select id="training-device" value={config.device} disabled={busy || !resources?.runtime.available} onChange={event => setConfig(current => ({ ...current, device: event.target.value, numEnvs: resources ? taskDefaults(current.taskId, resources, event.target.value).numEnvs : 4 }))} className={inputClass}><option value="cpu">CPU（小规模 / 实验）</option>{resources?.gpus.map(gpu => <option key={gpu.id} value={`cuda:${gpu.id}`} disabled={resources.runtime.cudaBuild === false}>GPU {gpu.id} · {gpu.name}</option>)}</select></label>
        <div className="grid grid-cols-2 gap-3">{numberField('numEnvs', '并行环境数', 1, config.device === 'cpu' ? 64 : 4096)}{numberField('cpuThreads', 'CPU 线程数', 1, resources?.cpu.availableThreads)}{numberField('iterations', '训练迭代数', 1, 20000)}{numberField('saveInterval', '保存间隔（迭代）', 1, 1000)}{numberField('maxSeconds', '最长运行秒数', 30, 86400)}{numberField('seed', '随机种子', 0, 2147483647)}</div>
        <p className="text-xs leading-5 text-slate-400">GPU 训练也使用 CPU。线程数和环境数控制计算规模，不保证固定 GPU 利用率或显存上限。</p>
        <button onClick={start} disabled={busy || !deviceAvailable || !config.taskId || !!connectionError} className="w-full rounded bg-blue-600 px-3 py-2 text-sm hover:bg-blue-500 disabled:opacity-40">{busy ? '正在处理…' : activeJob ? '加入训练队列' : config.resumeJobId ? '继续训练' : '开始训练'}</button>
        {error && <div role="alert" className="rounded border border-rose-900 bg-rose-950/30 p-3 text-xs text-rose-200"><p>{error}</p><button onClick={() => setError('')} className="mt-2 underline">清除提示</button></div>}
        <p className="text-xs leading-5 text-slate-400">关闭页面后，训练服务中的任务会继续运行。需要结束任务时，请点击“停止并保存”。本服务一次运行一个训练任务，其余任务按队列等待。</p>
        {!!resources?.checkpoints.length && <details className="rounded border border-slate-800 p-3 text-xs"><summary className="cursor-pointer">检测到的已有权重（只读列表，{resources.checkpoints.length} 个）</summary><p className="my-2 text-slate-400">外部训练权重不作为本服务的续训检查点。</p><ul className="space-y-2">{resources.checkpoints.map(checkpoint => <li key={checkpoint.path} className="break-all">{checkpoint.name} · {(checkpoint.bytes / 1024 ** 2).toFixed(1)} MiB<span className="block text-slate-500">{checkpoint.path}</span></li>)}</ul></details>}
      </section>
      <section className="mt-6 min-w-0 space-y-4 lg:mt-0">
        <label htmlFor="training-job" className="block text-sm">训练记录<select id="training-job" value={selected?.id ?? ''} onChange={event => {
          if (selected?.livePreviewEnabled && activeStatuses.has(selected.status)) void client.preview(selected.id, false).catch(() => {});
          setSelectedId(event.target.value);
        }} className={inputClass}><option value="" disabled>暂无训练任务</option>{jobs.map(job => <option key={job.id} value={job.id}>{job.id} · {statusLabel[job.status]}</option>)}</select></label>
        {selected ? <>
          <div className="space-y-3 rounded border border-slate-800 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3"><span role="status" className="text-sm text-sky-300">{statusLabel[selected.status]} · {selected.config.device}</span><span className="text-xs text-slate-400">{new Date(selected.createdAt).toLocaleString()}</span></div>
            <progress aria-label="训练进度" value={selected.metrics.iteration} max={Math.max(1, selected.metrics.totalIterations)} className="h-2 w-full accent-sky-500" />
            <div className="flex flex-wrap gap-4 text-xs"><span>迭代 {selected.metrics.iteration} / {selected.metrics.totalIterations}</span>{selected.metrics.reward !== undefined && <span>奖励 {selected.metrics.reward.toFixed(3)}</span>}{selected.metrics.fps !== undefined && <span>{selected.metrics.fps.toFixed(0)} 环境步/s</span>}{selected.metrics.loss !== undefined && <span>损失 {selected.metrics.loss.toFixed(4)}</span>}</div>
            {selected.livePreviewEnabled && <TrainingLiveView frame={selected.liveFrame} jobId={selected.id} client={client} />}
            {selected.history && <RewardHistory history={selected.history} />}
            <RewardTermBars current={selected.metrics.rewardTerms} history={selected.history} card={cardSettings} defaults={cardTask?.taskCard} disabled={busy} onCommit={value => {
              if (cardJob) void act(async signal => acceptJob(await client.taskCard(cardJob.id, value, signal)));
              else setConfig(current => ({ ...current, taskCard: value }));
            }} />
            {selected.error && <p className="text-xs text-rose-300" role="alert">{selected.error}</p>}
            <div className="flex flex-wrap gap-2">{(activeStatuses.has(selected.status) || selected.liveFrame) && <button disabled={busy || selected.status === 'stopping'} onClick={() => void act(async signal => acceptJob(await client.preview(selected.id, !selected.livePreviewEnabled, signal)))} className="flex items-center gap-2 rounded bg-sky-800 px-3 py-2 text-xs disabled:opacity-40">{selected.livePreviewEnabled ? <EyeOff size={14} /> : <Eye size={14} />}{selected.livePreviewEnabled ? '关闭训练画面，加速训练' : '显示训练画面'}</button>}{activeStatuses.has(selected.status) && <button disabled={busy || selected.status === 'stopping'} onClick={() => void act(async signal => acceptJob(await client.stop(selected.id, signal)))} className="flex items-center gap-2 rounded bg-amber-800 px-3 py-2 text-xs disabled:opacity-40"><Square size={14} />停止并保存</button>}{selected.bundleReady && <button disabled={busy} onClick={() => void act(async signal => { const bundle = await client.bundle(selected.id, signal); if (mounted.current && !signal.aborted) onReplay(bundle); })} className="flex items-center gap-2 rounded bg-emerald-700 px-3 py-2 text-xs disabled:opacity-40"><Play size={14} />用 MuJoCo 回放训练结果</button>}{selected.bundleReady && <a href={trainingBundleUrl(selected.id)} download={`${selected.id}-replay.json`} className="flex items-center gap-2 rounded bg-slate-800 px-3 py-2 text-xs hover:bg-slate-700"><Download size={14} />下载完整回放包（JSON）</a>}</div>
            {!selected.livePreviewEnabled && activeStatuses.has(selected.status) && <p className="text-xs text-emerald-300">训练画面已关闭：worker不复制姿态数据，浏览器不加载WebGL，后台保持最快训练。</p>}
            <p className="text-xs text-slate-400">训练导出的模型、权重和观测定义一起回放，使用 MuJoCo；历史 PhysX 策略在原回放入口加载。</p>
          </div>
          <section><h3 className="mb-2 text-sm">检查点</h3>{selected.checkpoints.length ? <ul className="space-y-2">{selected.checkpoints.map(checkpoint => <li key={checkpoint.name} className="flex flex-wrap items-center justify-between gap-2 rounded border border-slate-800 px-3 py-2 text-xs"><span>{checkpoint.name} · {(checkpoint.bytes / 1024 ** 2).toFixed(1)} MiB</span><a href={trainingCheckpointUrl(selected.id, checkpoint.name)} download className="flex items-center gap-1 text-sky-300 hover:underline"><Download size={14} />下载检查点</a></li>)}</ul> : <p className="text-xs text-slate-500">尚未保存检查点。</p>}</section>
          <section><h3 className="mb-2 text-sm">训练日志</h3><pre aria-label="训练日志" className="max-h-[420px] overflow-auto whitespace-pre-wrap break-words rounded border border-slate-800 bg-black/30 p-3 font-mono text-[11px] leading-5 text-slate-300">{selected.logTail || '等待训练输出…'}</pre></section>
        </> : <p className="rounded border border-dashed border-slate-800 p-8 text-center text-sm text-slate-500">选择任务和计算设备后开始训练。</p>}
      </section>
    </div>
  </div>;
}
