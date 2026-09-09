import React, { useMemo, useState } from 'react';
import { GaitConfig, SimMetrics, ZbotConfiguration } from '../types/zbot';
import { forwardKinematics } from '../utils/kinematics';

interface Props {
  config: ZbotConfiguration;
  gait: GaitConfig;
  metrics: SimMetrics;
  friction: number;
  kp: number;
  selfCollision: boolean;
  xml: string;
  disabled: boolean;
}
interface Experiment {
  id: string;
  recordedAt: string;
  config: ZbotConfiguration;
  gait: GaitConfig;
  physics: { friction: number; kp: number; selfCollision: boolean };
  metrics: SimMetrics;
  xml: string;
}
const formatVector = (v?: number[]) => v ? `[${v.map(x => x.toFixed(3)).join(', ')}]` : '—';
function download(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const csvCell = (value: unknown) => `"${String(value ?? '').replace(/"/g, '""')}"`;

export function ResearchPanel({ config, gait, metrics, friction, kp, selfCollision, xml, disabled }: Props) {
  const [records, setRecords] = useState<Experiment[]>([]);
  const [tab, setTab] = useState<'analysis' | 'records'>('analysis');
  const kinematics = useMemo(() => forwardKinematics(config), [config]);
  const axisAngles = kinematics.joints.slice(1).map((joint, i) => {
    const previous = kinematics.joints[i].axis;
    return Math.acos(Math.max(-1, Math.min(1, joint.axis.reduce((sum, x, j) => sum + x * previous[j], 0)))) * 180 / Math.PI;
  });
  const record = () => {
    const item: Experiment = structuredClone({ id: crypto.randomUUID(), recordedAt: new Date().toISOString(), config, gait, physics: { friction, kp, selfCollision }, metrics, xml });
    setRecords(previous => [...previous, item].slice(-50)); setTab('records');
  };
  const exportCsv = () => {
    const header = ['构型','基座','模块数','仿真时间_s','基座路程_m','基座速度_m_s','末端X_m','末端Y_m','末端Z_m','接触数','峰值关节力矩_Nm','摩擦','Kp','自碰撞','控制类型','频率_Hz','振幅_deg','相位差_deg','偏置_deg','控制速度','sigma_deg','q0_deg','目标q_deg','记录时间'];
    const rows = records.map(r => [r.config.name, r.config.baseMode, r.config.modules.length, r.metrics.time, r.metrics.totalDistance, r.metrics.speed, ...(r.metrics.endEffectorPos ?? ['','','']), r.metrics.contactCount, Math.max(0,...Object.values(r.metrics.jointTorques).map(Math.abs)), r.physics.friction, r.physics.kp, r.physics.selfCollision, r.gait.type, r.gait.frequency, r.gait.amplitude, r.gait.phaseLag, r.gait.steering, r.gait.speed, r.config.modules.slice(1).map(m => m.dockAngle).join(' '), r.config.modules.map(m => m.initialAngle ?? 0).join(' '), JSON.stringify(r.gait.manualAngles), r.recordedAt]);
    download('zbot-experiments.csv', '\uFEFF' + [header,...rows].map(row => row.map(csvCell).join(',')).join('\r\n'), 'text/csv;charset=utf-8');
  };
  return <div className="p-4 space-y-4 text-xs bg-slate-900 min-h-full">
    <div><h2 className="font-semibold text-slate-100">构型—功能研究</h2><p className="text-slate-400 mt-1 leading-relaxed">{config.hypothesis ?? config.description}</p></div>
    <div className="flex gap-2"><button className={`px-3 py-1.5 rounded ${tab === 'analysis' ? 'bg-blue-600' : 'bg-slate-800'}`} onClick={() => setTab('analysis')}>轴系与测量</button><button className={`px-3 py-1.5 rounded ${tab === 'records' ? 'bg-blue-600' : 'bg-slate-800'}`} onClick={() => setTab('records')}>实验记录 ({records.length})</button></div>
    {tab === 'analysis' ? <>
      <div className="rounded-lg border border-slate-700 bg-slate-950 p-3 space-y-2">
        <h3 className="font-medium text-blue-300">机械程序 · σ 与 q₀</h3>
        <p className="font-mono break-words">σ = [{config.modules.slice(1).map(m => m.dockAngle).join(', ')}]°</p>
        <p className="font-mono break-words">q₀ = [{config.modules.map(m => m.initialAngle ?? 0).join(', ')}]°</p>
        <p className="text-slate-400">相邻关节轴夹角（初始姿态）</p><p className="font-mono text-amber-300">[{axisAngles.map(a => a.toFixed(1)).join(', ')}]°</p>
        <p className="text-[11px] text-slate-500">标准斜轴满足 cos α = (1 + cos σ) / 2。自定义轴向或完整欧拉旋转时以上显示按实际变换计算。</p>
      </div>
      <div className="rounded-lg border border-slate-700 p-3 space-y-2">
        <h3 className="font-medium text-blue-300">末端与接触</h3>
        <p className="text-slate-400">设计末端 / m <span className="block text-slate-200 font-mono">{formatVector(kinematics.tip)}</span></p>
        <p className="text-slate-400">当前仿真末端 / m <span className="block text-emerald-300 font-mono">{formatVector(metrics.endEffectorPos)}</span></p>
        <p className="text-slate-400">当前接触点 <span className="float-right text-slate-200">{metrics.contactCount ?? '—'}</span></p>
        <p className="text-slate-400">当前最大 |τ| <span className="float-right text-slate-200">{Math.max(0, ...Object.values(metrics.jointTorques).map(Math.abs)).toFixed(3)} N·m</span></p>
        <p className="text-[11px] text-slate-500">设计末端来自正运动学；自由基座落地后会改变世界坐标。接触数量和瞬时力矩不能单独证明稳定支撑。</p>
      </div>
      <div className="text-slate-400 leading-relaxed space-y-2"><p>建议实验：固定基座检查轴系与末端轨迹；自由基座检查接地、抬升和运动表现。分别记录相同时间点的数据。</p><p>前四类按连接排列分类，腿型与功能分区型按任务组织分类，允许交叉。运动、承载及攀爬功能均需进一步验证。</p></div>
    </> : <>
      <p className="text-slate-400 leading-relaxed">记录当前状态快照，最多保留 50 条；刷新页面前请导出。JSON 包含完整构型、控制、物理参数及 XML，CSV 便于比较测量值。</p>
      <div className="flex gap-2"><button disabled={!records.length} onClick={exportCsv} className="bg-slate-800 px-3 py-2 rounded disabled:opacity-40">导出 CSV</button><button disabled={!records.length} onClick={() => download('zbot-experiments.json', JSON.stringify({ schemaVersion: 1, units: { angle: 'degree', distance: 'm', torque: 'N*m' }, records }, null, 2), 'application/json')} className="bg-slate-800 px-3 py-2 rounded disabled:opacity-40">导出完整 JSON</button></div>
      {!records.length && <p className="text-slate-500 py-6 text-center">尚无记录。选择构型，运行实验后记录状态。</p>}
      {records.map((r, index) => <div key={r.id} className="border border-slate-700 rounded-lg p-3 bg-slate-950 space-y-1.5"><div className="flex justify-between gap-2"><h3 className="font-medium text-slate-200">{index + 1}. {r.config.name}</h3><button aria-label={`删除记录 ${index + 1}`} className="text-slate-500 hover:text-rose-400" onClick={() => setRecords(prev => prev.filter(x => x.id !== r.id))}>×</button></div><p className="text-slate-400">{r.config.baseMode === 'fixed' ? '固定' : '自由'}基座 · {r.metrics.time.toFixed(3)} s · μ {r.physics.friction} · Kp {r.physics.kp}</p><p className="font-mono text-emerald-300">路程 {r.metrics.totalDistance.toFixed(4)} m · 接触 {r.metrics.contactCount ?? '—'}</p><p className="text-slate-400 font-mono">末端 {formatVector(r.metrics.endEffectorPos)}</p></div>)}
      <p className="text-amber-200/80 text-[11px]">比较时请核对基座模式、控制参数、采样时刻及碰撞设置。自定义 XML 中的设置以所记录 XML 为准。</p>
    </>}
    <button disabled={disabled} onClick={record} className="w-full py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-40 font-medium">记录当前实验状态</button>
  </div>;
}
