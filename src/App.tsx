import React, { useState, useEffect, useRef, useMemo } from 'react';
import { PRESET_CONFIGURATIONS } from './data/presets';
import { ZbotConfiguration, GaitConfig, SimMetrics } from './types/zbot';
import { generateMujocoXML } from './utils/xmlGenerator';
import { mujocoEngine } from './mujoco/MujocoEngine';
import { meshManager } from './utils/meshManager';
import { SimulationViewport } from './components/SimulationViewport';
import { ConfigurationEditor } from './components/ConfigurationEditor';
import { GaitControlPanel } from './components/GaitControlPanel';
import { XmlModal } from './components/XmlModal';
import { MeshUploadModal } from './components/MeshUploadModal';
import { TelemetryDrawer } from './components/TelemetryDrawer';
import { ResearchPanel } from './components/ResearchPanel';
import { ArmControlPanel, ArmControlMode, ArmSolveStatus } from './components/ArmControlPanel';
import { EndEffectorTarget, getEndEffectorPose, isOrthogonalArm, solveInverseKinematics } from './utils/inverseKinematics';
import { Bot, Code2, UploadCloud, AlertCircle, HelpCircle, X } from 'lucide-react';

export default function App() {
  const [config, setConfig] = useState<ZbotConfiguration>(() => structuredClone(PRESET_CONFIGURATIONS[0]));
  const [gait, setGait] = useState<GaitConfig>(() => structuredClone(PRESET_CONFIGURATIONS[0].defaultGait));
  const [isRunning, setIsRunning] = useState(false);
  const [speedMultiplier, setSpeedMultiplier] = useState(1);
  const [friction, setFriction] = useState(1.2);
  const [kp, setKp] = useState(80);
  const [selfCollision, setSelfCollision] = useState(false);
  const [simMetrics, setSimMetrics] = useState<SimMetrics>(() => mujocoEngine.getMetrics());
  const [isMujocoReady, setIsMujocoReady] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isXmlModalOpen, setIsXmlModalOpen] = useState(false);
  const [isMeshModalOpen, setIsMeshModalOpen] = useState(false);
  const [isTelemetryOpen, setIsTelemetryOpen] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [meshRevision, setMeshRevision] = useState(0);
  const [customXmlOverride, setCustomXmlOverride] = useState<string | null>(null);
  const [panel, setPanel] = useState<'create' | 'research'>('create');
  const [armTarget, setArmTarget] = useState<EndEffectorTarget | null>(null);
  const [armMode, setArmMode] = useState<ArmControlMode>('translate');
  const [armStatus, setArmStatus] = useState<ArmSolveStatus | null>(null);
  const armSupported = isOrthogonalArm(config) && !customXmlOverride;
  const runningRef = useRef(false);
  const gaitRef = useRef(gait);
  const speedRef = useRef(speedMultiplier);
  const metricsRef = useRef(simMetrics);
  const epochRef = useRef(0);
  useEffect(() => { runningRef.current = isRunning; }, [isRunning]);
  useEffect(() => { gaitRef.current = gait; }, [gait]);
  useEffect(() => { speedRef.current = speedMultiplier; }, [speedMultiplier]);

  const xmlResult = useMemo(() => {
    try { return { xml: customXmlOverride ?? generateMujocoXML(config, { friction, kp, selfCollision }), error: null }; }
    catch (err) { return { xml: '', error: String(err) }; }
  }, [config, customXmlOverride, friction, kp, selfCollision]);

  useEffect(() => {
    let mounted = true;
    mujocoEngine.init().then(() => { if (mounted) setIsMujocoReady(true); })
      .catch(err => { if (mounted) setErrorMessage(`MuJoCo 初始化失败：${String(err)}`); });
    return () => { mounted = false; mujocoEngine.cleanupModel(); };
  }, []);

  const updateMetrics = (previous?: SimMetrics) => {
    const next = mujocoEngine.getMetrics(previous);
    metricsRef.current = next;
    setSimMetrics(next);
  };

  useEffect(() => {
    if (!isMujocoReady) return;
    runningRef.current = false;
    setIsRunning(false);
    epochRef.current++;
    setArmTarget(null); setArmStatus(null);
    try {
      if (xmlResult.error) throw new Error(xmlResult.error);
      mujocoEngine.loadModelFromXml(xmlResult.xml);
      mujocoEngine.resetSimulation(customXmlOverride ? undefined : config);
      updateMetrics();
      setErrorMessage(null);
    } catch (err) { setErrorMessage(`模型载入失败：${String(err)}`); }
  }, [isMujocoReady, xmlResult, meshRevision]);

  useEffect(() => {
    let animId = 0;
    let lastTime = performance.now();
    let accumulator = 0;
    let epoch = epochRef.current;
    const loop = (now: number) => {
      const dt = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;
      if (epoch !== epochRef.current) { epoch = epochRef.current; accumulator = 0; }
      if (runningRef.current && mujocoEngine.isReady()) {
        try {
          const timestep = Number(mujocoEngine.getModel().opt.timestep);
          accumulator += dt * speedRef.current;
          const steps = Math.floor(accumulator / timestep);
          if (steps > 0) {
            mujocoEngine.step(steps, gaitRef.current);
            accumulator -= steps * timestep;
            const next = mujocoEngine.getMetrics(metricsRef.current);
            if (![next.time, ...next.rootPos, ...Object.values(next.jointAngles)].every(Number.isFinite)) throw new Error('仿真状态出现非有限数值，请重置或降低控制增益');
            metricsRef.current = next;
            setSimMetrics(next);
          }
        } catch (err) {
          runningRef.current = false; setIsRunning(false); setErrorMessage(String(err));
        }
      } else accumulator = 0;
      animId = requestAnimationFrame(loop);
    };
    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, []);

  const handleConfigChange = (next: ZbotConfiguration) => {
    setArmTarget(null); setArmStatus(null);
    runningRef.current = false; setIsRunning(false);
    setConfig(structuredClone(next));
    setGait(structuredClone(next.defaultGait));
    setCustomXmlOverride(null);
  };
  const handleGaitChange = (next: GaitConfig) => {
    setArmTarget(null); setArmStatus(null);
    setGait(next); gaitRef.current = next;
    // At t=0 the paused workbench previews q directly; after a run controls remain physical targets.
    if (!isRunning && !errorMessage && mujocoEngine.isReady() && mujocoEngine.getTime() === 0 && next.type === 'manual') {
      mujocoEngine.applyPose(next.manualAngles); updateMetrics();
    }
  };
  const handleReset = () => {
    setArmTarget(null); setArmStatus(null);
    runningRef.current = false; setIsRunning(false); epochRef.current++;
    if (!mujocoEngine.isReady() || errorMessage) return;
    mujocoEngine.resetSimulation(customXmlOverride ? undefined : config);
    setGait(structuredClone(config.defaultGait));
    updateMetrics();
  };
  const handleStep = () => {
    if (errorMessage || !mujocoEngine.isReady()) return;
    mujocoEngine.step(1, gaitRef.current); updateMetrics(metricsRef.current);
  };
  const disabled = !isMujocoReady || !!errorMessage;

  const enableArm = () => {
    if (disabled || !armSupported) return;
    const observed = mujocoEngine.getMetrics().jointAngles;
    const q = config.modules.map((m,i) => observed[`joint_${i}`] ?? m.initialAngle ?? 0);
    const next = { ...gaitRef.current, type: 'manual' as const, manualAngles: Object.fromEntries(q.map((v,i) => [`joint_${i}`,v])) };
    setGait(next); gaitRef.current = next;
    setArmTarget(getEndEffectorPose(config, q));
    setArmStatus({ converged: true, positionError: 0, orientationError: 0 });
  };
  const changeArmTarget = (target: EndEffectorTarget) => {
    if (disabled || !armSupported || !armTarget) return;
    const seed = config.modules.map((m,i) => gaitRef.current.manualAngles[`joint_${i}`] ?? m.initialAngle ?? 0);
    const result = solveInverseKinematics(config, target, seed);
    setArmTarget(target); setArmStatus(result);
    if (!result.converged) return;
    const next = { ...gaitRef.current, type: 'manual' as const, manualAngles: Object.fromEntries(result.angles.map((v,i) => [`joint_${i}`,v])) };
    setGait(next); gaitRef.current = next;
    if (!runningRef.current) { mujocoEngine.applyPose(next.manualAngles); updateMetrics(); }
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      <header className="bg-slate-900 border-b border-slate-800 px-4 py-3 flex flex-wrap gap-3 items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center"><Bot className="w-5 h-5" /></div>
          <div><h1 className="font-bold text-sm">ZBot 六类构型实验台</h1><p className="text-[11px] text-slate-400">连接序列 σ → 关节轴拓扑 → 运动与接触 → 功能验证</p></div>
        </div>
        <nav aria-label="六类构型" className="flex gap-1 flex-wrap">
          {PRESET_CONFIGURATIONS.map((preset, i) => <button key={preset.id} onClick={() => handleConfigChange(preset)} className={`px-2 py-1.5 rounded text-xs ${config.id === preset.id ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'}`}>{String(i + 1).padStart(2, '0')} {['平面','正交','同手螺旋','异手均衡','单腿链','功能分区'][i]}</button>)}
        </nav>
        <div className="flex gap-2">
          <button title="CAD资产" onClick={() => setIsMeshModalOpen(true)} className="p-2 bg-slate-800 rounded"><UploadCloud size={16} /></button>
          <button title="MuJoCo XML" onClick={() => setIsXmlModalOpen(true)} className="p-2 bg-slate-800 rounded"><Code2 size={16} /></button>
          <button title="使用指南" onClick={() => setIsHelpOpen(true)} className="p-2 bg-slate-800 rounded"><HelpCircle size={16} /></button>
        </div>
      </header>
      <div className="px-4 py-1.5 flex flex-wrap gap-x-4 gap-y-1 text-[11px] border-b border-slate-800 text-slate-400">
        <span className={errorMessage ? 'text-rose-400' : isMujocoReady ? 'text-emerald-400' : 'text-amber-400'}>{errorMessage ? '模型异常 · 已暂停' : !isMujocoReady ? '物理引擎加载中…' : isRunning ? '动力学运行中' : simMetrics.time === 0 ? '构型预览 · 可编辑初始姿态' : '动力学已暂停'}</span>
        <span>{config.baseMode === 'fixed' ? '固定基座' : '自由基座'} · {config.modules.length} 模块</span>
        <span>{meshManager.hasCadMeshA() && meshManager.hasCadMeshB() ? 'OBJ 网格已载入' : '使用程序网格 / 等待资产'}</span>
        <label className="flex gap-1.5 items-center"><input type="checkbox" checked={selfCollision} disabled={!!customXmlOverride} onChange={e => setSelfCollision(e.target.checked)} />启用非相邻自碰撞</label>
        <span>{customXmlOverride ? '自定义 XML 模式：物理参数以 XML 为准' : '候选构型 · 接触采用凸包近似 · 功能需实验验证'}</span>
      </div>
      {errorMessage && <div role="alert" className="bg-rose-950 px-4 py-2 text-xs text-rose-200 flex gap-2"><AlertCircle size={16} /><span>{errorMessage}</span><button className="underline ml-auto" onClick={() => handleConfigChange(PRESET_CONFIGURATIONS[0])}>恢复平面预设</button></div>}
      <main className="flex-1 flex flex-col xl:flex-row overflow-y-auto xl:overflow-hidden min-h-0">
        <section className="flex-1 flex flex-col min-w-0 xl:min-h-0">
          <div className="flex-1 min-h-[360px] relative p-2"><SimulationViewport config={config} simMetrics={simMetrics} armTarget={armSupported && !disabled ? armTarget : null} armMode={armMode} armValid={armStatus?.converged ?? true} onArmTargetChange={changeArmTarget} onApplyImpulse={(x,y,z) => { if (!disabled) mujocoEngine.applyImpulse(x,y,z); }} /></div>
          <TelemetryDrawer metrics={simMetrics} isOpen={isTelemetryOpen} onToggle={() => setIsTelemetryOpen(!isTelemetryOpen)} />
          <fieldset disabled={disabled} className="shrink-0 disabled:opacity-50">
            <GaitControlPanel isRunning={isRunning} onTogglePlay={() => setIsRunning(v => !v)} onStep={handleStep} onReset={handleReset}
              gait={gait} onGaitChange={handleGaitChange} speedMultiplier={speedMultiplier} onSpeedMultiplierChange={setSpeedMultiplier}
              friction={friction} onFrictionChange={setFriction} kp={kp} onKpChange={setKp} metrics={simMetrics} jointCount={config.modules.length} jointRanges={config.modules.map(m => m.jointRange)} physicsDisabled={!!customXmlOverride} />
          </fieldset>
        </section>
        <aside className="w-full xl:w-[390px] shrink-0 flex flex-col min-h-[500px] xl:min-h-0 border-l border-slate-800">
          {armSupported && <ArmControlPanel target={armTarget} mode={armMode} status={armStatus} disabled={disabled} onEnable={enableArm} onDisable={() => { setArmTarget(null); setArmStatus(null); }} onModeChange={setArmMode} onTargetChange={changeArmTarget} />}
          <div className="grid grid-cols-2 p-2 gap-2 bg-slate-900 border-b border-slate-800 text-xs">
            <button onClick={() => setPanel('create')} className={`py-2 rounded ${panel === 'create' ? 'bg-blue-600' : 'bg-slate-800'}`}>构型创建</button>
            <button onClick={() => setPanel('research')} className={`py-2 rounded ${panel === 'research' ? 'bg-blue-600' : 'bg-slate-800'}`}>分析与实验记录</button>
          </div>
          <div className={panel === 'create' ? 'flex-1 min-h-0' : 'hidden'}><ConfigurationEditor currentConfig={config} onConfigChange={handleConfigChange} onOpenXmlModal={() => setIsXmlModalOpen(true)} onOpenMeshModal={() => setIsMeshModalOpen(true)} /></div>
          <div className={panel === 'research' ? 'flex-1 min-h-0 overflow-y-auto' : 'hidden'}><ResearchPanel config={config} gait={gait} metrics={simMetrics} friction={friction} kp={kp} selfCollision={selfCollision} xml={xmlResult.xml} disabled={disabled} /></div>
        </aside>
      </main>
      {isXmlModalOpen && <XmlModal isOpen onClose={() => setIsXmlModalOpen(false)} xmlContent={xmlResult.xml} onApplyCustomXml={setCustomXmlOverride} onResetToAutoXml={() => setCustomXmlOverride(null)} />}
      {isMeshModalOpen && <MeshUploadModal isOpen onClose={() => setIsMeshModalOpen(false)} onUpdateMeshes={(a,b) => { mujocoEngine.updateVfsMeshes(a,b); setMeshRevision(v => v + 1); }} hasCustomMeshA={meshManager.hasCadMeshA()} hasCustomMeshB={meshManager.hasCadMeshB()} />}
      {isHelpOpen && <div className="fixed inset-0 z-50 bg-black/75 flex items-center justify-center p-4"><div role="dialog" aria-label="使用指南" className="bg-slate-900 border border-slate-700 rounded-xl max-w-2xl p-6 text-sm space-y-4 max-h-[85vh] overflow-y-auto">
        <div className="flex justify-between"><h2 className="font-semibold">从构型设计到可复现实验</h2><button aria-label="关闭指南" onClick={() => setIsHelpOpen(false)}><X size={18}/></button></div>
        <p>1. 选择六类预设，查看五个连接角 σ 与六个初始关节角 q。σ 绕输出端局部 z 轴；q 绕模块内部 [0, −1, 1] 斜轴。0° / 90° / 180° / 270° 接口旋转对应标准相邻轴夹角 0° / 60° / 90° / 60°。</p>
        <p>2. 初始状态暂停，可直接预览 q；编辑连接角、初始姿态或基座模式会重建模型。固定基座用于关节与末端研究，自由基座用于地面接触实验。自由基座初始化会整体抬升以避免初始穿地。</p>
        <p>3. 启动动力学后，手动角度成为伺服目标。波形控制围绕 q 偏置振荡，不保证移动成功。重置恢复保存的初始构型。摩擦、增益和碰撞开关变化将暂停并重置实验。</p>
        <p>4. 在“分析与实验记录”中记录状态、导出 CSV 和含构型、控制参数、XML 的 JSON。比较时保持模块数、基座、控制、时间与物理参数一致。</p>
        <p className="text-amber-200">腿型为单腿链；未建立闭环或多足机器人。候选功能不等于已验证性能。OBJ碰撞使用凸包近似；质量、摩擦、驱动力矩尚未按实物标定，不能直接作承载结论。接口方位能否锁定也需实物确认。</p>
        <a className="text-blue-400 underline" href="https://chatgpt.com/share/6aa0f22a-9538-83ee-91b8-631f6cb2e845" target="_blank" rel="noreferrer">参考：ZBot 构型功能研究对话</a>
      </div></div>}
    </div>
  );
}
