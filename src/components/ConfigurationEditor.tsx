import React, { useEffect, useRef, useState } from 'react';
import { ZbotConfiguration, ZbotModule } from '../types/zbot';
import { PRESET_CONFIGURATIONS } from '../data/presets';
import { SEVEN_DOF_PRESETS } from '../data/sevenDofPresets';
import { createConfiguration, insertModule, parseConfiguration, validateConfiguration, withSerialModules } from '../utils/configuration';
import { Plus, Trash2, Copy, ArrowUp, ArrowDown, Code2, UploadCloud } from 'lucide-react';

interface ConfigurationEditorProps {
  currentConfig: ZbotConfiguration;
  onConfigChange: (newConfig: ZbotConfiguration) => void;
  onOpenXmlModal: () => void;
  onOpenMeshModal: () => void;
}
const field = 'w-full px-2 py-1.5 bg-slate-950 border border-slate-700 rounded text-xs text-slate-200';
const button = 'px-2 py-1.5 rounded border border-slate-700 bg-slate-800 hover:bg-slate-700 text-xs disabled:opacity-40';
const storageKey = 'zbot.configuration.v1';

export const ConfigurationEditor: React.FC<ConfigurationEditorProps> = ({ currentConfig: config, onConfigChange, onOpenXmlModal, onOpenMeshModal }) => {
  const [tab, setTab] = useState<'presets' | 'modules' | 'environment'>('modules');
  const [selectedId, setSelectedId] = useState(config.modules[0]?.id);
  const [message, setMessage] = useState('');
  const [name, setName] = useState(config.name);
  useEffect(() => { setName(config.name); }, [config.name]);
  const fileInput = useRef<HTMLInputElement>(null);
  const selected = config.modules.find(m => m.id === selectedId) ?? config.modules[0];
  const index = config.modules.indexOf(selected);
  const apply = (next: ZbotConfiguration, custom = true) => {
    const errors = validateConfiguration(next);
    if (errors.length) { setMessage(errors.join('；')); return; }
    setMessage('');
    onConfigChange(custom ? { ...next, id: 'custom', category: 'custom' } : structuredClone(next));
  };
  const update = (changes: Partial<ZbotModule>) => {
    const next = { ...config, modules: config.modules.map(m => m.id === selected.id ? { ...m, ...changes } : m) };
    if (changes.initialAngle !== undefined) next.defaultGait = { ...config.defaultGait, manualAngles: { ...config.defaultGait.manualAngles, [`joint_${index}`]: changes.initialAngle } };
    apply(next);
  };
  const insert = (after: number, copy = false) => {
    const next = insertModule(config, after, copy);
    apply(next); setSelectedId(next.modules[after + 1].id);
  };
  const move = (offset: number) => {
    const modules = [...config.modules];
    [modules[index], modules[index + offset]] = [modules[index + offset], modules[index]];
    apply(withSerialModules(config, modules));
  };
  const run = (action: () => void) => { try { action(); } catch (error) { setMessage(error instanceof Error ? error.message : String(error)); } };
  const namedConfiguration = () => {
    const next = { ...config, name: name.trim() };
    const errors = validateConfiguration(next);
    if (errors.length) throw new Error(errors.join('；'));
    if (next.name !== config.name) apply(next);
    return next;
  };
  const saveLocal = () => {
    const next = namedConfiguration();
    localStorage.setItem(storageKey, JSON.stringify(next));
    setMessage(`已保存“${next.name}”，可在此浏览器读取；导出 JSON 可另存文件`);
  };
  const exportJson = () => {
    const next = namedConfiguration();
    const url = URL.createObjectURL(new Blob([JSON.stringify(next, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `${next.name.replace(/[\\/:*?"<>|]/g, '_')}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <section className="flex flex-col h-full min-h-0 bg-slate-900 text-slate-200" aria-label="构型编辑器">
    <div className="p-3 space-y-3 border-b border-slate-800 shrink-0">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">ZBot 构型创建</h2>
        <button className="px-3 py-2 rounded bg-blue-600 hover:bg-blue-500 text-xs font-medium" onClick={() => {
          const next = createConfiguration(); apply(next, false); setSelectedId(next.modules[0].id); setTab('modules'); setName(next.name);
        }}><Plus className="inline w-3 h-3"/> 新建构型（1 模块）</button>
      </div>
      <p className="text-[11px] text-slate-400 leading-relaxed">从一个模块开始，增加或删除模块，调整连接方位与关节角度，实时查看三维构型。</p>
      <label className="block text-xs text-slate-400">构型名称<input className={`${field} mt-1`} maxLength={200} value={name} onChange={e => setName(e.target.value)} onBlur={() => run(namedConfiguration)}/></label>
      <div className="grid grid-cols-2 gap-2">
        <button className={`${button} border-blue-700 text-blue-200`} onClick={() => run(saveLocal)}>保存构型</button>
        <button className={button} onClick={() => run(() => { const text = localStorage.getItem(storageKey); if (!text) throw new Error('尚无本地保存的构型'); const next = parseConfiguration(text); apply(next, false); setName(next.name); setSelectedId(next.modules[0].id); setTab('modules'); setMessage(`已读取“${next.name}”`); })}>读取构型</button>
        <button className={button} onClick={() => run(exportJson)}>导出 JSON 文件</button>
        <button className={button} onClick={() => fileInput.current?.click()}>导入 JSON 文件</button>
      </div>
      <p className="text-[11px] text-slate-500">保存构型保留在此浏览器，后续保存会覆盖；导出文件可保留多份构型。</p>
    </div>
    <div className="flex border-b border-slate-800 px-2 pt-2 shrink-0">
      {([['modules', '模块编辑'], ['presets', '构型库'], ['environment', '基座设置']] as const).map(([key, title]) => <button key={key} onClick={() => setTab(key)} className={`px-3 py-2 text-xs border-b-2 ${tab === key ? 'text-blue-300 border-blue-500' : 'text-slate-400 border-transparent'}`}>{title}</button>)}
    </div>
    {message && <div role="status" className="m-3 p-2 border border-amber-700 text-amber-200 rounded text-xs break-words">{message}</div>}
    <div className="flex-1 overflow-auto p-3 space-y-3">
      {tab === 'presets' && <>
        <p className="text-xs text-slate-400 leading-relaxed">按参考对话复现六种候选姿态。每类 6 模块、5 个连接方位 σ、6 个动作角 q。点击构型可恢复原始参数。</p>
        {PRESET_CONFIGURATIONS.map(p => <button key={p.id} onClick={() => apply(p, false)} className={`w-full text-left p-3 rounded-lg border transition ${config.id === p.id ? 'border-blue-500 bg-blue-950/50' : 'border-slate-700 bg-slate-800/60 hover:border-slate-500'}`}>
          <span className="block text-xs font-semibold text-slate-100">{p.name}</span>
          <span className="block text-[11px] text-slate-400 mt-1 leading-relaxed">{p.hypothesis}</span>
          <span className="block font-mono text-[11px] text-blue-300 mt-2">σ = [{p.modules.slice(1).map(m => m.dockAngle).join(', ')}]°</span>
          <span className="block font-mono text-[11px] text-slate-300">q = [{p.modules.map(m => m.initialAngle).join(', ')}]°</span>
        </button>)}
        <h3 className="text-sm font-semibold text-cyan-200 pt-3 border-t border-slate-700">七自由度 · 等包络对照</h3>
        <p className="text-xs text-slate-400">统一7模块、106 mm安装节距；ZBot原始网格，蛇形/YuMi风格使用关节壳体与连接件。ZBot全部±180°；YuMi纵轴±180°、横轴±90°；蛇形全部±90°。三个预设从相同末端目标开始。</p>
        {SEVEN_DOF_PRESETS.map(p => <button key={p.id} onClick={() => apply(p, false)} className={`w-full text-left p-3 rounded-lg border transition-colors ${config.id === p.id ? 'border-cyan-500 bg-cyan-950/50' : 'border-slate-700 bg-slate-800/60 hover:border-cyan-600'}`}>
          <span className="block text-xs font-semibold">{p.name}</span><span className="block text-[11px] text-slate-400 mt-1 leading-relaxed">{p.hypothesis}</span>
          <span className="block text-[11px] text-cyan-300 mt-2">7 DOF · 固定基座 · {p.geometryMode === 'cad' ? 'ZBot OBJ' : '关节与连接结构'}</span>
        </button>)}
        <p className="text-[11px] text-amber-200/80 leading-relaxed">前四类按轴系排列，后两类按任务组织，分类可交叉。所有功能仍为待验证假设。</p>
      </>}
      {tab === 'modules' && <>
        <div className="flex justify-between items-center"><span className="text-xs text-slate-400">单链 · {config.modules.length}/24 模块</span><button className={button} disabled={config.modules.length >= 24} onClick={() => run(() => insert(config.modules.length - 1))}><Plus className="inline w-3 h-3"/> 末端追加</button></div>
        <div className="space-y-1">{config.modules.map((m, i) => <button key={m.id} onClick={() => setSelectedId(m.id)} className={`w-full text-left p-2 rounded border text-xs flex items-center justify-between ${m.id === selected.id ? 'border-blue-500 bg-blue-950/40' : 'border-slate-700 bg-slate-800/50'}`}><span><span className="inline-block w-2 h-2 mr-2 rounded-full" style={{ background: m.colorA }}/>{i + 1}. {m.name}</span><span className="font-mono text-slate-400">{i ? `σ ${m.dockAngle}° · ` : ''}q {m.initialAngle ?? 0}°</span></button>)}</div>
        <div className="flex flex-wrap gap-1">
          <button className={button} disabled={config.modules.length >= 24} onClick={() => run(() => insert(index))} title="在当前模块后插入"><Plus className="inline w-3 h-3"/> 插入</button>
          <button className={button} disabled={config.modules.length >= 24} onClick={() => run(() => insert(index, true))}><Copy className="inline w-3 h-3"/> 复制</button>
          <button className={button} disabled={index === 0} onClick={() => move(-1)} aria-label="模块前移"><ArrowUp className="w-3 h-3"/></button>
          <button className={button} disabled={index === config.modules.length - 1} onClick={() => move(1)} aria-label="模块后移"><ArrowDown className="w-3 h-3"/></button>
          <button className={button} disabled={config.modules.length <= 1} onClick={() => apply(withSerialModules(config, config.modules.filter(m => m.id !== selected.id)))}><Trash2 className="inline w-3 h-3"/> 删除</button>
        </div>
        <label className="block text-xs text-slate-400">模块名称<input className={`${field} mt-1`} value={selected.name} onChange={e => update({ name: e.target.value })}/></label>
        {index > 0 && <label className="block text-xs text-slate-400">连接方位 σ（°）<select className={`${field} mt-1`} value={selected.dockAngle} onChange={e => update({ dockAngle: Number(e.target.value), customEuler: undefined })}>{[0, 90, 180, 270].map(v => <option key={v} value={v}>{v}°</option>)}{![0,90,180,270].includes(selected.dockAngle) && <option value={selected.dockAngle}>{selected.dockAngle}°（导入值）</option>}</select></label>}
        <p className="text-xs text-cyan-300">关节范围：{selected.jointRange[0]}° ～ {selected.jointRange[1]}°</p>
        <label className="block text-xs text-slate-400">关节角度 q（°）<input className={`${field} mt-1`} type="number" min={selected.jointRange[0]} max={selected.jointRange[1]} value={selected.initialAngle ?? 0} onChange={e => update({ initialAngle: e.target.valueAsNumber })}/><input className="w-full mt-2 accent-blue-500" aria-label="初始动作角滑块" type="range" min={selected.jointRange[0]} max={selected.jointRange[1]} value={selected.initialAngle ?? 0} onChange={e => update({ initialAngle: Number(e.target.value) })}/></label>
        <div className="text-xs text-slate-400">关节轴 [x, y, z]<div className="grid grid-cols-3 gap-2 mt-1">{selected.jointAxis.map((v, i) => <input key={i} className={field} aria-label={`关节轴 ${'XYZ'[i]}`} type="number" value={v} onChange={e => { const axis = [...selected.jointAxis] as [number, number, number]; axis[i] = e.target.valueAsNumber; update({ jointAxis: axis }); }}/>)}</div></div>
        <div className="flex gap-3 text-xs text-slate-400">{(['colorA', 'colorB'] as const).map(key => <label key={key}>半模块 {key.slice(-1)}<input aria-label={key} className="ml-2 w-8 h-6 align-middle" type="color" value={selected[key] ?? '#3687b4'} onChange={e => update({ [key]: e.target.value })}/></label>)}</div>
        <p className="text-[11px] text-slate-500">增删与排序自动重接单链；q 随模块身份保留。σ 是安装方位，不等于相邻关节轴夹角。</p>
      </>}
      {tab === 'environment' && <>
        {config.modules.length === 7 && <div className="rounded-lg border border-cyan-800 p-3 space-y-2">
          <p className="text-xs text-cyan-200">七轴结构 · 106 mm安装节距</p>
          <label className="block text-xs text-slate-300">统一关节限位<select aria-label="统一关节限位" className={`${field} mt-1`} value={config.modules.every(m=>m.jointRange[0]===-90 && m.jointRange[1]===90) ? '90' : config.modules.every(m=>m.jointRange[0]===-180 && m.jointRange[1]===180) ? '180' : 'custom'} onChange={e=>{
            const limit=Number(e.target.value); if(!Number.isFinite(limit))return;
            const clamp=(v:number)=>Math.max(-limit,Math.min(limit,v));
            apply({...config, modules:config.modules.map(m=>({...m,jointRange:[-limit,limit],initialAngle:clamp(m.initialAngle??0)})), defaultGait:{...config.defaultGait,manualAngles:Object.fromEntries(config.modules.map((m,i)=>[`joint_${i}`,clamp(config.defaultGait.manualAngles[`joint_${i}`]??m.initialAngle??0)]))}});
          }}><option value="90">±90° · 共同基础条件</option><option value="180">±180° · 大转角对照</option><option value="custom" disabled>混合／自定义限位</option></select></label>
          <p className="text-[11px] text-slate-400">切换会暂停并重建；超出新范围的初始角会截断。概念结构每模块标称1 kg，ZBot沿用网格密度估算；均非实测质量。YuMi风格不代表ABB原机。</p>
        </div>}

        <label className="block text-xs text-slate-400">基座模式<select className={`${field} mt-1`} value={config.baseMode ?? 'free'} onChange={e => apply({ ...config, baseMode: e.target.value as 'fixed' | 'free' })}><option value="fixed">固定基座 · 姿态与操作实验</option><option value="free">自由基座 · 接触与运动实验</option></select></label>
        {(['rootPos', 'rootEuler'] as const).map(key => <div key={key} className="text-xs text-slate-400">{key === 'rootPos' ? '基座位置 X / Y / Z（m）' : '基座姿态 Roll / Pitch / Yaw（°）'}<div className="grid grid-cols-3 gap-2 mt-1">{config[key].map((v, i) => <input key={i} className={field} aria-label={`${key} ${'XYZ'[i]}`} type="number" step={key === 'rootPos' ? .01 : 5} value={v} onChange={e => { const vec = [...config[key]] as [number, number, number]; vec[i] = e.target.valueAsNumber; apply({ ...config, [key]: vec }); }}/>)}</div></div>)}
        <label className="block text-xs text-slate-400">研究假设<textarea rows={3} className={`${field} mt-1`} value={config.hypothesis ?? config.description} onChange={e => apply({ ...config, hypothesis: e.target.value })}/></label>
        <div className="grid grid-cols-2 gap-2 pt-3 border-t border-slate-800">
          <button className={button} onClick={onOpenXmlModal}><Code2 className="inline w-3 h-3"/> MuJoCo XML</button>
          <button className={button} onClick={onOpenMeshModal}><UploadCloud className="inline w-3 h-3"/> CAD 网格</button>
        </div>
        <p className="text-[11px] text-slate-500">几何：模块轴向节距 106 mm，斜轴铰链中心 z = 53 mm；坐标与尺寸按原模型解释。</p>
      </>}
    </div>
    <input ref={fileInput} className="hidden" type="file" accept=".json,application/json" onChange={async e => { const file = e.target.files?.[0]; e.target.value = ''; if (!file) return; try { if (file.size > 200_000) throw new Error('构型文件不得超过 200 KB'); const next = parseConfiguration(await file.text()); apply(next, false); setName(next.name); setSelectedId(next.modules[0].id); setTab('modules'); setMessage(`已导入“${next.name}”`); } catch (error) { setMessage(error instanceof Error ? error.message : String(error)); } }}/>
  </section>;
};
