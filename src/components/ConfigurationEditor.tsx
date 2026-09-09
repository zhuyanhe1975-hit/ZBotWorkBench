import React, { useRef, useState } from 'react';
import { ZbotConfiguration, ZbotModule } from '../types/zbot';
import { PRESET_CONFIGURATIONS } from '../data/presets';
import { parseConfiguration, validateConfiguration, withSerialModules } from '../utils/configuration';
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
  const [tab, setTab] = useState<'presets' | 'modules' | 'environment'>('presets');
  const [selectedId, setSelectedId] = useState(config.modules[0]?.id);
  const [message, setMessage] = useState('');
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
    const source = config.modules[after];
    const m: ZbotModule = copy ? { ...source, id: `mod_${crypto.randomUUID()}`, name: `${source.name} 副本` } : {
      id: `mod_${crypto.randomUUID()}`, name: `模块 ${config.modules.length + 1}`, parentId: null,
      dockAngle: 180, initialAngle: 0, jointAxis: [0, -1, 1], jointRange: [-180, 180], colorA: '#3687b4', colorB: '#57ab98',
    };
    const modules = [...config.modules];
    modules.splice(after + 1, 0, m);
    apply(withSerialModules(config, modules)); setSelectedId(m.id);
  };
  const move = (offset: number) => {
    const modules = [...config.modules];
    [modules[index], modules[index + offset]] = [modules[index + offset], modules[index]];
    apply(withSerialModules(config, modules));
  };
  const run = (action: () => void) => { try { action(); } catch (error) { setMessage(error instanceof Error ? error.message : String(error)); } };
  const exportJson = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `${config.id}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <section className="flex flex-col h-full min-h-0 bg-slate-900 text-slate-200" aria-label="构型编辑器">
    <div className="flex border-b border-slate-800 px-2 pt-2 shrink-0">
      {([['presets', '六类构型库'], ['modules', '串联编辑'], ['environment', '基座与保存']] as const).map(([key, title]) => <button key={key} onClick={() => setTab(key)} className={`px-3 py-2 text-xs border-b-2 ${tab === key ? 'text-blue-300 border-blue-500' : 'text-slate-400 border-transparent'}`}>{title}</button>)}
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
        <p className="text-[11px] text-amber-200/80 leading-relaxed">前四类按轴系排列，后两类按任务组织，分类可交叉。所有功能仍为待验证假设。</p>
      </>}
      {tab === 'modules' && <>
        <div className="flex justify-between items-center"><span className="text-xs text-slate-400">单链 · {config.modules.length}/24 模块</span><button className={button} disabled={config.modules.length >= 24} onClick={() => insert(config.modules.length - 1)}><Plus className="inline w-3 h-3"/> 末端追加</button></div>
        <div className="space-y-1">{config.modules.map((m, i) => <button key={m.id} onClick={() => setSelectedId(m.id)} className={`w-full text-left p-2 rounded border text-xs flex items-center justify-between ${m.id === selected.id ? 'border-blue-500 bg-blue-950/40' : 'border-slate-700 bg-slate-800/50'}`}><span><span className="inline-block w-2 h-2 mr-2 rounded-full" style={{ background: m.colorA }}/>{i + 1}. {m.name}</span><span className="font-mono text-slate-400">{i ? `σ ${m.dockAngle}° · ` : ''}q {m.initialAngle ?? 0}°</span></button>)}</div>
        <div className="flex flex-wrap gap-1">
          <button className={button} disabled={config.modules.length >= 24} onClick={() => insert(index)} title="在当前模块后插入"><Plus className="inline w-3 h-3"/> 插入</button>
          <button className={button} disabled={config.modules.length >= 24} onClick={() => insert(index, true)}><Copy className="inline w-3 h-3"/> 复制</button>
          <button className={button} disabled={index === 0} onClick={() => move(-1)} aria-label="模块前移"><ArrowUp className="w-3 h-3"/></button>
          <button className={button} disabled={index === config.modules.length - 1} onClick={() => move(1)} aria-label="模块后移"><ArrowDown className="w-3 h-3"/></button>
          <button className={button} disabled={config.modules.length <= 1} onClick={() => apply(withSerialModules(config, config.modules.filter(m => m.id !== selected.id)))}><Trash2 className="inline w-3 h-3"/> 删除</button>
        </div>
        <label className="block text-xs text-slate-400">模块名称<input className={`${field} mt-1`} value={selected.name} onChange={e => update({ name: e.target.value })}/></label>
        {index > 0 && <label className="block text-xs text-slate-400">连接方位 σ（°）<select className={`${field} mt-1`} value={selected.dockAngle} onChange={e => update({ dockAngle: Number(e.target.value), customEuler: undefined })}>{[0, 90, 180, 270].map(v => <option key={v} value={v}>{v}°</option>)}{![0,90,180,270].includes(selected.dockAngle) && <option value={selected.dockAngle}>{selected.dockAngle}°（导入值）</option>}</select></label>}
        <label className="block text-xs text-slate-400">初始动作角 q（°）<input className={`${field} mt-1`} type="number" min={selected.jointRange[0]} max={selected.jointRange[1]} value={selected.initialAngle ?? 0} onChange={e => update({ initialAngle: e.target.valueAsNumber })}/><input className="w-full mt-2 accent-blue-500" aria-label="初始动作角滑块" type="range" min={selected.jointRange[0]} max={selected.jointRange[1]} value={selected.initialAngle ?? 0} onChange={e => update({ initialAngle: Number(e.target.value) })}/></label>
        <div className="text-xs text-slate-400">关节轴 [x, y, z]<div className="grid grid-cols-3 gap-2 mt-1">{selected.jointAxis.map((v, i) => <input key={i} className={field} aria-label={`关节轴 ${'XYZ'[i]}`} type="number" value={v} onChange={e => { const axis = [...selected.jointAxis] as [number, number, number]; axis[i] = e.target.valueAsNumber; update({ jointAxis: axis }); }}/>)}</div></div>
        <div className="flex gap-3 text-xs text-slate-400">{(['colorA', 'colorB'] as const).map(key => <label key={key}>半模块 {key.slice(-1)}<input aria-label={key} className="ml-2 w-8 h-6 align-middle" type="color" value={selected[key] ?? '#3687b4'} onChange={e => update({ [key]: e.target.value })}/></label>)}</div>
        <p className="text-[11px] text-slate-500">增删与排序自动重接单链；q 随模块身份保留。σ 是安装方位，不等于相邻关节轴夹角。</p>
      </>}
      {tab === 'environment' && <>
        <label className="block text-xs text-slate-400">构型名称<input className={`${field} mt-1`} value={config.name} onChange={e => apply({ ...config, name: e.target.value })}/></label>
        <label className="block text-xs text-slate-400">基座模式<select className={`${field} mt-1`} value={config.baseMode ?? 'free'} onChange={e => apply({ ...config, baseMode: e.target.value as 'fixed' | 'free' })}><option value="fixed">固定基座 · 姿态与操作实验</option><option value="free">自由基座 · 接触与运动实验</option></select></label>
        {(['rootPos', 'rootEuler'] as const).map(key => <div key={key} className="text-xs text-slate-400">{key === 'rootPos' ? '基座位置 X / Y / Z（m）' : '基座姿态 Roll / Pitch / Yaw（°）'}<div className="grid grid-cols-3 gap-2 mt-1">{config[key].map((v, i) => <input key={i} className={field} aria-label={`${key} ${'XYZ'[i]}`} type="number" step={key === 'rootPos' ? .01 : 5} value={v} onChange={e => { const vec = [...config[key]] as [number, number, number]; vec[i] = e.target.valueAsNumber; apply({ ...config, [key]: vec }); }}/>)}</div></div>)}
        <label className="block text-xs text-slate-400">研究假设<textarea rows={3} className={`${field} mt-1`} value={config.hypothesis ?? config.description} onChange={e => apply({ ...config, hypothesis: e.target.value })}/></label>
        <div className="grid grid-cols-2 gap-2 pt-3 border-t border-slate-800">
          <button className={button} onClick={() => run(() => { localStorage.setItem(storageKey, JSON.stringify(config)); setMessage('已保存到此浏览器'); })}>本地保存</button>
          <button className={button} onClick={() => run(() => { const text = localStorage.getItem(storageKey); if (!text) throw new Error('尚无本地保存的构型'); apply(parseConfiguration(text), false); })}>读取本地</button>
          <button className={button} onClick={() => run(exportJson)}>导出 JSON</button>
          <button className={button} onClick={() => fileInput.current?.click()}>导入 JSON</button>
          <button className={button} onClick={onOpenXmlModal}><Code2 className="inline w-3 h-3"/> MuJoCo XML</button>
          <button className={button} onClick={onOpenMeshModal}><UploadCloud className="inline w-3 h-3"/> CAD 网格</button>
        </div>
        <input ref={fileInput} className="hidden" type="file" accept=".json,application/json" onChange={async e => { const file = e.target.files?.[0]; e.target.value = ''; if (!file) return; try { if (file.size > 200_000) throw new Error('构型文件不得超过 200 KB'); apply(parseConfiguration(await file.text()), false); } catch (error) { setMessage(error instanceof Error ? error.message : String(error)); } }}/>
        <p className="text-[11px] text-slate-500">几何：模块轴向节距 106 mm，斜轴铰链中心 z = 53 mm；坐标与尺寸按原模型解释。</p>
      </>}
    </div>
  </section>;
};
