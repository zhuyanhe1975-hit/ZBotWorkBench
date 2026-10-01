import React, { useEffect, useRef, useState } from 'react';
import { ZbotConfiguration, ZbotModule } from '../types/zbot';
import { PRESET_CONFIGURATIONS } from '../data/presets';
import { SEVEN_DOF_PRESETS } from '../data/sevenDofPresets';
import { appendConnectorModule, createTetrahedronConfiguration, createCubeConfiguration, createCubeQuadruped, removeConnectorBranch, createConfiguration, insertModule, parseConfiguration, validateConfiguration, withSerialModules } from '../utils/configuration';
import { createZip, modelExportFiles } from '../utils/modelExport';
import { readConfigurationLibrary, saveConfigurationToLibrary, deleteConfigurationFromLibrary } from '../utils/configurationLibrary';
import { meshManager } from '../utils/meshManager';
import { connectorFaces, connectorName, faceLabel } from '../utils/moduleMount';
import { XmlGeneratorOptions } from '../utils/xmlGenerator';
import { Plus, Trash2, Copy, ArrowUp, ArrowDown, Code2, UploadCloud } from 'lucide-react';

interface ConfigurationEditorProps {
  currentConfig: ZbotConfiguration;
  exportOptions?: XmlGeneratorOptions;
  onConfigChange: (newConfig: ZbotConfiguration) => void;
  onOpenXmlModal: () => void;
  onOpenMeshModal: () => void;
}
const field = 'w-full px-2 py-1.5 bg-slate-950 border border-slate-700 rounded text-xs text-slate-200';
const button = 'px-2 py-1.5 rounded border border-slate-700 bg-slate-800 hover:bg-slate-700 text-xs disabled:opacity-40';

const displayAngle = (angle: number) => Number(angle.toFixed(2));

export const ConfigurationEditor: React.FC<ConfigurationEditorProps> = ({ currentConfig: config, exportOptions, onConfigChange, onOpenXmlModal, onOpenMeshModal }) => {
  const [tab, setTab] = useState<'saved' | 'presets' | 'modules' | 'environment'>('modules');
  const [selectedId, setSelectedId] = useState(config.modules[0]?.id);
  const [message, setMessage] = useState('');
  const [name, setName] = useState(config.name);
  const [savedConfigurations, setSavedConfigurations] = useState<ZbotConfiguration[]>([]);
  useEffect(() => {
    try { setSavedConfigurations(readConfigurationLibrary(localStorage)); }
    catch (error) { setMessage(error instanceof Error ? error.message : String(error)); }
  }, []);
  useEffect(() => { setName(config.name); }, [config.name]);
  const fileInput = useRef<HTMLInputElement>(null);
  const selected = config.modules.find(m => m.id === selectedId) ?? config.modules[0];
  const index = selected ? config.modules.indexOf(selected) : -1;
  const rootName = connectorName(config.rootConnector?.type);
  const rootFaces = connectorFaces(config.rootConnector?.type);
  const outputOccupied = !!selected && config.modules.some(m => m.parentId === selected.id);
  const apply = (next: ZbotConfiguration, custom = true) => {
    const errors = validateConfiguration(next);
    if (errors.length) { setMessage(errors.join('；')); return; }
    setMessage('');
    onConfigChange(custom ? { ...next, id: 'custom', category: 'custom' } : structuredClone(next));
  };
  const update = (changes: Partial<ZbotModule>) => {
    if (!selected) return;
    const next = { ...config, modules: config.modules.map(m => m.id === selected?.id ? { ...m, ...changes } : m) };
    if (changes.initialAngle !== undefined) next.defaultGait = { ...config.defaultGait, manualAngles: { ...config.defaultGait.manualAngles, [`joint_${index}`]: changes.initialAngle } };
    apply(next);
  };
  const insert = (after: number, copy = false) => {
    const next = insertModule(config, after, copy);
    apply(next); setSelectedId(next.modules[config.rootConnector ? next.modules.length - 1 : after + 1].id);
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
    setSavedConfigurations(saveConfigurationToLibrary(localStorage, next));
    setTab('saved');
    setMessage(`已将“${next.name}”加入构型列表`);
  };
  const exportJson = () => {
    const next = namedConfiguration();
    const url = URL.createObjectURL(new Blob([JSON.stringify(next, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `${next.name.replace(/[\\/:*?"<>|]/g, '_')}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const exportModel = (format: 'mjcf' | 'urdf') => {
    const next = namedConfiguration();
    const files = modelExportFiles(next, format, { ma: meshManager.getMeshAText(), mb: meshManager.getMeshBText() }, exportOptions);
    const zip = createZip(files);
    const url = URL.createObjectURL(new Blob([zip as BlobPart], { type: 'application/zip' }));
    const link = document.createElement('a'); link.href = url;
    link.download = `${next.name.replace(/[\\/:*?"<>|]/g, '_')}-${format}.zip`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setMessage(`已导出 ${format.toUpperCase()}，解压后使用模型文件；包内附构型与初始关节位置${format === 'urdf' ? '。URDF 为运动学模型，动力学使用前需补充惯量与实际电机限值' : ''}`);
  };
  return <section className="flex flex-col h-full min-h-0 bg-slate-900 text-slate-200" aria-label="构型编辑器">
    <div className="p-3 space-y-3 border-b border-slate-800 shrink-0">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">ZBot 构型创建</h2>
        <button className="px-3 py-2 rounded bg-blue-600 hover:bg-blue-500 text-xs font-medium" onClick={() => {
          const next = createConfiguration(); apply(next, false); setSelectedId(next.modules[0]?.id); setTab('modules'); setName(next.name);
        }}><Plus className="inline w-3 h-3"/> 新建构型（1 模块）</button>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <button className={button} onClick={() => { const next = createCubeConfiguration(); apply(next, false); setSelectedId(undefined); setTab('modules'); setName(next.name); }}>新建立方体根（100 mm）</button>
        <button className={button} onClick={() => { const next = createCubeQuadruped(); apply(next, false); setSelectedId(next.modules[0]?.id); setTab('modules'); setName(next.name); }}>立方体四足示例</button>
        <button className={`${button} col-span-2`} onClick={() => { const next = createTetrahedronConfiguration(); apply(next, false); setSelectedId(undefined); setTab('modules'); setName(next.name); }}>新建正四面体根（面内切圆 Ø103.92 mm）</button>
      </div>
      <label className="block text-xs text-slate-400">基座模式<select aria-label="基座模式" className={`${field} mt-1`} value={config.baseMode ?? (config.category === 'arm' ? 'fixed' : 'free')} onChange={e => apply({ ...config, baseMode: e.target.value as 'fixed' | 'free' })}><option value="fixed">固定基座 · 姿态与操作实验</option><option value="free">浮动基座 · 六自由度接触与运动实验</option></select><span className="block mt-1 text-[11px] text-slate-500">所有构型均可切换；切换会暂停并重建模型。浮动基座可在重力、接触和关节驱动下平移与旋转。</span></label>
      <p className="text-[11px] text-slate-400 leading-relaxed">从模块或连接件根开始，增加或删除模块，调整连接方位与关节角度，实时查看三维构型。</p>
      <label className="block text-xs text-slate-400">构型名称<input className={`${field} mt-1`} maxLength={200} value={name} onChange={e => setName(e.target.value)} onBlur={() => run(namedConfiguration)}/></label>
      <div className="grid grid-cols-2 gap-2">
        <button className={`${button} border-blue-700 text-blue-200`} onClick={() => run(saveLocal)}>保存构型</button>
        <button className={button} onClick={() => run(() => { setSavedConfigurations(readConfigurationLibrary(localStorage)); setTab('saved'); })}>构型列表</button>
        <button className={button} onClick={() => run(exportJson)}>导出 JSON 文件</button>
        <button className={button} onClick={() => fileInput.current?.click()}>导入 JSON 文件</button>
        <button className={button} onClick={() => run(() => exportModel('mjcf'))}>保存 MJCF（ZIP）</button>
        <button className={button} onClick={() => run(() => exportModel('urdf'))}>保存 URDF（ZIP）</button>
      </div>
      <p className="text-[11px] text-slate-500">每次保存都会向构型列表添加一份快照，同名构型也可保留多份；刷新后仍可读取。列表保存在此浏览器，导出 JSON 可备份。MJCF / URDF 压缩包包含模型及所需网格，解压后使用。</p>
    </div>
    <div className="flex border-b border-slate-800 px-2 pt-2 shrink-0">
      {([['modules', '模块编辑'], ['saved', '构型列表'], ['presets', '预设构型'], ['environment', '基座设置']] as const).map(([key, title]) => <button key={key} onClick={() => setTab(key)} className={`px-3 py-2 text-xs border-b-2 ${tab === key ? 'text-blue-300 border-blue-500' : 'text-slate-400 border-transparent'}`}>{title}</button>)}
    </div>
    {message && <div role="status" className="m-3 p-2 border border-amber-700 text-amber-200 rounded text-xs break-words">{message}</div>}
    <div className="flex-1 overflow-auto p-3 space-y-3">
      {tab === 'saved' && <>
        <h3 className="text-sm font-semibold text-blue-200">我的构型 · {savedConfigurations.length}</h3>
        {savedConfigurations.length === 0 && <p className="text-xs text-slate-400">尚未保存构型。点击“保存构型”添加到列表。</p>}
        {savedConfigurations.map(saved => <div key={saved.id} className="flex gap-2 items-center rounded-lg border border-slate-700 bg-slate-800/60 p-2">
          <button className="flex-1 min-w-0 text-left p-1" onClick={() => { apply(saved, false); setName(saved.name); setSelectedId(saved.modules[0]?.id); setTab('modules'); setMessage(`已读取“${saved.name}”`); }}>
            <span className="block text-xs font-semibold break-words">{saved.name}</span>
            <span className="block text-[11px] text-slate-400 mt-1">{saved.modules.length} 模块 · {saved.rootConnector ? connectorName(saved.rootConnector.type) : '单链'} · {saved.baseMode === 'fixed' ? '固定基座' : '浮动基座'}</span>
          </button>
          <button className={button} aria-label={`删除已保存构型 ${saved.name}`} onClick={() => run(() => { setSavedConfigurations(deleteConfigurationFromLibrary(localStorage, saved.id)); setMessage(`已从列表删除“${saved.name}”`); })}><Trash2 className="w-3 h-3"/></button>
        </div>)}
      </>}
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
          <span className="block text-[11px] text-cyan-300 mt-2">7 DOF · 默认固定基座（可切换） · {p.geometryMode === 'cad' ? 'ZBot OBJ' : '关节与连接结构'}</span>
        </button>)}
        <p className="text-[11px] text-amber-200/80 leading-relaxed">前四类按轴系排列，后两类按任务组织，分类可交叉。所有功能仍为待验证假设。</p>
      </>}
      {tab === 'modules' && <>
        <div className="flex justify-between items-center"><span className="text-xs text-slate-400">{config.rootConnector ? `${rootName}多分支` : '单链'} · {config.modules.length}/24 模块</span><button className={button} disabled={config.modules.length >= 24 || (config.rootConnector && (!selected || outputOccupied))} onClick={() => run(() => insert(config.rootConnector ? index : config.modules.length - 1))}><Plus className="inline w-3 h-3"/> 末端追加</button></div>
        {config.rootConnector && <div className="rounded border border-slate-600 bg-slate-950 p-3 space-y-2">
          <p className="text-xs text-slate-200">根结构件 · {rootName} · {config.rootConnector.type === 'tetrahedron' ? '面内切圆 Ø103.92 mm · 棱长 180 mm' : '边长 100 mm'}</p>
          <p className="text-[11px] text-slate-400">点击空闲面添加分支；每面一个端口，模块局部 +Z 朝外。仿真质量标称 1 kg。</p>
          <div className={`grid ${config.rootConnector.type === 'tetrahedron' ? 'grid-cols-2' : 'grid-cols-3'} gap-2`}>{rootFaces.map(face => {
            const occupied = config.modules.some(m => m.parentId === null && m.mountFace === face);
            return <button key={face} className={button} disabled={occupied || config.modules.length >= 24} onClick={() => run(() => { const next = appendConnectorModule(config, undefined, face); apply(next); setSelectedId(next.modules[next.modules.length - 1].id); })}>{faceLabel(face)} {occupied ? '已连接' : '添加'}</button>;
          })}</div>
        </div>}
        <div className="space-y-1">{config.modules.map((m, i) => <button key={m.id} onClick={() => setSelectedId(m.id)} className={`w-full text-left p-2 rounded border text-xs flex items-center justify-between ${m.id === selected?.id ? 'border-blue-500 bg-blue-950/40' : 'border-slate-700 bg-slate-800/50'}`}><span><span className="inline-block w-2 h-2 mr-2 rounded-full" style={{ background: m.colorA }}/>{i + 1}. {m.name}</span><span className="font-mono text-slate-400">{config.rootConnector ? `${m.parentId === null ? `${rootName} ${faceLabel(m.mountFace!)}` : `↳ ${config.modules.find(p => p.id === m.parentId)?.name}`} · ` : ''}{i || config.rootConnector ? `σ ${displayAngle(m.dockAngle)}° · ` : ''}q {displayAngle(m.initialAngle ?? 0)}°</span></button>)}</div>
        {selected && <>
        <div className="flex flex-wrap gap-1">
          <button className={button} disabled={config.modules.length >= 24 || (config.rootConnector && outputOccupied)} onClick={() => run(() => insert(index))} title="在当前模块后插入"><Plus className="inline w-3 h-3"/> 插入</button>
          <button className={button} disabled={config.modules.length >= 24 || (config.rootConnector && outputOccupied)} onClick={() => run(() => insert(index, true))}><Copy className="inline w-3 h-3"/> 复制</button>
          <button className={button} disabled={!!config.rootConnector || index === 0} onClick={() => move(-1)} aria-label="模块前移"><ArrowUp className="w-3 h-3"/></button>
          <button className={button} disabled={!!config.rootConnector || index === config.modules.length - 1} onClick={() => move(1)} aria-label="模块后移"><ArrowDown className="w-3 h-3"/></button>
          <button className={button} disabled={!config.rootConnector && config.modules.length <= 1} onClick={() => apply(config.rootConnector ? removeConnectorBranch(config, selected.id) : withSerialModules(config, config.modules.filter(m => m.id !== selected.id)))}><Trash2 className="inline w-3 h-3"/> {config.rootConnector ? '删除此分支' : '删除'}</button>
        </div>
        <label className="block text-xs text-slate-400">模块名称<input className={`${field} mt-1`} value={selected.name} onChange={e => update({ name: e.target.value })}/></label>
        {config.rootConnector && selected.parentId === null && <label className="block text-xs text-slate-400">{rootName}连接面<select aria-label={`${rootName}连接面`} className={`${field} mt-1`} value={selected.mountFace} onChange={e => update({ mountFace: e.target.value as ZbotModule['mountFace'] })}>{rootFaces.map(face => <option key={face} value={face} disabled={config.modules.some(m => m.id !== selected.id && m.parentId === null && m.mountFace === face)}>{faceLabel(face)}</option>)}</select></label>}
        {(index > 0 || config.rootConnector) && <label className="block text-xs text-slate-400">连接方位 σ（°）<select className={`${field} mt-1`} value={selected.dockAngle} onChange={e => update({ dockAngle: Number(e.target.value), customEuler: undefined })}>{[0, 90, 180, 270].map(v => <option key={v} value={v}>{v}°</option>)}{![0,90,180,270].includes(selected.dockAngle) && <option value={selected.dockAngle}>{displayAngle(selected.dockAngle)}°（自定义）</option>}</select></label>}
        <p className="text-xs text-cyan-300">关节范围：{selected.jointRange[0]}° ～ {selected.jointRange[1]}°</p>
        <label className="block text-xs text-slate-400">关节角度 q（°）<input className={`${field} mt-1`} type="number" min={selected.jointRange[0]} max={selected.jointRange[1]} value={selected.initialAngle ?? 0} onChange={e => update({ initialAngle: e.target.valueAsNumber })}/><input className="w-full mt-2 accent-blue-500" aria-label="初始动作角滑块" type="range" min={selected.jointRange[0]} max={selected.jointRange[1]} value={selected.initialAngle ?? 0} onChange={e => update({ initialAngle: Number(e.target.value) })}/></label>
        <div className="text-xs text-slate-400">关节轴 [x, y, z]<div className="grid grid-cols-3 gap-2 mt-1">{selected.jointAxis.map((v, i) => <input key={i} className={field} aria-label={`关节轴 ${'XYZ'[i]}`} type="number" value={v} onChange={e => { const axis = [...selected.jointAxis] as [number, number, number]; axis[i] = e.target.valueAsNumber; update({ jointAxis: axis }); }}/>)}</div></div>
        <div className="flex gap-3 text-xs text-slate-400">{(['colorA', 'colorB'] as const).map(key => <label key={key}>半模块 {key.slice(-1)}<input aria-label={key} className="ml-2 w-8 h-6 align-middle" type="color" value={selected[key] ?? '#3687b4'} onChange={e => update({ [key]: e.target.value })}/></label>)}</div>
        </>}
        <p className="text-[11px] text-slate-500">{config.rootConnector ? '分支末端可追加或复制模块；删除会移除此模块及其后代，其他分支保持连接。' : '增删与排序自动重接单链；q 随模块身份保留。'}σ 是安装方位，不等于相邻关节轴夹角。</p>
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

        {(['rootPos', 'rootEuler'] as const).map(key => <div key={key} className="text-xs text-slate-400">{key === 'rootPos' ? '基座位置 X / Y / Z（m）' : '基座姿态 Roll / Pitch / Yaw（°）'}<div className="grid grid-cols-3 gap-2 mt-1">{config[key].map((v, i) => <input key={i} className={field} aria-label={`${key} ${'XYZ'[i]}`} type="number" step={key === 'rootPos' ? .01 : 5} value={v} onChange={e => { const vec = [...config[key]] as [number, number, number]; vec[i] = e.target.valueAsNumber; apply({ ...config, [key]: vec }); }}/>)}</div></div>)}
        <label className="block text-xs text-slate-400">研究假设<textarea rows={3} className={`${field} mt-1`} value={config.hypothesis ?? config.description} onChange={e => apply({ ...config, hypothesis: e.target.value })}/></label>
        <div className="grid grid-cols-2 gap-2 pt-3 border-t border-slate-800">
          <button className={button} onClick={onOpenXmlModal}><Code2 className="inline w-3 h-3"/> MuJoCo XML</button>
          <button className={button} onClick={onOpenMeshModal}><UploadCloud className="inline w-3 h-3"/> CAD 网格</button>
        </div>
        <p className="text-[11px] text-slate-500">几何：模块轴向节距 106 mm，斜轴铰链中心 z = 53 mm；坐标与尺寸按原模型解释。</p>
      </>}
    </div>
    <input ref={fileInput} className="hidden" type="file" accept=".json,application/json" onChange={async e => { const file = e.target.files?.[0]; e.target.value = ''; if (!file) return; try { if (file.size > 200_000) throw new Error('构型文件不得超过 200 KB'); const next = parseConfiguration(await file.text()); apply(next, false); setName(next.name); setSelectedId(next.modules[0]?.id); setTab('modules'); setMessage(`已导入“${next.name}”`); } catch (error) { setMessage(error instanceof Error ? error.message : String(error)); } }}/>
  </section>;
};
