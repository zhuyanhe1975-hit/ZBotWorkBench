import React from 'react';
import { Euler, Quaternion } from 'three';
import { EndEffectorTarget } from '../utils/inverseKinematics';

export type ArmControlMode = 'translate' | 'rotate';
export interface ArmSolveStatus { converged: boolean; positionError: number; orientationError: number }
interface Props {
  target: EndEffectorTarget | null;
  mode: ArmControlMode;
  positionOnly: boolean;
  onPositionOnlyChange: (value: boolean) => void;
  status: ArmSolveStatus | null;
  disabled: boolean;
  onEnable: () => void;
  onDisable: () => void;
  onModeChange: (mode: ArmControlMode) => void;
  onTargetChange: (target: EndEffectorTarget) => void;
}
export function ArmControlPanel({ target, mode, positionOnly, onPositionOnlyChange, status, disabled, onEnable, onDisable, onModeChange, onTargetChange }: Props) {
  const euler = target ? new Euler().setFromQuaternion(new Quaternion(...target.quaternion), 'XYZ').toArray().slice(0, 3).map(v => Number(v) * 180 / Math.PI) : [];
  return <fieldset disabled={disabled} className="p-3 bg-cyan-950/30 border-b border-cyan-900 text-xs space-y-2 disabled:opacity-50">
    <div className="flex justify-between items-center gap-2"><h2 className="font-semibold text-cyan-200">串联构型 · 末端控制球</h2><button onClick={target ? onDisable : onEnable} className="px-2 py-1.5 bg-cyan-700 hover:bg-cyan-600 rounded" aria-pressed={!!target}>{target ? '关闭末端控制' : '开启末端控制'}</button></div>
    <label className="flex items-center gap-2 text-slate-300"><input type="checkbox" checked={positionOnly} onChange={e => onPositionOnlyChange(e.target.checked)} />仅控制位置（允许姿态变化）</label>
    {target && <>
      <div className="flex gap-2"><button aria-pressed={mode === 'translate'} onClick={() => onModeChange('translate')} className={`px-3 py-1.5 rounded ${mode === 'translate' ? 'bg-cyan-600' : 'bg-slate-800'}`}>平移 XYZ</button><button disabled={positionOnly} aria-pressed={mode === 'rotate'} onClick={() => onModeChange('rotate')} className={`px-3 py-1.5 rounded ${mode === 'rotate' ? 'bg-cyan-600' : 'bg-slate-800'} disabled:opacity-40`}>旋转 XYZ</button><button onClick={onEnable} className="ml-auto text-cyan-300">贴合当前末端</button></div>
      <p className="text-slate-400 text-[11px]">拖动球心可沿视平面平移；拖动箭头或旋转环调整目标。{positionOnly ? '当前只约束位置，姿态随关节自然变化。' : '平移保持姿态，旋转保持位置。'}</p>
      <div className="grid grid-cols-3 gap-2">{target.position.map((v, i) => <label key={i} className="text-slate-400">{'XYZ'[i]} / m<input aria-label={`末端目标 ${'XYZ'[i]}`} type="number" step="0.005" value={Number(v.toFixed(4))} onChange={e => { if (!Number.isFinite(e.target.valueAsNumber)) return; const position = [...target.position] as [number,number,number]; position[i] = e.target.valueAsNumber; onTargetChange({ ...target, position }); }} className="w-full bg-slate-950 border border-slate-700 rounded p-1 mt-1 text-slate-100" /></label>)}</div>
      <div className="grid grid-cols-3 gap-2">{!positionOnly && euler.map((v, i) => <label key={i} className="text-slate-400">{['Roll','Pitch','Yaw'][i]} / °<input aria-label={`末端目标 ${['Roll','Pitch','Yaw'][i]}`} type="number" step="2" value={Number(v.toFixed(2))} onChange={e => { if (!Number.isFinite(e.target.valueAsNumber)) return; const next = [...euler]; next[i] = e.target.valueAsNumber; const quaternion = new Quaternion().setFromEuler(new Euler(next[0] * Math.PI / 180, next[1] * Math.PI / 180, next[2] * Math.PI / 180, 'XYZ')).toArray() as [number,number,number,number]; onTargetChange({ ...target, quaternion }); }} className="w-full bg-slate-950 border border-slate-700 rounded p-1 mt-1 text-slate-100" /></label>)}</div>
      <p role="status" className={status && !status.converged ? 'text-amber-300' : 'text-emerald-300'}>{status && !status.converged ? '目标未收敛，保持上一关节目标。' : '目标已求解。暂停时预览，运行时由伺服跟踪。'}{status && <span className="block text-[11px]">{positionOnly ? '位置残差 / 未约束姿态差 ' : '求解残差 '}{(status.positionError * 1000).toFixed(2)} mm / {(status.orientationError * 180 / Math.PI).toFixed(2)}°</span>}</p>
    </>}
  </fieldset>;
}
