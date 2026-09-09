import React from 'react';
import { GaitConfig, GaitType, SimMetrics } from '../types/zbot';
import {
  Play,
  Pause,
  RotateCcw,
  StepForward,
  Gauge,
  Sliders,
  Compass,
  Activity,
  Waves,
  Zap,
  FastForward,
} from 'lucide-react';

interface GaitControlPanelProps {
  isRunning: boolean;
  onTogglePlay: () => void;
  onStep: () => void;
  onReset: () => void;
  gait: GaitConfig;
  onGaitChange: (newGait: GaitConfig) => void;
  speedMultiplier: number;
  onSpeedMultiplierChange: (speed: number) => void;
  friction: number;
  onFrictionChange: (val: number) => void;
  kp: number;
  onKpChange: (val: number) => void;
  metrics: SimMetrics;
  jointCount: number;
  jointRanges?: [number, number][];
  physicsDisabled?: boolean;
}

const GAIT_OPTIONS: { type: GaitType; label: string; desc: string }[] = [
  { type: 'serpentine', label: '蛇形行进波', desc: '围绕初始姿态传递正弦波；推进效果需接触实验验证' },
  { type: 'inchworm', label: '尺蠖蠕动步态', desc: '周期性弓背收缩与展平爬行' },
  { type: 'rolling', label: '反向行波', desc: '反向相位传播；当前拓扑为开放单链' },
  { type: 'sidewind', label: '侧伏蜿蜒', desc: '双轴正交相位波侧向滑行' },
  { type: 'trot', label: '分组振荡', desc: '关节分组反相运动；不代表完整四足步态' },
  { type: 'manual', label: '手动关节调试', desc: '独立调节每个铰链关节的角度' },
];

export const GaitControlPanel: React.FC<GaitControlPanelProps> = ({
  isRunning,
  onTogglePlay,
  onStep,
  onReset,
  gait,
  onGaitChange,
  speedMultiplier,
  onSpeedMultiplierChange,
  friction,
  onFrictionChange,
  kp,
  onKpChange,
  metrics,
  jointCount,
  jointRanges,
  physicsDisabled,
}) => {
  return (
    <div className="bg-slate-900 border-t border-slate-800 p-3 text-slate-200">
      {/* Top Primary Bar: Run / Pause / Metrics HUD */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
        {/* Playback Controls */}
        <div className="flex flex-wrap items-center gap-2 [&_button]:whitespace-nowrap">
          <button
            onClick={onTogglePlay}
            className={`px-4 py-2 rounded-lg font-medium text-xs flex items-center gap-2 shadow-lg transition active:scale-95 ${
              isRunning
                ? 'bg-amber-600 hover:bg-amber-500 text-white'
                : 'bg-emerald-600 hover:bg-emerald-500 text-white'
            }`}
          >
            {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-current" />}
            <span>{isRunning ? '暂停仿真' : '启动动力学'}</span>
          </button>

          <button
            onClick={onStep}
            disabled={isRunning}
            className="px-3 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:pointer-events-none text-slate-200 rounded-lg text-xs font-medium flex items-center gap-1.5 border border-slate-700 transition"
            title="单步步进仿真"
          >
            <StepForward className="w-3.5 h-3.5" />
            单步
          </button>

          <button
            onClick={onReset}
            className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-medium flex items-center gap-1.5 border border-slate-700 transition"
            title="重置机器人姿态与位置"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            重置
          </button>

          {/* Speed Multiplier */}
          <div className="flex items-center bg-slate-950 rounded-lg border border-slate-800 p-0.5 ml-2 text-xs">
            {[0.5, 1.0, 2.0].map((s) => (
              <button
                key={s}
                onClick={() => onSpeedMultiplierChange(s)}
                className={`px-2 py-1 rounded font-mono transition ${
                  speedMultiplier === s ? 'bg-blue-600 text-white font-medium' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        {/* Live Metrics Quick Badges */}
        <div className="flex flex-wrap items-center gap-2 text-xs [&>div]:whitespace-nowrap">
          <div className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800/80 flex items-center gap-2">
            <span className="text-slate-400 text-[11px]">仿真时间:</span>
            <span className="font-mono text-slate-200 font-medium">
              {metrics.time.toFixed(2)} s
            </span>
          </div>

          <div className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800/80 flex items-center gap-2">
            <span className="text-slate-400 text-[11px]">基座速度:</span>
            <span className="font-mono text-emerald-400 font-medium">
              {metrics.speed.toFixed(3)} m/s
            </span>
          </div>

          <div className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800/80 flex items-center gap-2">
            <span className="text-slate-400 text-[11px]">累计路程:</span>
            <span className="font-mono text-sky-400 font-medium">
              {metrics.totalDistance.toFixed(3)} m
            </span>
          </div>

          <div className="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800/80 flex items-center gap-2">
            <span className="text-slate-400 text-[11px]">主动关节:</span>
            <span className="font-mono text-purple-400 font-medium">
              {jointCount}
            </span>
          </div>
        </div>
      </div>

      {/* Gait Type Selector Pills */}
      <div className="pt-3 pb-2 flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold text-slate-400 mr-1 flex items-center gap-1">
          <Waves className="w-3.5 h-3.5 text-blue-400" />
          步态控制器:
        </span>
        {GAIT_OPTIONS.map((g) => (
          <button
            key={g.type}
            onClick={() => onGaitChange({ ...gait, type: g.type })}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition ${
              gait.type === g.type
                ? 'bg-blue-600/30 border-blue-500 text-blue-300 shadow-sm'
                : 'bg-slate-800/50 border-slate-700/60 text-slate-400 hover:text-slate-200 hover:bg-slate-800'
            }`}
            title={g.desc}
          >
            {g.label}
          </button>
        ))}
      </div>

      {/* Gait Parameter Sliders or Manual Sliders */}
      {gait.type !== 'manual' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 pt-2">
          {/* Frequency */}
          <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800">
            <div className="flex justify-between text-[11px] mb-1">
              <span className="text-slate-400">振荡频率 (f)</span>
              <span className="font-mono text-blue-400">{gait.frequency.toFixed(2)} Hz</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="2.5"
              step="0.05"
              value={gait.frequency}
              onChange={(e) => onGaitChange({ ...gait, frequency: parseFloat(e.target.value) })}
              className="w-full accent-blue-500 h-1 bg-slate-700 rounded cursor-pointer"
            />
          </div>

          {/* Amplitude */}
          <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800">
            <div className="flex justify-between text-[11px] mb-1">
              <span className="text-slate-400">关节振幅 (A)</span>
              <span className="font-mono text-emerald-400">{gait.amplitude.toFixed(0)}°</span>
            </div>
            <input
              type="range"
              min="0"
              max="80"
              step="1"
              value={gait.amplitude}
              onChange={(e) => onGaitChange({ ...gait, amplitude: parseFloat(e.target.value) })}
              className="w-full accent-emerald-500 h-1 bg-slate-700 rounded cursor-pointer"
            />
          </div>

          {/* Phase Lag */}
          <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800">
            <div className="flex justify-between text-[11px] mb-1">
              <span className="text-slate-400">相邻模块相位差 (Δφ)</span>
              <span className="font-mono text-amber-400">{gait.phaseLag.toFixed(0)}°</span>
            </div>
            <input
              type="range"
              min="0"
              max="180"
              step="5"
              value={gait.phaseLag}
              onChange={(e) => onGaitChange({ ...gait, phaseLag: parseFloat(e.target.value) })}
              className="w-full accent-amber-500 h-1 bg-slate-700 rounded cursor-pointer"
            />
          </div>

          {/* Steering bias */}
          <div className="bg-slate-950/60 p-2 rounded-lg border border-slate-800">
            <div className="flex justify-between text-[11px] mb-1">
              <span className="text-slate-400">转弯转向偏置 (Bias)</span>
              <span className="font-mono text-sky-400">
                {gait.steering > 0 ? `+${gait.steering.toFixed(0)}°` : gait.steering < 0 ? `${gait.steering.toFixed(0)}°` : '0°'}
              </span>
            </div>
            <input
              type="range"
              min="-25"
              max="25"
              step="1"
              value={gait.steering}
              onChange={(e) => onGaitChange({ ...gait, steering: parseFloat(e.target.value) })}
              className="w-full accent-sky-500 h-1 bg-slate-700 rounded cursor-pointer"
            />
          </div>
        </div>
      ) : (
        /* Manual Sliders for Joints */
        <div className="pt-2">
          <div className="text-[11px] text-slate-400 mb-2">关节目标 q (°) · 初始暂停时直接预览，运行后由伺服跟踪：</div>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2 max-h-28 overflow-y-auto">
            {Array.from({ length: jointCount }).map((_, i) => {
              const jKey = `joint_${i}`;
              const val = gait.manualAngles[jKey] ?? 0;
              return (
                <div key={jKey} className="bg-slate-950 p-2 rounded border border-slate-800">
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-slate-400">J{i}</span>
                    <span className="font-mono text-blue-400">{Number(val.toFixed(1))}°</span>
                  </div>
                  <input
                    type="range"
                    min={jointRanges?.[i]?.[0] ?? -180}
                    max={jointRanges?.[i]?.[1] ?? 180}
                    aria-label={`关节 J${i} 目标角`}
                    step="1"
                    value={val}
                    onChange={(e) => {
                      const nVal = parseFloat(e.target.value);
                      onGaitChange({
                        ...gait,
                        manualAngles: { ...gait.manualAngles, [jKey]: nVal },
                      });
                    }}
                    className="w-full accent-blue-500 h-1 bg-slate-700 rounded cursor-pointer"
                  />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Physics Environment Toggles: Floor Friction & Servo Gain */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 text-[11px]">
        <div className="flex items-center gap-2">
          <span className="text-slate-400 whitespace-nowrap">地面摩擦因数 (μ):</span>
          <input
            type="range"
            min="0.1"
            max="2.5"
            step="0.1"
            value={friction}
            disabled={physicsDisabled}
            aria-label="地面摩擦因数"
            onChange={(e) => onFrictionChange(parseFloat(e.target.value))}
            className="w-32 accent-slate-400 h-1 bg-slate-700 rounded cursor-pointer"
          />
          <span className="font-mono text-slate-300 text-xs">{friction.toFixed(1)}</span>
          <span className="text-slate-500 text-[10px]">
            ({friction < 0.4 ? '光滑冰面' : friction > 1.4 ? '高粘附胶面' : '标准实验台面'})
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-slate-400 whitespace-nowrap">伺服刚度增益 (Kp):</span>
          <input
            type="range"
            min="20"
            max="200"
            step="5"
            value={kp}
            disabled={physicsDisabled}
            aria-label="伺服刚度增益"
            onChange={(e) => onKpChange(parseFloat(e.target.value))}
            className="w-32 accent-slate-400 h-1 bg-slate-700 rounded cursor-pointer"
          />
          <span className="font-mono text-slate-300 text-xs">{kp}</span>
          <span className="text-slate-500 text-[10px]">(PD位置闭环抗扰能力)</span>
        </div>
      </div>
    </div>
  );
};
