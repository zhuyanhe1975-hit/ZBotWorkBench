import React from 'react';
import { SimMetrics } from '../types/zbot';
import { Activity, Gauge, TrendingUp, Cpu } from 'lucide-react';

interface TelemetryDrawerProps {
  metrics: SimMetrics;
  isOpen: boolean;
  onToggle: () => void;
}

export const TelemetryDrawer: React.FC<TelemetryDrawerProps> = ({
  metrics,
  isOpen,
  onToggle,
}) => {
  const jointEntries = Object.entries(metrics.jointAngles || {});

  return (
    <div className="border-t border-slate-800 bg-slate-950/90 backdrop-blur-md text-slate-200">
      <div
        onClick={onToggle}
        className="px-4 py-2 flex items-center justify-between cursor-pointer hover:bg-slate-900/60 transition text-xs"
      >
        <div className="flex flex-wrap items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-blue-400" />
          <span className="font-semibold text-slate-300">动力学状态与实时关节遥测数据</span>
          <span className="text-[11px] text-slate-500 font-mono">
            t = {metrics.time.toFixed(2)}s | v = {metrics.speed.toFixed(3)} m/s
          </span>
        </div>
        <span className="text-[11px] text-blue-400 hover:underline shrink-0 ml-2">
          {isOpen ? '收起面板 ▲' : '展开详情 ▼'}
        </span>
      </div>

      {isOpen && (
        <div className="p-4 pt-2 border-t border-slate-800/60 grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          {/* Spatial Kinematics */}
          <div className="bg-slate-900/70 p-3 rounded-lg border border-slate-800 space-y-2">
            <div className="font-medium text-slate-300 flex items-center gap-1.5">
              <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />
              <span>基座空间位姿与运动学</span>
            </div>
            <div className="space-y-1 text-[11px] text-slate-400">
              <div className="flex justify-between">
                <span>位置坐标 [X, Y, Z]</span>
                <span className="font-mono text-slate-200">
                  [{metrics.rootPos[0].toFixed(2)}, {metrics.rootPos[1].toFixed(2)}, {metrics.rootPos[2].toFixed(2)}] m
                </span>
              </div>
              <div className="flex justify-between">
                <span>速度矢量 [Vx, Vy, Vz]</span>
                <span className="font-mono text-slate-200">
                  [{metrics.rootVelocity[0].toFixed(2)}, {metrics.rootVelocity[1].toFixed(2)}, {metrics.rootVelocity[2].toFixed(2)}] m/s
                </span>
              </div>
              <div className="flex justify-between">
                <span>累计三维路程</span>
                <span className="font-mono text-emerald-400 font-medium">
                  {metrics.totalDistance.toFixed(3)} m
                </span>
              </div>
            </div>
          </div>

          {/* Actuator State Bar Visualizer */}
          <div className="bg-slate-900/70 p-3 rounded-lg border border-slate-800 md:col-span-2 space-y-2">
            <div className="font-medium text-slate-300 flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-purple-400" />
                <span>各模块倾斜铰链关节实时角度 (qpos / degree)</span>
              </div>
              <span className="text-[10px] text-slate-500 font-mono">角度 ° / 驱动力矩 N·m</span>
            </div>

            {jointEntries.length === 0 ? (
              <div className="text-slate-500 text-center py-3 text-[11px]">暂无活动的关节数据</div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">
                {jointEntries.map(([jName, rawAngle]) => {
                  const angle = Number(rawAngle) || 0;
                  const pct = Math.max(0, Math.min(100, ((angle + 180) / 360) * 100));
                  return (
                    <div key={jName} className="bg-slate-950 p-2 rounded border border-slate-800/80">
                      <div className="flex justify-between text-[10px] mb-1">
                        <span className="text-slate-400">{jName.replace('joint_', 'J')}</span>
                        <span className="font-mono text-purple-300">{angle.toFixed(1)}°</span>
                      </div>
                      <div className="text-[10px] text-slate-400 mb-1">τ {(metrics.jointTorques[jName] ?? 0).toFixed(3)} N·m</div>
                      <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-purple-500 h-full rounded-full transition-all duration-75"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
