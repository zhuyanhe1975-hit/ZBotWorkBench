import React, { useState } from 'react';
import {
  X,
  Upload,
  Check,
  Download,
  RotateCcw,
  Box,
  FileText,
  CheckCircle2,
  ClipboardPaste,
  Layers,
  RefreshCw
} from 'lucide-react';
import { generateZbotOBJ } from '../utils/zbotMeshGenerator';
import { meshManager } from '../utils/meshManager';

interface MeshUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUpdateMeshes: (customA?: string, customB?: string) => void;
  hasCustomMeshA: boolean;
  hasCustomMeshB: boolean;
}

export const MeshUploadModal: React.FC<MeshUploadModalProps> = ({
  isOpen,
  onClose,
  onUpdateMeshes,
  hasCustomMeshA,
  hasCustomMeshB,
}) => {
  const [activeMode, setActiveMode] = useState<'upload' | 'paste'>('upload');
  const [meshAStatus, setMeshAStatus] = useState<string | null>(null);
  const [meshBStatus, setMeshBStatus] = useState<string | null>(null);

  const [pasteTextA, setPasteTextA] = useState('');
  const [pasteTextB, setPasteTextB] = useState('');

  if (!isOpen) return null;

  const applyMesh = (part: 'ma' | 'mb', text: string, label: string) => {
    const status = part === 'ma' ? setMeshAStatus : setMeshBStatus;
    try {
      onUpdateMeshes(part === 'ma' ? text : undefined, part === 'mb' ? text : undefined);
      status(label);
    } catch (error) { status(`错误：${error instanceof Error ? error.message : String(error)}`); }
  };
  const handleFileUpload = async (part: 'ma' | 'mb', file: File) => {
    const status = part === 'ma' ? setMeshAStatus : setMeshBStatus;
    try {
      if (file.size > 20 * 1024 * 1024) throw new Error('OBJ 文件不得超过 20 MB');
      applyMesh(part, await file.text(), `已载入 ${file.name}`);
    } catch (error) { status(`错误：${String(error)}`); }
  };
  const handleApplyPasted = (part: 'ma' | 'mb') => {
    applyMesh(part, part === 'ma' ? pasteTextA : pasteTextB, `已应用 ${part}.obj`);
  };

  const handleDownloadActiveObj = (part: 'ma' | 'mb') => {
    const objText = part === 'ma' ? meshManager.getMeshAText() : meshManager.getMeshBText();
    const blob = new Blob([objText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${part}.obj`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleResetToStandard = (part: 'ma' | 'mb') => {
    meshManager.resetToDefault(part);
    onUpdateMeshes();
    (part === 'ma' ? setMeshAStatus : setMeshBStatus)('已恢复内置参数化网格');
  };

  const handleReloadPublicAssets = async () => {
    try {
      const res = await meshManager.init();
      onUpdateMeshes();
      setMeshAStatus(meshManager.hasCadMeshA() ? '已重新载入 CAD ma.obj' : 'CAD 不可用，已使用参数化网格');
      setMeshBStatus(meshManager.hasCadMeshB() ? '已重新载入 CAD mb.obj' : 'CAD 不可用，已使用参数化网格');
    } catch (e: any) {
      setMeshAStatus(`重载失败: ${e?.message || e}`);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-2xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-950">
          <div className="flex items-center gap-2">
            <Box className="w-4 h-4 text-blue-400" />
            <span className="font-semibold text-sm text-slate-100">
              Zbot 模块 3D CAD 网格文件资产管理 (ma.obj / mb.obj)
            </span>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-100 transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Tab switch */}
        <div className="flex items-center px-4 pt-2 border-b border-slate-800 bg-slate-950/60 gap-4">
          <button
            onClick={() => setActiveMode('upload')}
            className={`py-2 text-xs font-medium border-b-2 flex items-center gap-1.5 transition ${
              activeMode === 'upload'
                ? 'border-blue-500 text-blue-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Upload className="w-3.5 h-3.5" />
            选择 OBJ 文件
          </button>
          <button
            onClick={() => setActiveMode('paste')}
            className={`py-2 text-xs font-medium border-b-2 flex items-center gap-1.5 transition ${
              activeMode === 'paste'
                ? 'border-blue-500 text-blue-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <ClipboardPaste className="w-3.5 h-3.5" />
            直接粘贴 OBJ 文本代码
          </button>
        </div>

        {/* Content Area */}
        <div className="p-4 space-y-4 text-xs overflow-y-auto flex-1">
          <div className="bg-blue-950/40 border border-blue-900/60 p-2.5 rounded-lg text-blue-200 text-[11px] leading-relaxed">
            <strong>提示：</strong>Zbot 机器人单模块由<strong>半模块A (ma.obj)</strong> 与 <strong>半模块B (mb.obj)</strong> 沿 45° 倾斜对角截面铰链配合组成。MuJoCo 会将网格写入内存虚拟文件系统 (VFS)，用于凸包近似接触计算与外观显示。
          </div>

          {activeMode === 'upload' ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Part A Card */}
              <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 flex flex-col justify-between space-y-3">
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <div className="font-semibold text-slate-200 flex items-center gap-1.5">
                      <div className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                      <span>半模块 A (ma.obj)</span>
                    </div>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-950 text-blue-300 border border-blue-800">
                      {hasCustomMeshA ? '自定义CAD' : '标准参数化'}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    基座半段（底面 z=0 至 45° 倾斜截面 z=0.053m）。
                  </p>

                  {meshAStatus && (
                    <div className="mt-2 text-[11px] text-emerald-400 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3 shrink-0" />
                      <span>{meshAStatus}</span>
                    </div>
                  )}
                </div>

                <div className="space-y-1.5 pt-2 border-t border-slate-800/80">
                  <label className="w-full py-2 px-3 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 border border-slate-700 cursor-pointer transition">
                    <Upload className="w-3.5 h-3.5 text-blue-400" />
                    <span>选择 ma.obj</span>
                    <input
                      type="file"
                      accept=".obj"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) handleFileUpload('ma', file);
                      }}
                    />
                  </label>

                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => handleDownloadActiveObj('ma')}
                      className="flex-1 py-1 px-2 bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 rounded border border-slate-800 text-[11px] flex items-center justify-center gap-1 transition"
                    >
                      <Download className="w-3 h-3" />
                      下载当前OBJ
                    </button>
                    {hasCustomMeshA && (
                      <button
                        onClick={() => handleResetToStandard('ma')}
                        className="py-1 px-2 bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 rounded border border-slate-800 text-[11px] flex items-center gap-1 transition"
                        title="恢复内置标准参数化模型"
                      >
                        <RotateCcw className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Part B Card */}
              <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 flex flex-col justify-between space-y-3">
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <div className="font-semibold text-slate-200 flex items-center gap-1.5">
                      <div className="w-2.5 h-2.5 rounded-full bg-cyan-500" />
                      <span>半模块 B (mb.obj)</span>
                    </div>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                      {hasCustomMeshB ? '自定义CAD' : '标准参数化'}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    输出动力半段（45° 截面至顶部对接端面 z=0.106m）。
                  </p>

                  {meshBStatus && (
                    <div className="mt-2 text-[11px] text-emerald-400 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3 shrink-0" />
                      <span>{meshBStatus}</span>
                    </div>
                  )}
                </div>

                <div className="space-y-1.5 pt-2 border-t border-slate-800/80">
                  <label className="w-full py-2 px-3 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 border border-slate-700 cursor-pointer transition">
                    <Upload className="w-3.5 h-3.5 text-cyan-400" />
                    <span>选择 mb.obj</span>
                    <input
                      type="file"
                      accept=".obj"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) handleFileUpload('mb', file);
                      }}
                    />
                  </label>

                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => handleDownloadActiveObj('mb')}
                      className="flex-1 py-1 px-2 bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 rounded border border-slate-800 text-[11px] flex items-center justify-center gap-1 transition"
                    >
                      <Download className="w-3 h-3" />
                      下载当前OBJ
                    </button>
                    {hasCustomMeshB && (
                      <button
                        onClick={() => handleResetToStandard('mb')}
                        className="py-1 px-2 bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 rounded border border-slate-800 text-[11px] flex items-center gap-1 transition"
                        title="恢复内置标准参数化模型"
                      >
                        <RotateCcw className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Paste Mode */
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-slate-200">粘贴 ma.obj 内容:</span>
                  <button
                    onClick={() => handleApplyPasted('ma')}
                    className="px-2.5 py-1 bg-blue-600 hover:bg-blue-500 text-white rounded text-[11px] transition"
                  >
                    更新 ma.obj
                  </button>
                </div>
                <textarea
                  rows={8}
                  placeholder="v 0.035 0.0 0.0&#10;vn 0 -0.707 0.707&#10;f 1 2 3..."
                  value={pasteTextA}
                  onChange={(e) => setPasteTextA(e.target.value)}
                  className="w-full p-2 bg-slate-950 font-mono text-[11px] border border-slate-800 rounded-lg text-slate-200 focus:outline-none focus:border-blue-500 resize-none"
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-slate-200">粘贴 mb.obj 内容:</span>
                  <button
                    onClick={() => handleApplyPasted('mb')}
                    className="px-2.5 py-1 bg-cyan-600 hover:bg-cyan-500 text-white rounded text-[11px] transition"
                  >
                    更新 mb.obj
                  </button>
                </div>
                <textarea
                  rows={8}
                  placeholder="v 0.035 0.0 0.053&#10;vn 0 0.707 -0.707&#10;f 1 2 3..."
                  value={pasteTextB}
                  onChange={(e) => setPasteTextB(e.target.value)}
                  className="w-full p-2 bg-slate-950 font-mono text-[11px] border border-slate-800 rounded-lg text-slate-200 focus:outline-none focus:border-cyan-500 resize-none"
                />
              </div>
            </div>
          )}
        </div>

        {activeMode === 'paste' && <div role="status" className="px-4 pb-3 text-xs text-amber-200">{meshAStatus && <p>A: {meshAStatus}</p>}{meshBStatus && <p>B: {meshBStatus}</p>}</div>}
        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-slate-800 bg-slate-950">
          <button
            onClick={handleReloadPublicAssets}
            className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs flex items-center gap-1.5 transition"
            title="重新从 public/assets/ma.obj 和 mb.obj 读取并加载"
          >
            <RefreshCw className="w-3.5 h-3.5 text-emerald-400" />
            <span>重新载入 public/assets CAD 资产</span>
          </button>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition"
          >
            完成并关闭
          </button>
        </div>
      </div>
    </div>
  );
};
