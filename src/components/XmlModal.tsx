import React, { useState, useEffect } from 'react';
import { X, Copy, Check, Download, Play, AlertCircle, RefreshCw } from 'lucide-react';

interface XmlModalProps {
  isOpen: boolean;
  onClose: () => void;
  xmlContent: string;
  onApplyCustomXml: (xml: string) => void;
  onResetToAutoXml: () => void;
}

export const XmlModal: React.FC<XmlModalProps> = ({
  isOpen,
  onClose,
  xmlContent,
  onApplyCustomXml,
  onResetToAutoXml,
}) => {
  const [currentText, setCurrentText] = useState(xmlContent);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setCurrentText(xmlContent);
    setError(null);
  }, [xmlContent]);

  if (!isOpen) return null;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(currentText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // fallback
    }
  };

  const handleDownload = () => {
    const blob = new Blob([currentText], { type: 'text/xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'zbot_mujoco_model.xml';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleApply = () => {
    try {
      setError(null);
      onApplyCustomXml(currentText);
      onClose();
    } catch (err: any) {
      setError(err?.message || 'MuJoCo XML 解析失败，请检查标签与语法');
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-4xl max-h-[88vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-950">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm text-slate-100">
              MuJoCo MJCF XML 描述文件查看与编辑
            </span>
            <span className="text-[11px] px-2 py-0.5 rounded bg-blue-950 text-blue-300 border border-blue-800 font-mono">
              WASM Compatible
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 flex items-center gap-1.5 border border-slate-700 transition"
              title="复制代码"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? '已复制' : '复制XML'}</span>
            </button>

            <button
              onClick={handleDownload}
              className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 flex items-center gap-1.5 border border-slate-700 transition"
              title="下载为.xml文件"
            >
              <Download className="w-3.5 h-3.5" />
              <span>下载XML</span>
            </button>

            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-100 transition"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Info banner */}
        <div className="px-4 py-2 bg-slate-800/40 text-xs text-slate-400 border-b border-slate-800 flex items-center justify-between">
          <span>
            此XML由当前构型拓扑自动生成。您可直接在此修改XML或粘贴您已有的Zbot XML，然后点击“编译并装载”。
          </span>
          <button
            onClick={onResetToAutoXml}
            className="text-blue-400 hover:underline flex items-center gap-1 text-[11px] shrink-0 ml-2"
          >
            <RefreshCw className="w-3 h-3" />
            重置为当前拓扑生成版
          </button>
        </div>

        {/* Error notice if any */}
        {error && (
          <div className="px-4 py-2 bg-rose-950/60 border-b border-rose-800 text-rose-300 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Code Editor Area */}
        <div className="flex-1 p-3 bg-slate-950 overflow-hidden flex flex-col">
          <textarea
            value={currentText}
            onChange={(e) => setCurrentText(e.target.value)}
            className="w-full flex-1 bg-slate-950 text-slate-200 font-mono text-xs leading-relaxed p-3 border border-slate-800 rounded-lg focus:outline-none focus:border-blue-500 resize-none selection:bg-blue-900"
            spellCheck={false}
          />
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-slate-800 bg-slate-950">
          <span className="text-[11px] text-slate-500">
            支持标准 MuJoCo 3.x 标签体系：&lt;compiler&gt;, &lt;default&gt;, &lt;worldbody&gt;, &lt;actuator&gt;
          </span>

          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-4 py-1.5 rounded-lg text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition"
            >
              取消
            </button>
            <button
              onClick={handleApply}
              className="px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium flex items-center gap-1.5 shadow transition active:scale-95"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>编译并装载到仿真器</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
