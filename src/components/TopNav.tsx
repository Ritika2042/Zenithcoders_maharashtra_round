import React from 'react';
import { Plus, Download, Sparkles } from 'lucide-react';

interface TopNavProps {
  activeView: 'relearn' | 'studio' | 'explorer' | 'codex' | 'auditor';
  onNavigate: (view: 'relearn' | 'studio' | 'explorer' | 'codex' | 'auditor') => void;
  onAddNewRecord: () => void;
  onOpenExportImport: () => void;
  recordCount: number;
}

export const TopNav: React.FC<TopNavProps> = ({
  activeView,
  onNavigate,
  onAddNewRecord,
  onOpenExportImport,
  recordCount,
}) => {
  return (
    <header className="flex items-center justify-between px-6 py-3.5 bg-white border-b border-slate-200 sticky top-0 z-40">
      {/* Zone 1: Single text element wordmark */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => onNavigate('relearn')}
          className="text-base font-bold tracking-tight text-slate-900 hover:text-blue-600 transition-colors flex items-center gap-2"
        >
          <span>ReLearn Workbench</span>
          <span className="text-[11px] font-mono font-medium text-blue-700 bg-blue-50 px-1.5 py-0.5 rounded border border-blue-200">
            Person 1 + 2
          </span>
        </button>
      </div>

      {/* Zone 2: Navigation links */}
      <nav className="hidden md:flex items-center gap-5 text-xs font-semibold text-slate-600">
        <button
          type="button"
          onClick={() => onNavigate('relearn')}
          className={`py-1 transition-colors relative flex items-center gap-1.5 ${
            activeView === 'relearn'
              ? 'text-blue-600 after:absolute after:bottom-0 after:left-0 after:right-0 after:h-0.5 after:bg-blue-600 font-bold'
              : 'hover:text-slate-900 text-slate-800'
          }`}
        >
          <Sparkles className="w-3.5 h-3.5 text-blue-500" />
          <span>ReLearn Adaptive Learning</span>
        </button>

        <button
          type="button"
          onClick={() => onNavigate('studio')}
          className={`py-1 transition-colors relative ${
            activeView === 'studio'
              ? 'text-blue-600 after:absolute after:bottom-0 after:left-0 after:right-0 after:h-0.5 after:bg-blue-600'
              : 'hover:text-slate-900'
          }`}
        >
          Annotator Studio
        </button>

        <button
          type="button"
          onClick={() => onNavigate('explorer')}
          className={`py-1 transition-colors relative flex items-center gap-1.5 ${
            activeView === 'explorer'
              ? 'text-blue-600 after:absolute after:bottom-0 after:left-0 after:right-0 after:h-0.5 after:bg-blue-600'
              : 'hover:text-slate-900'
          }`}
        >
          <span>Dataset Explorer</span>
          <span className="text-[10px] font-mono bg-slate-100 px-1.5 py-0.2 rounded-full text-slate-600">
            {recordCount}
          </span>
        </button>

        <button
          type="button"
          onClick={() => onNavigate('codex')}
          className={`py-1 transition-colors relative ${
            activeView === 'codex'
              ? 'text-blue-600 after:absolute after:bottom-0 after:left-0 after:right-0 after:h-0.5 after:bg-blue-600'
              : 'hover:text-slate-900'
          }`}
        >
          Taxonomy & Codex
        </button>

        <button
          type="button"
          onClick={() => onNavigate('auditor')}
          className={`py-1 transition-colors relative ${
            activeView === 'auditor'
              ? 'text-blue-600 after:absolute after:bottom-0 after:left-0 after:right-0 after:h-0.5 after:bg-blue-600'
              : 'hover:text-slate-900'
          }`}
        >
          Integrity Auditor
        </button>
      </nav>

      {/* Zone 3: 1-2 primary actions */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onOpenExportImport}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors whitespace-nowrap"
        >
          <Download className="w-3.5 h-3.5" />
          <span>Export / Import</span>
        </button>

        <button
          type="button"
          onClick={onAddNewRecord}
          className="flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors whitespace-nowrap shadow-xs"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>New Sample</span>
        </button>
      </div>
    </header>
  );
};
