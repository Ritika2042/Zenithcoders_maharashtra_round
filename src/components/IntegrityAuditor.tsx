import React from 'react';
import { SampleRecord } from '../types/dataset';
import { auditDataset } from '../utils/validator';
import { CheckCircle2, AlertTriangle, AlertCircle, ArrowUpRight, ShieldCheck, PieChart, GitFork, Info } from 'lucide-react';

interface IntegrityAuditorProps {
  records: SampleRecord[];
  onSelectRecordToEdit: (sampleId: string) => void;
  onAddViolationSample: () => void;
}

export const IntegrityAuditor: React.FC<IntegrityAuditorProps> = ({
  records,
  onSelectRecordToEdit,
  onAddViolationSample,
}) => {
  const auditResult = auditDataset(records);
  const complianceRate = records.length > 0 
    ? Math.round(((records.length - auditResult.invalidCount) / records.length) * 100) 
    : 100;

  // Distribution counters
  const labelCounts: Record<string, number> = {};
  const errorTypeCounts: Record<string, number> = {};
  const splitCounts: Record<string, number> = { train: 0, val: 0, test: 0 };
  const groupCounts: Record<string, number> = {};

  records.forEach((r) => {
    labelCounts[r.misconception_id] = (labelCounts[r.misconception_id] || 0) + 1;
    errorTypeCounts[r.error_type] = (errorTypeCounts[r.error_type] || 0) + 1;
    splitCounts[r.split] = (splitCounts[r.split] || 0) + 1;
    if (r.question_group) {
      groupCounts[r.question_group] = (groupCounts[r.question_group] || 0) + 1;
    }
  });

  return (
    <div className="space-y-6">
      {/* Overview Top Card */}
      <div className="border border-slate-200 bg-white rounded-xl p-5 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200 flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5" /> Schema Validation Engine (v1.0 Frozen)
              </span>
              <span className="text-xs text-slate-500">·</span>
              <span className="text-xs text-slate-500">13 Strict Annotation Rules Audited</span>
            </div>
            <h1 className="text-xl font-bold text-slate-900 mt-1.5">Dataset Integrity & Compliance Auditor</h1>
            <p className="text-xs text-slate-600 mt-0.5">
              Live automated auditing across all active benchmark records against the frozen taxonomy constraints and Rule 13 split leakage rules.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onAddViolationSample}
              className="text-xs text-slate-700 hover:text-slate-900 font-medium px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50"
            >
              + Inject Constraint Violation Test
            </button>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
          <div className="border border-slate-200 rounded-lg p-3.5 bg-slate-50/50">
            <div className="text-xs font-medium text-slate-500">Total Samples</div>
            <div className="text-2xl font-bold text-slate-900 font-mono mt-1 tabular-nums">
              {records.length}
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">Active benchmark records</div>
          </div>

          <div className="border border-slate-200 rounded-lg p-3.5 bg-slate-50/50">
            <div className="text-xs font-medium text-slate-500">Schema Compliance</div>
            <div className={`text-2xl font-bold font-mono mt-1 tabular-nums ${complianceRate === 100 ? 'text-emerald-600' : 'text-amber-600'}`}>
              {complianceRate}%
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">
              {auditResult.invalidCount === 0 ? 'Zero constraint violations' : `${auditResult.invalidCount} invalid record(s)`}
            </div>
          </div>

          <div className="border border-slate-200 rounded-lg p-3.5 bg-slate-50/50">
            <div className="text-xs font-medium text-slate-500">Fully Valid Records</div>
            <div className="text-2xl font-bold text-emerald-600 font-mono mt-1 tabular-nums">
              {auditResult.validCount}
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">Pass all frozen schema constraints</div>
          </div>

          <div className="border border-slate-200 rounded-lg p-3.5 bg-slate-50/50">
            <div className="text-xs font-medium text-slate-500">Guideline Warnings</div>
            <div className="text-2xl font-bold text-amber-600 font-mono mt-1 tabular-nums">
              {auditResult.warningCount}
            </div>
            <div className="text-[11px] text-slate-500 mt-0.5">Rule 9/10/12/13 guideline alerts</div>
          </div>
        </div>
      </div>

      {/* Rule 13 Question Group Split Leakage Auditor Box */}
      <div className="border border-slate-200 bg-white rounded-xl p-5 shadow-xs">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <GitFork className="w-4 h-4 text-blue-600" />
            <h2 className="text-sm font-semibold text-slate-900">
              Rule 13 Question Group Integrity (Split Leakage Check)
            </h2>
          </div>
          <span className="text-xs text-slate-500 font-mono">
            {Object.keys(groupCounts).length} Distinct Question Groups
          </span>
        </div>

        <div className="mt-4">
          {auditResult.splitLeakageGroups.length === 0 ? (
            <div className="flex items-start gap-3 p-3.5 bg-emerald-50/50 border border-emerald-200 rounded-lg text-xs text-emerald-900">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold">Rule 13 Pass:</span> All near-duplicate / related questions sharing a <code className="font-mono bg-white px-1 rounded border border-emerald-200">question_group</code> are cleanly partitioned into the same split (train, val, or test). Zero split leakage detected across {records.length} records.
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex items-start gap-3 p-3 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-900">
                <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold">Rule 13 Split Leakage Alert:</span> The following {auditResult.splitLeakageGroups.length} question group(s) have records spread across multiple splits. To satisfy Rule 13, assign all records in each group to the same split:
                </div>
              </div>

              <div className="grid grid-cols-1 gap-2">
                {auditResult.splitLeakageGroups.map((lg) => (
                  <div key={lg.question_group} className="p-3 border border-slate-200 rounded-lg bg-slate-50 flex items-center justify-between text-xs">
                    <div>
                      <span className="font-mono font-bold text-slate-900">{lg.question_group}</span>
                      <span className="text-slate-500 ml-2">Spans splits: [{lg.splits.join(', ')}]</span>
                      <div className="text-[11px] text-slate-400 font-mono mt-0.5">
                        Samples: {lg.sample_ids.join(', ')}
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => onSelectRecordToEdit(lg.sample_ids[0])}
                      className="text-xs font-semibold text-blue-600 hover:text-blue-800 bg-white px-2.5 py-1 rounded border border-slate-200"
                    >
                      Inspect First Sample
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Distribution Breakdowns */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Misconception Class Breakdown */}
        <div className="border border-slate-200 bg-white rounded-xl p-4 shadow-xs">
          <h2 className="text-xs font-semibold text-slate-900 uppercase tracking-wider mb-3 flex items-center gap-1.5">
            <PieChart className="w-3.5 h-3.5 text-blue-600" /> Label Distribution
          </h2>
          <div className="space-y-1.5 text-xs">
            {Object.entries(labelCounts)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([label, count]) => {
                const percent = Math.round((count / records.length) * 100) || 0;
                return (
                  <div key={label} className="flex items-center justify-between py-1 border-b border-slate-100 last:border-0">
                    <span className="font-mono font-medium text-slate-700">{label}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-slate-400 font-mono text-[11px] tabular-nums">{percent}%</span>
                      <span className="font-mono font-bold text-slate-900 bg-slate-100 px-1.5 py-0.5 rounded text-[11px] tabular-nums min-w-6 text-center">
                        {count}
                      </span>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>

        {/* Error Type Breakdown */}
        <div className="border border-slate-200 bg-white rounded-xl p-4 shadow-xs">
          <h2 className="text-xs font-semibold text-slate-900 uppercase tracking-wider mb-3">
            Error Type Distribution
          </h2>
          <div className="space-y-1.5 text-xs">
            {Object.entries(errorTypeCounts)
              .sort(([, a], [, b]) => b - a)
              .map(([errorType, count]) => {
                const percent = Math.round((count / records.length) * 100) || 0;
                return (
                  <div key={errorType} className="flex items-center justify-between py-1 border-b border-slate-100 last:border-0">
                    <span className="text-slate-700 font-medium capitalize">{errorType.replace('_', ' ')}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-slate-400 font-mono text-[11px] tabular-nums">{percent}%</span>
                      <span className="font-mono font-bold text-slate-900 bg-slate-100 px-1.5 py-0.5 rounded text-[11px] tabular-nums min-w-6 text-center">
                        {count}
                      </span>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>

        {/* Split Distribution */}
        <div className="border border-slate-200 bg-white rounded-xl p-4 shadow-xs">
          <h2 className="text-xs font-semibold text-slate-900 uppercase tracking-wider mb-3">
            Split Partitioning
          </h2>
          <div className="space-y-1.5 text-xs">
            {(['train', 'val', 'test'] as const).map((splitName) => {
              const count = splitCounts[splitName] || 0;
              const percent = Math.round((count / records.length) * 100) || 0;
              return (
                <div key={splitName} className="flex items-center justify-between py-1 border-b border-slate-100 last:border-0">
                  <span className="text-slate-700 font-medium uppercase font-mono">{splitName}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-slate-400 font-mono text-[11px] tabular-nums">{percent}%</span>
                    <span className="font-mono font-bold text-slate-900 bg-slate-100 px-1.5 py-0.5 rounded text-[11px] tabular-nums min-w-6 text-center">
                      {count}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Itemized Audit Report / Violations List */}
      <div className="border border-slate-200 bg-white rounded-xl overflow-hidden shadow-xs">
        <div className="px-5 py-3 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
            {auditResult.violationsByRecord.length === 0 ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            ) : (
              <AlertCircle className="w-4 h-4 text-amber-600" />
            )}
            Audit Findings & Integrity Log
          </h2>
          <span className="text-xs text-slate-500">
            {auditResult.violationsByRecord.length} flagged record(s)
          </span>
        </div>

        {auditResult.violationsByRecord.length === 0 ? (
          <div className="p-8 text-center space-y-2">
            <CheckCircle2 className="w-8 h-8 text-emerald-600 mx-auto" />
            <div className="text-sm font-semibold text-slate-900">Zero Integrity Violations Detected</div>
            <p className="text-xs text-slate-500 max-w-md mx-auto">
              All records in the current dataset strictly satisfy all 7 frozen schema constraints and follow the 13 frozen annotation rules.
            </p>
          </div>
        ) : (
          <div className="divide-y divide-slate-100">
            {auditResult.violationsByRecord.map((item) => (
              <div key={item.sample_id} className="p-4 hover:bg-slate-50/60 flex items-start justify-between gap-4">
                <div className="space-y-1.5 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-bold text-xs text-slate-900 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                      {item.sample_id}
                    </span>
                  </div>

                  <div className="space-y-1 mt-1">
                    {item.errors.map((err, idx) => (
                      <div
                        key={idx}
                        className={`text-xs flex items-start gap-2 p-2 rounded ${
                          err.severity === 'error'
                            ? 'bg-red-50 text-red-900 border border-red-100'
                            : 'bg-amber-50 text-amber-900 border border-amber-100'
                        }`}
                      >
                        <span className="font-semibold shrink-0 uppercase text-[10px] px-1 py-0.5 rounded bg-white/70">
                          {err.ruleNumber}
                        </span>
                        <span className="text-xs">{err.message}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => onSelectRecordToEdit(item.sample_id)}
                  className="flex items-center gap-1 text-xs font-semibold text-blue-600 hover:text-blue-800 bg-blue-50 hover:bg-blue-100 px-3 py-1.5 rounded-lg border border-blue-200 shrink-0 transition-colors"
                >
                  Inspect & Fix <ArrowUpRight className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
