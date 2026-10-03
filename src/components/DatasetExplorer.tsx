import React, { useState, useMemo } from 'react';
import { SampleRecord } from '../types/dataset';
import { validateRecord, isMisconceptionClass } from '../utils/validator';
import { Search, Filter, Plus, Edit3, Copy, Trash2, CheckCircle2, AlertTriangle, AlertCircle, FileText } from 'lucide-react';

interface DatasetExplorerProps {
  records: SampleRecord[];
  onSelectRecord: (record: SampleRecord) => void;
  onDuplicateRecord: (record: SampleRecord) => void;
  onDeleteRecord: (sampleId: string) => void;
  onAddNewRecord: () => void;
}

export const DatasetExplorer: React.FC<DatasetExplorerProps> = ({
  records,
  onSelectRecord,
  onDuplicateRecord,
  onDeleteRecord,
  onAddNewRecord,
}) => {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [filterSplit, setFilterSplit] = useState<string>('all');
  const [filterLabel, setFilterLabel] = useState<string>('all');
  const [filterErrorType, setFilterErrorType] = useState<string>('all');
  const [filterValidation, setFilterValidation] = useState<'all' | 'valid' | 'invalid'>('all');

  // Filter records
  const filteredRecords = useMemo(() => {
    return records.filter((r) => {
      // Search
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matches =
          r.sample_id.toLowerCase().includes(q) ||
          r.question.toLowerCase().includes(q) ||
          r.student_reasoning.toLowerCase().includes(q) ||
          r.student_answer.toLowerCase().includes(q) ||
          r.annotator_rationale.toLowerCase().includes(q) ||
          r.concept.toLowerCase().includes(q);
        if (!matches) return false;
      }

      // Split
      if (filterSplit !== 'all' && r.split !== filterSplit) return false;

      // Label
      if (filterLabel !== 'all') {
        if (filterLabel === 'M_ALL') {
          if (!isMisconceptionClass(r.misconception_id)) return false;
        } else if (r.misconception_id !== filterLabel) {
          return false;
        }
      }

      // Error Type
      if (filterErrorType !== 'all' && r.error_type !== filterErrorType) return false;

      // Validation
      if (filterValidation !== 'all') {
        const issues = validateRecord(r);
        const hasError = issues.some((i) => i.severity === 'error');
        if (filterValidation === 'valid' && hasError) return false;
        if (filterValidation === 'invalid' && !hasError) return false;
      }

      return true;
    });
  }, [records, searchQuery, filterSplit, filterLabel, filterErrorType, filterValidation]);

  return (
    <div className="space-y-4">
      {/* Top Controls Bar */}
      <div className="border border-slate-200 bg-white rounded-xl p-4 shadow-xs space-y-3">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
          {/* Search bar */}
          <div className="relative flex-1 max-w-md">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search sample ID, code question, reasoning, or rationale..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full text-xs pl-9 pr-3 py-2 border border-slate-200 rounded-lg focus:outline-hidden focus:border-blue-500 bg-slate-50/50"
            />
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 font-mono tabular-nums">
              Showing {filteredRecords.length} of {records.length} records
            </span>
            <button
              type="button"
              onClick={onAddNewRecord}
              className="flex items-center gap-1.5 text-xs bg-blue-600 text-white font-medium py-1.5 px-3.5 rounded-lg hover:bg-blue-700 transition-colors shadow-xs"
            >
              <Plus className="w-3.5 h-3.5" /> New Record
            </button>
          </div>
        </div>

        {/* Filter bar */}
        <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-slate-100 text-xs">
          {/* Split filter */}
          <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg border border-slate-200">
            {['all', 'train', 'val', 'test'].map((s) => (
              <button
                key={s}
                onClick={() => setFilterSplit(s)}
                className={`px-2.5 py-1 rounded-md text-xs font-medium capitalize transition-colors ${
                  filterSplit === s ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {s}
              </button>
            ))}
          </div>

          {/* Label selector */}
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500">Label:</span>
            <select
              value={filterLabel}
              onChange={(e) => setFilterLabel(e.target.value)}
              className="text-xs bg-white border border-slate-200 rounded-lg px-2.5 py-1 text-slate-700 focus:outline-hidden"
            >
              <option value="all">All Labels</option>
              <option value="M_ALL">All M01–M08</option>
              <option value="M01">M01 (Type Confusion)</option>
              <option value="M02">M02 (Division)</option>
              <option value="M03">M03 (= vs ==)</option>
              <option value="M04">M04 (Precedence)</option>
              <option value="M05">M05 (Index/Position)</option>
              <option value="M06">M06 (Loops/Bounds)</option>
              <option value="M07">M07 (Arg-Param Binding)</option>
              <option value="M08">M08 (Recursion)</option>
              <option value="NONE">NONE</option>
              <option value="INSUFFICIENT">INSUFFICIENT</option>
              <option value="OOS">OOS (Out of Scope)</option>
            </select>
          </div>

          {/* Error Type selector */}
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500">Error Type:</span>
            <select
              value={filterErrorType}
              onChange={(e) => setFilterErrorType(e.target.value)}
              className="text-xs bg-white border border-slate-200 rounded-lg px-2.5 py-1 text-slate-700 focus:outline-hidden"
            >
              <option value="all">All Types</option>
              <option value="conceptual">conceptual</option>
              <option value="trace_error">trace_error</option>
              <option value="correct">correct</option>
              <option value="careless">careless</option>
              <option value="typo">typo</option>
              <option value="syntax_error">syntax_error</option>
              <option value="other">other</option>
            </select>
          </div>

          {/* Validation Status filter */}
          <div className="flex items-center gap-1.5 ml-auto">
            <span className="text-slate-500">Validation:</span>
            <div className="flex items-center bg-slate-100 p-0.5 rounded-lg border border-slate-200">
              <button
                onClick={() => setFilterValidation('all')}
                className={`px-2 py-0.5 rounded text-xs font-medium ${
                  filterValidation === 'all' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
                }`}
              >
                All
              </button>
              <button
                onClick={() => setFilterValidation('valid')}
                className={`px-2 py-0.5 rounded text-xs font-medium ${
                  filterValidation === 'valid' ? 'bg-white text-emerald-700 shadow-xs' : 'text-slate-600'
                }`}
              >
                Valid Only
              </button>
              <button
                onClick={() => setFilterValidation('invalid')}
                className={`px-2 py-0.5 rounded text-xs font-medium ${
                  filterValidation === 'invalid' ? 'bg-white text-red-700 shadow-xs' : 'text-slate-600'
                }`}
              >
                With Errors
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* High-density Table */}
      <div className="border border-slate-200 bg-white rounded-xl overflow-hidden shadow-xs">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold select-none">
              <tr>
                <th className="py-2.5 px-3.5 w-28">Sample ID</th>
                <th className="py-2.5 px-3 w-16">Split</th>
                <th className="py-2.5 px-3.5">Question & Concept</th>
                <th className="py-2.5 px-3 w-32">Student Answer</th>
                <th className="py-2.5 px-3 w-40">Misconception ID</th>
                <th className="py-2.5 px-3 w-28">Error Type</th>
                <th className="py-2.5 px-3 w-24">Status</th>
                <th className="py-2.5 px-3 w-28 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredRecords.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-slate-500">
                    <FileText className="w-8 h-8 text-slate-300 mx-auto mb-2" />
                    No records match the active search and filter criteria.
                  </td>
                </tr>
              ) : (
                filteredRecords.map((r) => {
                  const issues = validateRecord(r);
                  const hasError = issues.some((i) => i.severity === 'error');
                  const hasWarning = issues.some((i) => i.severity === 'warning');

                  return (
                    <tr
                      key={r.sample_id}
                      onClick={() => onSelectRecord(r)}
                      className="hover:bg-blue-50/30 cursor-pointer transition-colors group"
                    >
                      <td className="py-2.5 px-3.5 font-mono font-medium text-slate-900 whitespace-nowrap">
                        {r.sample_id}
                      </td>
                      <td className="py-2.5 px-3 text-slate-500 uppercase font-mono text-[11px]">
                        {r.split}
                      </td>
                      <td className="py-2.5 px-3.5 max-w-xs">
                        <div className="flex items-center gap-1.5 truncate">
                          <span className="font-medium text-slate-800 truncate" title={r.concept}>
                            {r.concept}
                          </span>
                          <span className="text-[10px] font-mono text-slate-500 bg-slate-100 px-1 rounded shrink-0 border border-slate-200" title={`Rule 13 Question Group: ${r.question_group}`}>
                            {r.question_group}
                          </span>
                        </div>
                        <div className="text-slate-500 truncate text-[11px] font-mono mt-0.5" title={r.question}>
                          {r.question.split('\n')[0]}
                        </div>
                      </td>
                      <td className="py-2.5 px-3 max-w-[120px]">
                        <div className="truncate font-mono text-slate-800" title={r.student_answer}>
                          {r.student_answer}
                        </div>
                        <div className="text-[11px] text-slate-400 truncate" title={`Correct: ${r.correct_answer}`}>
                          Ref: {r.correct_answer}
                        </div>
                      </td>
                      <td className="py-2.5 px-3">
                        <div className="flex items-center gap-1.5 flex-wrap">
                          <span
                            className={`font-mono font-bold text-[11px] px-1.5 py-0.5 rounded border ${
                              r.misconception_id.startsWith('M')
                                ? 'bg-blue-50 text-blue-800 border-blue-200'
                                : r.misconception_id === 'OOS'
                                ? 'bg-amber-50 text-amber-800 border-amber-200'
                                : 'bg-slate-100 text-slate-700 border-slate-200'
                            }`}
                          >
                            {r.misconception_id}
                          </span>
                          {r.secondary_misconception_ids?.length > 0 && (
                            <span className="font-mono text-[10px] text-slate-500 bg-slate-50 px-1 rounded border border-slate-200" title={`Secondary: ${r.secondary_misconception_ids.join(', ')}`}>
                              +{r.secondary_misconception_ids.length}
                            </span>
                          )}
                        </div>
                        {r.misconception_variant && (
                          <div className="text-[10px] text-slate-500 truncate mt-0.5 font-mono max-w-[150px]" title={r.misconception_variant}>
                            {r.misconception_variant}
                          </div>
                        )}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-[11px] text-slate-700">
                        {r.error_type}
                      </td>
                      <td className="py-2.5 px-3">
                        {hasError ? (
                          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-red-700 bg-red-50 border border-red-200 px-1.5 py-0.5 rounded" title={issues.map(i => i.message).join(' | ')}>
                            <AlertCircle className="w-3 h-3" /> Error
                          </span>
                        ) : hasWarning ? (
                          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-amber-700 bg-amber-50 border border-amber-200 px-1.5 py-0.5 rounded" title={issues.map(i => i.message).join(' | ')}>
                            <AlertTriangle className="w-3 h-3" /> Warn
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-1.5 py-0.5 rounded">
                            <CheckCircle2 className="w-3 h-3" /> Valid
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 px-3 text-right" onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-center justify-end gap-1">
                          <button
                            type="button"
                            onClick={() => onSelectRecord(r)}
                            title="Annotate & Edit"
                            className="p-1 text-slate-400 hover:text-blue-600 rounded hover:bg-slate-100"
                          >
                            <Edit3 className="w-3.5 h-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => onDuplicateRecord(r)}
                            title="Duplicate Record"
                            className="p-1 text-slate-400 hover:text-slate-700 rounded hover:bg-slate-100"
                          >
                            <Copy className="w-3.5 h-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              if (confirm(`Delete record ${r.sample_id}?`)) {
                                onDeleteRecord(r.sample_id);
                              }
                            }}
                            title="Delete Record"
                            className="p-1 text-slate-400 hover:text-red-600 rounded hover:bg-slate-100"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
