import React, { useState, useEffect, useRef } from 'react';
import { SampleRecord, MisconceptionId, ErrorType, MisconceptionClassId, DatasetSplit } from '../types/dataset';
import { FROZEN_TAXONOMY, NON_CLASS_OUTCOMES } from '../data/taxonomy';
import { validateRecord, isMisconceptionClass } from '../utils/validator';
import {
  CheckCircle2,
  AlertCircle,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  Quote,
  Save,
  RotateCcw,
  Compass,
  FileCode2,
  Check,
  Plus,
} from 'lucide-react';

interface AnnotatorStudioProps {
  currentRecord: SampleRecord;
  records: SampleRecord[];
  onSaveRecord: (updated: SampleRecord) => void;
  onSelectRecordById: (sampleId: string) => void;
  onOpenDecisionWizard: () => void;
  onAddNewRecord: () => void;
}

export const AnnotatorStudio: React.FC<AnnotatorStudioProps> = ({
  currentRecord,
  records,
  onSaveRecord,
  onSelectRecordById,
  onOpenDecisionWizard,
  onAddNewRecord,
}) => {
  const [form, setForm] = useState<SampleRecord>({ ...currentRecord });
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState<boolean>(false);
  const [selectedQuote, setSelectedQuote] = useState<string>('');
  const [showSavedFeedback, setShowSavedFeedback] = useState<boolean>(false);

  // Text selection tracking
  const reasoningRef = useRef<HTMLDivElement>(null);

  // Sync state when currentRecord changes
  useEffect(() => {
    setForm({ ...currentRecord });
    setHasUnsavedChanges(false);
  }, [currentRecord.sample_id]);

  // Validation
  const issues = validateRecord(form);
  const errors = issues.filter((i) => i.severity === 'error');
  const warnings = issues.filter((i) => i.severity === 'warning');

  // Index navigation
  const currentIndex = records.findIndex((r) => r.sample_id === currentRecord.sample_id);
  const prevRecord = currentIndex > 0 ? records[currentIndex - 1] : null;
  const nextRecord = currentIndex < records.length - 1 ? records[currentIndex + 1] : null;

  // Selected M-Class meta
  const isM = isMisconceptionClass(form.misconception_id);
  const currentTaxonomyItem = FROZEN_TAXONOMY.find((t) => t.id === form.misconception_id);

  // Handle text selection in student reasoning
  const handleMouseUp = () => {
    const sel = window.getSelection();
    if (sel && sel.toString().trim().length > 0) {
      setSelectedQuote(sel.toString().trim());
    } else {
      setSelectedQuote('');
    }
  };

  const handleApplySelectedQuoteAsEvidence = () => {
    if (!selectedQuote) return;
    setForm((prev) => ({
      ...prev,
      evidence_basis: selectedQuote,
    }));
    setHasUnsavedChanges(true);
    setSelectedQuote('');
  };

  const handleFieldChange = <K extends keyof SampleRecord>(key: K, val: SampleRecord[K]) => {
    setForm((prev) => {
      const next = { ...prev, [key]: val };

      // Automatic constraint enforcement assistance:
      if (key === 'misconception_id') {
        const nextIsM = isMisconceptionClass(val as string);
        if (!nextIsM) {
          next.misconception_variant = null;
          next.secondary_misconception_ids = [];
          if (next.error_type === 'conceptual') {
            next.error_type = val === 'NONE' ? (next.answer_correct ? 'correct' : 'careless') : (val === 'INSUFFICIENT' ? 'trace_error' : 'other');
          }
        } else {
          if (next.error_type !== 'conceptual') {
            next.error_type = 'conceptual';
          }
          const tax = FROZEN_TAXONOMY.find((t) => t.id === val);
          if (tax && tax.variants.length > 0 && !next.misconception_variant) {
            next.misconception_variant = tax.variants[0];
          }
        }
      }

      if (key === 'answer_correct' && val === true && next.misconception_id === 'NONE') {
        next.error_type = 'correct';
      }

      return next;
    });
    setHasUnsavedChanges(true);
  };

  const handleSave = () => {
    onSaveRecord(form);
    setHasUnsavedChanges(false);
    setShowSavedFeedback(true);
    setTimeout(() => setShowSavedFeedback(false), 2000);
  };

  const handleGenerateRationaleTemplate = () => {
    let draft = '';
    if (isM) {
      const quotePart = form.evidence_basis ? `("${form.evidence_basis.slice(0, 45)}...")` : 'reasoning';
      draft = `Student explicitly demonstrates ${form.misconception_id} by stating in evidence ${quotePart}.`;
    } else if (form.misconception_id === 'OOS') {
      draft = `Student states an out-of-scope conceptual belief (${form.evidence_basis || 'concept'}), which is excluded from M01-M08 under Rule 2.`;
    } else if (form.misconception_id === 'INSUFFICIENT') {
      draft = `Under Rule 3, an arithmetic or trace mistake with absence of an explicit conceptual belief is coded INSUFFICIENT.`;
    } else {
      draft = form.answer_correct
        ? 'Student response is correct with sound reasoning, demonstrating absence of any misconception.'
        : `Student error was a ${form.error_type} slip with absence of any conceptual misconception.`;
    }
    handleFieldChange('annotator_rationale', draft);
  };

  const toggleSecondaryId = (mId: MisconceptionClassId) => {
    const list = [...(form.secondary_misconception_ids || [])];
    const index = list.indexOf(mId);
    if (index >= 0) {
      list.splice(index, 1);
    } else {
      if (list.length >= 2) return; // max 2
      list.push(mId);
    }
    handleFieldChange('secondary_misconception_ids', list);
  };

  return (
    <div className="space-y-4">
      {/* Top Studio Control Bar */}
      <div className="border border-slate-200 bg-white rounded-xl px-5 py-3 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        {/* Record Navigator */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            <button
              type="button"
              disabled={!prevRecord}
              onClick={() => prevRecord && onSelectRecordById(prevRecord.sample_id)}
              className="p-1.5 rounded-lg border border-slate-200 text-slate-600 hover:text-slate-900 hover:bg-slate-50 disabled:opacity-30 disabled:pointer-events-none"
              title="Previous Record"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              type="button"
              disabled={!nextRecord}
              onClick={() => nextRecord && onSelectRecordById(nextRecord.sample_id)}
              className="p-1.5 rounded-lg border border-slate-200 text-slate-600 hover:text-slate-900 hover:bg-slate-50 disabled:opacity-30 disabled:pointer-events-none"
              title="Next Record"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          <div className="flex items-center gap-2">
            <span className="font-mono text-xs font-bold text-slate-900 bg-slate-100 px-2 py-1 rounded border border-slate-200">
              {form.sample_id}
            </span>
            <span className="text-xs text-slate-400 font-mono">
              ({currentIndex + 1} of {records.length})
            </span>
          </div>

          {hasUnsavedChanges && (
            <span className="text-[11px] font-semibold text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
              Unsaved Changes
            </span>
          )}
        </div>

        {/* Studio Actions */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onOpenDecisionWizard}
            className="flex items-center gap-1.5 text-xs text-blue-700 bg-blue-50 hover:bg-blue-100 font-medium py-1.5 px-3 rounded-lg border border-blue-200 transition-colors"
          >
            <Compass className="w-3.5 h-3.5" /> Decision Wizard (Rule 2)
          </button>

          <button
            type="button"
            onClick={() => setForm({ ...currentRecord })}
            disabled={!hasUnsavedChanges}
            className="flex items-center gap-1 text-xs text-slate-600 hover:text-slate-900 font-medium py-1.5 px-2.5 rounded-lg border border-slate-200 hover:bg-slate-50 disabled:opacity-40"
          >
            <RotateCcw className="w-3.5 h-3.5" /> Revert
          </button>

          <button
            type="button"
            onClick={handleSave}
            className="flex items-center gap-1.5 text-xs bg-slate-900 text-white font-medium py-1.5 px-4 rounded-lg hover:bg-slate-800 transition-colors shadow-xs"
          >
            {showSavedFeedback ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Save className="w-3.5 h-3.5" />}
            {showSavedFeedback ? 'Saved!' : 'Save Record'}
          </button>
        </div>
      </div>

      {/* Main Dual-Pane Studio Canvas */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* LEFT PANE: Problem & Student Response (5 Cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="border border-slate-200 bg-white rounded-xl p-5 shadow-xs space-y-4">
            {/* Metadata Bar */}
            <div className="flex items-center justify-between text-xs pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-slate-700">Split:</span>
                <select
                  value={form.split}
                  onChange={(e) => handleFieldChange('split', e.target.value as DatasetSplit)}
                  className="font-mono text-xs bg-slate-50 border border-slate-200 rounded px-2 py-0.5"
                >
                  <option value="train">TRAIN</option>
                  <option value="val">VAL</option>
                  <option value="test">TEST</option>
                </select>
              </div>

              <div className="text-slate-500 font-mono text-[11px] truncate max-w-[200px]" title={form.source}>
                Source: {form.source}
              </div>
            </div>

            {/* Question Group (Rule 13) & Concept */}
            <div className="space-y-2 text-xs">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="flex items-center justify-between mb-0.5">
                    <label className="text-[11px] font-semibold text-slate-800">
                      Question Group (Rule 13)
                    </label>
                  </div>
                  <input
                    type="text"
                    value={form.question_group}
                    onChange={(e) => handleFieldChange('question_group', e.target.value)}
                    placeholder="e.g. fn_greet_binding"
                    className="w-full text-xs font-mono px-2.5 py-1.5 border border-slate-200 rounded-md bg-slate-50/50"
                  />
                  <div className="text-[10px] text-slate-500 mt-0.5">
                    Rule 13: Keep near-duplicates in same split.
                  </div>
                </div>

                <div>
                  <label className="text-[11px] font-semibold text-slate-800 block mb-0.5">Concept</label>
                  <input
                    type="text"
                    value={form.concept}
                    onChange={(e) => handleFieldChange('concept', e.target.value)}
                    className="w-full text-xs px-2.5 py-1.5 border border-slate-200 rounded-md bg-slate-50/50"
                  />
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    Format: {form.question_format}
                  </div>
                </div>
              </div>
            </div>

            {/* Question / Code Box */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <label className="font-semibold text-slate-800 flex items-center gap-1.5">
                  <FileCode2 className="w-3.5 h-3.5 text-blue-600" /> Question & Python Code
                </label>
              </div>
              <textarea
                value={form.question}
                onChange={(e) => handleFieldChange('question', e.target.value)}
                rows={5}
                className="w-full text-xs font-mono bg-slate-900 text-slate-100 rounded-lg p-3.5 border border-slate-800 focus:outline-hidden leading-relaxed"
              />
            </div>

            {/* Answers Comparison */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="space-y-1 bg-emerald-50/40 border border-emerald-200 rounded-lg p-3">
                <label className="text-[11px] font-semibold text-emerald-900 block">Reference Correct Answer</label>
                <textarea
                  value={form.correct_answer}
                  onChange={(e) => handleFieldChange('correct_answer', e.target.value)}
                  rows={2}
                  className="w-full text-xs font-mono bg-white border border-emerald-200 rounded p-2 text-slate-900"
                />
              </div>

              <div className={`space-y-1 rounded-lg p-3 border ${
                form.answer_correct ? 'bg-emerald-50/30 border-emerald-200' : 'bg-rose-50/30 border-rose-200'
              }`}>
                <div className="flex items-center justify-between">
                  <label className="text-[11px] font-semibold text-slate-900 block">Student Answer</label>
                  <label className="flex items-center gap-1 text-[11px] cursor-pointer text-slate-700">
                    <input
                      type="checkbox"
                      checked={form.answer_correct}
                      onChange={(e) => handleFieldChange('answer_correct', e.target.checked)}
                      className="rounded border-slate-300 text-emerald-600"
                    />
                    <span>Correct?</span>
                  </label>
                </div>
                <textarea
                  value={form.student_answer}
                  onChange={(e) => handleFieldChange('student_answer', e.target.value)}
                  rows={2}
                  className="w-full text-xs font-mono bg-white border border-slate-200 rounded p-2 text-slate-900"
                />
              </div>
            </div>

            {/* Student Reasoning (with interactive quote selection) */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <label className="font-semibold text-slate-800 flex items-center gap-1">
                  <Quote className="w-3.5 h-3.5 text-indigo-600" /> Student Stated Reasoning
                </label>
                <span className="text-[11px] text-slate-500">Highlight text to set evidence</span>
              </div>

              <div
                ref={reasoningRef}
                onMouseUp={handleMouseUp}
                className="p-3 bg-amber-50/20 border border-amber-200/80 rounded-lg text-xs text-slate-800 leading-relaxed font-sans cursor-text select-text"
              >
                {form.student_reasoning || <span className="text-slate-400 italic">No reasoning provided by student.</span>}
              </div>

              {/* Editable Reasoning fallback */}
              <details className="text-[11px] text-slate-500 cursor-pointer">
                <summary className="hover:text-slate-700">Edit raw student reasoning text</summary>
                <textarea
                  value={form.student_reasoning}
                  onChange={(e) => handleFieldChange('student_reasoning', e.target.value)}
                  rows={3}
                  className="w-full text-xs font-sans bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-slate-800 mt-1"
                />
              </details>

              {/* Floating or docked highlight action */}
              {selectedQuote && (
                <div className="flex items-center justify-between p-2.5 bg-blue-50 border border-blue-200 rounded-lg text-xs animate-in fade-in duration-150">
                  <div className="truncate mr-2 text-blue-900">
                    <span className="font-semibold">Selected quote:</span> "{selectedQuote}"
                  </div>
                  <button
                    type="button"
                    onClick={handleApplySelectedQuoteAsEvidence}
                    className="shrink-0 font-semibold text-white bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded text-xs transition-colors shadow-xs"
                  >
                    Set as Evidence Basis
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* RIGHT PANE: Frozen Annotation Inspector & Rules Enforcement (7 Cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="border border-slate-200 bg-white rounded-xl p-5 shadow-xs space-y-5">
            {/* Section Header */}
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div>
                <h2 className="text-sm font-semibold text-slate-900">Annotation Specification & Validation</h2>
                <p className="text-xs text-slate-500">Live enforcement of frozen taxonomy v1.0 rules and constraints</p>
              </div>

              {/* Validation pill status */}
              {errors.length === 0 ? (
                <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-lg">
                  <CheckCircle2 className="w-3.5 h-3.5" /> All Constraints Pass
                </div>
              ) : (
                <div className="flex items-center gap-1.5 text-xs font-semibold text-red-700 bg-red-50 border border-red-200 px-2.5 py-1 rounded-lg">
                  <AlertCircle className="w-3.5 h-3.5" /> {errors.length} Constraint Violation(s)
                </div>
              )}
            </div>

            {/* Violation Alert Banners if any */}
            {issues.length > 0 && (
              <div className="space-y-2">
                {issues.map((iss, i) => (
                  <div
                    key={i}
                    className={`p-3 rounded-lg border text-xs flex items-start gap-2.5 ${
                      iss.severity === 'error'
                        ? 'bg-red-50 border-red-200 text-red-900'
                        : 'bg-amber-50 border-amber-200 text-amber-900'
                    }`}
                  >
                    {iss.severity === 'error' ? (
                      <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                    )}
                    <div className="space-y-0.5 flex-1">
                      <div className="font-semibold">{iss.ruleNumber} Violation: {iss.field}</div>
                      <div className="text-xs leading-relaxed">{iss.message}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Row 1: Primary Misconception ID */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-800 block">
                Primary Label (`misconception_id`):
              </label>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {FROZEN_TAXONOMY.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => handleFieldChange('misconception_id', item.id)}
                    className={`p-2.5 rounded-lg border text-left transition-all ${
                      form.misconception_id === item.id
                        ? 'border-blue-600 bg-blue-50/80 text-blue-900 ring-1 ring-blue-600 font-semibold'
                        : 'border-slate-200 hover:border-slate-300 text-slate-700 bg-white'
                    }`}
                  >
                    <div className="font-mono text-xs">{item.id}</div>
                    <div className="text-[11px] truncate mt-0.5 text-slate-600">{item.name}</div>
                  </button>
                ))}
              </div>

              {/* Non-class outcomes */}
              <div className="grid grid-cols-3 gap-2 pt-1">
                {NON_CLASS_OUTCOMES.map((nc) => (
                  <button
                    key={nc.id}
                    type="button"
                    onClick={() => handleFieldChange('misconception_id', nc.id)}
                    className={`p-2 rounded-lg border text-center transition-all ${
                      form.misconception_id === nc.id
                        ? nc.id === 'OOS'
                          ? 'border-amber-600 bg-amber-50 text-amber-900 ring-1 ring-amber-600 font-semibold'
                          : nc.id === 'INSUFFICIENT'
                          ? 'border-slate-700 bg-slate-100 text-slate-900 ring-1 ring-slate-700 font-semibold'
                          : 'border-emerald-600 bg-emerald-50 text-emerald-900 ring-1 ring-emerald-600 font-semibold'
                        : 'border-slate-200 hover:border-slate-300 text-slate-700 bg-white'
                    }`}
                  >
                    <div className="font-mono text-xs font-bold">{nc.id}</div>
                    <div className="text-[10px] text-slate-500 truncate">{nc.meaning}</div>
                  </button>
                ))}
              </div>

              {/* Definition callout for selected label */}
              {currentTaxonomyItem && (
                <div className="p-3 bg-blue-50/40 border border-blue-200 rounded-lg text-xs space-y-1">
                  <div className="font-semibold text-blue-900">
                    {currentTaxonomyItem.id}: {currentTaxonomyItem.name} ({currentTaxonomyItem.status})
                  </div>
                  <div className="text-slate-700 leading-relaxed">{currentTaxonomyItem.definition}</div>
                  {currentTaxonomyItem.exclusionNotes && (
                    <div className="text-amber-800 font-medium text-[11px] pt-1">
                      ⚠️ {currentTaxonomyItem.exclusionNotes}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Row 2: Variant & Secondary IDs */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
              {/* Variant */}
              <div className="space-y-1.5">
                <label className="font-semibold text-slate-800 block">
                  Misconception Variant:
                  {!isM && <span className="text-slate-400 font-normal ml-1">(Must be null for non-M classes)</span>}
                </label>
                {isM ? (
                  <select
                    value={form.misconception_variant || ''}
                    onChange={(e) => handleFieldChange('misconception_variant', e.target.value || null)}
                    className="w-full text-xs bg-white border border-slate-200 rounded-lg px-3 py-2 text-slate-800 focus:outline-hidden focus:border-blue-500 font-mono"
                  >
                    <option value="">-- Select Variant --</option>
                    {currentTaxonomyItem?.variants.map((v) => (
                      <option key={v} value={v}>
                        {v}
                      </option>
                    ))}
                  </select>
                ) : (
                  <div className="p-2 border border-slate-200 rounded-lg bg-slate-50 text-slate-400 font-mono text-xs">
                    null (Constraint 7 enforced)
                  </div>
                )}
              </div>

              {/* Error Type */}
              <div className="space-y-1.5">
                <label className="font-semibold text-slate-800 block">
                  Error Type (`error_type`):
                </label>
                <select
                  value={form.error_type}
                  onChange={(e) => handleFieldChange('error_type', e.target.value as ErrorType)}
                  className="w-full text-xs bg-white border border-slate-200 rounded-lg px-3 py-2 text-slate-800 focus:outline-hidden focus:border-blue-500 font-mono capitalize"
                >
                  <option value="conceptual">conceptual (Only valid with M01-M08)</option>
                  <option value="trace_error">trace_error (Expected with INSUFFICIENT)</option>
                  <option value="correct">correct (Required when answer_correct=true)</option>
                  <option value="careless">careless (Allowed with NONE)</option>
                  <option value="typo">typo (Allowed with NONE)</option>
                  <option value="syntax_error">syntax_error (Allowed with NONE)</option>
                  <option value="other">other (For OOS)</option>
                </select>
              </div>
            </div>

            {/* Row 3: Secondary Misconception IDs */}
            <div className="space-y-1.5 text-xs">
              <label className="font-semibold text-slate-800 block">
                Secondary Misconceptions (Max 2, optional):
                {!isM && <span className="text-slate-400 font-normal ml-1">(Must be empty when primary is not M01-M08)</span>}
              </label>

              {isM ? (
                <div className="space-y-2">
                  <div className="flex flex-wrap gap-1.5">
                    {FROZEN_TAXONOMY.filter((t) => t.id !== form.misconception_id).map((t) => {
                      const isSelected = form.secondary_misconception_ids?.includes(t.id);
                      return (
                        <button
                          key={t.id}
                          type="button"
                          onClick={() => toggleSecondaryId(t.id)}
                          className={`px-2.5 py-1 rounded-md text-xs font-mono transition-colors ${
                            isSelected
                              ? 'bg-blue-600 text-white font-bold'
                              : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
                          }`}
                        >
                          {isSelected ? `✓ ${t.id}` : `+ ${t.id}`}
                        </button>
                      );
                    })}
                  </div>
                  <div className="text-[11px] text-slate-500">
                    Selected: [{form.secondary_misconception_ids?.join(', ') || 'none'}] (Rule 7: Max 2, each requires explicit evidence in reasoning)
                  </div>
                </div>
              ) : (
                <div className="p-2 border border-slate-200 rounded-lg bg-slate-50 text-slate-400 font-mono text-xs">
                  [] (Constraint 1 enforced: empty list)
                </div>
              )}
            </div>

            {/* Row 4: Evidence Basis */}
            <div className="space-y-1.5 text-xs">
              <div className="flex items-center justify-between">
                <label className="font-semibold text-slate-800 block">
                  Evidence Basis (`evidence_basis`):
                  {isM && <span className="text-blue-700 ml-1 font-semibold">*Required for M-classes</span>}
                </label>
                <span className="text-[11px] text-slate-400">Points to student's own words or absence</span>
              </div>
              <textarea
                value={form.evidence_basis}
                onChange={(e) => handleFieldChange('evidence_basis', e.target.value)}
                rows={2}
                placeholder="Quote the exact words from student reasoning supporting the label, or state evidence absence..."
                className="w-full text-xs font-mono bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-slate-800 focus:outline-hidden focus:border-blue-500"
              />
            </div>

            {/* Row 5: Annotator Rationale */}
            <div className="space-y-1.5 text-xs">
              <div className="flex items-center justify-between">
                <label className="font-semibold text-slate-800 block">
                  Annotator Rationale (`annotator_rationale`):
                  <span className="text-slate-500 ml-1 font-normal">(Rule 10: Exactly one sentence naming evidence or absence)</span>
                </label>
                <button
                  type="button"
                  onClick={handleGenerateRationaleTemplate}
                  className="flex items-center gap-1 text-[11px] text-blue-600 hover:text-blue-800 font-semibold"
                >
                  <Sparkles className="w-3 h-3" /> Draft Rationale Template
                </button>
              </div>
              <textarea
                value={form.annotator_rationale}
                onChange={(e) => handleFieldChange('annotator_rationale', e.target.value)}
                rows={2}
                placeholder="One sentence justification explicitly naming the student evidence or its absence..."
                className="w-full text-xs bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-slate-800 focus:outline-hidden focus:border-blue-500"
              />
            </div>

            {/* Bottom Commit Bar */}
            <div className="pt-3 border-t border-slate-100 flex items-center justify-between">
              <button
                type="button"
                onClick={onAddNewRecord}
                className="flex items-center gap-1.5 text-xs text-slate-600 hover:text-slate-900 font-medium py-1.5 px-3 rounded-lg border border-slate-200 hover:bg-slate-50"
              >
                <Plus className="w-3.5 h-3.5" /> New Blank Record
              </button>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleSave}
                  className="flex items-center gap-1.5 text-xs bg-blue-600 text-white font-medium py-2 px-5 rounded-lg hover:bg-blue-700 transition-colors shadow-xs"
                >
                  {showSavedFeedback ? <Check className="w-3.5 h-3.5" /> : <Save className="w-3.5 h-3.5" />}
                  {showSavedFeedback ? 'Changes Saved' : 'Save Changes'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
