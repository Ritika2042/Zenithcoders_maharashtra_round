import React, { useState, useEffect } from 'react';
import { SampleRecord, MisconceptionId, ErrorType } from './types/dataset';
import { isMisconceptionClass } from './utils/validator';
import { SEED_DATASET } from './data/seedData';
import { TopNav } from './components/TopNav';
import { AnnotatorStudio } from './components/AnnotatorStudio';
import { DatasetExplorer } from './components/DatasetExplorer';
import { TaxonomyCodex } from './components/TaxonomyCodex';
import { IntegrityAuditor } from './components/IntegrityAuditor';
import { ExportImportModal } from './components/ExportImportModal';
import { DecisionWizard } from './components/DecisionWizardModal';
import { InterventionStudio } from './components/InterventionStudio';

const STORAGE_KEY = 'cs1_misconceptions_dataset_v2';

export default function App() {
  const [records, setRecords] = useState<SampleRecord[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          // If R0018-R0020 are not present, append them from SEED_DATASET
          const existingIds = new Set(parsed.map((r: SampleRecord) => r.sample_id));
          const missing = SEED_DATASET.filter((r) => !existingIds.has(r.sample_id));
          if (missing.length > 0) {
            return [...parsed, ...missing];
          }
          return parsed;
        }
      }
    } catch (e) {
      console.error('Failed to load records from storage', e);
    }
    return SEED_DATASET;
  });

  const [activeView, setActiveView] = useState<'relearn' | 'studio' | 'explorer' | 'codex' | 'auditor'>('relearn');
  const [selectedRecordId, setSelectedRecordId] = useState<string>(() => records[0]?.sample_id || 'CS1-M01-001');
  const [isExportImportOpen, setIsExportImportOpen] = useState<boolean>(false);
  const [isDecisionWizardOpen, setIsDecisionWizardOpen] = useState<boolean>(false);

  // Sync to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
    } catch (e) {
      console.error('Failed to save to localStorage', e);
    }
  }, [records]);

  const currentRecord = records.find((r) => r.sample_id === selectedRecordId) || records[0];

  const handleSaveRecord = (updated: SampleRecord) => {
    setRecords((prev) => prev.map((r) => (r.sample_id === updated.sample_id ? updated : r)));
  };

  const handleAddNewRecord = () => {
    const nextId = `CS1-NEW-${String(records.length + 1).padStart(3, '0')}`;
    const newRecord: SampleRecord = {
      sample_id: nextId,
      question_group: 'general',
      concept: 'python_fundamentals',
      question_format: 'predict_output',
      question: '# Enter Python snippet here\nx = 1\nprint(x)',
      correct_answer: '1',
      student_answer: '',
      student_reasoning: '',
      answer_correct: false,
      misconception_id: 'INSUFFICIENT',
      misconception_variant: null,
      secondary_misconception_ids: [],
      evidence_basis: 'absence of stated student reasoning',
      annotator_rationale: 'Student provided no reasoning, so evidence is insufficient to code an M-label under Rule 1.',
      error_type: 'other',
      source: 'Instructor Assessment',
      split: 'train',
    };

    setRecords((prev) => [newRecord, ...prev]);
    setSelectedRecordId(nextId);
    setActiveView('studio');
  };

  const handleDuplicateRecord = (rec: SampleRecord) => {
    const dupId = `${rec.sample_id}-COPY`;
    const duplicated: SampleRecord = {
      ...rec,
      sample_id: dupId,
    };
    setRecords((prev) => [duplicated, ...prev]);
    setSelectedRecordId(dupId);
    setActiveView('studio');
  };

  const handleDeleteRecord = (sampleId: string) => {
    if (records.length <= 1) {
      alert('Cannot delete the last remaining record in the dataset.');
      return;
    }
    const filtered = records.filter((r) => r.sample_id !== sampleId);
    setRecords(filtered);
    if (selectedRecordId === sampleId) {
      setSelectedRecordId(filtered[0].sample_id);
    }
  };

  const handleImportRecords = (newRecords: SampleRecord[], overwrite: boolean) => {
    if (overwrite) {
      setRecords(newRecords);
      if (newRecords.length > 0) setSelectedRecordId(newRecords[0].sample_id);
    } else {
      // Append without duplicate IDs
      const existingIds = new Set(records.map((r) => r.sample_id));
      const filteredNew = newRecords.map((r) => {
        if (existingIds.has(r.sample_id)) {
          return { ...r, sample_id: `${r.sample_id}-IMP` };
        }
        return r;
      });
      setRecords((prev) => [...prev, ...filteredNew]);
    }
  };

  const handleResetToSeed = () => {
    if (confirm('Reset dataset to the default 17 curated benchmark cases?')) {
      setRecords(SEED_DATASET);
      setSelectedRecordId(SEED_DATASET[0].sample_id);
      localStorage.removeItem(STORAGE_KEY);
    }
  };

  const handleAddViolationSample = () => {
    const violationId = `CS1-ERR-TEST-${records.length + 1}`;
    const brokenRecord: SampleRecord = {
      sample_id: violationId,
      question_group: 'syntax_and_types',
      concept: 'type_conversion',
      question_format: 'predict_output',
      question: 'x = "10"\ny = 5\nprint(x + y)',
      correct_answer: 'TypeError',
      student_answer: '15',
      student_reasoning: 'Adding string and int works like normal math.',
      answer_correct: false,
      misconception_id: 'NONE', // VIOLATION: NONE with error_type = conceptual and non-empty secondary_misconception_ids
      misconception_variant: 'treats_numeric_string_as_int', // VIOLATION: variant on non-M class
      secondary_misconception_ids: ['M05'], // VIOLATION: secondary on non-M class
      evidence_basis: 'Adding string and int works like normal math',
      annotator_rationale: 'Student had a misconception.',
      error_type: 'conceptual', // VIOLATION: conceptual error_type on NONE
      source: 'Synthetic Constraint Test',
      split: 'test',
    };

    setRecords((prev) => [brokenRecord, ...prev]);
    setSelectedRecordId(violationId);
    setActiveView('auditor');
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans antialiased selection:bg-blue-100 selection:text-blue-900">
      {/* 3-Zone Top Bar Contract */}
      <TopNav
        activeView={activeView}
        onNavigate={(view) => setActiveView(view)}
        onAddNewRecord={handleAddNewRecord}
        onOpenExportImport={() => setIsExportImportOpen(true)}
        recordCount={records.length}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6">
        {activeView === 'relearn' && <InterventionStudio />}

        {activeView === 'studio' && currentRecord && (
          <AnnotatorStudio
            currentRecord={currentRecord}
            records={records}
            onSaveRecord={handleSaveRecord}
            onSelectRecordById={(id) => setSelectedRecordId(id)}
            onOpenDecisionWizard={() => setIsDecisionWizardOpen(true)}
            onAddNewRecord={handleAddNewRecord}
          />
        )}

        {activeView === 'explorer' && (
          <DatasetExplorer
            records={records}
            onSelectRecord={(r) => {
              setSelectedRecordId(r.sample_id);
              setActiveView('studio');
            }}
            onDuplicateRecord={handleDuplicateRecord}
            onDeleteRecord={handleDeleteRecord}
            onAddNewRecord={handleAddNewRecord}
          />
        )}

        {activeView === 'codex' && <TaxonomyCodex />}

        {activeView === 'auditor' && (
          <IntegrityAuditor
            records={records}
            onSelectRecordToEdit={(id) => {
              setSelectedRecordId(id);
              setActiveView('studio');
            }}
            onAddViolationSample={handleAddViolationSample}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-4 px-6 text-xs text-slate-500 mt-auto">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-700">Python CS1 Misconceptions Workbench</span>
            <span>·</span>
            <span>Taxonomy v1.0 Frozen</span>
            <span>·</span>
            <span className="tabular-nums font-mono">{records.length} Records</span>
          </div>

          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={handleResetToSeed}
              className="text-slate-500 hover:text-slate-900 transition-colors"
            >
              Reset to Standard Seed Dataset
            </button>
            <button
              type="button"
              onClick={() => setIsExportImportOpen(true)}
              className="text-blue-600 hover:text-blue-800 font-medium"
            >
              Export Locked Schema (CSV/JSON)
            </button>
          </div>
        </div>
      </footer>

      {/* Export / Import Modal */}
      {isExportImportOpen && (
        <ExportImportModal
          records={records}
          onImportRecords={handleImportRecords}
          onClose={() => setIsExportImportOpen(false)}
        />
      )}

      {/* Decision Wizard Modal */}
      {isDecisionWizardOpen && (
        <DecisionWizard
          onClose={() => setIsDecisionWizardOpen(false)}
          onApplyRecommendation={(rec) => {
            if (currentRecord) {
              const updated: SampleRecord = {
                ...currentRecord,
                misconception_id: rec.misconception_id,
                error_type: rec.error_type,
                annotator_rationale: rec.rationaleSuggestion,
                misconception_variant: isMisconceptionClass(rec.misconception_id)
                  ? currentRecord.misconception_variant || null
                  : null,
                secondary_misconception_ids: isMisconceptionClass(rec.misconception_id)
                  ? currentRecord.secondary_misconception_ids || []
                  : [],
              };
              handleSaveRecord(updated);
            }
          }}
        />
      )}
    </div>
  );
}
