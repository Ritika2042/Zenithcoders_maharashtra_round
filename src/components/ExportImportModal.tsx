import React, { useState } from 'react';
import { SampleRecord } from '../types/dataset';
import { validateRecord } from '../utils/validator';
import { Download, Upload, Copy, Check, AlertCircle } from 'lucide-react';

interface ExportImportModalProps {
  records: SampleRecord[];
  onImportRecords: (newRecords: SampleRecord[], overwrite: boolean) => void;
  onClose: () => void;
}

export const ExportImportModal: React.FC<ExportImportModalProps> = ({
  records,
  onImportRecords,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<'export' | 'import'>('export');
  const [exportFormat, setExportFormat] = useState<'json' | 'csv'>('json');
  const [importText, setImportText] = useState<string>('');
  const [overwriteMode, setOverwriteMode] = useState<boolean>(false);
  const [importStatus, setImportStatus] = useState<{
    totalParsed: number;
    errorsCount: number;
    parsedRecords: SampleRecord[];
  } | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  // Generate JSON string
  const jsonExportString = JSON.stringify(records, null, 2);

  // Generate CSV string
  const generateCSV = (data: SampleRecord[]) => {
    const headers = [
      'sample_id',
      'question_group',
      'concept',
      'question_format',
      'question',
      'correct_answer',
      'student_answer',
      'student_reasoning',
      'answer_correct',
      'misconception_id',
      'misconception_variant',
      'secondary_misconception_ids',
      'evidence_basis',
      'annotator_rationale',
      'error_type',
      'source',
      'split',
    ];

    const escapeCSV = (val: unknown): string => {
      if (val === null || val === undefined) return '';
      if (Array.isArray(val)) {
        return `"${val.join(';').replace(/"/g, '""')}"`;
      }
      const str = String(val);
      if (str.includes(',') || str.includes('\n') || str.includes('"')) {
        return `"${str.replace(/"/g, '""')}"`;
      }
      return str;
    };

    const rows = data.map((r) =>
      [
        r.sample_id,
        r.question_group,
        r.concept,
        r.question_format,
        r.question,
        r.correct_answer,
        r.student_answer,
        r.student_reasoning,
        r.answer_correct,
        r.misconception_id,
        r.misconception_variant,
        r.secondary_misconception_ids,
        r.evidence_basis,
        r.annotator_rationale,
        r.error_type,
        r.source,
        r.split,
      ]
        .map(escapeCSV)
        .join(',')
    );

    return [headers.join(','), ...rows].join('\n');
  };

  const csvExportString = generateCSV(records);

  const handleDownload = () => {
    const content = exportFormat === 'json' ? jsonExportString : csvExportString;
    const mimeType = exportFormat === 'json' ? 'application/json' : 'text/csv';
    const filename = `cs1_misconceptions_v1_${exportFormat}.${exportFormat}`;

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleCopy = () => {
    const content = exportFormat === 'json' ? jsonExportString : csvExportString;
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleParseImport = () => {
    try {
      const trimmed = importText.trim();
      let parsed: SampleRecord[] = [];

      if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
        const raw = JSON.parse(trimmed);
        parsed = Array.isArray(raw) ? raw : [raw];
      } else {
        // Simple CSV parse
        const lines = trimmed.split('\n');
        if (lines.length < 2) throw new Error('CSV must have a header and at least one data row.');
        const headers = lines[0].split(',').map((h) => h.trim().replace(/^"|"$/g, ''));

        for (let i = 1; i < lines.length; i++) {
          if (!lines[i].trim()) continue;
          // Split on commas not inside quotes
          const regex = /(".*?"|[^",\s]+)(?=\s*,|\s*$)/g;
          const match = lines[i].match(/(?:[^\s",]+|"[^"]*")+/g) || [];
          const record: any = {};
          headers.forEach((h, idx) => {
            let val = match[idx] ? match[idx].replace(/^"|"$/g, '').replace(/""/g, '"') : '';
            if (h === 'answer_correct') {
              record[h] = val === 'true';
            } else if (h === 'secondary_misconception_ids') {
              record[h] = val ? val.split(';').map((s: string) => s.trim()).filter(Boolean) : [];
            } else if (h === 'misconception_variant') {
              record[h] = val === 'null' || !val ? null : val;
            } else {
              record[h] = val;
            }
          });
          parsed.push(record as SampleRecord);
        }
      }

      let errorsCount = 0;
      parsed.forEach((rec) => {
        const issues = validateRecord(rec);
        if (issues.some((i) => i.severity === 'error')) errorsCount++;
      });

      setImportStatus({
        totalParsed: parsed.length,
        errorsCount,
        parsedRecords: parsed,
      });
    } catch (err: any) {
      alert(`Parse Error: ${err.message}`);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white rounded-xl shadow-xl border border-slate-200 w-full max-w-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div>
            <h2 className="text-base font-semibold text-slate-900">Dataset Schema Import & Export</h2>
            <p className="text-xs text-slate-500">Restored locked schema specification (17 fields, frozen v1.0 constraints)</p>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600 text-sm font-medium">
            ✕
          </button>
        </div>

        {/* Tab switch */}
        <div className="flex border-b border-slate-200 px-6 bg-white gap-4">
          <button
            onClick={() => setActiveTab('export')}
            className={`py-3 text-xs font-semibold border-b-2 flex items-center gap-1.5 ${
              activeTab === 'export'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-slate-500 hover:text-slate-900'
            }`}
          >
            <Download className="w-3.5 h-3.5" /> Export Dataset ({records.length} records)
          </button>
          <button
            onClick={() => setActiveTab('import')}
            className={`py-3 text-xs font-semibold border-b-2 flex items-center gap-1.5 ${
              activeTab === 'import'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-slate-500 hover:text-slate-900'
            }`}
          >
            <Upload className="w-3.5 h-3.5" /> Import & Validate
          </button>
        </div>

        {/* Tab content */}
        <div className="p-6 overflow-y-auto space-y-4 flex-1">
          {activeTab === 'export' ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-600 font-medium">Format:</span>
                  <div className="flex items-center bg-slate-100 p-0.5 rounded-lg border border-slate-200">
                    <button
                      type="button"
                      onClick={() => setExportFormat('json')}
                      className={`px-3 py-1 text-xs rounded-md font-medium transition-colors ${
                        exportFormat === 'json' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
                      }`}
                    >
                      JSON
                    </button>
                    <button
                      type="button"
                      onClick={() => setExportFormat('csv')}
                      className={`px-3 py-1 text-xs rounded-md font-medium transition-colors ${
                        exportFormat === 'csv' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
                      }`}
                    >
                      CSV
                    </button>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={handleCopy}
                    className="flex items-center gap-1 text-xs font-medium px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-700"
                  >
                    {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                    {copied ? 'Copied!' : 'Copy to Clipboard'}
                  </button>
                  <button
                    type="button"
                    onClick={handleDownload}
                    className="flex items-center gap-1 text-xs font-medium px-3.5 py-1.5 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors shadow-xs"
                  >
                    <Download className="w-3.5 h-3.5" /> Download File
                  </button>
                </div>
              </div>

              <div>
                <label className="text-xs font-medium text-slate-700 block mb-1">
                  Preview ({exportFormat.toUpperCase()}):
                </label>
                <textarea
                  readOnly
                  value={exportFormat === 'json' ? jsonExportString : csvExportString}
                  rows={12}
                  className="w-full text-xs font-mono bg-slate-50 border border-slate-200 rounded-lg p-3 text-slate-800 focus:outline-hidden"
                />
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <p className="text-xs text-slate-600">
                Paste JSON (array of record objects) or CSV (comma-separated with headers matching the 17 locked fields).
              </p>

              <textarea
                value={importText}
                onChange={(e) => {
                  setImportText(e.target.value);
                  setImportStatus(null);
                }}
                placeholder="Paste JSON or CSV data here..."
                rows={9}
                className="w-full text-xs font-mono bg-slate-50 border border-slate-200 rounded-lg p-3 text-slate-800 focus:outline-hidden focus:border-blue-500"
              />

              <div className="flex items-center justify-between">
                <label className="flex items-center gap-2 text-xs text-slate-700 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={overwriteMode}
                    onChange={(e) => setOverwriteMode(e.target.checked)}
                    className="rounded border-slate-300 text-blue-600 focus:ring-0"
                  />
                  <span>Replace existing dataset completely (otherwise appends)</span>
                </label>

                <button
                  type="button"
                  onClick={handleParseImport}
                  disabled={!importText.trim()}
                  className="px-4 py-1.5 text-xs font-medium rounded-lg bg-slate-900 text-white hover:bg-slate-800 disabled:opacity-50"
                >
                  Parse & Pre-validate
                </button>
              </div>

              {importStatus && (
                <div className={`p-4 rounded-lg border text-xs ${
                  importStatus.errorsCount > 0 ? 'bg-amber-50 border-amber-200 text-amber-900' : 'bg-emerald-50 border-emerald-200 text-emerald-900'
                }`}>
                  <div className="font-semibold flex items-center gap-1.5">
                    {importStatus.errorsCount > 0 ? <AlertCircle className="w-4 h-4 text-amber-600" /> : <Check className="w-4 h-4 text-emerald-600" />}
                    <span>Parsed {importStatus.totalParsed} records</span>
                  </div>
                  <div className="mt-1">
                    {importStatus.errorsCount > 0
                      ? `${importStatus.errorsCount} records have schema constraint warnings or errors. You can still import and inspect them.`
                      : 'All records conform to the v1.0 schema constraints.'}
                  </div>
                  <div className="mt-3">
                    <button
                      type="button"
                      onClick={() => {
                        onImportRecords(importStatus.parsedRecords, overwriteMode);
                        onClose();
                      }}
                      className="px-4 py-1.5 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-md shadow-xs"
                    >
                      Confirm Import ({importStatus.totalParsed} records)
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-200 bg-slate-50 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="text-xs font-medium text-slate-700 px-4 py-1.5 rounded-lg border border-slate-200 hover:bg-white"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
