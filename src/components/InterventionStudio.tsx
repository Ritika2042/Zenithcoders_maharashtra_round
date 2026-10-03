import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  ArrowDown,
  RotateCcw,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  HelpCircle,
  BookOpen,
  Code2,
  Terminal,
  Activity,
  Layers,
  Copy,
  Check,
  Send,
  Info
} from 'lucide-react';

export interface DiagnosisResult {
  misconception_id: string;
  confidence: number;
  evidence: string;
  rationale: string;
  decision_source: 'model' | 'evidence_rule' | 'abstention_rule' | string;
}

export interface InterventionData {
  status: string;
  misconception_id: string;
  title: string;
  short_explanation: string;
  concrete_example?: string;
  contrast_example?: string;
  key_rule: string;
  pedagogical_basis?: string;
  suggested_action?: string;
}

export interface FollowupQuestion {
  question: string;
  expected_answer: string;
  evaluation_type?: string;
  accepted_keywords?: string[];
  forbidden_patterns?: string[];
}

export interface SecondInterventionData {
  status: string;
  misconception_id: string;
  title: string;
  pedagogical_approach: string;
  explanation: string;
  step_by_step_trace: string;
  key_takeaway: string;
  reference_resource?: string;
}

export interface ResolutionData {
  status: 'RESOLVED' | 'NOT_RESOLVED';
  confidence: number;
  evidence: string;
  rationale: string;
  persistent_misconception?: boolean;
}

export interface PipelineSession {
  context: {
    question: string;
    correct_answer: string;
    student_answer: string;
    student_reasoning: string;
  };
  diagnosis: DiagnosisResult;
  intervention: InterventionData | null;
  followup: FollowupQuestion | null;
  status: string;
  message?: string;
  next_action: string;
}

export interface ResolutionResponse {
  context: any;
  diagnosis: DiagnosisResult;
  intervention: InterventionData | null;
  followup: FollowupQuestion | null;
  student_followup_submission: {
    answer: string;
    reasoning: string;
  };
  resolution: ResolutionData;
  next_action: string;
  second_intervention?: SecondInterventionData;
}

interface PresetCase {
  id: string;
  tag: string;
  title: string;
  category: 'misconception' | 'abstention';
  question: string;
  correct_answer: string;
  student_answer: string;
  student_reasoning: string;
  correct_followup?: {
    answer: string;
    reasoning: string;
  };
  incorrect_followup?: {
    answer: string;
    reasoning: string;
  };
}

/**
 * Benchmark Presets matching the EXACT verified scenarios from baseline_pipeline.py & intervention_engine.py
 */
export const PRESET_CASES: PresetCase[] = [
  {
    id: 'M01',
    tag: 'M01',
    title: 'String vs Number Type Confusion',
    category: 'misconception',
    question: 'x = "5"\ny = 2\nprint(x + y)',
    correct_answer: 'TypeError',
    student_answer: '7',
    student_reasoning: 'The quotes around "5" don\'t matter; Python treats it as integer 5 so adding gives 7.',
    correct_followup: {
      answer: '34',
      reasoning: 'Both are strings in quotes, so the + operator concatenates them into "34".'
    },
    incorrect_followup: {
      answer: '7',
      reasoning: 'String + string adds to 7 like regular numbers.'
    }
  },
  {
    id: 'M02',
    tag: 'M02',
    title: 'Division Semantics (/ vs //)',
    category: 'misconception',
    question: 'print(7 // 2)',
    correct_answer: '3',
    student_answer: '3.5',
    student_reasoning: 'Floor division computes division normally and keeps the decimal portion as 3.5.',
    correct_followup: {
      answer: '2',
      reasoning: '11 // 4 performs floor division, which discards the remainder and rounds down to the whole integer 2.'
    },
    incorrect_followup: {
      answer: '2.75',
      reasoning: 'Floor division keeps decimal so 2.75.'
    }
  },
  {
    id: 'M03',
    tag: 'M03',
    title: 'Assignment vs Equality (= vs ==)',
    category: 'misconception',
    question: 'if x = 5:\n    print(True)',
    correct_answer: 'SyntaxError',
    student_answer: 'True',
    student_reasoning: 'The single equals sign = tests whether x equals 5, so it compares the values.',
    correct_followup: {
      answer: 'False',
      reasoning: 'score == 50 is an equality comparison check, and since 100 != 50 it evaluates to False without modifying score.'
    },
    incorrect_followup: {
      answer: 'True',
      reasoning: 'score == 50 sets score to 50 so it assigns 50 and is True.'
    }
  },
  {
    id: 'M04',
    tag: 'M04',
    title: 'Operator Precedence',
    category: 'misconception',
    question: 'val = 6\nif val & 2 == 2:\n    print("Bit set")',
    correct_answer: 'Bit not set',
    student_answer: 'Bit set',
    student_reasoning: 'The & operator evaluates before == because bitwise has higher precedence than comparison.',
    correct_followup: {
      answer: '4',
      reasoning: 'Multiplication has higher precedence than subtraction, so 2 * 3 evaluates first to 6, then 10 - 6 gives 4.'
    },
    incorrect_followup: {
      answer: '24',
      reasoning: 'Left to right subtraction first gives 10 - 2 = 8, then 8 * 3 = 24.'
    }
  },
  {
    id: 'M05',
    tag: 'M05',
    title: 'Index / Position',
    category: 'misconception',
    question: 's = "Python"\nprint(s[1])',
    correct_answer: 'y',
    student_answer: 'P',
    student_reasoning: 'Indexing starts at 1, so index 1 refers to the first letter of the string.',
    correct_followup: {
      answer: 'red',
      reasoning: 'Python uses 0-based indexing, so index 0 accesses the initial first item "red".'
    },
    incorrect_followup: {
      answer: 'green',
      reasoning: 'Starts at 1 so index 0 is invalid or accesses green.'
    }
  },
  {
    id: 'M06',
    tag: 'M06',
    title: 'Loop Values / Boundaries / Iteration Interval',
    category: 'misconception',
    question: 'for x in range(1, 5):\n    print(x)',
    correct_answer: '1 2 3 4',
    student_answer: '1 2 3 4 5',
    student_reasoning: 'range(1, 5) includes 5 because the stop value is inclusive.',
    correct_followup: {
      answer: '[2, 3, 4]',
      reasoning: 'The stop boundary 5 in range(2, 5) is exclusive, so the loop stops before 5 and produces [2, 3, 4].'
    },
    incorrect_followup: {
      answer: '[2, 3, 4, 5]',
      reasoning: 'range includes 5 because the stop value is inclusive.'
    }
  },
  {
    id: 'M07',
    tag: 'M07',
    title: 'Function Argument–Parameter Binding',
    category: 'misconception',
    question: 'def greet(first, last):\n    print(first, last)\ngreet(last="Smith", "John")',
    correct_answer: 'SyntaxError',
    student_answer: 'John Smith',
    student_reasoning: 'Parameters are mapped by variable name regardless of whether positional or keyword syntax is used.',
    correct_followup: {
      answer: '-8',
      reasoning: 'Positional arguments bind by left-to-right order: first argument y=2 binds to a, and x=10 binds to b, so 2 - 10 gives -8.'
    },
    incorrect_followup: {
      answer: '8',
      reasoning: 'Caller names match parameter names so x is 10 and y is 2, giving 8.'
    }
  },
  {
    id: 'M08',
    tag: 'M08',
    title: 'Recursion Termination / Base Case',
    category: 'misconception',
    question: 'def f(n):\n    if n == 0:\n        return 0\n    return f(n - 1)\nprint(f(3))',
    correct_answer: '0',
    student_answer: 'RecursionError',
    student_reasoning: 'The function calls itself recursively without ever reaching an exit, believing recursion stops automatically without base cases.',
    correct_followup: {
      answer: 'Go!',
      reasoning: 'When n decrements to 1, the base case if n == 1 triggers and returns "Go!", terminating the recursion safely.'
    },
    incorrect_followup: {
      answer: 'RecursionError',
      reasoning: 'Recursion never stops and never reaches exit because recursion is infinite.'
    }
  },
  {
    id: 'NONE',
    tag: 'NONE',
    title: 'No Misconception (Typo / Arithmetic Slip)',
    category: 'abstention',
    question: 'total = 14 + 8\nprint(total)',
    correct_answer: '22',
    student_answer: '21',
    student_reasoning: 'I meant to type 22 but hit the 1 key by accident; 14 + 8 is 22.'
  },
  {
    id: 'INSUFFICIENT',
    tag: 'INSUFFICIENT',
    title: 'Insufficient Evidence (Guess / Unexplained)',
    category: 'abstention',
    question: 'total = 0\nfor i in range(3):\n    for j in range(2):\n        total += i * j\nprint(total)',
    correct_answer: '3',
    student_answer: '7',
    student_reasoning: 'I guessed 7.'
  },
  {
    id: 'OOS',
    tag: 'OOS',
    title: 'Out of Scope (Aliasing / Mutability)',
    category: 'abstention',
    question: 'a = [1, 2]\nb = a\nb.append(3)\nprint(a)',
    correct_answer: '[1, 2, 3]',
    student_answer: '[1, 2]',
    student_reasoning: 'Setting b = a creates a separate copy of the list, so appending to b does not touch list a because aliasing does not happen.'
  }
];

export const InterventionStudio: React.FC = () => {
  // Input fields
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>('M01');
  const [question, setQuestion] = useState<string>(PRESET_CASES[0].question);
  const [correctAnswer, setCorrectAnswer] = useState<string>(PRESET_CASES[0].correct_answer);
  const [studentAnswer, setStudentAnswer] = useState<string>(PRESET_CASES[0].student_answer);
  const [studentReasoning, setStudentReasoning] = useState<string>(PRESET_CASES[0].student_reasoning);

  // Stage 2 & 3: Diagnosis & Intervention
  const [session, setSession] = useState<PipelineSession | null>(null);
  const [isDiagnosing, setIsDiagnosing] = useState<boolean>(false);
  const [diagnoseError, setDiagnoseError] = useState<string | null>(null);
  const [diagnoseLatency, setDiagnoseLatency] = useState<number | null>(null);

  // Stage 4: Follow-up input
  const [followupAnswer, setFollowupAnswer] = useState<string>('');
  const [followupReasoning, setFollowupReasoning] = useState<string>('');

  // Stage 5: Resolution Assessment
  const [resolutionResponse, setResolutionResponse] = useState<ResolutionResponse | null>(null);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [evaluateError, setEvaluateError] = useState<string | null>(null);
  const [evaluateLatency, setEvaluateLatency] = useState<number | null>(null);

  // UI state
  const [copiedCode, setCopiedCode] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'offline'>('checking');

  // Check backend health on mount
  useEffect(() => {
    fetch('/api/relearn/health')
      .then((r) => r.json())
      .then(() => setBackendStatus('connected'))
      .catch(() => setBackendStatus('offline'));
  }, []);

  // Handle manual input modification
  const handleInputChange = (field: 'question' | 'correctAnswer' | 'studentAnswer' | 'studentReasoning', val: string) => {
    // Clear active preset indicator when user modifies any field
    setSelectedPresetId(null);
    setSession(null);
    setResolutionResponse(null);
    setDiagnoseError(null);
    setEvaluateError(null);
    setFollowupAnswer('');
    setFollowupReasoning('');

    if (field === 'question') setQuestion(val);
    else if (field === 'correctAnswer') setCorrectAnswer(val);
    else if (field === 'studentAnswer') setStudentAnswer(val);
    else if (field === 'studentReasoning') setStudentReasoning(val);
  };

  // Handle Preset selection
  const handleSelectPreset = (preset: PresetCase) => {
    setSelectedPresetId(preset.id);
    setQuestion(preset.question);
    setCorrectAnswer(preset.correct_answer);
    setStudentAnswer(preset.student_answer);
    setStudentReasoning(preset.student_reasoning);

    // Clear previous downstream stages
    setSession(null);
    setResolutionResponse(null);
    setDiagnoseError(null);
    setEvaluateError(null);
    setFollowupAnswer('');
    setFollowupReasoning('');
  };

  // Quick fill helper for follow-up inputs based on current active diagnosis
  const handleQuickFillFollowup = (type: 'correct' | 'incorrect') => {
    const targetId = session?.diagnosis?.misconception_id || selectedPresetId;
    if (!targetId) return;
    const preset = PRESET_CASES.find((p) => p.id === targetId);
    if (!preset) return;
    if (type === 'correct' && preset.correct_followup) {
      setFollowupAnswer(preset.correct_followup.answer);
      setFollowupReasoning(preset.correct_followup.reasoning);
    } else if (type === 'incorrect' && preset.incorrect_followup) {
      setFollowupAnswer(preset.incorrect_followup.answer);
      setFollowupReasoning(preset.incorrect_followup.reasoning);
    }
  };

  // Execute Stage 1 -> Stage 2 & 3: Diagnose & Intervene
  const handleDiagnoseAndIntervene = async () => {
    setIsDiagnosing(true);
    setDiagnoseError(null);
    setResolutionResponse(null);
    setFollowupAnswer('');
    setFollowupReasoning('');
    const startTime = performance.now();

    const payload = {
      question: question.trim(),
      correct_answer: correctAnswer.trim(),
      student_answer: studentAnswer.trim(),
      student_reasoning: studentReasoning.trim(),
    };

    console.log('[ReLearn UI] Diagnose & Intervene payload:', payload);

    try {
      const response = await fetch('/api/relearn/diagnose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Diagnosis server error (HTTP ${response.status})`);
      }

      const data: PipelineSession = await response.json();
      console.log('[ReLearn UI] Diagnosis response received:', data);
      setSession(data);
      setDiagnoseLatency(Math.round(performance.now() - startTime));
      setBackendStatus('connected');
    } catch (err: any) {
      console.error('API call failed:', err);
      setDiagnoseError(err.message || 'Diagnosis failed. Check backend status.');
      setBackendStatus('offline');
    } finally {
      setIsDiagnosing(false);
    }
  };

  // Execute Stage 4 -> Stage 5: Evaluate Follow-up Resolution
  const handleEvaluateFollowup = async () => {
    if (!session) return;
    setIsEvaluating(true);
    setEvaluateError(null);
    const startTime = performance.now();

    try {
      const response = await fetch('/api/relearn/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session,
          followup_answer: followupAnswer,
          followup_reasoning: followupReasoning,
        }),
      });

      if (!response.ok) {
        throw new Error(`Evaluation server error (HTTP ${response.status})`);
      }

      const data: ResolutionResponse = await response.json();
      setResolutionResponse(data);
      setEvaluateLatency(Math.round(performance.now() - startTime));
      setBackendStatus('connected');
    } catch (err: any) {
      console.error('Follow-up evaluation failed:', err);
      setEvaluateError(err.message || 'Evaluation failed.');
      setBackendStatus('offline');
    } finally {
      setIsEvaluating(false);
    }
  };

  // Reset all state
  const handleReset = () => {
    const defaultPreset = PRESET_CASES[0];
    handleSelectPreset(defaultPreset);
  };

  // Copy code helper
  const handleCopyCode = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 1800);
  };

  const currentPreset = PRESET_CASES.find((p) => p.id === selectedPresetId);
  const isAbstention = session?.diagnosis?.misconception_id && ['NONE', 'INSUFFICIENT', 'OOS'].includes(session.diagnosis.misconception_id);

  return (
    <div className="space-y-8 pb-16">
      {/* Page Header */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1.5 text-xs text-slate-500 font-medium">
              <span>Person 1 Misconception Classifier</span>
              <span aria-hidden="true">·</span>
              <span>Person 2 Adaptive Intervention Engine</span>
              <span aria-hidden="true">·</span>
              <span className="flex items-center gap-1">
                <span
                  className={`w-2 h-2 rounded-full ${
                    backendStatus === 'connected'
                      ? 'bg-emerald-500'
                      : backendStatus === 'checking'
                      ? 'bg-amber-400'
                      : 'bg-rose-500'
                  }`}
                />
                <span className="capitalize">{backendStatus} Backend</span>
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900">
              ReLearn — Adaptive Learning
            </h1>
            <p className="mt-1 text-sm text-slate-600 font-normal">
              Understand the mistake. Target the misconception. Verify learning.
            </p>
          </div>

          <div className="flex items-center gap-2 self-start md:self-center">
            <button
              type="button"
              onClick={handleReset}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors shadow-xs"
            >
              <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
              <span>Reset / Clear</span>
            </button>
            <button
              type="button"
              onClick={handleDiagnoseAndIntervene}
              disabled={isDiagnosing}
              className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors shadow-xs"
            >
              {isDiagnosing ? (
                <>
                  <Activity className="w-4 h-4 animate-spin" />
                  <span>Diagnosing...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Diagnose & Intervene</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Preset Selector */}
        <div className="mt-6 pt-5 border-t border-slate-100">
          <div className="flex items-center justify-between mb-2.5">
            <span className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
              1-Click Benchmark Presets (11 Verified Canonical Cases)
            </span>
            <span className="text-[11px] text-slate-500">
              Click any benchmark to load verified student response
            </span>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {PRESET_CASES.map((preset) => {
              const isActive = selectedPresetId === preset.id;
              const isAbst = preset.category === 'abstention';
              return (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => handleSelectPreset(preset)}
                  className={`px-2.5 py-1.5 text-xs font-medium rounded-lg transition-all border ${
                    isActive
                      ? 'bg-blue-600 text-white border-blue-600 shadow-xs'
                      : isAbst
                      ? 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                      : 'bg-white text-slate-700 border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                  }`}
                >
                  <span className="font-mono font-bold mr-1">{preset.tag}</span>
                  <span className="text-[11px] opacity-90">{preset.title.split(' ')[0]}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* PIPELINE STAGES CONTAINER */}
      <div className="space-y-6">
        {/* ============================================================== */}
        {/* STAGE 1: STUDENT RESPONSE */}
        {/* ============================================================== */}
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
          <div className="px-5 py-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="flex items-center justify-center w-5 h-5 rounded-full bg-blue-100 text-blue-700 text-xs font-bold font-mono">
                1
              </span>
              <h2 className="text-sm font-bold tracking-tight text-slate-900 uppercase">
                Student Response Input
              </h2>
            </div>
            <div className="text-xs text-slate-500 flex items-center gap-2">
              <span>Source Mode:</span>
              {selectedPresetId && currentPreset ? (
                <span className="font-semibold text-slate-800 font-mono">
                  Preset: {currentPreset.tag} — {currentPreset.title}
                </span>
              ) : (
                <span className="font-semibold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-200 font-mono text-[11px]">
                  Custom User Input
                </span>
              )}
            </div>
          </div>

          <div className="p-5 grid grid-cols-1 lg:grid-cols-2 gap-5">
            {/* Left Column: Code Question & Expected */}
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
                    <Code2 className="w-3.5 h-3.5 text-slate-500" />
                    <span>Python Code Question</span>
                  </label>
                  <button
                    type="button"
                    onClick={() => handleCopyCode(question, 'question')}
                    className="text-[11px] text-slate-500 hover:text-slate-800 flex items-center gap-1"
                  >
                    {copiedCode === 'question' ? <Check className="w-3 h-3 text-emerald-600" /> : <Copy className="w-3 h-3" />}
                    <span>{copiedCode === 'question' ? 'Copied' : 'Copy'}</span>
                  </button>
                </div>
                <textarea
                  rows={4}
                  value={question}
                  onChange={(e) => handleInputChange('question', e.target.value)}
                  className="w-full font-mono text-xs p-3 bg-slate-900 text-slate-100 rounded-lg border border-slate-800 focus:outline-hidden focus:ring-2 focus:ring-blue-500"
                  placeholder="# Python code snippet"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Correct Answer (Ground Truth)
                </label>
                <input
                  type="text"
                  value={correctAnswer}
                  onChange={(e) => handleInputChange('correctAnswer', e.target.value)}
                  className="w-full font-mono text-xs px-3 py-2 bg-slate-50 text-slate-900 rounded-lg border border-slate-200 focus:outline-hidden focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>

            {/* Right Column: Student Answer & Reasoning */}
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Student Submitted Answer
                </label>
                <input
                  type="text"
                  value={studentAnswer}
                  onChange={(e) => handleInputChange('studentAnswer', e.target.value)}
                  className="w-full font-mono text-xs px-3 py-2 bg-white text-slate-900 rounded-lg border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g. 7 or [1, 2, 3, 4, 5]"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5 flex items-center justify-between">
                  <span>Student Stated Reasoning (Key Diagnostic Evidence)</span>
                  <span className="text-[11px] font-normal text-slate-500">Editable for custom live testing</span>
                </label>
                <textarea
                  rows={4}
                  value={studentReasoning}
                  onChange={(e) => handleInputChange('studentReasoning', e.target.value)}
                  className="w-full text-xs p-3 bg-white text-slate-900 rounded-lg border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-blue-500"
                  placeholder="Explain why you chose this answer..."
                />
              </div>
            </div>
          </div>

          <div className="px-5 py-3 bg-slate-50 border-t border-slate-200 flex flex-wrap items-center justify-between gap-3">
            <span className="text-xs text-slate-500">
              Click <strong className="text-slate-700">Diagnose & Intervene</strong> to run Person 1 Misconception Classifier + Person 2 Adaptive Remediation.
            </span>
            <button
              type="button"
              onClick={handleDiagnoseAndIntervene}
              disabled={isDiagnosing}
              className="flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors shadow-xs ml-auto"
            >
              {isDiagnosing ? <Activity className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
              <span>Diagnose & Intervene</span>
              <ArrowDown className="w-3.5 h-3.5 ml-0.5" />
            </button>
          </div>
        </div>

        {diagnoseError && (
          <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-800 flex items-start gap-2.5">
            <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold">Diagnosis Error</div>
              <div className="mt-0.5">{diagnoseError}</div>
            </div>
          </div>
        )}

        {/* Connector Arrow */}
        {session && (
          <div className="flex justify-center my-2 text-slate-400">
            <ArrowDown className="w-5 h-5 animate-pulse" />
          </div>
        )}

        {/* ============================================================== */}
        {/* STAGE 2: MISCONCEPTION DETECTION (PERSON 1) */}
        {/* ============================================================== */}
        {session && (
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
            <div className="px-5 py-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="flex items-center justify-center w-5 h-5 rounded-full bg-indigo-100 text-indigo-700 text-xs font-bold font-mono">
                  2
                </span>
                <h2 className="text-sm font-bold tracking-tight text-slate-900 uppercase">
                  Misconception Detection (Person 1 Diagnosis)
                </h2>
              </div>
              <div className="flex items-center gap-3 text-xs">
                {diagnoseLatency !== null && (
                  <span className="text-slate-500 font-mono text-[11px]">
                    Latency: {diagnoseLatency}ms
                  </span>
                )}
                <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-200 text-slate-700 font-semibold">
                  Source: {session.diagnosis.decision_source}
                </span>
              </div>
            </div>

            <div className="p-5 space-y-5">
              {/* Header metrics card */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between p-4 bg-slate-50 border border-slate-200 rounded-xl gap-4">
                <div>
                  <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                    Authoritative Diagnosis
                  </div>
                  <div className="flex items-center gap-2.5 mt-1">
                    <span className="font-mono font-extrabold text-xl text-blue-600">
                      {session.diagnosis.misconception_id}
                    </span>
                    <span className="text-slate-400">·</span>
                    <span className="font-bold text-slate-900 text-base">
                      {session.intervention?.title ||
                        (session.diagnosis.misconception_id === 'NONE'
                          ? 'Sound Conceptual Grasp (No Misconception)'
                          : session.diagnosis.misconception_id === 'INSUFFICIENT'
                          ? 'Insufficient Evidence (Abstention)'
                          : 'Out of Scope (Topic Beyond M01-M08)')}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-6">
                  <div>
                    <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                      Confidence
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <div className="w-20 bg-slate-200 h-2.5 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${
                            session.diagnosis.confidence >= 0.7
                              ? 'bg-emerald-500'
                              : session.diagnosis.confidence >= 0.4
                              ? 'bg-blue-500'
                              : 'bg-amber-500'
                          }`}
                          style={{ width: `${Math.round(session.diagnosis.confidence * 100)}%` }}
                        />
                      </div>
                      <span className="font-mono font-bold text-xs text-slate-700">
                        {Math.round(session.diagnosis.confidence * 100)}%
                      </span>
                    </div>
                  </div>

                  <div>
                    <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                      Decision Source
                    </div>
                    <div className="font-mono text-xs font-semibold text-slate-800 mt-1 capitalize">
                      {session.diagnosis.decision_source}
                    </div>
                  </div>
                </div>
              </div>

              {/* Evidence & Rationale */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-white border border-slate-200 rounded-xl space-y-1.5">
                  <div className="text-xs font-bold text-slate-700 flex items-center gap-1.5">
                    <Terminal className="w-3.5 h-3.5 text-slate-500" />
                    <span>Evidence Extracted From Student Reasoning</span>
                  </div>
                  <div className="text-xs text-slate-700 bg-amber-50/60 border border-amber-200/60 rounded-lg p-2.5 font-mono">
                    {session.diagnosis.evidence || 'No explicit textual markers detected.'}
                  </div>
                </div>

                <div className="p-4 bg-white border border-slate-200 rounded-xl space-y-1.5">
                  <div className="text-xs font-bold text-slate-700 flex items-center gap-1.5">
                    <BookOpen className="w-3.5 h-3.5 text-slate-500" />
                    <span>Diagnostic Rationale</span>
                  </div>
                  <div className="text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded-lg p-2.5">
                    {session.diagnosis.rationale}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Connector Arrow */}
        {session && (
          <div className="flex justify-center my-2 text-slate-400">
            <ArrowDown className="w-5 h-5 animate-pulse" />
          </div>
        )}

        {/* ============================================================== */}
        {/* STAGE 3: TARGETED INTERVENTION / ROUTING GUIDANCE (PERSON 2) */}
        {/* ============================================================== */}
        {session && (
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
            <div className="px-5 py-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="flex items-center justify-center w-5 h-5 rounded-full bg-emerald-100 text-emerald-700 text-xs font-bold font-mono">
                  3
                </span>
                <h2 className="text-sm font-bold tracking-tight text-slate-900 uppercase">
                  Targeted Pedagogical Intervention (Person 2)
                </h2>
              </div>
              <div className="text-xs font-mono font-medium text-slate-600 bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded border border-emerald-200">
                Action: {session.next_action}
              </div>
            </div>

            <div className="p-5 space-y-5">
              {/* If M01-M08: Active Targeted Intervention */}
              {session.intervention ? (
                <>
                  {/* Strategy & Short Explanation */}
                  <div className="p-4 bg-blue-50/70 border border-blue-200 rounded-xl">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-bold text-blue-900 uppercase tracking-wider">
                        Strategy: Direct Conceptual Remediation
                      </span>
                      <span className="text-[11px] font-mono font-semibold text-blue-700">
                        {session.intervention.title}
                      </span>
                    </div>
                    <p className="text-xs text-blue-950 leading-relaxed font-medium mt-1">
                      {session.intervention.short_explanation}
                    </p>
                  </div>

                  {/* Concrete vs Contrast Examples */}
                  {session.intervention.concrete_example && session.intervention.contrast_example && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Concrete Example */}
                      <div className="border border-slate-200 rounded-xl p-4 bg-slate-900 text-slate-100">
                        <div className="text-xs font-bold text-emerald-400 mb-2 flex items-center justify-between">
                          <span>Concrete Model</span>
                          <span className="text-[10px] text-slate-400 uppercase font-mono">Execution Flow</span>
                        </div>
                        <pre className="text-xs font-mono whitespace-pre-wrap leading-relaxed text-slate-200">
                          {session.intervention.concrete_example}
                        </pre>
                      </div>

                      {/* Contrast Example */}
                      <div className="border border-slate-200 rounded-xl p-4 bg-slate-900 text-slate-100">
                        <div className="text-xs font-bold text-amber-400 mb-2 flex items-center justify-between">
                          <span>Contrast Comparison</span>
                          <span className="text-[10px] text-slate-400 uppercase font-mono">Side-by-Side</span>
                        </div>
                        <pre className="text-xs font-mono whitespace-pre-wrap leading-relaxed text-slate-200">
                          {session.intervention.contrast_example}
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* Key Rule */}
                  <div className="p-3.5 bg-slate-50 border-l-4 border-l-blue-600 border-r border-t border-b border-slate-200 rounded-r-xl flex items-start gap-3">
                    <Sparkles className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
                    <div>
                      <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                        Key Grounding Rule
                      </div>
                      <div className="text-xs font-semibold text-slate-900 mt-0.5">
                        {session.intervention.key_rule}
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                /* Non-M Classes (NONE, INSUFFICIENT, OOS): Special Pedagogical Routing */
                <div className="space-y-4">
                  <div className="p-5 bg-slate-50 border border-slate-200 rounded-xl flex items-start gap-4">
                    <div className="p-2.5 rounded-full bg-slate-100 shrink-0">
                      {session.diagnosis.misconception_id === 'NONE' ? (
                        <CheckCircle2 className="w-6 h-6 text-emerald-600" />
                      ) : session.diagnosis.misconception_id === 'INSUFFICIENT' ? (
                        <HelpCircle className="w-6 h-6 text-amber-600" />
                      ) : (
                        <BookOpen className="w-6 h-6 text-blue-600" />
                      )}
                    </div>
                    <div className="space-y-1">
                      <div className="text-xs font-bold uppercase tracking-wider text-slate-700">
                        {session.diagnosis.misconception_id === 'NONE'
                          ? 'Strategy: Curriculum Continuity (Non-Misconception Slip)'
                          : session.diagnosis.misconception_id === 'INSUFFICIENT'
                          ? 'Strategy: Reasoning Elicitation & Clarification'
                          : 'Strategy: Out-of-Scope Redirection'}
                      </div>
                      <p className="text-xs text-slate-700 leading-relaxed">
                        {session.message ||
                          (session.diagnosis.misconception_id === 'NONE'
                            ? 'Your reasoning demonstrates sound conceptual logic. The error was an ordinary clerical or arithmetic typo, so no remediation is required.'
                            : session.diagnosis.misconception_id === 'INSUFFICIENT'
                            ? 'I need to understand how you arrived at that answer. Please explain the step you used to trace the code.'
                            : 'This concept relates to topics outside the current supported CS1 misconception set (e.g. object aliasing or default arguments).')}
                      </p>
                    </div>
                  </div>

                  <div className="p-3.5 bg-white border-l-4 border-l-slate-600 border border-slate-200 rounded-r-xl flex items-start gap-3">
                    <Info className="w-4 h-4 text-slate-600 shrink-0 mt-0.5" />
                    <div>
                      <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                        Pedagogical Recommendation
                      </div>
                      <div className="text-xs font-semibold text-slate-900 mt-0.5">
                        {session.diagnosis.misconception_id === 'NONE' &&
                          'Verify arithmetic and typing keystrokes carefully before submitting. Continue normal curriculum sequence.'}
                        {session.diagnosis.misconception_id === 'INSUFFICIENT' &&
                          'Re-prompt student to articulate the specific evaluation step they used before asserting an authoritative diagnosis.'}
                        {session.diagnosis.misconception_id === 'OOS' &&
                          'Refer student to Python documentation on object references, aliasing, and mutable data types.'}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Connector Arrow */}
        {session && (
          <div className="flex justify-center my-2 text-slate-400">
            <ArrowDown className="w-5 h-5 animate-pulse" />
          </div>
        )}

        {/* ============================================================== */}
        {/* STAGE 4: FOLLOW-UP QUESTION (VERIFICATION PROBE) */}
        {/* ============================================================== */}
        {session && (
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
            <div className="px-5 py-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="flex items-center justify-center w-5 h-5 rounded-full bg-purple-100 text-purple-700 text-xs font-bold font-mono">
                  4
                </span>
                <h2 className="text-sm font-bold tracking-tight text-slate-900 uppercase">
                  Follow-up Question (Verification Probe)
                </h2>
              </div>
              <div className="text-xs text-slate-500">
                {session.followup ? 'Tests whether the misconception is resolved' : 'Not required for this state'}
              </div>
            </div>

            <div className="p-5 space-y-5">
              {session.followup ? (
                <>
                  {/* Follow-up Prompt Box */}
                  <div className="p-4 bg-purple-50/60 border border-purple-200 rounded-xl space-y-2">
                    <div className="text-xs font-bold text-purple-900 uppercase tracking-wider flex items-center justify-between">
                      <span>Follow-Up Assessment Question</span>
                      <span className="text-[11px] font-mono font-normal text-purple-700">
                        Target: {session.diagnosis.misconception_id}
                      </span>
                    </div>
                    <pre className="text-xs font-mono text-purple-950 whitespace-pre-wrap leading-relaxed bg-white/70 p-3 rounded-lg border border-purple-200/60">
                      {session.followup.question}
                    </pre>
                  </div>

                  {/* Fast test buttons */}
                  <div className="flex flex-wrap items-center justify-between gap-2 p-3 bg-slate-50 rounded-lg border border-slate-200">
                    <span className="text-xs text-slate-600 font-medium">
                      Evaluator Fast-Test Quick Fills:
                    </span>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => handleQuickFillFollowup('correct')}
                        className="px-2.5 py-1 text-xs font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 rounded hover:bg-emerald-100 transition-colors"
                      >
                        Fill Correct Answer (RESOLVED)
                      </button>
                      <button
                        type="button"
                        onClick={() => handleQuickFillFollowup('incorrect')}
                        className="px-2.5 py-1 text-xs font-medium text-amber-700 bg-amber-50 border border-amber-200 rounded hover:bg-amber-100 transition-colors"
                      >
                        Fill Misconception Recurrence (NOT_RESOLVED)
                      </button>
                    </div>
                  </div>

                  {/* Student Follow-up Input */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Student Follow-up Answer
                      </label>
                      <input
                        type="text"
                        value={followupAnswer}
                        onChange={(e) => setFollowupAnswer(e.target.value)}
                        className="w-full font-mono text-xs px-3 py-2 bg-white text-slate-900 rounded-lg border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-purple-500"
                        placeholder="Enter follow-up answer..."
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Student Follow-up Reasoning
                      </label>
                      <input
                        type="text"
                        value={followupReasoning}
                        onChange={(e) => setFollowupReasoning(e.target.value)}
                        className="w-full text-xs px-3 py-2 bg-white text-slate-900 rounded-lg border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-purple-500"
                        placeholder="Explain why this output occurs..."
                      />
                    </div>
                  </div>

                  <div className="pt-2 flex items-center justify-between">
                    <span className="text-xs text-slate-500">
                      Click <strong className="text-slate-700">Evaluate Follow-up</strong> to evaluate learning resolution and trigger second intervention if unresolved.
                    </span>
                    <button
                      type="button"
                      onClick={handleEvaluateFollowup}
                      disabled={isEvaluating || !followupAnswer}
                      className="flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-white bg-purple-600 rounded-lg hover:bg-purple-700 disabled:opacity-50 transition-colors shadow-xs"
                    >
                      {isEvaluating ? (
                        <>
                          <Activity className="w-3.5 h-3.5 animate-spin" />
                          <span>Evaluating...</span>
                        </>
                      ) : (
                        <>
                          <Send className="w-3.5 h-3.5" />
                          <span>Evaluate Follow-up</span>
                        </>
                      )}
                    </button>
                  </div>
                </>
              ) : (
                /* No Followup Needed (NONE, INSUFFICIENT, OOS) */
                <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl text-center space-y-1">
                  <div className="text-xs font-semibold text-slate-700">
                    Follow-up Verification Probe Not Applicable
                  </div>
                  <div className="text-xs text-slate-500">
                    {session.diagnosis.misconception_id === 'NONE' &&
                      'The student demonstrates sound conceptual grasp; curriculum progression continues immediately.'}
                    {session.diagnosis.misconception_id === 'INSUFFICIENT' &&
                      'Follow-up probe is paused while awaiting clarification of student reasoning.'}
                    {session.diagnosis.misconception_id === 'OOS' &&
                      'External topic referral active; no M01-M08 verification test is queued.'}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {evaluateError && (
          <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-800 flex items-start gap-2.5">
            <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold">Evaluation Error</div>
              <div className="mt-0.5">{evaluateError}</div>
            </div>
          </div>
        )}

        {/* Connector Arrow */}
        {session && (
          <div className="flex justify-center my-2 text-slate-400">
            <ArrowDown className="w-5 h-5 animate-pulse" />
          </div>
        )}

        {/* ============================================================== */}
        {/* STAGE 5: RESOLUTION ASSESSMENT & OUTCOME */}
        {/* ============================================================== */}
        {session && (
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
            <div className="px-5 py-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="flex items-center justify-center w-5 h-5 rounded-full bg-slate-900 text-white text-xs font-bold font-mono">
                  5
                </span>
                <h2 className="text-sm font-bold tracking-tight text-slate-900 uppercase">
                  Resolution Assessment & Outcome
                </h2>
              </div>
              <div className="flex items-center gap-3 text-xs">
                {evaluateLatency !== null && (
                  <span className="text-slate-500 font-mono text-[11px]">
                    Latency: {evaluateLatency}ms
                  </span>
                )}
                <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-200 text-slate-800 font-semibold">
                  Next: {resolutionResponse?.next_action || session.next_action}
                </span>
              </div>
            </div>

            <div className="p-5 space-y-5">
              {resolutionResponse ? (
                <>
                  {/* Resolution Status Card */}
                  <div
                    className={`p-5 rounded-xl border flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${
                      resolutionResponse.resolution.status === 'RESOLVED'
                        ? 'bg-emerald-50 border-emerald-200'
                        : 'bg-amber-50 border-amber-200'
                    }`}
                  >
                    <div className="flex items-center gap-3.5">
                      <div
                        className={`p-2.5 rounded-full ${
                          resolutionResponse.resolution.status === 'RESOLVED'
                            ? 'bg-emerald-100 text-emerald-700'
                            : 'bg-amber-100 text-amber-700'
                        }`}
                      >
                        {resolutionResponse.resolution.status === 'RESOLVED' ? (
                          <CheckCircle2 className="w-7 h-7" />
                        ) : (
                          <XCircle className="w-7 h-7" />
                        )}
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <span
                            className={`text-lg font-black tracking-tight ${
                              resolutionResponse.resolution.status === 'RESOLVED'
                                ? 'text-emerald-900'
                                : 'text-amber-900'
                            }`}
                          >
                            {resolutionResponse.resolution.status}
                          </span>
                          <span className="text-slate-400">·</span>
                          <span className="text-xs font-semibold text-slate-700">
                            Confidence: {Math.round(resolutionResponse.resolution.confidence * 100)}%
                          </span>
                        </div>
                        <div className="text-xs text-slate-700 mt-1 leading-relaxed">
                          {resolutionResponse.resolution.rationale}
                        </div>
                      </div>
                    </div>

                    <div className="text-right sm:border-l sm:border-slate-200 sm:pl-5">
                      <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                        Curriculum State
                      </div>
                      <div className="font-bold text-sm text-slate-900 font-mono mt-0.5">
                        {resolutionResponse.next_action === 'COMPLETED'
                          ? 'MASTERY CONFIRMED'
                          : 'TIER-2 INTERVENTION'}
                      </div>
                    </div>
                  </div>

                  {/* Follow-up Evidence */}
                  <div className="p-4 bg-white border border-slate-200 rounded-xl space-y-1.5">
                    <div className="text-xs font-bold text-slate-700 flex items-center gap-1.5">
                      <Terminal className="w-3.5 h-3.5 text-slate-500" />
                      <span>Evaluation Evidence from Follow-up</span>
                    </div>
                    <div className="text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded-lg p-3 font-mono">
                      {resolutionResponse.resolution.evidence}
                    </div>
                  </div>

                  {/* SECOND TARGETED INTERVENTION (IF NOT_RESOLVED) */}
                  {resolutionResponse.second_intervention && (
                    <div className="border-2 border-amber-300 bg-amber-50/40 rounded-xl p-5 space-y-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Layers className="w-4 h-4 text-amber-700" />
                          <h3 className="text-sm font-extrabold text-amber-950 uppercase tracking-wide">
                            Tier-2 Reinforced Intervention: {resolutionResponse.second_intervention.title}
                          </h3>
                        </div>
                        <span className="text-xs font-mono bg-amber-200 text-amber-900 px-2 py-0.5 rounded font-bold">
                          {resolutionResponse.second_intervention.pedagogical_approach}
                        </span>
                      </div>

                      <p className="text-xs text-amber-900 font-medium leading-relaxed">
                        {resolutionResponse.second_intervention.explanation}
                      </p>

                      {/* Step by step execution trace */}
                      <div className="bg-slate-900 rounded-lg p-4 text-slate-100 space-y-2">
                        <div className="text-[11px] font-bold text-amber-400 uppercase tracking-wider">
                          Step-by-Step Execution Model Trace
                        </div>
                        <pre className="text-xs font-mono text-slate-200 whitespace-pre-wrap leading-relaxed">
                          {resolutionResponse.second_intervention.step_by_step_trace}
                        </pre>
                      </div>

                      {/* Key takeaway */}
                      <div className="p-3 bg-white border border-amber-200 rounded-lg text-xs text-amber-950 font-semibold flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-amber-600 shrink-0" />
                        <span>
                          <strong>Key Takeaway:</strong> {resolutionResponse.second_intervention.key_takeaway}
                        </span>
                      </div>
                    </div>
                  )}
                </>
              ) : isAbstention ? (
                /* Non-M Status (NONE, INSUFFICIENT, OOS) */
                <div
                  className={`p-5 rounded-xl border flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${
                    session.diagnosis.misconception_id === 'NONE'
                      ? 'bg-emerald-50 border-emerald-200'
                      : session.diagnosis.misconception_id === 'INSUFFICIENT'
                      ? 'bg-amber-50 border-amber-200'
                      : 'bg-slate-50 border-slate-200'
                  }`}
                >
                  <div className="flex items-center gap-3.5">
                    <div
                      className={`p-2.5 rounded-full ${
                        session.diagnosis.misconception_id === 'NONE'
                          ? 'bg-emerald-100 text-emerald-700'
                          : session.diagnosis.misconception_id === 'INSUFFICIENT'
                          ? 'bg-amber-100 text-amber-700'
                          : 'bg-slate-200 text-slate-700'
                      }`}
                    >
                      {session.diagnosis.misconception_id === 'NONE' ? (
                        <CheckCircle2 className="w-7 h-7" />
                      ) : session.diagnosis.misconception_id === 'INSUFFICIENT' ? (
                        <HelpCircle className="w-7 h-7" />
                      ) : (
                        <BookOpen className="w-7 h-7" />
                      )}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span
                          className={`text-lg font-black tracking-tight ${
                            session.diagnosis.misconception_id === 'NONE'
                              ? 'text-emerald-900'
                              : session.diagnosis.misconception_id === 'INSUFFICIENT'
                              ? 'text-amber-900'
                              : 'text-slate-900'
                          }`}
                        >
                          {session.diagnosis.misconception_id === 'NONE'
                            ? 'RESOLVED / SOUND UNDERSTANDING'
                            : session.diagnosis.misconception_id === 'INSUFFICIENT'
                            ? 'AWAITING CLARIFICATION'
                            : 'OUT OF SCOPE'}
                        </span>
                        <span className="text-slate-400">·</span>
                        <span className="text-xs font-semibold text-slate-700">
                          Confidence: {Math.round(session.diagnosis.confidence * 100)}%
                        </span>
                      </div>
                      <div className="text-xs text-slate-700 mt-1 leading-relaxed">
                        {session.message || session.diagnosis.rationale}
                      </div>
                    </div>
                  </div>

                  <div className="text-right sm:border-l sm:border-slate-200 sm:pl-5">
                    <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                      Next System Action
                    </div>
                    <div className="font-bold text-sm text-slate-900 font-mono mt-0.5">
                      {session.next_action}
                    </div>
                  </div>
                </div>
              ) : (
                /* M01-M08 awaiting follow-up submission */
                <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl text-center space-y-1">
                  <div className="text-xs font-semibold text-slate-700">
                    Awaiting Student Follow-up Submission
                  </div>
                  <div className="text-xs text-slate-500">
                    Submit a response in Stage 4 and click &quot;Evaluate Follow-up&quot; to verify learning resolution.
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
