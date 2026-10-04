import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  RotateCcw,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  HelpCircle,
  BookOpen,
  Code2,
  Activity,
  Send,
  ArrowRight,
  Lightbulb,
  Plus,
} from 'lucide-react';

export interface DiagnosisResult {
  misconception_id: string;
  confidence: number;
  evidence: string;
  rationale: string;
  decision_source: 'model' | 'evidence_rule' | 'abstention_rule' | string;
  misconception_variant?: string | null;
  secondary_misconception_ids?: string[];
  error_type?: string;
  answer_correct?: boolean;
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
  resolution_type?: string;
  confidence: number;
  evidence: string;
  rationale: string;
  persistent_misconception?: boolean;
}

export interface MisconceptionProgress {
  detected: number;
  resolved: number;
  unresolved: number;
  current_status?: 'AWAITING_FOLLOWUP' | 'NEEDS_PRACTICE' | 'MASTERED';
}

export interface LearnerState {
  attempts: number;
  misconception_detections: number;
  resolved_count: number;
  unresolved_count: number;
  mastered_count?: number;
  needs_practice_count?: number;
  awaiting_followup_count?: number;
  last_status: string | null;
  misconceptions: Record<string, MisconceptionProgress>;
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
  learner_state?: LearnerState;
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
  learner_state?: LearnerState;
}

interface QuestionBankItem {
  id: string;
  label: string;
  highlight?: boolean;
  prompt: string;
  question: string;
  correct_answer: string; // Stored internally; never exposed in Student Mode
  sample_answer?: string;
  sample_reasoning?: string;
}

interface QuestionCategory {
  id: string;
  label: string;
  questions: QuestionBankItem[];
}

const MISCONCEPTION_NAMES: Record<string, string> = {
  M01: 'M01 — String vs Number Type Confusion',
  M02: 'M02 — Division Semantics (/ vs //)',
  M03: 'M03 — Assignment vs Equality (= vs ==)',
  M04: 'M04 — Operator Precedence',
  M05: 'M05 — Index / Position',
  M06: 'M06 — Loop Values / Boundaries / Iteration Interval',
  M07: 'M07 — Function Argument–Parameter Binding',
  M08: 'M08 — Recursion Termination / Base Case',
  NONE: 'NONE — No Conceptual Misconception',
  INSUFFICIENT: 'INSUFFICIENT — Insufficient Conceptual Evidence',
  OOS: 'OOS — Out of Scope Concept',
};

const PROGRESS_MISCONCEPTION_LABELS: Record<string, string> = {
  M01: 'M01 — String vs Number',
  M02: 'M02 — Division Semantics (/ vs //)',
  M03: 'M03 — Assignment vs Equality (= vs ==)',
  M04: 'M04 — Operator Precedence',
  M05: 'M05 — Index / Position',
  M06: 'M06 — Loop Boundaries',
  M07: 'M07 — Function Argument–Parameter Binding',
  M08: 'M08 — Recursion Termination / Base Case',
};

const INITIAL_LEARNER_STATE: LearnerState = {
  attempts: 0,
  misconception_detections: 0,
  resolved_count: 0,
  unresolved_count: 0,
  last_status: 'READY',
  misconceptions: {},
};

/**
 * Initial 7-Category Question Bank.
 * Each question stores { id, label, question, correct_answer } internally.
 * In Student Mode, only `question` is shown to the learner when answering;
 * `correct_answer` is never displayed and is passed internally to the backend diagnosis endpoint.
 */
const INITIAL_QUESTION_CATEGORIES: QuestionCategory[] = [
  {
    id: 'floor_division',
    label: 'Floor Division',
    questions: [
      {
        id: 'floor_division_q1',
        label: 'Q1',
        prompt: 'What is the output of this Python code?',
        question: 'print(7 // 2)',
        correct_answer: '3',
        sample_answer: '4',
        sample_reasoning: '7 divided by 2 is 3.5, so // rounds up to 4.',
      },
      {
        id: 'floor_division_q2',
        label: 'Q2',
        prompt: 'What is the output of this Python code?',
        question: 'print(9 // 4)',
        correct_answer: '2',
        sample_answer: '2.25',
        sample_reasoning: '9 divided by 4 is 2.25, and // is division so it gives 2.25.',
      },
      {
        id: 'floor_division_q3',
        label: 'Q3',
        prompt: 'What is the output of this Python code?',
        question: 'print(15 // 4)',
        correct_answer: '3',
        sample_answer: '4',
        sample_reasoning: '15 divided by 4 is 3.75, which rounds to the nearest integer 4.',
      },
    ],
  },
  {
    id: 'operator_precedence',
    label: 'Operator Precedence',
    questions: [
      {
        id: 'operator_precedence_q1',
        label: 'Q1',
        highlight: true,
        prompt: 'What is the output of this Python code?',
        question: 'print(2 ** 3 * 1)',
        correct_answer: '8',
        sample_answer: '8',
        sample_reasoning: 'First multiply 3 by 1, then calculate 2 to the power 3.',
      },
      {
        id: 'operator_precedence_q2',
        label: 'Q2',
        prompt: 'What is the output of this Python code?',
        question: 'print(2 * 3 ** 2)',
        correct_answer: '18',
        sample_answer: '36',
        sample_reasoning: '2 * 3 is 6, and then 6 squared is 36 because we go left to right.',
      },
      {
        id: 'operator_precedence_q3',
        label: 'Q3',
        prompt: 'What is the output of this Python code?',
        question: 'print(10 - 2 ** 2)',
        correct_answer: '6',
        sample_answer: '64',
        sample_reasoning: 'First subtract 10 - 2 to get 8, then raise 8 to the power 2 to get 64.',
      },
    ],
  },
  {
    id: 'range_boundaries',
    label: 'range() Boundaries',
    questions: [
      {
        id: 'range_boundaries_q1',
        label: 'Q1',
        prompt: 'What values are produced by this Python code?',
        question: 'for i in range(1, 5):\n    print(i)',
        correct_answer: '1 2 3 4',
        sample_answer: '1 2 3 4 5',
        sample_reasoning: 'range(1, 5) includes both the starting value 1 and the ending value 5.',
      },
      {
        id: 'range_boundaries_q2',
        label: 'Q2',
        prompt: 'What values are produced by this Python code?',
        question: 'for i in range(3):\n    print(i)',
        correct_answer: '0 1 2',
        sample_answer: '1 2 3',
        sample_reasoning: 'range(3) counts 3 numbers starting from 1 up to 3.',
      },
      {
        id: 'range_boundaries_q3',
        label: 'Q3',
        prompt: 'What values are produced by this Python code?',
        question: 'for i in range(2, 6):\n    print(i)',
        correct_answer: '2 3 4 5',
        sample_answer: '2 3 4 5 6',
        sample_reasoning: 'The loop starts at 2 and stops after printing the stop value 6.',
      },
    ],
  },
  {
    id: 'string_indexing',
    label: 'String Indexing',
    questions: [
      {
        id: 'string_indexing_q1',
        label: 'Q1',
        prompt: 'What is the output of this Python code?',
        question: 's = "Python"\nprint(s[1])',
        correct_answer: 'y',
        sample_answer: 'P',
        sample_reasoning: 'Indexing starts at 1, so index 1 refers to the first character P.',
      },
      {
        id: 'string_indexing_q2',
        label: 'Q2',
        prompt: 'What is the output of this Python code?',
        question: 'word = "code"\nprint(word[0])',
        correct_answer: 'c',
        sample_answer: 'o',
        sample_reasoning: 'Index 0 skips the first letter and gives the next character.',
      },
      {
        id: 'string_indexing_q3',
        label: 'Q3',
        prompt: 'What is the output of this Python code?',
        question: 'text = "Data"\nprint(text[2])',
        correct_answer: 't',
        sample_answer: 'a',
        sample_reasoning: 'Position 2 is the second letter in Data, which is a.',
      },
    ],
  },
  {
    id: 'string_integer',
    label: 'String + Integer',
    questions: [
      {
        id: 'string_integer_q1',
        label: 'Q1',
        prompt: 'What happens when this Python code runs?',
        question: 'x = "5"\ny = 2\nprint(x + y)',
        correct_answer: 'TypeError',
        sample_answer: '7',
        sample_reasoning:
          'The quotes around "5" do not matter; Python treats it as integer 5 so adding gives 7.',
      },
      {
        id: 'string_integer_q2',
        label: 'Q2',
        prompt: 'What is the output of this Python code?',
        question: 'print("3" + "4")',
        correct_answer: '34',
        sample_answer: '7',
        sample_reasoning: '"3" and "4" are numbers so the + operator adds them numerically to get 7.',
      },
      {
        id: 'string_integer_q3',
        label: 'Q3',
        prompt: 'What is the output of this Python code?',
        question: 'a = "10"\nprint(a * 2)',
        correct_answer: '1010',
        sample_answer: '20',
        sample_reasoning: 'Multiplying "10" by 2 multiplies the number 10 by 2 to produce 20.',
      },
    ],
  },
  {
    id: 'arithmetic_order',
    label: 'Arithmetic Order',
    questions: [
      {
        id: 'arithmetic_order_q1',
        label: 'Q1',
        prompt: 'What is the output of this Python code?',
        question: 'print(2 + 3 * 4)',
        correct_answer: '14',
        sample_answer: '20',
        sample_reasoning: 'Evaluate left to right: 2 + 3 is 5, and then 5 * 4 is 20.',
      },
      {
        id: 'arithmetic_order_q2',
        label: 'Q2',
        prompt: 'What is the output of this Python code?',
        question: 'print(20 - 4 * 3)',
        correct_answer: '8',
        sample_answer: '48',
        sample_reasoning: 'First do 20 - 4 to get 16, then multiply 16 by 3 to get 48.',
      },
      {
        id: 'arithmetic_order_q3',
        label: 'Q3',
        prompt: 'What is the output of this Python code?',
        question: 'print((2 + 3) * 4)',
        correct_answer: '20',
        sample_answer: '14',
        sample_reasoning: 'Multiplication always happens before addition even with parentheses, so 3 * 4 is 12 plus 2 is 14.',
      },
    ],
  },
  {
    id: 'other',
    label: 'Other',
    questions: [
      {
        id: 'other_q1',
        label: 'Q1',
        prompt: 'What is the output of this Python code?',
        question: 'x = 5\nprint(x == 5)',
        correct_answer: 'True',
        sample_answer: '5',
        sample_reasoning: 'x == 5 assigns 5 to x and prints the value 5.',
      },
      {
        id: 'other_q2',
        label: 'Q2',
        prompt: 'What is the output of this Python code?',
        question: 'def greet(name):\n    print(name)\n\ngreet("Riya")',
        correct_answer: 'Riya',
        sample_answer: 'name',
        sample_reasoning: 'The function prints the parameter name literally instead of the argument passed in.',
      },
      {
        id: 'other_q3',
        label: 'Q3',
        prompt: 'What values are produced by this Python code?',
        question:
          'def countdown(n):\n    if n == 0:\n        return\n    print(n)\n    countdown(n - 1)\n\ncountdown(3)',
        correct_answer: '3 2 1',
        sample_answer: '3 2 1 0',
        sample_reasoning: 'The recursive function prints n down to and including the base case 0.',
      },
    ],
  },
];

const QUESTION_BANK: QuestionBankItem[] = INITIAL_QUESTION_CATEGORIES.flatMap(
  (category) => category.questions
);

export const InterventionStudio: React.FC = () => {
  // Mode: 'student' (default learner-facing UI) vs 'instructor' (custom question + correct_answer config)
  const [uiMode, setUiMode] = useState<'student' | 'instructor'>('student');

  // Dynamic Question Bank categories state (7 categories, supports '+ Add Question')
  const [categories, setCategories] = useState<QuestionCategory[]>(INITIAL_QUESTION_CATEGORIES);
  const [selectedCategoryId, setSelectedCategoryId] = useState<string>(
    INITIAL_QUESTION_CATEGORIES[0].id
  );
  const [selectedQuestionId, setSelectedQuestionId] = useState<string>(
    INITIAL_QUESTION_CATEGORIES[0].questions[0].id
  );

  // Add Question Form state
  const [isAddQuestionOpen, setIsAddQuestionOpen] = useState<boolean>(false);
  const [newQuestionCode, setNewQuestionCode] = useState<string>('');
  const [newCorrectAnswer, setNewCorrectAnswer] = useState<string>('');
  const [newSampleAnswer, setNewSampleAnswer] = useState<string>('');
  const [newSampleReasoning, setNewSampleReasoning] = useState<string>('');
  const [addQuestionError, setAddQuestionError] = useState<string | null>(null);

  // Internal question & correct_answer state + learner-entered answer & reasoning
  const [question, setQuestion] = useState<string>(
    INITIAL_QUESTION_CATEGORIES[0].questions[0].question
  );
  const [correctAnswer, setCorrectAnswer] = useState<string>(
    INITIAL_QUESTION_CATEGORIES[0].questions[0].correct_answer
  );
  const [studentAnswer, setStudentAnswer] = useState<string>('');
  const [studentReasoning, setStudentReasoning] = useState<string>('');

  // Stage 2: Diagnosis state
  const [session, setSession] = useState<PipelineSession | null>(null);
  const [isDiagnosing, setIsDiagnosing] = useState<boolean>(false);
  const [diagnoseError, setDiagnoseError] = useState<string | null>(null);

  // Stage 3: Targeted Intervention visibility & follow-up state
  const [showIntervention, setShowIntervention] = useState<boolean>(false);
  const [followupAnswer, setFollowupAnswer] = useState<string>('');
  const [followupReasoning, setFollowupReasoning] = useState<string>('');
  const [resolutionResponse, setResolutionResponse] = useState<ResolutionResponse | null>(null);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [evaluateError, setEvaluateError] = useState<string | null>(null);

  // Learner Progress state (persisted in-memory across attempts)
  const [learnerState, setLearnerState] = useState<LearnerState>(INITIAL_LEARNER_STATE);
  const [isResettingProgress, setIsResettingProgress] = useState<boolean>(false);

  // Backend connectivity indicator
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'offline'>(
    'checking'
  );

  useEffect(() => {
    fetch('/api/relearn/health')
      .then((r) => r.json())
      .then(() => setBackendStatus('connected'))
      .catch(() => setBackendStatus('offline'));

    fetch('/api/relearn/learner')
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data && typeof data.attempts === 'number') {
          setLearnerState(data);
        }
      })
      .catch(() => {});
  }, []);

  const currentCategory =
    categories.find((cat) => cat.id === selectedCategoryId) || categories[0];

  const currentBankItem: QuestionBankItem | undefined =
    currentCategory.questions.find((item) => item.id === selectedQuestionId) ||
    currentCategory.questions[0];

  const clearDownstreamStages = () => {
    setSession(null);
    setShowIntervention(false);
    setResolutionResponse(null);
    setDiagnoseError(null);
    setEvaluateError(null);
    setFollowupAnswer('');
    setFollowupReasoning('');
  };

  const handleSelectCategory = (category: QuestionCategory) => {
    clearDownstreamStages();
    setSelectedCategoryId(category.id);
    setAddQuestionError(null);
    setStudentAnswer('');
    setStudentReasoning('');

    const firstQuestion = category.questions[0];
    if (firstQuestion) {
      setSelectedQuestionId(firstQuestion.id);
      setQuestion(firstQuestion.question);
      setCorrectAnswer(firstQuestion.correct_answer);
    } else {
      setSelectedQuestionId('');
      if (uiMode === 'student') {
        setQuestion('');
        setCorrectAnswer('');
      }
    }
  };

  const handleSelectQuestion = (item: QuestionBankItem) => {
    clearDownstreamStages();
    setIsAddQuestionOpen(false);
    setAddQuestionError(null);
    setSelectedQuestionId(item.id);
    setQuestion(item.question);
    setCorrectAnswer(item.correct_answer);
    setStudentAnswer('');
    setStudentReasoning('');
  };

  const handleFillSampleResponse = (item: QuestionBankItem) => {
    clearDownstreamStages();
    setIsAddQuestionOpen(false);
    setAddQuestionError(null);
    setSelectedQuestionId(item.id);
    setQuestion(item.question);
    setCorrectAnswer(item.correct_answer);
    setStudentAnswer(item.sample_answer || '');
    setStudentReasoning(item.sample_reasoning || '');
  };

  const handleOpenAddQuestion = () => {
    setIsAddQuestionOpen(true);
    setAddQuestionError(null);
    setNewQuestionCode('');
    setNewCorrectAnswer('');
    setNewSampleAnswer('');
    setNewSampleReasoning('');
  };

  const handleCancelAddQuestion = () => {
    setIsAddQuestionOpen(false);
    setAddQuestionError(null);
    setNewQuestionCode('');
    setNewCorrectAnswer('');
    setNewSampleAnswer('');
    setNewSampleReasoning('');
  };

  const handleAddQuestion = () => {
    const trimmedQuestion = newQuestionCode.trim();
    const trimmedCorrect = newCorrectAnswer.trim();

    if (!trimmedQuestion || !trimmedCorrect) {
      setAddQuestionError('Please enter both the Question / Python Code and the Correct Answer.');
      return;
    }

    const nextNumber = currentCategory.questions.length + 1;
    const nextLabel = `Q${nextNumber}`;
    const newItem: QuestionBankItem = {
      id: `${currentCategory.id}_q${nextNumber}_${Date.now()}`,
      label: nextLabel,
      prompt: 'What is the output of this Python code?',
      question: trimmedQuestion,
      correct_answer: trimmedCorrect,
      sample_answer: newSampleAnswer.trim() || undefined,
      sample_reasoning: newSampleReasoning.trim() || undefined,
    };

    setCategories((prevCategories) =>
      prevCategories.map((cat) =>
        cat.id === currentCategory.id
          ? { ...cat, questions: [...cat.questions, newItem] }
          : cat
      )
    );

    // Close Add Question form and select the newly added question immediately
    setIsAddQuestionOpen(false);
    setAddQuestionError(null);
    setNewQuestionCode('');
    setNewCorrectAnswer('');
    setNewSampleAnswer('');
    setNewSampleReasoning('');

    // Clear current attempt state while keeping Learner Progress intact
    clearDownstreamStages();
    setSelectedQuestionId(newItem.id);
    setQuestion(newItem.question);
    setCorrectAnswer(newItem.correct_answer);
    setStudentAnswer('');
    setStudentReasoning('');
  };

  const handleFieldChange = (
    field: 'question' | 'correctAnswer' | 'studentAnswer' | 'studentReasoning',
    value: string
  ) => {
    clearDownstreamStages();
    if (field === 'question') setQuestion(value);
    else if (field === 'correctAnswer') setCorrectAnswer(value);
    else if (field === 'studentAnswer') setStudentAnswer(value);
    else if (field === 'studentReasoning') setStudentReasoning(value);
  };

  const handleClearForm = () => {
    setStudentAnswer('');
    setStudentReasoning('');
    setSession(null);
    setShowIntervention(false);
    setResolutionResponse(null);
    setFollowupAnswer('');
    setFollowupReasoning('');
    setIsDiagnosing(false);
    setIsEvaluating(false);
    setDiagnoseError(null);
    setEvaluateError(null);
    if (uiMode === 'instructor') {
      setQuestion('');
      setCorrectAnswer('');
    }
  };

  const handleSwitchMode = (nextMode: 'student' | 'instructor') => {
    clearDownstreamStages();
    setUiMode(nextMode);
    if (nextMode === 'student' && currentBankItem) {
      // Ensure question and internal correct_answer match the selected question bank item
      setQuestion(currentBankItem.question);
      setCorrectAnswer(currentBankItem.correct_answer);
    }
  };

  const handleResetProgress = async () => {
    setIsResettingProgress(true);
    try {
      const response = await fetch('/api/relearn/learner/reset', {
        method: 'POST',
      });
      if (response.ok) {
        const data: LearnerState = await response.json();
        setLearnerState(data);
      } else {
        setLearnerState(INITIAL_LEARNER_STATE);
      }
    } catch {
      setLearnerState(INITIAL_LEARNER_STATE);
    } finally {
      setIsResettingProgress(false);
    }
  };

  // Stage 1 -> Stage 2: Live diagnosis call sending { question, correct_answer, student_answer, student_reasoning }
  const handleDiagnose = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setIsDiagnosing(true);
    setDiagnoseError(null);
    setSession(null);
    setShowIntervention(false);
    setResolutionResponse(null);
    setFollowupAnswer('');
    setFollowupReasoning('');

    const activeQuestion =
      uiMode === 'student' ? currentBankItem?.question || '' : question;
    const activeCorrectAnswer =
      uiMode === 'student' ? currentBankItem?.correct_answer || '' : correctAnswer;

    const payload = {
      question: activeQuestion.trim(),
      correct_answer: activeCorrectAnswer.trim(),
      student_answer: studentAnswer.trim(),
      student_reasoning: studentReasoning.trim(),
    };

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
      setSession(data);
      if (data.learner_state) {
        setLearnerState(data.learner_state);
      }
      setBackendStatus('connected');
    } catch (err: any) {
      setDiagnoseError(err.message || 'Diagnosis failed. Please check backend connection.');
      setBackendStatus('offline');
    } finally {
      setIsDiagnosing(false);
    }
  };

  // Stage 3: Follow-up resolution check
  const handleEvaluateFollowup = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!session) return;
    setIsEvaluating(true);
    setEvaluateError(null);

    try {
      const response = await fetch('/api/relearn/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session,
          followup_answer: followupAnswer.trim(),
          followup_reasoning: followupReasoning.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error(`Evaluation server error (HTTP ${response.status})`);
      }

      const data: ResolutionResponse = await response.json();
      setResolutionResponse(data);
      if (data.learner_state) {
        setLearnerState(data.learner_state);
      }
      setBackendStatus('connected');
    } catch (err: any) {
      setEvaluateError(err.message || 'Evaluation failed.');
      setBackendStatus('offline');
    } finally {
      setIsEvaluating(false);
    }
  };

  const diag = session?.diagnosis;
  const isMisconception =
    diag && ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08'].includes(diag.misconception_id);
  const isRightAnswerWrongReason = Boolean(isMisconception && diag?.answer_correct === true);
  const encounteredMisconceptions = Object.entries(learnerState.misconceptions || {}).filter(
    ([, stats]) => stats.detected > 0
  );

  return (
    <div className="max-w-4xl mx-auto space-y-6 pb-16">
      {/* Header Banner */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-medium text-slate-500 mb-1">
              <span>Adaptive Python CS1 Learning</span>
              <span>·</span>
              <span className="flex items-center gap-1.5">
                <span
                  className={`w-2 h-2 rounded-full ${
                    backendStatus === 'connected'
                      ? 'bg-emerald-500'
                      : backendStatus === 'checking'
                      ? 'bg-amber-400'
                      : 'bg-rose-500'
                  }`}
                />
                <span className="capitalize">{backendStatus}</span>
              </span>
            </div>
            <h1 className="text-2xl font-extrabold tracking-tight text-slate-900">
              ReLearn — Adaptive Python Practice
            </h1>
            <p className="mt-1 text-sm text-slate-600">
              ReLearn analyzes your <strong>answer and reasoning</strong> to identify underlying
              conceptual misconceptions — even when your final answer happens to be right.
            </p>
          </div>

          <div className="flex items-center gap-2 self-start sm:self-center">
            <div className="inline-flex rounded-lg border border-slate-200 bg-slate-100 p-0.5 text-xs">
              <button
                type="button"
                onClick={() => handleSwitchMode('student')}
                className={`px-2.5 py-1 rounded-md font-semibold transition-colors ${
                  uiMode === 'student'
                    ? 'bg-white text-slate-900 shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Student Mode
              </button>
              <button
                type="button"
                onClick={() => handleSwitchMode('instructor')}
                className={`px-2.5 py-1 rounded-md font-semibold transition-colors ${
                  uiMode === 'instructor'
                    ? 'bg-white text-slate-900 shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Demo / Instructor Mode
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ===================================================================== */}
      {/* LEARNER PROGRESS (Across Attempts)                                    */}
      {/* ===================================================================== */}
      <section className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
        <div className="px-6 py-3.5 bg-slate-50 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-4">
            <h2 className="text-sm font-bold text-slate-900">Learner Progress</h2>
            <div className="flex flex-wrap items-center gap-3 text-xs text-slate-600">
              <span>
                Attempts:{' '}
                <strong className="font-mono text-slate-900">{learnerState.attempts ?? 0}</strong>
              </span>
              <span>·</span>
              <span>
                Misconceptions Encountered:{' '}
                <strong className="font-mono text-slate-900">
                  {encounteredMisconceptions.length}
                </strong>
              </span>
              <span>·</span>
              <span>
                Resolved:{' '}
                <strong className="font-mono text-emerald-700">
                  {learnerState.resolved_count ?? 0}
                </strong>
              </span>
              <span>·</span>
              <span>
                Needs Practice:{' '}
                <strong className="font-mono text-amber-700">
                  {learnerState.needs_practice_count ?? learnerState.unresolved_count ?? 0}
                </strong>
              </span>
            </div>
          </div>

          <button
            type="button"
            onClick={handleResetProgress}
            disabled={isResettingProgress}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold text-slate-600 bg-white border border-slate-200 rounded-md hover:bg-slate-100 disabled:opacity-50 transition-colors"
          >
            <RotateCcw className="w-3 h-3" />
            <span>Reset Progress</span>
          </button>
        </div>

        <div className="p-4">
          {encounteredMisconceptions.length === 0 ? (
            <div className="text-xs text-slate-500">
              {(learnerState.attempts ?? 0) === 0
                ? 'No attempts recorded yet. Answer a Python question below to track misconceptions and demonstrated understanding across attempts.'
                : `Attempts: ${learnerState.attempts ?? 0} — No M01–M08 misconceptions detected yet.`}
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {encounteredMisconceptions.map(([mId, stats]) => {
                const hasExplicitStatus = Boolean(stats.current_status);
                const demonstratedUnderstanding = hasExplicitStatus
                  ? stats.current_status === 'MASTERED'
                  : stats.resolved >= stats.detected && stats.detected > 0;
                const demonstratedPartialResolution =
                  !hasExplicitStatus && stats.resolved > 0 && stats.resolved < stats.detected;
                const needsPractice = hasExplicitStatus
                  ? stats.current_status === 'NEEDS_PRACTICE'
                  : stats.unresolved > 0;
                const awaitingFollowup = hasExplicitStatus
                  ? stats.current_status === 'AWAITING_FOLLOWUP'
                  : !demonstratedUnderstanding && !demonstratedPartialResolution && !needsPractice;

                return (
                  <div
                    key={mId}
                    className="p-3.5 rounded-lg border border-slate-200 bg-slate-50/70 flex flex-col justify-between gap-2"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="text-xs font-bold text-slate-900">
                        {PROGRESS_MISCONCEPTION_LABELS[mId] || MISCONCEPTION_NAMES[mId] || mId}
                      </div>
                      {stats.detected > 1 && (
                        <span className="shrink-0 text-[11px] font-semibold px-2 py-0.5 rounded bg-indigo-50 text-indigo-800 border border-indigo-200">
                          Encountered {stats.detected} times
                        </span>
                      )}
                    </div>

                    <div className="flex flex-wrap items-center gap-3 text-xs text-slate-700 font-mono">
                      <span>Detected: {stats.detected ?? 0}</span>
                      <span>
                        Resolved: {stats.resolved ?? 0}
                        {(stats.resolved ?? 0) > 0 ? ' ✓' : ''}
                      </span>
                      {(stats.unresolved ?? 0) > 0 && <span>Unresolved: {stats.unresolved}</span>}
                    </div>

                    <div className="flex flex-wrap items-center gap-2 pt-0.5">
                      {demonstratedUnderstanding && (
                        <span className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200">
                          ✓ Demonstrated understanding
                        </span>
                      )}
                      {demonstratedPartialResolution && (
                        <span className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200">
                          ✓ Demonstrated resolution
                        </span>
                      )}
                      {needsPractice && (
                        <span className="inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded bg-amber-50 text-amber-800 border border-amber-200">
                          Needs practice
                        </span>
                      )}
                      {awaitingFollowup && (
                        <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">
                          Awaiting follow-up
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </section>

      {/* ===================================================================== */}
      {/* STAGE 1: PYTHON QUESTION & LEARNER RESPONSE                           */}
      {/* ===================================================================== */}
      <section className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
        <div className="px-6 py-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <span className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-600 text-white text-xs font-bold font-mono">
              1
            </span>
            <h2 className="text-base font-bold text-slate-900">
              {uiMode === 'student' ? 'Python Question' : 'Demo / Instructor Custom Question'}
            </h2>
          </div>
          <span className="text-xs text-slate-500">
            {uiMode === 'student'
              ? 'Select a category and question, or add a new question'
              : 'Configure custom question and internal expected output'}
          </span>
        </div>

        <form onSubmit={handleDiagnose} className="p-6 space-y-5">
          {/* Category & Question Selector Bar */}
          <div className="space-y-3">
            {/* Row 1: 7 Categories */}
            <div className="space-y-1.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-semibold text-slate-600">Categories:</span>
                {uiMode === 'student' &&
                  currentBankItem &&
                  (currentBankItem.sample_answer || currentBankItem.sample_reasoning) && (
                    <button
                      type="button"
                      onClick={() => handleFillSampleResponse(currentBankItem)}
                      className="text-xs font-medium text-blue-600 hover:text-blue-800 underline underline-offset-2"
                    >
                      Fill sample student response for this question
                    </button>
                  )}
              </div>
              <div className="flex flex-wrap gap-1.5">
                {categories.map((category) => {
                  const isCategorySelected = category.id === currentCategory.id;
                  return (
                    <button
                      key={category.id}
                      type="button"
                      onClick={() => handleSelectCategory(category)}
                      className={`px-3 py-1.5 text-xs rounded-md border transition-colors ${
                        isCategorySelected
                          ? 'bg-slate-900 text-white border-slate-900 font-semibold shadow-2xs'
                          : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100 font-medium'
                      }`}
                    >
                      {category.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Row 2: Questions in Selected Category + "+ Add Question" */}
            <div className="space-y-1.5 pt-1">
              <span className="block text-xs font-semibold text-slate-600">
                Questions ({currentCategory.label}):
              </span>
              <div className="flex flex-wrap items-center gap-1.5">
                {currentCategory.questions.length === 0 ? (
                  <span className="text-xs text-slate-500 italic mr-2">No questions yet.</span>
                ) : (
                  currentCategory.questions.map((qItem) => {
                    const isQuestionSelected = currentBankItem?.id === qItem.id;
                    return (
                      <button
                        key={qItem.id}
                        type="button"
                        onClick={() =>
                          uiMode === 'student'
                            ? handleSelectQuestion(qItem)
                            : handleFillSampleResponse(qItem)
                        }
                        className={`px-3 py-1 text-xs rounded-md border transition-colors ${
                          isQuestionSelected
                            ? 'bg-blue-600 text-white border-blue-600 font-semibold shadow-2xs'
                            : qItem.highlight
                            ? 'bg-amber-50 text-amber-900 border-amber-300 hover:bg-amber-100 font-semibold'
                            : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100 font-medium'
                        }`}
                      >
                        {qItem.label}
                      </button>
                    );
                  })
                )}

                <button
                  type="button"
                  onClick={handleOpenAddQuestion}
                  className={`inline-flex items-center gap-1 px-3 py-1 text-xs font-semibold rounded-md border transition-colors ${
                    isAddQuestionOpen
                      ? 'bg-emerald-600 text-white border-emerald-600'
                      : 'bg-emerald-50 text-emerald-800 border-emerald-300 hover:bg-emerald-100'
                  }`}
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>+ Add Question</span>
                </button>
              </div>
            </div>

            {/* Add New Question Card / Form */}
            {isAddQuestionOpen && (
              <div className="mt-3 p-4 rounded-xl border border-emerald-200 bg-emerald-50/40 space-y-4">
                <div className="flex items-center justify-between gap-2 border-b border-emerald-200/80 pb-2.5">
                  <div>
                    <h3 className="text-sm font-bold text-slate-900">Add New Question</h3>
                    <p className="text-xs text-slate-600">
                      Category:{' '}
                      <strong className="font-semibold text-slate-900">
                        {currentCategory.label}
                      </strong>{' '}
                      (will be added as{' '}
                      <strong className="font-mono text-emerald-800">
                        Q{currentCategory.questions.length + 1}
                      </strong>
                      )
                    </p>
                  </div>
                </div>

                {addQuestionError && (
                  <div className="p-2.5 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-800 font-medium flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0" />
                    <span>{addQuestionError}</span>
                  </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label
                      htmlFor="new-question-code"
                      className="block text-xs font-semibold text-slate-700 mb-1"
                    >
                      Question / Python Code <span className="text-rose-600">*</span>
                    </label>
                    <textarea
                      id="new-question-code"
                      rows={4}
                      value={newQuestionCode}
                      onChange={(e) => {
                        setNewQuestionCode(e.target.value);
                        if (addQuestionError) setAddQuestionError(null);
                      }}
                      className="w-full font-mono text-xs p-2.5 bg-slate-900 text-slate-100 rounded-lg border border-slate-800 focus:outline-hidden focus:ring-2 focus:ring-emerald-500"
                      placeholder={'e.g.\nx = 10\nprint(x // 3)'}
                    />
                  </div>

                  <div>
                    <label
                      htmlFor="new-correct-answer"
                      className="block text-xs font-semibold text-slate-700 mb-1"
                    >
                      Correct Answer (Internal Reference — hidden from students){' '}
                      <span className="text-rose-600">*</span>
                    </label>
                    <textarea
                      id="new-correct-answer"
                      rows={4}
                      value={newCorrectAnswer}
                      onChange={(e) => {
                        setNewCorrectAnswer(e.target.value);
                        if (addQuestionError) setAddQuestionError(null);
                      }}
                      className="w-full font-mono text-xs p-2.5 bg-white text-slate-900 rounded-lg border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-emerald-500"
                      placeholder="e.g. 3"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label
                      htmlFor="new-sample-answer"
                      className="block text-xs font-semibold text-slate-600 mb-1"
                    >
                      Optional Sample Student Answer
                    </label>
                    <textarea
                      id="new-sample-answer"
                      rows={2}
                      value={newSampleAnswer}
                      onChange={(e) => setNewSampleAnswer(e.target.value)}
                      className="w-full font-mono text-xs p-2.5 bg-white text-slate-900 rounded-lg border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-emerald-500"
                      placeholder="Optional sample student answer..."
                    />
                  </div>

                  <div>
                    <label
                      htmlFor="new-sample-reasoning"
                      className="block text-xs font-semibold text-slate-600 mb-1"
                    >
                      Optional Sample Student Reasoning
                    </label>
                    <textarea
                      id="new-sample-reasoning"
                      rows={2}
                      value={newSampleReasoning}
                      onChange={(e) => setNewSampleReasoning(e.target.value)}
                      className="w-full text-xs p-2.5 bg-white text-slate-900 rounded-lg border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-emerald-500"
                      placeholder="Optional sample student reasoning..."
                    />
                  </div>
                </div>

                <div className="flex items-center justify-end gap-2.5 pt-1">
                  <button
                    type="button"
                    onClick={handleCancelAddQuestion}
                    className="px-3.5 py-1.5 text-xs font-semibold text-slate-600 bg-white border border-slate-200 rounded-lg hover:bg-slate-100 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={handleAddQuestion}
                    className="inline-flex items-center gap-1.5 px-4 py-1.5 text-xs font-semibold text-white bg-emerald-600 rounded-lg hover:bg-emerald-700 transition-colors shadow-2xs"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    <span>Add Question</span>
                  </button>
                </div>
              </div>
            )}
          </div>

          {uiMode === 'student' ? (
            /* STUDENT MODE (DEFAULT): Question displayed read-only; Correct Answer NEVER shown */
            <div className="space-y-5">
              <div className="space-y-2">
                <div className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
                  <Code2 className="w-3.5 h-3.5 text-slate-500" />
                  <span>
                    Question ({currentCategory.label}
                    {currentBankItem ? ` — ${currentBankItem.label}` : ''})
                  </span>
                </div>
                {currentBankItem ? (
                  <>
                    <p className="text-sm font-medium text-slate-800">{currentBankItem.prompt}</p>
                    <pre className="w-full font-mono text-sm p-4 bg-slate-900 text-slate-100 rounded-xl border border-slate-800 whitespace-pre-wrap leading-relaxed">
                      {currentBankItem.question}
                    </pre>
                  </>
                ) : (
                  <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600">
                    No questions yet in <strong>{currentCategory.label}</strong>. Click{' '}
                    <strong>+ Add Question</strong> above to add one.
                  </div>
                )}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div>
                  <label
                    htmlFor="student-answer-input"
                    className="block text-xs font-semibold text-slate-700 mb-1.5"
                  >
                    Your Answer
                  </label>
                  <input
                    id="student-answer-input"
                    type="text"
                    value={studentAnswer}
                    onChange={(e) => handleFieldChange('studentAnswer', e.target.value)}
                    className="w-full font-mono text-xs px-3 py-2.5 bg-white text-slate-900 rounded-lg border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-blue-500"
                    placeholder="Enter the output of the code..."
                  />
                </div>

                <div>
                  <label
                    htmlFor="student-reasoning-input"
                    className="block text-xs font-semibold text-slate-700 mb-1.5"
                  >
                    Why did you choose this answer?
                  </label>
                  <textarea
                    id="student-reasoning-input"
                    rows={3}
                    value={studentReasoning}
                    onChange={(e) => handleFieldChange('studentReasoning', e.target.value)}
                    className="w-full text-xs p-3 bg-white text-slate-900 rounded-lg border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-blue-500"
                    placeholder="Explain how you worked out your answer..."
                  />
                </div>
              </div>
            </div>
          ) : (
            /* DEMO / INSTRUCTOR MODE: Allows custom Question and internal Correct Answer */
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1.5 flex items-center gap-1.5">
                    <Code2 className="w-3.5 h-3.5 text-slate-500" />
                    <span>Question (Custom Python Code)</span>
                  </label>
                  <textarea
                    rows={4}
                    value={question}
                    onChange={(e) => handleFieldChange('question', e.target.value)}
                    className="w-full font-mono text-xs p-3 bg-slate-900 text-slate-100 rounded-lg border border-slate-800 focus:outline-hidden focus:ring-2 focus:ring-blue-500"
                    placeholder="Enter Python code or question..."
                    required
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                    Expected Output (Internal Reference)
                  </label>
                  <input
                    type="text"
                    value={correctAnswer}
                    onChange={(e) => handleFieldChange('correctAnswer', e.target.value)}
                    className="w-full font-mono text-xs px-3 py-2.5 bg-slate-50 text-slate-900 rounded-lg border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-blue-500"
                    placeholder="Expected output or error (e.g. 3, 8, TypeError)"
                    required
                  />
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                    Your Answer
                  </label>
                  <input
                    type="text"
                    value={studentAnswer}
                    onChange={(e) => handleFieldChange('studentAnswer', e.target.value)}
                    className="w-full font-mono text-xs px-3 py-2.5 bg-white text-slate-900 rounded-lg border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-blue-500"
                    placeholder="Enter the output of the code..."
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                    Why did you choose this answer?
                  </label>
                  <textarea
                    rows={4}
                    value={studentReasoning}
                    onChange={(e) => handleFieldChange('studentReasoning', e.target.value)}
                    className="w-full text-xs p-3 bg-white text-slate-900 rounded-lg border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-blue-500"
                    placeholder="Explain how you worked out your answer..."
                  />
                </div>
              </div>
            </div>
          )}

          <div className="pt-2 flex items-center justify-end gap-3">
            <button
              type="button"
              onClick={handleClearForm}
              className="flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium text-slate-600 bg-slate-50 border border-slate-200 rounded-lg hover:bg-slate-100 hover:text-slate-900 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Clear Form</span>
            </button>
            <button
              type="submit"
              disabled={
                isDiagnosing ||
                !(uiMode === 'student'
                  ? (currentBankItem?.question || '').trim()
                  : question.trim())
              }
              className="flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors shadow-xs"
            >
              {isDiagnosing ? (
                <>
                  <Activity className="w-4 h-4 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Submit Answer</span>
                </>
              )}
            </button>
          </div>
        </form>
      </section>

      {diagnoseError && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-800 flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold">Diagnosis Error</div>
            <div className="mt-0.5">{diagnoseError}</div>
          </div>
        </div>
      )}

      {/* ===================================================================== */}
      {/* STAGE 2: DIAGNOSIS                                                    */}
      {/* ===================================================================== */}
      {session && diag && (
        <section className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
          <div className="px-6 py-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <span className="flex items-center justify-center w-6 h-6 rounded-full bg-indigo-600 text-white text-xs font-bold font-mono">
                2
              </span>
              <h2 className="text-base font-bold text-slate-900">Diagnosis</h2>
            </div>
            {diag.answer_correct !== undefined && (
              <span
                className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${
                  isRightAnswerWrongReason
                    ? 'bg-amber-50 text-amber-900 border-amber-300'
                    : diag.answer_correct
                    ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                    : 'bg-slate-100 text-slate-700 border-slate-200'
                }`}
              >
                {isRightAnswerWrongReason
                  ? 'Final Answer Correct · Misconception in Reasoning'
                  : diag.answer_correct
                  ? 'Final Answer Correct'
                  : 'Final Answer Incorrect'}
              </span>
            )}
          </div>

          <div className="p-6 space-y-5">
            {isRightAnswerWrongReason && (
              <div className="p-3.5 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-950 flex items-start gap-2.5">
                <Lightbulb className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                <div>
                  <strong>Right Answer for the Wrong Reason:</strong> Although the student&apos;s
                  final answer (
                  <code className="font-mono font-bold">
                    {studentAnswer.trim() || '(correct output)'}
                  </code>
                  ) matches the expected answer, ReLearn detected a conceptual misconception in their
                  reasoning.
                </div>
              </div>
            )}

            {/* 1. Misconception Name, 2. Confidence, 3. Error Type */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-4 bg-slate-50 border border-slate-200 rounded-xl">
              <div className="sm:col-span-1">
                <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                  Misconception Name
                </div>
                <div className="mt-1 font-bold text-sm text-slate-900">
                  {MISCONCEPTION_NAMES[diag.misconception_id] ||
                    diag.misconception_id ||
                    'INSUFFICIENT — Insufficient Evidence'}
                </div>
              </div>

              <div>
                <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                  Confidence
                </div>
                <div className="flex items-center gap-2.5 mt-1.5">
                  {(() => {
                    const pct = Number.isFinite(Number(diag.confidence))
                      ? Math.max(0, Math.min(100, Math.round(Number(diag.confidence) * 100)))
                      : 0;
                    return (
                      <>
                        <div className="w-24 bg-slate-200 h-2.5 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-600"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="font-mono font-bold text-sm text-slate-900">
                          {pct}%
                        </span>
                      </>
                    );
                  })()}
                </div>
              </div>

              <div>
                <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                  Error Type
                </div>
                <div className="mt-1">
                  <span className="inline-block font-mono text-xs font-semibold px-2.5 py-0.5 rounded bg-white border border-slate-200 text-slate-800">
                    {diag.error_type || (isMisconception ? 'conceptual' : 'other')}
                  </span>
                </div>
              </div>
            </div>

            {/* 4. Conceptual Evidence & 5. Short Rationale */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-white border border-slate-200 rounded-xl space-y-1.5">
                <div className="text-xs font-bold text-slate-700">Conceptual Evidence</div>
                <div className="text-xs text-slate-800 bg-amber-50/70 border border-amber-200/70 rounded-lg p-3 font-mono leading-relaxed">
                  {diag.evidence || 'No explicit conceptual clause extracted.'}
                </div>
              </div>

              <div className="p-4 bg-white border border-slate-200 rounded-xl space-y-1.5">
                <div className="text-xs font-bold text-slate-700">Short Rationale</div>
                <div className="text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded-lg p-3 leading-relaxed">
                  {diag.rationale ||
                    'Reasoning does not provide enough evidence to diagnose a specific misconception.'}
                </div>
              </div>
            </div>

            {/* Primary Button: Get Targeted Help */}
            {!showIntervention && (
              <div className="pt-1 flex justify-end">
                <button
                  type="button"
                  onClick={() => setShowIntervention(true)}
                  className="flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 transition-colors shadow-xs"
                >
                  <span>Get Targeted Help</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>
        </section>
      )}

      {/* ===================================================================== */}
      {/* STAGE 3: TARGETED INTERVENTION                                        */}
      {/* ===================================================================== */}
      {session && diag && showIntervention && (
        <section className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-xs">
          <div className="px-6 py-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <span className="flex items-center justify-center w-6 h-6 rounded-full bg-emerald-600 text-white text-xs font-bold font-mono">
                3
              </span>
              <h2 className="text-base font-bold text-slate-900">Targeted Intervention</h2>
            </div>
            <span className="text-xs font-mono font-semibold text-emerald-700 bg-emerald-50 px-2.5 py-0.5 rounded border border-emerald-200">
              {session.intervention?.title || diag.misconception_id || 'Guidance'}
            </span>
          </div>

          <div className="p-6 space-y-6">
            {session.intervention ? (
              <>
                {/* 1. Explanation */}
                <div className="p-4 bg-blue-50/70 border border-blue-200 rounded-xl space-y-1">
                  <div className="text-xs font-bold text-blue-900 uppercase tracking-wider">
                    Explanation
                  </div>
                  <p className="text-sm text-blue-950 leading-relaxed">
                    {session.intervention.short_explanation || diag.rationale}
                  </p>
                </div>

                {/* 2. Example */}
                {(session.intervention.concrete_example ||
                  session.intervention.contrast_example) && (
                  <div className="space-y-2">
                    <div className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                      Example
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {session.intervention.concrete_example && (
                        <div className="p-4 bg-slate-900 text-slate-100 rounded-xl border border-slate-800">
                          <div className="text-xs font-bold text-emerald-400 mb-2">
                            Concrete Code
                          </div>
                          <pre className="text-xs font-mono whitespace-pre-wrap leading-relaxed text-slate-200">
                            {session.intervention.concrete_example}
                          </pre>
                        </div>
                      )}
                      {session.intervention.contrast_example && (
                        <div className="p-4 bg-slate-900 text-slate-100 rounded-xl border border-slate-800">
                          <div className="text-xs font-bold text-amber-400 mb-2">
                            Contrast Comparison
                          </div>
                          <pre className="text-xs font-mono whitespace-pre-wrap leading-relaxed text-slate-200">
                            {session.intervention.contrast_example}
                          </pre>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* 3. Key Rule */}
                <div className="p-4 bg-slate-50 border-l-4 border-l-blue-600 border border-slate-200 rounded-r-xl">
                  <div className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                    Key Rule
                  </div>
                  <div className="text-sm font-semibold text-slate-900 mt-1">
                    {session.intervention.key_rule ||
                      'Apply the Python language rule step by step.'}
                  </div>
                </div>
              </>
            ) : (
              /* Non-M01..M08 guidance (NONE, INSUFFICIENT, OOS) */
              <div className="p-5 bg-slate-50 border border-slate-200 rounded-xl flex items-start gap-3.5">
                {diag.misconception_id === 'NONE' ? (
                  <CheckCircle2 className="w-6 h-6 text-emerald-600 shrink-0 mt-0.5" />
                ) : diag.misconception_id === 'INSUFFICIENT' ? (
                  <HelpCircle className="w-6 h-6 text-amber-600 shrink-0 mt-0.5" />
                ) : (
                  <BookOpen className="w-6 h-6 text-blue-600 shrink-0 mt-0.5" />
                )}
                <div className="space-y-1">
                  <div className="text-sm font-bold text-slate-900">
                    {diag.misconception_id === 'NONE'
                      ? 'No Misconception Remediation Needed'
                      : diag.misconception_id === 'INSUFFICIENT'
                      ? 'Clarification Needed'
                      : 'Topic Outside Supported CS1 Misconception Set'}
                  </div>
                  <p className="text-xs text-slate-700 leading-relaxed">
                    {session.message ||
                      diag.rationale ||
                      'Please explain the step-by-step rule you used to arrive at your answer.'}
                  </p>
                </div>
              </div>
            )}

            {/* 4. Follow-up Question & 5. Answer Input */}
            {session.followup && session.followup.question && (
              <div className="pt-4 border-t border-slate-200 space-y-4">
                <div className="p-4 bg-indigo-50/60 border border-indigo-200 rounded-xl space-y-2">
                  <div className="text-xs font-bold text-indigo-950 uppercase tracking-wider">
                    Follow-up Question
                  </div>
                  <pre className="text-xs font-mono text-indigo-950 whitespace-pre-wrap leading-relaxed bg-white p-3 rounded-lg border border-indigo-200/70">
                    {session.followup.question}
                  </pre>
                </div>

                <form onSubmit={handleEvaluateFollowup} className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Your Follow-up Answer
                      </label>
                      <input
                        type="text"
                        value={followupAnswer}
                        onChange={(e) => setFollowupAnswer(e.target.value)}
                        className="w-full font-mono text-xs px-3 py-2.5 bg-white text-slate-900 rounded-lg border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-indigo-500"
                        placeholder="Enter the exact output..."
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Explain Why (One Sentence)
                      </label>
                      <input
                        type="text"
                        value={followupReasoning}
                        onChange={(e) => setFollowupReasoning(e.target.value)}
                        className="w-full text-xs px-3 py-2.5 bg-white text-slate-900 rounded-lg border border-slate-300 focus:outline-hidden focus:ring-2 focus:ring-indigo-500"
                        placeholder="Explain the rule you used..."
                      />
                    </div>
                  </div>

                  <div className="flex justify-end">
                    <button
                      type="submit"
                      disabled={isEvaluating}
                      className="flex items-center gap-2 px-5 py-2 text-xs font-semibold text-white bg-emerald-600 rounded-lg hover:bg-emerald-700 disabled:opacity-50 transition-colors shadow-xs"
                    >
                      {isEvaluating ? (
                        <>
                          <Activity className="w-3.5 h-3.5 animate-spin" />
                          <span>Checking...</span>
                        </>
                      ) : (
                        <>
                          <Send className="w-3.5 h-3.5" />
                          <span>Submit Follow-up</span>
                        </>
                      )}
                    </button>
                  </div>
                </form>
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

            {/* 6. Resolution Result */}
            {resolutionResponse && resolutionResponse.resolution && (
              <div className="pt-4 border-t border-slate-200 space-y-4">
                <div
                  className={`p-5 rounded-xl border flex items-start gap-3.5 ${
                    resolutionResponse.resolution.status === 'RESOLVED'
                      ? 'bg-emerald-50 border-emerald-200'
                      : 'bg-amber-50 border-amber-200'
                  }`}
                >
                  {resolutionResponse.resolution.status === 'RESOLVED' ? (
                    <CheckCircle2 className="w-6 h-6 text-emerald-600 shrink-0 mt-0.5" />
                  ) : (
                    <XCircle className="w-6 h-6 text-amber-600 shrink-0 mt-0.5" />
                  )}
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span
                        className={`text-base font-extrabold ${
                          resolutionResponse.resolution.status === 'RESOLVED'
                            ? 'text-emerald-900'
                            : 'text-amber-900'
                        }`}
                      >
                        Resolution Result: {resolutionResponse.resolution.status || 'NOT_RESOLVED'}
                      </span>
                    </div>
                    <p className="text-xs text-slate-700 leading-relaxed">
                      {resolutionResponse.resolution.rationale ||
                        'Please review the conceptual rule and explain why your answer works.'}
                    </p>
                  </div>
                </div>

                {resolutionResponse.second_intervention && (
                  <div className="p-5 bg-amber-50/50 border border-amber-300 rounded-xl space-y-3">
                    <div className="text-xs font-bold text-amber-950 uppercase tracking-wider">
                      Step-by-Step Walkthrough:{' '}
                      {resolutionResponse.second_intervention.title || 'Conceptual Trace'}
                    </div>
                    {resolutionResponse.second_intervention.explanation && (
                      <p className="text-xs text-amber-900 leading-relaxed">
                        {resolutionResponse.second_intervention.explanation}
                      </p>
                    )}
                    {resolutionResponse.second_intervention.step_by_step_trace && (
                      <pre className="text-xs font-mono bg-slate-900 text-slate-100 p-4 rounded-lg whitespace-pre-wrap leading-relaxed">
                        {resolutionResponse.second_intervention.step_by_step_trace}
                      </pre>
                    )}
                    {resolutionResponse.second_intervention.key_takeaway && (
                      <div className="text-xs font-semibold text-amber-950">
                        Key Takeaway: {resolutionResponse.second_intervention.key_takeaway}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
};
