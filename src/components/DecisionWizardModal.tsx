import React, { useState } from 'react';
import { MisconceptionId, ErrorType } from '../types/dataset';
import { ArrowRight, CheckCircle2, RotateCcw, AlertTriangle, HelpCircle } from 'lucide-react';

interface DecisionWizardProps {
  onApplyRecommendation: (result: {
    misconception_id: MisconceptionId;
    error_type: ErrorType;
    rationaleSuggestion: string;
    suggestedSecondary?: string[];
  }) => void;
  onClose: () => void;
}

export const DecisionWizard: React.FC<DecisionWizardProps> = ({
  onApplyRecommendation,
  onClose,
}) => {
  const [step, setStep] = useState<number>(1);
  const [hasConceptualBelief, setHasConceptualBelief] = useState<boolean | null>(null);
  const [isOutOfScope, setIsOutOfScope] = useState<boolean | null>(null);
  const [oosReason, setOosReason] = useState<string>('');
  const [selectedMClass, setSelectedMClass] = useState<string>('');
  const [isTraceOnly, setIsTraceOnly] = useState<boolean | null>(null);
  const [isP23Fact, setIsP23Fact] = useState<boolean | null>(null);
  const [noneErrorType, setNoneErrorType] = useState<ErrorType>('careless');

  // Interactive M05 vs M06 Boundary Helper state
  const [m05VsM06Context, setM05VsM06Context] = useState<'sequence_position' | 'loop_values' | null>(null);

  const resetAll = () => {
    setStep(1);
    setHasConceptualBelief(null);
    setIsOutOfScope(null);
    setOosReason('');
    setSelectedMClass('');
    setIsTraceOnly(null);
    setIsP23Fact(null);
    setM05VsM06Context(null);
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white rounded-xl shadow-xl border border-slate-200 w-full max-w-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div>
            <h2 className="text-base font-semibold text-slate-900">Annotation Decision Wizard (13 Frozen Rules)</h2>
            <p className="text-xs text-slate-500">Step-by-step guidance following the frozen v1.0 taxonomy rules</p>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 text-sm font-medium px-2 py-1 rounded-md"
          >
            ✕
          </button>
        </div>

        {/* Wizard Body */}
        <div className="p-6 overflow-y-auto space-y-6 text-sm text-slate-700 flex-1">
          {/* Step indicator */}
          <div className="flex items-center gap-2 text-xs text-slate-500 border-b border-slate-100 pb-3">
            <span className={`font-semibold ${step === 1 ? 'text-blue-600' : ''}`}>1. Concept vs Slip</span>
            <span>→</span>
            <span className={`font-semibold ${step === 2 ? 'text-blue-600' : ''}`}>2. Out of Scope Check</span>
            <span>→</span>
            <span className={`font-semibold ${step === 3 ? 'text-blue-600' : ''}`}>3. M01-M08 Match</span>
            <span>→</span>
            <span className={`font-semibold ${step === 4 ? 'text-blue-600' : ''}`}>4. Decision</span>
          </div>

          {/* STEP 1: Conceptual belief check (Rule 1 & Rule 2 & Rule 3) */}
          {step === 1 && (
            <div className="space-y-4">
              <div className="bg-blue-50/70 border border-blue-100 rounded-lg p-3 text-xs text-blue-900">
                <span className="font-semibold">Rule 1, 2 & 3 Check:</span> An M-label requires explicit conceptual evidence in the student's reasoning or answer (Rule 1). A wrong answer alone NEVER proves a misconception (Rule 2). A trace mistake alone = INSUFFICIENT unless reasoning explicitly demonstrates a misconception (Rule 3).
              </div>

              <p className="font-medium text-slate-900">Does the student's reasoning state an explicit conceptual belief or mental rule?</p>

              <div className="space-y-2">
                <button
                  type="button"
                  onClick={() => {
                    setHasConceptualBelief(true);
                    setStep(2);
                  }}
                  className="w-full text-left p-3.5 border border-slate-200 rounded-lg hover:border-blue-500 hover:bg-blue-50/30 transition-colors flex items-start gap-3"
                >
                  <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" />
                  <div>
                    <div className="font-medium text-slate-900">Yes, student states a conceptual rule or belief</div>
                    <div className="text-xs text-slate-500 mt-0.5">E.g., "quotes are ignored", "single slash rounds down", "range is inclusive", "left to right evaluation"</div>
                  </div>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setHasConceptualBelief(false);
                    setStep(3); // branch to non-conceptual (trace slip, careless, correct, P23)
                  }}
                  className="w-full text-left p-3.5 border border-slate-200 rounded-lg hover:border-blue-500 hover:bg-blue-50/30 transition-colors flex items-start gap-3"
                >
                  <AlertTriangle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
                  <div>
                    <div className="font-medium text-slate-900">No explicit conceptual rule is stated</div>
                    <div className="text-xs text-slate-500 mt-0.5">Reasoning is empty, answers correctly, states an isolated math fact (Rule 5), makes a careless slip, or made a pure tracking/arithmetic trace mistake (Rule 3).</div>
                  </div>
                </button>
              </div>
            </div>
          )}

          {/* STEP 2: Out of Scope Check (Rules 4, 9, 10) */}
          {step === 2 && hasConceptualBelief && (
            <div className="space-y-4">
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs text-slate-700">
                <span className="font-semibold">Rules 4, 9, 10 (Out of Scope & Narrowed Classes):</span> If the belief is genuine but outside M01-M08, label as <strong className="text-amber-700">OOS</strong>.
                <div className="mt-1 text-slate-600">
                  • Print/output-format beliefs (spacing, quotes, newlines) → OOS (Rule 4)<br />
                  • Aliasing, memory references, mutable object identity → OOS (Rule 9: Do not use M03 for aliasing/references)<br />
                  • Caller-side mutable parameter mutation or type conversion → OOS (Rule 10: Do not use M07 for type conversion or caller-side mutation)
                </div>
              </div>

              <p className="font-medium text-slate-900">Does the student's belief fall into one of the Out-of-Scope (OOS) categories?</p>

              <div className="grid grid-cols-1 gap-2">
                <button
                  type="button"
                  onClick={() => {
                    setIsOutOfScope(true);
                    setOosReason('Aliasing, references, or object identity (narrowed M03 exclusion)');
                    setStep(4);
                  }}
                  className="text-left p-3 border border-slate-200 rounded-lg hover:border-amber-500 hover:bg-amber-50/30"
                >
                  <div className="font-medium text-slate-900">List Aliasing / Object Identity / Memory References</div>
                  <div className="text-xs text-slate-500">Learner believes b = a creates a separate copy or modifies separate memory (Narrowed M03 exclusion).</div>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setIsOutOfScope(true);
                    setOosReason('Caller-side parameter mutation or argument type conversion (narrowed M07 exclusion)');
                    setStep(4);
                  }}
                  className="text-left p-3 border border-slate-200 rounded-lg hover:border-amber-500 hover:bg-amber-50/30"
                >
                  <div className="font-medium text-slate-900">Caller-Side Parameter Mutation / Argument Type Conversion</div>
                  <div className="text-xs text-slate-500">Learner believes mutating a parameter in a function cannot alter the caller's object (Narrowed M07 exclusion).</div>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setIsOutOfScope(true);
                    setOosReason('Print output formatting: quotes, spaces, newlines (Rule 4)');
                    setStep(4);
                  }}
                  className="text-left p-3 border border-slate-200 rounded-lg hover:border-amber-500 hover:bg-amber-50/30"
                >
                  <div className="font-medium text-slate-900">Print Output Formatting (Quotes, Spacing, Commas, Newlines)</div>
                  <div className="text-xs text-slate-500">Learner believes print() outputs quotes or specific formatting (Rule 4).</div>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setIsOutOfScope(false);
                    setStep(3); // proceed to M01-M08 selection
                  }}
                  className="text-left p-3 border border-blue-200 bg-blue-50/20 rounded-lg hover:border-blue-500 font-medium text-blue-900"
                >
                  No, the belief relates to M01-M08 (Types, Division, = vs ==, Precedence, Indexing, Loops, Function Binding, Recursion)
                </button>
              </div>
            </div>
          )}

          {/* STEP 3 (Branch A): M01-M08 Match */}
          {step === 3 && hasConceptualBelief && !isOutOfScope && (
            <div className="space-y-4">
              <p className="font-medium text-slate-900">Select which frozen M-class matches the stated student belief:</p>
              
              <div className="space-y-2">
                <div className="p-3 border border-slate-200 rounded-lg bg-slate-50">
                  <span className="font-semibold text-xs text-slate-700">M05 vs M06 Decision Test:</span>
                  <div className="mt-1 text-xs text-slate-600 flex gap-2">
                    <button
                      type="button"
                      onClick={() => setM05VsM06Context('sequence_position')}
                      className={`px-2.5 py-1 rounded text-xs border ${m05VsM06Context === 'sequence_position' ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-slate-700 border-slate-300'}`}
                    >
                      Belief is about where an element sits in a sequence (M05)
                    </button>
                    <button
                      type="button"
                      onClick={() => setM05VsM06Context('loop_values')}
                      className={`px-2.5 py-1 rounded text-xs border ${m05VsM06Context === 'loop_values' ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-slate-700 border-slate-300'}`}
                    >
                      Belief is about loop values, range endpoint, or count (M06)
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  {[
                    { id: 'M01', name: 'M01: String vs Number Type Confusion' },
                    { id: 'M02', name: 'M02: Division Semantics (/ vs //)' },
                    { id: 'M03', name: 'M03: Assignment vs Equality (= vs == only)' },
                    { id: 'M04', name: 'M04: Operator Precedence' },
                    { id: 'M05', name: 'M05: Index / Position (0-based, negative, slices)' },
                    { id: 'M06', name: 'M06: Loop Values / Boundaries / Step' },
                    { id: 'M07', name: 'M07: Function Arg-Param Binding (order, names)' },
                    { id: 'M08', name: 'M08: Recursion Termination / Base Case' },
                  ].map((m) => (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => {
                        setSelectedMClass(m.id);
                        setStep(4);
                      }}
                      className={`p-2.5 border rounded-lg text-left transition-all ${
                        selectedMClass === m.id
                          ? 'border-blue-600 bg-blue-50 text-blue-900 font-semibold'
                          : 'border-slate-200 hover:border-slate-400 text-slate-800'
                      }`}
                    >
                      {m.name}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* STEP 3 (Branch B): Non-Conceptual Branch (Trace, P23, Careless, Typo, Correct) */}
          {step === 3 && !hasConceptualBelief && (
            <div className="space-y-4">
              <p className="font-medium text-slate-900">What is the nature of the student's answer or error?</p>

              <div className="space-y-2">
                <button
                  type="button"
                  onClick={() => {
                    setIsTraceOnly(true);
                    setStep(4);
                  }}
                  className="w-full text-left p-3 border border-slate-200 rounded-lg hover:border-blue-500 hover:bg-slate-50"
                >
                  <div className="font-medium text-slate-900">Trace / Mental Arithmetic Mistake (Rule 3)</div>
                  <div className="text-xs text-slate-500">Wrong intermediate calculation or loop variable tracking, but no conceptual rule stated. → INSUFFICIENT with error_type = trace_error.</div>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setIsP23Fact(true);
                    setStep(4);
                  }}
                  className="w-full text-left p-3 border border-slate-200 rounded-lg hover:border-blue-500 hover:bg-slate-50"
                >
                  <div className="font-medium text-slate-900">P23-Style Case: Isolated Math/Domain Knowledge (Rule 5)</div>
                  <div className="text-xs text-slate-500">Learner states a correct isolated fact (e.g. 0! = 1) with no demonstrated recursion misconception. → INSUFFICIENT.</div>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setNoneErrorType('correct');
                    setStep(4);
                  }}
                  className="w-full text-left p-3 border border-slate-200 rounded-lg hover:border-emerald-500 hover:bg-emerald-50/20"
                >
                  <div className="font-medium text-slate-900">Answer is Correct with sound/no misconception</div>
                  <div className="text-xs text-slate-500">Student arrived at correct solution. → NONE with error_type = correct.</div>
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setNoneErrorType('careless');
                    setStep(4);
                  }}
                  className="w-full text-left p-3 border border-slate-200 rounded-lg hover:border-slate-400 hover:bg-slate-50"
                >
                  <div className="font-medium text-slate-900">Careless Slip / Typo / Minor Syntax Error</div>
                  <div className="text-xs text-slate-500">Keyboard slip, typo, or missed colon with no conceptual misconception. → NONE with error_type in [careless, typo, syntax_error].</div>
                </button>
              </div>
            </div>
          )}

          {/* STEP 4: Summary & Recommendation */}
          {step === 4 && (
            <div className="space-y-4">
              <div className="p-4 rounded-xl border bg-slate-50 border-slate-200 space-y-3">
                <div className="text-xs font-semibold uppercase tracking-wider text-slate-500">Wizard Recommendation</div>
                
                {isOutOfScope && (
                  <div className="space-y-1">
                    <div className="text-lg font-bold text-amber-700">misconception_id = OOS</div>
                    <div className="text-xs text-slate-600">error_type = other</div>
                    <div className="text-xs text-slate-700 font-mono bg-white p-2 border border-slate-200 rounded">
                      Rationale: Student states an out-of-scope conceptual belief ({oosReason}), which is excluded from M01-M08.
                    </div>
                  </div>
                )}

                {selectedMClass && (
                  <div className="space-y-1">
                    <div className="text-lg font-bold text-blue-700">misconception_id = {selectedMClass}</div>
                    <div className="text-xs text-slate-600">error_type = conceptual</div>
                    <div className="text-xs text-slate-700 font-mono bg-white p-2 border border-slate-200 rounded">
                      Rationale: Student explicitly demonstrates {selectedMClass} by stating a conceptual belief in the reasoning.
                    </div>
                  </div>
                )}

                {isTraceOnly && (
                  <div className="space-y-1">
                    <div className="text-lg font-bold text-slate-700">misconception_id = INSUFFICIENT</div>
                    <div className="text-xs text-slate-600">error_type = trace_error</div>
                    <div className="text-xs text-slate-700 font-mono bg-white p-2 border border-slate-200 rounded">
                      Rationale: Under Rule 3, an arithmetic or execution trace mistake without an explicit conceptual misconception is coded INSUFFICIENT.
                    </div>
                  </div>
                )}

                {isP23Fact && (
                  <div className="space-y-1">
                    <div className="text-lg font-bold text-slate-700">misconception_id = INSUFFICIENT</div>
                    <div className="text-xs text-slate-600">error_type = trace_error</div>
                    <div className="text-xs text-slate-700 font-mono bg-white p-2 border border-slate-200 rounded">
                      Rationale: Under Rule 5, isolated factual knowledge with no demonstrated recursion misconception is coded INSUFFICIENT.
                    </div>
                  </div>
                )}

                {!isOutOfScope && !selectedMClass && !isTraceOnly && !isP23Fact && (
                  <div className="space-y-1">
                    <div className="text-lg font-bold text-emerald-700">misconception_id = NONE</div>
                    <div className="text-xs text-slate-600">error_type = {noneErrorType}</div>
                    <div className="text-xs text-slate-700 font-mono bg-white p-2 border border-slate-200 rounded">
                      Rationale: {noneErrorType === 'correct' 
                        ? 'Student answer is correct and no misconception is demonstrated.'
                        : `Student made a ${noneErrorType} error with absence of any conceptual misconception.`}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-200 bg-slate-50 flex items-center justify-between">
          <button
            type="button"
            onClick={resetAll}
            className="flex items-center gap-1.5 text-xs text-slate-600 hover:text-slate-900 font-medium py-1.5 px-3 rounded border border-slate-200 bg-white"
          >
            <RotateCcw className="w-3.5 h-3.5" /> Start Over
          </button>

          {step === 4 ? (
            <button
              type="button"
              onClick={() => {
                if (isOutOfScope) {
                  onApplyRecommendation({
                    misconception_id: 'OOS',
                    error_type: 'other',
                    rationaleSuggestion: `Student explicitly states an out-of-scope conceptual belief (${oosReason}), which is excluded from M01-M08.`,
                  });
                } else if (selectedMClass) {
                  onApplyRecommendation({
                    misconception_id: selectedMClass as MisconceptionId,
                    error_type: 'conceptual',
                    rationaleSuggestion: `Student explicitly demonstrates ${selectedMClass} by stating a conceptual belief in the reasoning.`,
                  });
                } else if (isTraceOnly || isP23Fact) {
                  onApplyRecommendation({
                    misconception_id: 'INSUFFICIENT',
                    error_type: 'trace_error',
                    rationaleSuggestion: isP23Fact
                      ? 'Under Rule 5, isolated factual knowledge with no demonstrated recursion misconception is coded INSUFFICIENT.'
                      : 'Under Rule 3, an arithmetic or execution trace mistake without an explicit conceptual misconception is coded INSUFFICIENT.',
                  });
                } else {
                  onApplyRecommendation({
                    misconception_id: 'NONE',
                    error_type: noneErrorType,
                    rationaleSuggestion: noneErrorType === 'correct'
                      ? 'Student answer is correct and no conceptual misconception is evidenced.'
                      : `Student committed a ${noneErrorType} error with absence of any conceptual misconception.`,
                  });
                }
                onClose();
              }}
              className="flex items-center gap-1.5 text-xs bg-blue-600 text-white font-medium py-1.5 px-4 rounded-lg hover:bg-blue-700 transition-colors shadow-xs"
            >
              Apply to Annotation Form <ArrowRight className="w-3.5 h-3.5" />
            </button>
          ) : (
            <button
              type="button"
              onClick={onClose}
              className="text-xs text-slate-600 hover:text-slate-900 font-medium py-1.5 px-3"
            >
              Cancel
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
