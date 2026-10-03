import React, { useState } from 'react';
import { FROZEN_TAXONOMY, NON_CLASS_OUTCOMES, ANNOTATION_RULES } from '../data/taxonomy';
import { BookOpen, HelpCircle, Layers, CheckCircle2, AlertCircle } from 'lucide-react';

export const TaxonomyCodex: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'taxonomy' | 'm05_m06' | 'narrowed' | 'rules' | 'schema'>('taxonomy');
  const [selectedMClassId, setSelectedMClassId] = useState<string>('M01');
  const [searchRule, setSearchRule] = useState<string>('');

  // Interactive M05 vs M06 tester
  const [testScenario, setTestScenario] = useState<number>(1);

  return (
    <div className="space-y-6">
      {/* Top Banner / Intro */}
      <div className="border border-slate-200 bg-white rounded-xl p-5 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                Taxonomy Status: FROZEN (v1.0)
              </span>
              <span className="text-xs text-slate-500">·</span>
              <span className="text-xs text-slate-500">M01–M08 · 13 Annotation Rules · Restored Locked Schema</span>
            </div>
            <h1 className="text-xl font-bold text-slate-900 mt-1.5">Taxonomy & Annotation Codex</h1>
            <p className="text-xs text-slate-600 mt-0.5">
              Definitive frozen definitions, boundary tests, and validation constraints for Python CS1 student misconceptions research.
            </p>
          </div>

          <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-lg border border-slate-200 shrink-0">
            <button
              onClick={() => setActiveTab('taxonomy')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                activeTab === 'taxonomy' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              M01–M08 & Outcomes
            </button>
            <button
              onClick={() => setActiveTab('m05_m06')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                activeTab === 'm05_m06' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              M05 vs M06 Boundary
            </button>
            <button
              onClick={() => setActiveTab('narrowed')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                activeTab === 'narrowed' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Narrowed Classes (M03/M07)
            </button>
            <button
              onClick={() => setActiveTab('rules')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                activeTab === 'rules' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              13 Annotation Rules
            </button>
            <button
              onClick={() => setActiveTab('schema')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                activeTab === 'schema' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Schema Constraints
            </button>
          </div>
        </div>
      </div>

      {/* Tab: Taxonomy M01-M08 & Outcomes */}
      {activeTab === 'taxonomy' && (
        <div className="space-y-6">
          <div className="border border-slate-200 bg-white rounded-xl overflow-hidden shadow-xs">
            <div className="px-5 py-3 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-blue-600" />
                Frozen Taxonomy Matrix (M01–M08)
              </h2>
              <span className="text-xs text-slate-500 font-mono">8 Core Classes</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold">
                  <tr>
                    <th className="py-2.5 px-4 w-16">ID</th>
                    <th className="py-2.5 px-4 w-60">Name</th>
                    <th className="py-2.5 px-4">Definition (Frozen)</th>
                    <th className="py-2.5 px-4 w-32">Status</th>
                    <th className="py-2.5 px-4 w-44">Typical Variants</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {FROZEN_TAXONOMY.map((item) => (
                    <tr
                      key={item.id}
                      onClick={() => setSelectedMClassId(item.id)}
                      className={`hover:bg-blue-50/40 cursor-pointer transition-colors ${
                        selectedMClassId === item.id ? 'bg-blue-50/60 font-medium' : ''
                      }`}
                    >
                      <td className="py-3 px-4 font-mono font-bold text-blue-700">{item.id}</td>
                      <td className="py-3 px-4 text-slate-900 font-semibold">{item.name}</td>
                      <td className="py-3 px-4 text-slate-700 leading-relaxed">
                        {item.definition}
                        {item.exclusionNotes && (
                          <div className="mt-1 text-slate-500 font-normal">
                            <span className="font-semibold text-amber-700">Boundary Note:</span> {item.exclusionNotes}
                          </div>
                        )}
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={`text-xs font-medium px-2 py-0.5 rounded ${
                            item.status === 'Narrowed'
                              ? 'bg-amber-100 text-amber-800 border border-amber-200'
                              : item.status.includes('clarified')
                              ? 'bg-blue-100 text-blue-800 border border-blue-200'
                              : 'bg-slate-100 text-slate-700 border border-slate-200'
                          }`}
                        >
                          {item.status}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-slate-500 font-mono text-[11px]">
                        {item.variants.map((v) => (
                          <div key={v} className="truncate" title={v}>
                            • {v}
                          </div>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Non-class outcomes */}
          <div className="border border-slate-200 bg-white rounded-xl overflow-hidden shadow-xs">
            <div className="px-5 py-3 border-b border-slate-200 bg-slate-50">
              <h2 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                <Layers className="w-4 h-4 text-indigo-600" />
                Non-Class Outcomes (Allowed Values of misconception_id)
              </h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-slate-200">
              {NON_CLASS_OUTCOMES.map((outcome) => (
                <div key={outcome.id} className="p-5 space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-bold text-sm text-slate-900 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                      {outcome.id}
                    </span>
                    <span className="text-xs font-semibold text-slate-700">{outcome.meaning}</span>
                  </div>
                  <p className="text-xs text-slate-600 leading-relaxed">{outcome.description}</p>
                  <div className="text-[11px] text-slate-500 pt-1 border-t border-slate-100">
                    <span className="font-semibold text-slate-700">Schema Rule:</span> secondary_misconception_ids and misconception_variant must be null/empty.
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Tab: M05 vs M06 Boundary */}
      {activeTab === 'm05_m06' && (
        <div className="space-y-6">
          <div className="border border-slate-200 bg-white rounded-xl p-5 shadow-xs space-y-4">
            <h2 className="text-sm font-semibold text-slate-900">M05 vs M06 Definitive Decision Test</h2>
            <p className="text-xs text-slate-600 leading-relaxed">
              When loops and sequences appear together, apply this four-part decision test to establish the primary misconception:
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div className="border border-blue-200 bg-blue-50/30 rounded-lg p-4 space-y-2">
                <div className="font-semibold text-blue-900 text-sm flex items-center gap-1.5">
                  <span className="font-mono bg-blue-600 text-white px-2 py-0.5 rounded text-xs">M05</span>
                  Index / Position
                </div>
                <div className="text-slate-700 font-medium">Core test: Belief is about where an element sits in a sequence.</div>
                <ul className="list-disc pl-4 space-y-1 text-slate-600">
                  <li>Zero-based vs 1-based indexing (`colors[1]` returns 1st item)</li>
                  <li>Negative indices relative to end or 0</li>
                  <li>Slice endpoints as positions within the sequence</li>
                  <li>Lookup off-by-one: accessing `x[i+1]` or `x[i-1]` based on wrong position belief</li>
                </ul>
              </div>

              <div className="border border-indigo-200 bg-indigo-50/30 rounded-lg p-4 space-y-2">
                <div className="font-semibold text-indigo-900 text-sm flex items-center gap-1.5">
                  <span className="font-mono bg-indigo-600 text-white px-2 py-0.5 rounded text-xs">M06</span>
                  Loop Values / Boundaries / Interval
                </div>
                <div className="text-slate-700 font-medium">Core test: Belief is about which values a loop takes or how many times it runs.</div>
                <ul className="list-disc pl-4 space-y-1 text-slate-600">
                  <li>Belief that `range(1, 5)` includes 5 (end-inclusive)</li>
                  <li>Belief that `range(n)` starts at 1 by default</li>
                  <li>Iteration count off-by-one (e.g. runs 5 times instead of 4)</li>
                  <li>Step interval misunderstanding (skipping values or step semantics)</li>
                </ul>
              </div>
            </div>

            {/* Interactive Scenario Walkthrough */}
            <div className="border border-slate-200 rounded-lg p-4 bg-slate-50 space-y-3">
              <div className="text-xs font-semibold text-slate-900 uppercase tracking-wider">
                Boundary Scenario Demonstration
              </div>

              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setTestScenario(1)}
                  className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                    testScenario === 1 ? 'bg-blue-600 text-white' : 'bg-white border border-slate-200 text-slate-700'
                  }`}
                >
                  Scenario A: Loop with x[i] Lookup
                </button>
                <button
                  type="button"
                  onClick={() => setTestScenario(2)}
                  className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                    testScenario === 2 ? 'bg-blue-600 text-white' : 'bg-white border border-slate-200 text-slate-700'
                  }`}
                >
                  Scenario B: range(len(x)) Endpoint
                </button>
              </div>

              {testScenario === 1 ? (
                <div className="bg-white p-3.5 rounded border border-slate-200 text-xs space-y-2">
                  <div className="font-mono bg-slate-900 text-slate-100 p-2 rounded">
                    {'for i in range(len(nums)):\n    print(nums[i + 1])  # Student thinks index 1 is first element'}
                  </div>
                  <div className="text-slate-700">
                    <span className="font-semibold text-blue-700">Code as M05:</span> The student understands that `i` takes values 0, 1, 2, but believes `nums[0+1]` points to the first element in the list because list positions start at 1. The root failure is element positioning (M05).
                  </div>
                </div>
              ) : (
                <div className="bg-white p-3.5 rounded border border-slate-200 text-xs space-y-2">
                  <div className="font-mono bg-slate-900 text-slate-100 p-2 rounded">
                    {'for i in range(len(nums)):  # nums has 3 elements\n    print(nums[i])  # Student thinks loop visits i = 0, 1, 2, 3'}
                  </div>
                  <div className="text-slate-700">
                    <span className="font-semibold text-indigo-700">Code as M06:</span> The student understands 0-based indexing for `nums[i]`, but believes `range(3)` produces `[0, 1, 2, 3]` because they think range endpoints are inclusive. The root failure is loop boundary generation (M06).
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tab: Narrowed Classes (M03 / M07) */}
      {activeTab === 'narrowed' && (
        <div className="space-y-6">
          <div className="border border-slate-200 bg-white rounded-xl p-5 shadow-xs space-y-4">
            <h2 className="text-sm font-semibold text-slate-900">Narrowed Classes & Explicit Exclusions</h2>
            <p className="text-xs text-slate-600 leading-relaxed">
              In the frozen v1.0 taxonomy, M03 and M07 are strictly narrowed. Concepts regarding memory aliasing, mutable references, or caller-side mutations are NOT permitted under M03 or M07.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5 text-xs">
              {/* M03 */}
              <div className="border border-amber-200 bg-amber-50/20 rounded-lg p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-mono font-bold text-amber-900 bg-amber-100 px-2 py-0.5 rounded">M03: Assignment vs Equality</span>
                  <span className="text-[11px] font-semibold text-amber-800">Status: Narrowed</span>
                </div>
                <div className="space-y-1.5 text-slate-700">
                  <div className="font-semibold text-slate-900">What is INCLUDED (Code M03):</div>
                  <div className="pl-3 border-l-2 border-amber-400">
                    Confusing assignment <code className="font-mono bg-amber-100 px-1 rounded">=</code> with equality comparison <code className="font-mono bg-amber-100 px-1 rounded">==</code> (e.g. writing `if x = 5:` expecting an equality check, or expecting `==` to store a value).
                  </div>

                  <div className="font-semibold text-slate-900 pt-2">What is EXCLUDED (Code OOS):</div>
                  <div className="pl-3 border-l-2 border-red-400 text-slate-600">
                    Beliefs about <strong>aliasing</strong>, object identity, reference copies (e.g. `b = a; b.append(1)` believing `b` is an isolated copy). Must be coded as <strong>OOS</strong> with error_type = 'other'.
                  </div>
                </div>
              </div>

              {/* M07 */}
              <div className="border border-amber-200 bg-amber-50/20 rounded-lg p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-mono font-bold text-amber-900 bg-amber-100 px-2 py-0.5 rounded">M07: Argument-Parameter Binding</span>
                  <span className="text-[11px] font-semibold text-amber-800">Status: Narrowed</span>
                </div>
                <div className="space-y-1.5 text-slate-700">
                  <div className="font-semibold text-slate-900">What is INCLUDED (Code M07):</div>
                  <div className="pl-3 border-l-2 border-amber-400">
                    Order of parameters, positional vs keyword matching, count mismatch, or believing parameter names must equal argument variable names in caller scope.
                  </div>

                  <div className="font-semibold text-slate-900 pt-2">What is EXCLUDED (Code OOS):</div>
                  <div className="pl-3 border-l-2 border-red-400 text-slate-600">
                    Beliefs about <strong>type conversion</strong> during passing or <strong>caller-side mutation</strong> (e.g., believing mutating a list parameter inside a function cannot affect the caller's list). Must be coded as <strong>OOS</strong>.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab: 11 Annotation Rules */}
      {activeTab === 'rules' && (
        <div className="space-y-4">
          <div className="border border-slate-200 bg-white rounded-xl p-5 shadow-xs">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
              <div>
                <h2 className="text-sm font-semibold text-slate-900">13 Final Frozen Annotation Rules</h2>
                <p className="text-xs text-slate-500">Every annotator must follow these 13 rules to ensure research validity and inter-rater reliability.</p>
              </div>
              <input
                type="text"
                placeholder="Search rule keywords..."
                value={searchRule}
                onChange={(e) => setSearchRule(e.target.value)}
                className="text-xs px-3 py-1.5 border border-slate-200 rounded-lg w-64 focus:outline-hidden focus:border-blue-500"
              />
            </div>

            <div className="space-y-3">
              {ANNOTATION_RULES.filter(
                (r) =>
                  !searchRule ||
                  r.title.toLowerCase().includes(searchRule.toLowerCase()) ||
                  r.summary.toLowerCase().includes(searchRule.toLowerCase()) ||
                  r.details.toLowerCase().includes(searchRule.toLowerCase())
              ).map((rule) => (
                <div key={rule.id} className="border border-slate-200 rounded-lg p-4 bg-slate-50/50 hover:bg-white transition-colors">
                  <div className="flex items-start gap-3">
                    <span className="font-mono text-xs font-bold bg-slate-900 text-white px-2 py-0.5 rounded shrink-0">
                      Rule {rule.id}
                    </span>
                    <div className="space-y-1 flex-1 text-xs">
                      <div className="font-semibold text-slate-900 text-sm">{rule.title}</div>
                      <div className="text-slate-800 font-medium">{rule.summary}</div>
                      <div className="text-slate-600 whitespace-pre-line leading-relaxed">{rule.details}</div>
                      <div className="bg-blue-50/60 border border-blue-100 rounded p-2 text-blue-900 font-mono text-[11px] mt-2">
                        <span className="font-bold">Practical check:</span> {rule.practicalCheck}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Tab: Schema Constraints */}
      {activeTab === 'schema' && (
        <div className="space-y-6">
          <div className="border border-slate-200 bg-white rounded-xl p-5 shadow-xs space-y-4">
            <h2 className="text-sm font-semibold text-slate-900">Restored Locked Dataset Schema & Constraints</h2>
            <p className="text-xs text-slate-600">
              The 17 restored fields and machine-validated integrity rules enforced by the studio:
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border border-slate-200 rounded-lg">
                <thead className="bg-slate-50 border-b border-slate-200 text-slate-700 font-semibold">
                  <tr>
                    <th className="py-2.5 px-4 w-48">Field</th>
                    <th className="py-2.5 px-4 w-32">Type</th>
                    <th className="py-2.5 px-4">Locked Constraint Specification</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono text-xs text-slate-700">
                  <tr>
                    <td className="py-2 px-4 font-semibold text-blue-700">sample_id</td>
                    <td className="py-2 px-4 text-slate-500">string</td>
                    <td className="py-2 px-4 text-slate-600">Unique record ID. Cannot be blank.</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-4 font-semibold text-blue-700">misconception_id</td>
                    <td className="py-2 px-4 text-slate-500">enum</td>
                    <td className="py-2 px-4 text-slate-600">One of M01-M08, NONE, INSUFFICIENT, or OOS.</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-4 font-semibold text-blue-700">misconception_variant</td>
                    <td className="py-2 px-4 text-slate-500">string | null</td>
                    <td className="py-2 px-4 text-slate-600">Must be NULL unless misconception_id is M01-M08.</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-4 font-semibold text-blue-700">secondary_misconception_ids</td>
                    <td className="py-2 px-4 text-slate-500">list&lt;M0x&gt;</td>
                    <td className="py-2 px-4 text-slate-600">Must be empty unless primary is M01-M08. Max 2 entries; no duplicates; cannot repeat primary.</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-4 font-semibold text-blue-700">evidence_basis</td>
                    <td className="py-2 px-4 text-slate-500">string</td>
                    <td className="py-2 px-4 text-slate-600">Must be non-null and point to learner's own words when misconception_id is M01-M08.</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-4 font-semibold text-blue-700">annotator_rationale</td>
                    <td className="py-2 px-4 text-slate-500">string</td>
                    <td className="py-2 px-4 text-slate-600">Required on every record. Exactly one sentence naming evidence or its absence.</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-4 font-semibold text-blue-700">error_type</td>
                    <td className="py-2 px-4 text-slate-500">enum</td>
                    <td className="py-2 px-4 text-slate-600">'conceptual' is allowed ONLY when misconception_id is M01-M08. 'NONE' requires correct/careless/typo/syntax_error. 'answer_correct=true' requires 'correct'.</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-4 font-semibold text-blue-700">split</td>
                    <td className="py-2 px-4 text-slate-500">'train'|'val'|'test'</td>
                    <td className="py-2 px-4 text-slate-600">Dataset partition.</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
