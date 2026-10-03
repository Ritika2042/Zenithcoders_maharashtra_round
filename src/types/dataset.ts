/**
 * Python CS1 Misconceptions Frozen Taxonomy (v1.0)
 * Dataset Schema and Annotation Types
 */

export type MisconceptionClassId =
  | 'M01'
  | 'M02'
  | 'M03'
  | 'M04'
  | 'M05'
  | 'M06'
  | 'M07'
  | 'M08';

export type NonClassOutcome = 'NONE' | 'INSUFFICIENT' | 'OOS';

export type MisconceptionId = MisconceptionClassId | NonClassOutcome;

export type ErrorType =
  | 'correct'
  | 'careless'
  | 'typo'
  | 'syntax_error'
  | 'trace_error'
  | 'conceptual'
  | 'other';

export type DatasetSplit = 'train' | 'val' | 'test';

export interface SampleRecord {
  id?: string;
  sample_id: string;
  question_group: string;
  concept: string;
  question_format: string;
  question: string;
  correct_answer: string;
  student_answer: string;
  student_reasoning: string;
  answer_correct: boolean;
  misconception_id: MisconceptionId;
  misconception_variant: string | null;
  secondary_misconception_ids: MisconceptionClassId[];
  evidence_basis: string;
  annotator_rationale: string;
  error_type: ErrorType;
  source: string;
  split: DatasetSplit;
}

export interface ValidationError {
  ruleNumber: string;
  field: keyof SampleRecord | 'general';
  message: string;
  severity: 'error' | 'warning';
}

export interface TaxonomyItem {
  id: MisconceptionClassId;
  name: string;
  definition: string;
  status: 'Unchanged' | 'Narrowed' | 'Unchanged; boundary clarified';
  exclusionNotes?: string;
  variants: string[];
}

export interface AnnotationRule {
  id: number;
  title: string;
  summary: string;
  details: string;
  practicalCheck: string;
}
