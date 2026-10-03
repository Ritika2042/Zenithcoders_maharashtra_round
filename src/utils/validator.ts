import { SampleRecord, ValidationError, MisconceptionClassId } from '../types/dataset';

const VALID_M_CLASSES: MisconceptionClassId[] = [
  'M01',
  'M02',
  'M03',
  'M04',
  'M05',
  'M06',
  'M07',
  'M08',
];

export function isMisconceptionClass(id: string): id is MisconceptionClassId {
  return VALID_M_CLASSES.includes(id as MisconceptionClassId);
}

/**
 * Validates a single SampleRecord against the frozen schema constraints and 13 annotation rules.
 */
export function validateRecord(record: SampleRecord): ValidationError[] {
  const errors: ValidationError[] = [];

  const isMClass = isMisconceptionClass(record.misconception_id);

  // Constraint 1 / Rule 8: secondary_misconception_ids must be empty unless misconception_id is M01-M08
  if (!isMClass && record.secondary_misconception_ids && record.secondary_misconception_ids.length > 0) {
    errors.push({
      ruleNumber: 'Constraint 1 / Rule 8',
      field: 'secondary_misconception_ids',
      message: `secondary_misconception_ids must be empty when misconception_id is '${record.misconception_id}'. Found: [${record.secondary_misconception_ids.join(', ')}]`,
      severity: 'error',
    });
  }

  // Rule 8: Secondary IDs limits: at most 2, no duplicates, cannot repeat primary
  if (record.secondary_misconception_ids) {
    if (record.secondary_misconception_ids.length > 2) {
      errors.push({
        ruleNumber: 'Rule 8',
        field: 'secondary_misconception_ids',
        message: `At most 2 secondary misconceptions are allowed. Found ${record.secondary_misconception_ids.length}.`,
        severity: 'error',
      });
    }

    if (isMisconceptionClass(record.misconception_id) && record.secondary_misconception_ids.includes(record.misconception_id)) {
      errors.push({
        ruleNumber: 'Rule 8',
        field: 'secondary_misconception_ids',
        message: `secondary_misconception_ids cannot repeat the primary label '${record.misconception_id}'.`,
        severity: 'error',
      });
    }

    const uniqueSet = new Set(record.secondary_misconception_ids);
    if (uniqueSet.size !== record.secondary_misconception_ids.length) {
      errors.push({
        ruleNumber: 'Rule 8',
        field: 'secondary_misconception_ids',
        message: 'secondary_misconception_ids cannot contain duplicate entries.',
        severity: 'error',
      });
    }
  }

  // Constraint 2 / Rule 1: evidence_basis must be non-null and point to the learner's own words when misconception_id is M01-M08
  if (isMClass) {
    if (!record.evidence_basis || record.evidence_basis.trim().length === 0) {
      errors.push({
        ruleNumber: 'Constraint 2 / Rule 1',
        field: 'evidence_basis',
        message: `evidence_basis is mandatory when misconception_id is '${record.misconception_id}'. An M01-M08 label requires explicit conceptual evidence in the student's reasoning or answer.`,
        severity: 'error',
      });
    } else {
      // Check if evidence basis appears in student reasoning or student answer (or has quote)
      const evidence = record.evidence_basis.toLowerCase().trim();
      const reasoning = (record.student_reasoning || '').toLowerCase();
      const answer = (record.student_answer || '').toLowerCase();
      
      const wordsMatch = evidence.split(/\s+/).some((word) => word.length > 3 && (reasoning.includes(word) || answer.includes(word)));
      if (!wordsMatch && evidence.length > 10 && !evidence.includes('absence') && !reasoning.includes(evidence)) {
        errors.push({
          ruleNumber: 'Rule 1',
          field: 'evidence_basis',
          message: 'evidence_basis should point directly to the learner\'s own words or explicitly state evidence presence.',
          severity: 'warning',
        });
      }
    }
  }

  // Rule 9: Do not use M03 for aliasing/references
  if (record.misconception_id === 'M03') {
    const textToCheck = `${record.student_reasoning} ${record.evidence_basis} ${record.annotator_rationale}`.toLowerCase();
    if (textToCheck.includes('alias') || textToCheck.includes('referen') || textToCheck.includes('copy of list') || textToCheck.includes('separate copy')) {
      errors.push({
        ruleNumber: 'Rule 9',
        field: 'misconception_id',
        message: 'Rule 9: Do not use M03 for aliasing/references. Beliefs about aliasing or memory references must be coded OOS.',
        severity: 'warning',
      });
    }
  }

  // Rule 10: Do not use M07 for type conversion or caller-side mutation
  if (record.misconception_id === 'M07') {
    const textToCheck = `${record.student_reasoning} ${record.evidence_basis} ${record.annotator_rationale}`.toLowerCase();
    if (textToCheck.includes('caller-side') || textToCheck.includes('mutate') || textToCheck.includes('mutation') || textToCheck.includes('type conversion')) {
      errors.push({
        ruleNumber: 'Rule 10',
        field: 'misconception_id',
        message: 'Rule 10: Do not use M07 for type conversion or caller-side mutation. These must be coded OOS.',
        severity: 'warning',
      });
    }
  }

  // Constraint 3: error_type = conceptual is allowed only when misconception_id is M01-M08 or OOS
  if (record.error_type === 'conceptual' && !isMClass && record.misconception_id !== 'OOS') {
    errors.push({
      ruleNumber: 'Constraint 3 / Rule 8',
      field: 'error_type',
      message: `error_type = 'conceptual' is allowed ONLY when misconception_id is M01-M08 or OOS. Current misconception_id is '${record.misconception_id}'.`,
      severity: 'error',
    });
  }

  // Constraint 4: misconception_id = NONE requires error_type of correct, careless, typo or syntax_error
  const allowedNoneErrorTypes = ['correct', 'careless', 'typo', 'syntax_error'];
  if (record.misconception_id === 'NONE' && !allowedNoneErrorTypes.includes(record.error_type)) {
    errors.push({
      ruleNumber: 'Constraint 4',
      field: 'error_type',
      message: `misconception_id = 'NONE' requires error_type to be one of: [${allowedNoneErrorTypes.join(', ')}]. Found '${record.error_type}'.`,
      severity: 'error',
    });
  }

  // Constraint 5: misconception_id = NONE with answer_correct = true requires error_type = correct
  if (record.misconception_id === 'NONE' && record.answer_correct === true && record.error_type !== 'correct') {
    errors.push({
      ruleNumber: 'Constraint 5',
      field: 'error_type',
      message: `When misconception_id = 'NONE' and answer_correct = true, error_type must be 'correct'. Found '${record.error_type}'.`,
      severity: 'error',
    });
  }

  // Constraint 7: misconception_variant must be null unless misconception_id is M01-M08
  if (!isMClass && record.misconception_variant !== null && record.misconception_variant !== '') {
    errors.push({
      ruleNumber: 'Constraint 7',
      field: 'misconception_variant',
      message: `misconception_variant must be null when misconception_id is '${record.misconception_id}'. Found '${record.misconception_variant}'.`,
      severity: 'error',
    });
  }

  // Rule 12: Every record must have a one-sentence annotator_rationale
  if (!record.annotator_rationale || record.annotator_rationale.trim().length === 0) {
    errors.push({
      ruleNumber: 'Rule 12',
      field: 'annotator_rationale',
      message: 'Rule 12: Every record must have a one-sentence annotator_rationale.',
      severity: 'error',
    });
  } else {
    const rationale = record.annotator_rationale.trim();
    // One sentence check: count sentence-ending punctuation not inside quotes/parens
    const sentenceTerminators = rationale.match(/[.!?](\s+[A-Z]|$)/g);
    if (sentenceTerminators && sentenceTerminators.length > 1) {
      errors.push({
        ruleNumber: 'Rule 12',
        field: 'annotator_rationale',
        message: 'Rule 12: annotator_rationale must be exactly one sentence.',
        severity: 'warning',
      });
    }

    const mentionsEvidence = /evidence|stated|words|stated belief|demonstrate|show|absence|quote|reasoning|careless|correct/i.test(rationale);
    if (!mentionsEvidence) {
      errors.push({
        ruleNumber: 'Rule 12',
        field: 'annotator_rationale',
        message: 'Rule 12: annotator_rationale must explicitly name the evidence or its absence.',
        severity: 'warning',
      });
    }
  }

  // Rule 13: Keep question_group IDs so related/near-duplicate questions can later be kept in the same train/test split.
  if (!record.question_group || record.question_group.trim() === '') {
    errors.push({
      ruleNumber: 'Rule 13',
      field: 'question_group',
      message: 'Rule 13: question_group ID is required to keep related/near-duplicate questions in the same split.',
      severity: 'error',
    });
  }

  // Schema baseline required fields
  if (!record.sample_id || record.sample_id.trim() === '') {
    errors.push({
      ruleNumber: 'Schema',
      field: 'sample_id',
      message: 'sample_id cannot be empty.',
      severity: 'error',
    });
  }
  if (!record.question || record.question.trim() === '') {
    errors.push({
      ruleNumber: 'Schema',
      field: 'question',
      message: 'question text cannot be empty.',
      severity: 'error',
    });
  }

  return errors;
}

export function auditDataset(records: SampleRecord[]): {
  total: number;
  validCount: number;
  invalidCount: number;
  warningCount: number;
  violationsByRecord: { sample_id: string; errors: ValidationError[] }[];
  splitLeakageGroups: { question_group: string; splits: string[]; sample_ids: string[] }[];
} {
  let validCount = 0;
  let invalidCount = 0;
  let warningCount = 0;
  const violationsByRecord: { sample_id: string; errors: ValidationError[] }[] = [];

  // Rule 13 audit: Group-level split distribution check
  const groupSplitsMap: Record<string, { splits: Set<string>; sample_ids: string[] }> = {};
  for (const record of records) {
    if (!record.question_group) continue;
    if (!groupSplitsMap[record.question_group]) {
      groupSplitsMap[record.question_group] = { splits: new Set(), sample_ids: [] };
    }
    groupSplitsMap[record.question_group].splits.add(record.split);
    groupSplitsMap[record.question_group].sample_ids.push(record.sample_id);
  }

  const splitLeakageGroups: { question_group: string; splits: string[]; sample_ids: string[] }[] = [];
  for (const [group, data] of Object.entries(groupSplitsMap)) {
    if (data.splits.size > 1) {
      splitLeakageGroups.push({
        question_group: group,
        splits: Array.from(data.splits),
        sample_ids: data.sample_ids,
      });
    }
  }

  for (const record of records) {
    const issues = validateRecord(record);

    // If this record belongs to a leakage group, add a Rule 13 warning
    const isLeaked = splitLeakageGroups.some((g) => g.question_group === record.question_group);
    if (isLeaked) {
      const groupData = splitLeakageGroups.find((g) => g.question_group === record.question_group);
      issues.push({
        ruleNumber: 'Rule 13',
        field: 'split',
        message: `Rule 13 Warning: question_group '${record.question_group}' has records spanning multiple splits ([${groupData?.splits.join(', ')}]). Near-duplicate questions must be kept in the same split.`,
        severity: 'warning',
      });
    }

    const hasError = issues.some((i) => i.severity === 'error');
    const hasWarning = issues.some((i) => i.severity === 'warning');

    if (hasError) {
      invalidCount++;
      violationsByRecord.push({ sample_id: record.sample_id, errors: issues });
    } else {
      validCount++;
      if (hasWarning) {
        warningCount++;
        violationsByRecord.push({ sample_id: record.sample_id, errors: issues });
      }
    }
  }

  return {
    total: records.length,
    validCount,
    invalidCount,
    warningCount,
    violationsByRecord,
    splitLeakageGroups,
  };
}
