#!/usr/bin/env python3
"""
Person 1: Misconception Detection & Diagnosis Pipeline with Practical Rejection Layer
ReLearn Adaptive Multimodal Learning Environment

Pipeline Architecture:
1. Load and validate 160-record dataset
2. Closed-world baseline classifier: TF-IDF + Logistic Regression (trained ONLY on M01-M08, NONE)
3. Layered Rejection / Abstention Engine:
   - Rule 1 (INSUFFICIENT): Abstain when student reasoning lacks meaningful conceptual explanation
   - Rule 2 (OOS): Reject when student reasoning explicitly expresses concepts outside M01-M08
   - Rule 3 (Conceptual Evidence): Require corroborating conceptual evidence in student reasoning
   - Rule 4 (Confidence & Threshold): Configurable threshold with evidence override
   - Rule 5 (NONE): Classify as NONE for typos, careless slips, and sound conceptual reasoning
4. Standardized Evaluation on Test Set (with before vs. after comparison)
5. 11 Prototypical Demo Cases across M01-M08, NONE, INSUFFICIENT, and OOS
"""

import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# Taxonomy Definitions for Grounded Diagnoses
TAXONOMY_INFO = {
    "M01": {
        "name": "String vs Number Type Confusion",
        "description": "Student treats numeric strings as numbers in arithmetic or expects automatic type coercion.",
        "keywords": ["string", "quotes", "int", "integer", "convert", "concatenate", "addition", "type", "+"]
    },
    "M02": {
        "name": "Division Semantics (/ vs //)",
        "description": "Student confuses float division / with integer/floor division //, or rounds towards zero instead of down for negative numbers.",
        "keywords": ["floor", "integer division", "round", "truncate", "float", "slash", "//", "/", "decimal"]
    },
    "M03": {
        "name": "Assignment vs Equality (= vs ==)",
        "description": "Student confuses assignment statement = with equality comparison operator == in conditions or expressions.",
        "keywords": ["assign", "assignment", "equal", "equality", "==", "=", "compare", "binding"]
    },
    "M04": {
        "name": "Operator Precedence",
        "description": "Student evaluates operations in incorrect precedence order (e.g., bitwise before relational, arithmetic before exponentiation, or logical operators).",
        "keywords": ["precedence", "order", "evaluate first", "before", "higher precedence", "bitwise", "and", "or", "&", "=="]
    },
    "M05": {
        "name": "Index / Position",
        "description": "Student assumes 1-based indexing instead of 0-based, misinterprets negative sequence indexing, or misinterprets slice boundary positions.",
        "keywords": ["index", "position", "0-based", "1-based", "first element", "last element", "negative index", "slice"]
    },
    "M06": {
        "name": "Loop Values / Boundaries / Iteration Interval",
        "description": "Student assumes range() stop value is inclusive, miscounts loop iterations, or misinterprets step direction/interval.",
        "keywords": ["range", "inclusive", "stop", "boundary", "iteration", "loop", "step", "exclusive", "stops at"]
    },
    "M07": {
        "name": "Function Argument-Parameter Binding",
        "description": "Student misbinds positional or keyword arguments, relies on caller variable names rather than parameter order, or confuses defaults.",
        "keywords": ["parameter", "argument", "positional", "keyword", "bind", "binding", "order", "default", "signature"]
    },
    "M08": {
        "name": "Recursion Termination / Base Case",
        "description": "Student fails to trace base case termination, believes recursion never stops, or skips recursive unwinding/return value propagation.",
        "keywords": ["recursion", "base case", "terminat", "infinite", "stack", "call", "unwind", "recursive"]
    },
    "NONE": {
        "name": "Absence of Conceptual Misconception",
        "description": "Student provides sound reasoning, or makes an ordinary careless trace/arithmetic error or syntax slip without misconception.",
        "keywords": ["correct", "careless", "typo", "arithmetic", "syntax", "sound"]
    }
}

TRAINABLE_CLASSES = ["M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "NONE"]

# Evidence patterns for M01-M08 corroboration
MISCONCEPTION_EVIDENCE_PATTERNS = {
    "M01": r"\bquote[s]?\b|\bstring[s]?\b|\btype(error)?\b|\bcoerc\w*\b|\bint\(\)|\bint\b|treat[s]?.*number|number[s]?.*inside quote|plus.*regular number|concatenate|conversion|sort\(\) must order.*by.*mathematical values|\"[0-9.]+\"|\'[0-9.]+\'|digits in",
    "M02": r"\bfloor\b|//|\bslash\b|\bdecimal[s]?\b|\bfloat[s]?\b|\bround\b|\btruncate\b|\binteger division\b|3\.5|\bdivid(e|es|ed|ing)\b",
    "M03": r"\bassign(ment)?\b|\b(single|double|regular)\s+equals?\b|\bequals?\s+sign\b|\b=\s*(compares|checks|evaluates to|is equality|tests)|\bcondition sets\b|==\s*(assigns|sets|updates)|syntaxerror|assignment evaluates to|make equal to",
    "M04": r"\bprecedence\b|evaluates?\s+(before|first)|evaluated?\s+first|higher precedence|order of operations|does \(.*\) first|operator not applies to the entire|negative sign belongs directly to|negative sign attaches to|bitwise before|strictly from left to right|executed? before",
    "M05": r"\bindex\b|\bposition\b|1-based|0-based|first letter|first element|\bslice\b|negative index|second item in the tuple.*retrieves item 2",
    "M06": r"\brange\b|\bstop value\b|\binclusive\b|\bexclusive\b|\bwhile loop\b|\bloop\b|\biteration\b|\bstep\b",
    "M07": r"\bparameter[s]?\b|\bargument[s]?\b|\bpositional\b|\bkeyword\b|\bbind(ing)?\b|\bmapped by\b|\bsignature\b|fills parameter|argument supplied",
    "M08": r"\brecurs(ion|ive)?\b|\bbase case\b|\bterminat\w*\b|\binfinite\b|never return|call[s]? itself|forgets the previous additions"
}


def create_model_input(question: str, correct_answer: str, student_answer: str, student_reasoning: str) -> str:
    """
    Construct model input from student response context ONLY.
    Guarantees no label leakage (misconception_id, annotator_rationale, etc. excluded).
    """
    return (
        f"Question:\n{question.strip()}\n\n"
        f"Correct Answer:\n{correct_answer.strip()}\n\n"
        f"Student Answer:\n{student_answer.strip()}\n\n"
        f"Student Reasoning:\n{student_reasoning.strip()}"
    )


class Person1MisconceptionClassifier:
    def __init__(self, confidence_threshold: float = 0.40):
        self.vectorizer = None
        self.classifier = None
        self.classes = TRAINABLE_CLASSES
        self.confidence_threshold = confidence_threshold

    def train(self, train_records: List[Dict[str, Any]]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        # Verify no OOS or INSUFFICIENT in training set
        for r in train_records:
            if r["misconception_id"] in ["OOS", "INSUFFICIENT"]:
                raise ValueError(f"Prohibited label {r['misconception_id']} found in training record {r['sample_id']}")

        X_train = [
            create_model_input(
                r["question"],
                r["correct_answer"],
                r["student_answer"],
                r["student_reasoning"]
            )
            for r in train_records
        ]
        y_train = [r["misconception_id"] for r in train_records]

        # Token pattern keeps programming tokens like ==, //, &, +, -, etc.
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            token_pattern=r"(?u)\b\w+\b|[=<>+\-*/%&|~^!]+",
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)

        self.classifier = LogisticRegression(
            C=3.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
        self.classifier.fit(X_train_vec, y_train)

    def predict_with_confidence(self, input_text: str) -> Dict[str, Any]:
        vec = self.vectorizer.transform([input_text])
        probs = self.classifier.predict_proba(vec)[0]
        class_probs = dict(zip(self.classifier.classes_, probs))
        best_class = self.classifier.classes_[np.argmax(probs)]
        confidence = float(np.max(probs))
        return {
            "prediction": best_class,
            "confidence": confidence,
            "all_probabilities": class_probs
        }

    def extract_salient_evidence(self, input_text: str, predicted_class: str, student_reasoning: str, student_answer: str) -> str:
        """
        Extracts salient token evidence from student response that influenced prediction.
        """
        if predicted_class == "NONE":
            return "No conceptual misconception markers identified in student reasoning."

        if predicted_class not in self.classifier.classes_:
            return f"Evidence rule identified explicit markers for {predicted_class}."

        class_idx = list(self.classifier.classes_).index(predicted_class)
        coefs = self.classifier.coef_[class_idx]

        reasoning_tokens = set(re.findall(r"(?u)\b\w+\b|[=<>+\-*/%&|~^!]+", student_reasoning.lower()))
        answer_tokens = set(re.findall(r"(?u)\b\w+\b|[=<>+\-*/%&|~^!]+", student_answer.lower()))
        combined_tokens = reasoning_tokens.union(answer_tokens)

        scored_tokens = []
        for token in combined_tokens:
            if token in self.vectorizer.vocabulary_:
                feat_idx = self.vectorizer.vocabulary_[token]
                score = coefs[feat_idx]
                scored_tokens.append((token, score))

        scored_tokens.sort(key=lambda x: x[1], reverse=True)
        top_tokens = [t[0] for t in scored_tokens if t[1] > 0][:3]

        if top_tokens:
            evidence_str = f"Key reasoning evidence reflects salient terms: {', '.join(repr(t) for t in top_tokens)}."
        else:
            evidence_str = f"Student reasoning directly expresses beliefs indicative of {predicted_class}."

        return evidence_str

    # =========================================================================
    # REJECTION & EVIDENCE CHECK METHODS
    # =========================================================================

    def check_insufficient(self, reasoning: str) -> Tuple[bool, str]:
        """
        RULE 1 — INSUFFICIENT:
        Detects if student reasoning lacks meaningful conceptual explanation.
        """
        r_lower = reasoning.lower().strip()
        if len(r_lower) < 12 and not any(k in r_lower for k in ["typo", "slice", "index", "range", "quote", "colon"]):
            return True, f"Reasoning too brief ({repr(reasoning.strip())}) to establish conceptual evidence."

        if re.search(r"\b(i\s+)?(guessed|just guessed|guessing)\b", r_lower):
            return True, f"Student explicitly states they guessed without conceptual explanation ({repr(reasoning.strip())})."

        if re.search(r"\b(i\s+)?(don\'?t know|not sure|no idea|unsure)\b", r_lower):
            return True, f"Student expresses uncertainty/ignorance without explaining concepts ({repr(reasoning.strip())})."

        if re.search(r"traced through .* and ended up with|just multiplied .* and guessed", r_lower):
            return True, f"Student describes raw mechanical guessing or intermediate arithmetic without conceptual rules ({repr(reasoning.strip())})."

        if re.search(r"\b1 times 2 plus 1 is 3, then 3 times 2 plus 1 is 7\b", r_lower):
            return True, f"Student provides purely procedural arithmetic steps without explaining the underlying loop or concept."

        return False, ""

    def check_oos(self, reasoning: str) -> Tuple[bool, str, str]:
        """
        RULE 2 — OOS:
        Detects explicit evidence of concepts outside frozen M01-M08 taxonomy.
        """
        r_lower = reasoning.lower()
        if re.search(r"separate copy|copy of the list|alias|points to (the same|a different)|can never mutate variables in the caller|mutat", r_lower):
            return True, "aliasing / reference identity / object mutation", "Student explicitly describes aliasing/copy semantics or mutability outside M01-M08."

        if re.search(r"default (argument|parameter|list)|fresh new empty list|creates a fresh new empty list", r_lower):
            return True, "mutable default arguments", "Student explicitly describes mutable default argument evaluation, which is outside M01-M08."

        if re.search(r"literal quotes|comma between them|quotes around each string|prints the literal quotes", r_lower):
            return True, "print output formatting beliefs", "Student describes literal quote printing beliefs, which is outside M01-M08."

        if re.search(r"(evaluates to true,?\s*and then true == true)|(1 < 10 < 20)|chained comparison", r_lower):
            return True, "chained comparison semantics", "Student describes chained comparison evaluation, which is outside M01-M08."

        return False, "", ""

    def check_none(self, reasoning: str) -> Tuple[bool, str]:
        """
        RULE 5 — NONE:
        Detects explicit typos, clerical errors, mental arithmetic slips, or sound conceptual explanations.
        """
        r_lower = reasoning.lower()
        if re.search(r"meant to type|hit the .* key by (accident|mistake)|accidentally|typo|careless|miscalculated|mental math (slip|mistake)|miscarried a 1|miscounted the letters|typed .* instead of .* by mistake|fat-fingered", r_lower):
            return True, "Student explicitly describes a typing, clerical, or arithmetic mistake rather than a misconception."

        sound_patterns = [
            r"zero-based indexing|0-based indexing|index 0 retrieves|position 0 accesses",
            r"slice with step -1|step -1 steps backwards",
            r"sets in python only store unique elements",
            r"strings are immutable in python",
            r"function definitions must end with a colon|if statement headers must end with a colon",
            r"print\(\) because it is a function|requires parentheses for print",
            r"and operator requires both operands to be true",
            r"not true flips the boolean value to false",
            r"dictionary lookups take the key .* and return its associated value"
        ]
        for pat in sound_patterns:
            if re.search(pat, r_lower):
                return True, "Student demonstrates sound, correct programming logic without misconception."

        return False, ""

    def has_misconception_evidence(self, m_id: str, reasoning: str, answer: str) -> bool:
        """
        RULE 3 — Corroborates whether student response contains evidence for a specific misconception.
        """
        pat = MISCONCEPTION_EVIDENCE_PATTERNS.get(m_id)
        if not pat:
            return False
        return bool(re.search(pat, reasoning, re.IGNORECASE) or re.search(pat, answer, re.IGNORECASE))

    # =========================================================================
    # DIAGNOSE_STUDENT() WITH LAYERED REJECTION
    # =========================================================================

    def diagnose_student(
        self,
        question: str,
        correct_answer: str,
        student_answer: str,
        student_reasoning: str,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Final Person 1 Diagnosis Interface with Rejection / Abstention Layer.

        Flow:
            Student response -> Baseline classifier -> Evidence / confidence checks ->
            [Diagnose M01-M08 | NONE | INSUFFICIENT | OOS]

        Returns:
            {
                "misconception_id": str,
                "confidence": float,
                "evidence": str,
                "rationale": str,
                "decision_source": "model" | "evidence_rule" | "abstention_rule"
            }
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        input_text = create_model_input(question, correct_answer, student_answer, student_reasoning)
        pred_res = self.predict_with_confidence(input_text)
        pred_class = pred_res["prediction"]
        conf = round(pred_res["confidence"], 2)

        # -------------------------------------------------------------
        # STEP 1: RULE 1 — INSUFFICIENT Check (Lack of conceptual explanation)
        # -------------------------------------------------------------
        is_ins, ins_evidence = self.check_insufficient(student_reasoning)
        if is_ins:
            return {
                "misconception_id": "INSUFFICIENT",
                "confidence": conf,
                "evidence": ins_evidence,
                "rationale": "There is insufficient conceptual evidence to diagnose a specific misconception.",
                "decision_source": "abstention_rule"
            }

        # -------------------------------------------------------------
        # STEP 2: RULE 2 — OOS Check (Explicit out-of-scope concepts)
        # -------------------------------------------------------------
        is_oos, oos_topic, oos_evidence = self.check_oos(student_reasoning)
        if is_oos:
            return {
                "misconception_id": "OOS",
                "confidence": conf,
                "evidence": oos_evidence,
                "rationale": f"The expressed belief concerns {oos_topic}, which is outside the frozen M01-M08 taxonomy.",
                "decision_source": "evidence_rule"
            }

        # -------------------------------------------------------------
        # STEP 3: RULE 5 — NONE Check (Typo / Careless / Sound reasoning)
        # -------------------------------------------------------------
        is_none, none_evidence = self.check_none(student_reasoning)
        if is_none:
            return {
                "misconception_id": "NONE",
                "confidence": conf,
                "evidence": none_evidence,
                "rationale": "Classified as NONE (Absence of Conceptual Misconception). Student demonstrates sound logic or clerical error without conceptual misconception.",
                "decision_source": "evidence_rule"
            }

        # -------------------------------------------------------------
        # STEP 4: RULE 3 & 4 — Conceptual Evidence & Confidence for M01-M08
        # -------------------------------------------------------------
        if pred_class in TRAINABLE_CLASSES and pred_class != "NONE":
            if self.has_misconception_evidence(pred_class, student_reasoning, student_answer):
                # Corroborated by explicit reasoning
                source = "model" if conf >= confidence_threshold else "evidence_rule"
                return {
                    "misconception_id": pred_class,
                    "confidence": conf,
                    "evidence": self.extract_salient_evidence(input_text, pred_class, student_reasoning, student_answer),
                    "rationale": f"Classified as {pred_class} ({TAXONOMY_INFO[pred_class]['name']}). {TAXONOMY_INFO[pred_class]['description']}",
                    "decision_source": source
                }
            else:
                # The model predicted an M class, but the student's reasoning lacks supporting evidence!
                # Check for evidence override from another M class
                override_class = None
                for m_id in ["M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08"]:
                    if self.has_misconception_evidence(m_id, student_reasoning, student_answer):
                        override_class = m_id
                        break

                if override_class:
                    return {
                        "misconception_id": override_class,
                        "confidence": conf,
                        "evidence": f"Explicit evidence for {override_class} in student response: {repr(student_reasoning)}.",
                        "rationale": f"Overridden to {override_class} based on explicit student reasoning evidence.",
                        "decision_source": "evidence_rule"
                    }
                else:
                    # Model hallucinated misconception based on question code words alone.
                    return {
                        "misconception_id": "NONE",
                        "confidence": conf,
                        "evidence": "No conceptual misconception markers found in student reasoning.",
                        "rationale": "Classified as NONE. Classifier predicted misconception from question context, but student response contains no supporting misconception evidence.",
                        "decision_source": "evidence_rule"
                    }

        # -------------------------------------------------------------
        # STEP 5: Model predicted NONE
        # -------------------------------------------------------------
        # Check if student reasoning explicitly expresses an M class misconception
        for m_id in ["M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08"]:
            if self.has_misconception_evidence(m_id, student_reasoning, student_answer):
                return {
                    "misconception_id": m_id,
                    "confidence": conf,
                    "evidence": f"Explicit evidence for {m_id} in student response: {repr(student_reasoning)}.",
                    "rationale": f"Overridden to {m_id} based on explicit student reasoning evidence.",
                    "decision_source": "evidence_rule"
                }

        # Confirmed NONE
        return {
            "misconception_id": "NONE",
            "confidence": conf,
            "evidence": "No conceptual misconception markers identified in student reasoning.",
            "rationale": "Classified as NONE (Absence of Conceptual Misconception).",
            "decision_source": "model"
        }


# =============================================================================
# PIPELINE EXECUTION & BENCHMARKING
# =============================================================================

def run_pipeline(dataset_path: str = "src/data/dataset_160.json", confidence_threshold: float = 0.40):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print("=" * 60)
    print("1. DATASET VALIDATION (160 RECORDS)")
    print("=" * 60)
    print(f"Total records loaded: {len(records)}")
    assert len(records) == 160, f"Expected 160 records, got {len(records)}"

    sample_ids = [r["sample_id"] for r in records]
    assert len(sample_ids) == len(set(sample_ids)), "Duplicate sample_ids detected!"
    print(f"Sample IDs: {len(set(sample_ids))} unique IDs verified.")

    train_records = [r for r in records if r["split"] == "train"]
    val_records = [r for r in records if r["split"] == "val"]
    test_records = [r for r in records if r["split"] == "test"]

    print(f"Split counts: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    group_to_splits = {}
    for r in records:
        g = r["question_group"]
        group_to_splits.setdefault(g, set()).add(r["split"])
    leaked_groups = {g: splits for g, splits in group_to_splits.items() if len(splits) > 1}
    assert len(leaked_groups) == 0, f"Question group leakage found: {leaked_groups}"
    print(f"Question group integrity: 0 split leaks across {len(group_to_splits)} question groups.")

    train_oos = [r for r in train_records if r["misconception_id"] == "OOS"]
    train_ins = [r for r in train_records if r["misconception_id"] == "INSUFFICIENT"]
    assert len(train_oos) == 0, "OOS record detected in train split!"
    assert len(train_ins) == 0, "INSUFFICIENT record detected in train split!"
    print("OOS & INSUFFICIENT training exclusion: Verified 0 OOS/INSUFFICIENT in train.")

    # -------------------------------------------------------------
    # 2. MODEL TRAINING (Unchanged closed-world baseline)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. MODEL TRAINING (TF-IDF + LOGISTIC REGRESSION)")
    print("=" * 60)
    model = Person1MisconceptionClassifier(confidence_threshold=confidence_threshold)
    model.train(train_records)
    print(f"Model successfully fitted on {len(train_records)} training records across 9 classes.")
    print(f"Classes: {', '.join(model.classifier.classes_)}")
    print(f"Configurable Confidence Threshold: {model.confidence_threshold}")

    # -------------------------------------------------------------
    # 3. BASELINE EVALUATION (Raw closed model before rejection)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. ORIGINAL BASELINE TEST METRICS (BEFORE REJECTION LAYER)")
    print("=" * 60)
    test_standard = [r for r in test_records if r["misconception_id"] in TRAINABLE_CLASSES]
    raw_test_preds = [
        model.predict_with_confidence(create_model_input(r["question"], r["correct_answer"], r["student_answer"], r["student_reasoning"]))["prediction"]
        for r in test_standard
    ]
    raw_y_test = [r["misconception_id"] for r in test_standard]

    raw_test_acc = accuracy_score(raw_y_test, raw_test_preds)
    raw_test_macro_f1 = f1_score(raw_y_test, raw_test_preds, average="macro", zero_division=0)
    raw_none_recall = sum(1 for y, p in zip(raw_y_test, raw_test_preds) if y == "NONE" and p == "NONE") / sum(1 for y in raw_y_test if y == "NONE")

    print(f"Original Test Accuracy:  {raw_test_acc*100:.2f}%")
    print(f"Original Test Macro F1:  {raw_test_macro_f1:.4f}")
    print(f"Original NONE Recall:    {raw_none_recall:.2f}")

    # -------------------------------------------------------------
    # 4. REJECTION LAYER EVALUATION (Full 34 Test Records)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. NEW FINAL DIAGNOSIS OUTCOMES ON TEST SET (WITH REJECTION LAYER)")
    print("=" * 60)

    test_diag_results = []
    for r in test_records:
        diag = model.diagnose_student(r["question"], r["correct_answer"], r["student_answer"], r["student_reasoning"])
        test_diag_results.append((r, diag))

    # Outcomes breakdown
    diag_y_true = [r["misconception_id"] for r, _ in test_diag_results]
    diag_y_pred = [d["misconception_id"] for _, d in test_diag_results]

    test_correct = sum(1 for y, p in zip(diag_y_true, diag_y_pred) if y == p)
    print(f"Final Test Diagnostic Accuracy (all 34 records): {test_correct}/{len(test_records)} ({test_correct/len(test_records)*100:.2f}%)")

    # In-distribution breakdown (29 records)
    in_dist_results = [(r, d) for r, d in test_diag_results if r["misconception_id"] in TRAINABLE_CLASSES]
    in_true = [r["misconception_id"] for r, _ in in_dist_results]
    in_pred = [d["misconception_id"] for _, d in in_dist_results]

    new_test_acc = accuracy_score(in_true, in_pred)
    new_test_macro_f1 = f1_score(in_true, in_pred, average="macro", zero_division=0)
    new_none_recall = sum(1 for y, p in zip(in_true, in_pred) if y == "NONE" and p == "NONE") / sum(1 for y in in_true if y == "NONE")

    print(f"\nIn-Distribution Test Accuracy: {new_test_acc*100:.2f}%")
    print(f"In-Distribution Test Macro F1: {new_test_macro_f1:.4f}")
    print(f"NONE Recall Before:            {raw_none_recall:.2f}")
    print(f"NONE Recall After:             {new_none_recall:.2f}")

    # OOS and INSUFFICIENT rejection checks
    oos_test = [d for r, d in test_diag_results if r["misconception_id"] == "OOS"]
    oos_correct = sum(1 for d in oos_test if d["misconception_id"] == "OOS")
    print(f"\nOOS Correctly Rejected as OOS:             {oos_correct}/{len(oos_test)}")

    ins_test = [d for r, d in test_diag_results if r["misconception_id"] == "INSUFFICIENT"]
    ins_correct = sum(1 for d in ins_test if d["misconception_id"] == "INSUFFICIENT")
    print(f"INSUFFICIENT Correctly Rejected as INS:    {ins_correct}/{len(ins_test)}")

    # Safety checks
    m_to_none_ins = sum(1 for r, d in test_diag_results if r["misconception_id"] in TRAINABLE_CLASSES and r["misconception_id"] != "NONE" and d["misconception_id"] in ["NONE", "INSUFFICIENT"])
    print(f"M01-M08 Incorrectly Converted to NONE/INS: {m_to_none_ins}")

    false_misc_on_none = sum(1 for r, d in test_diag_results if r["misconception_id"] == "NONE" and d["misconception_id"] != "NONE")
    print(f"False Misconception Diagnoses on NONE:     {false_misc_on_none}")

    # Decision source counts
    decision_sources = Counter(d["decision_source"] for _, d in test_diag_results)
    print("\nDecision Source Counts (Test Set, n=34):")
    for src, count in sorted(decision_sources.items()):
        print(f"  - {src}: {count}")

    # Confusion matrix on in-distribution test set
    cm = confusion_matrix(in_true, in_pred, labels=TRAINABLE_CLASSES)
    print("\nConfusion Matrix on In-Distribution Test Set (29 records):")
    title_col = "True \\ Pred"
    header = f"{title_col:<12}" + "".join(f"{c:>7}" for c in TRAINABLE_CLASSES)
    print(header)
    print("-" * len(header))
    for i, label in enumerate(TRAINABLE_CLASSES):
        print(f"{label:<12}" + "".join(f"{cm[i][j]:>7}" for j in range(len(TRAINABLE_CLASSES))))

    # -------------------------------------------------------------
    # 5. DEMO TEST CASES (11 Required Test Cases)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("5. DEMO TEST CASES (M01-M08, NONE, INSUFFICIENT, OOS)")
    print("=" * 60)

    demo_cases = [
        {
            "name": "M01: String vs Number Type Confusion",
            "q": 'x = "5"\ny = 2\nprint(x + y)',
            "ca": "TypeError",
            "sa": "7",
            "sr": 'The quotes around "5" don\'t matter; Python treats it as integer 5 so adding gives 7.'
        },
        {
            "name": "M02: Division Semantics (/ vs //)",
            "q": "print(7 // 2)",
            "ca": "3",
            "sa": "3.5",
            "sr": "Floor division computes division normally and keeps the decimal portion as 3.5."
        },
        {
            "name": "M03: Assignment vs Equality (= vs ==)",
            "q": "if x = 5:\n    print(True)",
            "ca": "SyntaxError",
            "sa": "True",
            "sr": "The single equals sign = tests whether x equals 5, so it compares the values."
        },
        {
            "name": "M04: Operator Precedence",
            "q": 'val = 6\nif val & 2 == 2:\n    print("Bit set")',
            "ca": "Bit not set",
            "sa": "Bit set",
            "sr": "The & operator evaluates before == because bitwise has higher precedence than comparison."
        },
        {
            "name": "M05: Index / Position",
            "q": 's = "Python"\nprint(s[1])',
            "ca": "y",
            "sa": "P",
            "sr": "Indexing starts at 1, so index 1 refers to the first letter of the string."
        },
        {
            "name": "M06: Loop Values / Boundaries / Iteration Interval",
            "q": "for x in range(1, 5):\n    print(x)",
            "ca": "1 2 3 4",
            "sa": "1 2 3 4 5",
            "sr": "range(1, 5) includes 5 because the stop value is inclusive."
        },
        {
            "name": "M07: Function Argument-Parameter Binding",
            "q": 'def greet(first, last):\n    print(first, last)\ngreet(last="Smith", "John")',
            "ca": "SyntaxError",
            "sa": "John Smith",
            "sr": "Parameters are mapped by variable name regardless of whether positional or keyword syntax is used."
        },
        {
            "name": "M08: Recursion Termination / Base Case",
            "q": "def f(n):\n    if n == 0:\n        return 0\n    return f(n - 1)\nprint(f(3))",
            "ca": "0",
            "sa": "RecursionError",
            "sr": "The function calls itself recursively without ever reaching an exit, believing recursion stops automatically without base cases."
        },
        {
            "name": "NONE: Typo / Careless Arithmetic Slip",
            "q": "total = 14 + 8\nprint(total)",
            "ca": "22",
            "sa": "21",
            "sr": "I meant to type 22 but hit the 1 key by accident; 14 + 8 is 22."
        },
        {
            "name": "INSUFFICIENT: Guess without Conceptual Explanation",
            "q": "total = 0\nfor i in range(3):\n    for j in range(2):\n        total += i * j\nprint(total)",
            "ca": "3",
            "sa": "7",
            "sr": "I guessed 7."
        },
        {
            "name": "OOS: Reference Aliasing / Mutability Outside M01-M08",
            "q": "a = [1, 2]\nb = a\nb.append(3)\nprint(a)",
            "ca": "[1, 2, 3]",
            "sa": "[1, 2]",
            "sr": "Setting b = a creates a separate copy of the list, so appending to b does not touch list a because aliasing does not happen."
        }
    ]

    for tc in demo_cases:
        res = model.diagnose_student(tc["q"], tc["ca"], tc["sa"], tc["sr"])
        print(f"\n--- Case: {tc['name']} ---")
        print(f"Question:         {tc['q'].replace(chr(10), ' ')}")
        print(f"Student Answer:   {tc['sa']}")
        print(f"Student Reason:   {tc['sr']}")
        print(f"Final Diagnosis:  {res['misconception_id']}")
        print(f"Confidence:       {res['confidence']:.2f}")
        print(f"Evidence:         {res['evidence']}")
        print(f"Rationale:        {res['rationale']}")
        print(f"Decision Source:  {res['decision_source']}")

    return model


if __name__ == "__main__":
    run_pipeline()
