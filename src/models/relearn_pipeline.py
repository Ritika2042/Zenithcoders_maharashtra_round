#!/usr/bin/env python3
"""
ReLearn End-to-End Pipeline: Person 1 Diagnosis -> Person 2 Adaptive Intervention & Resolution
ReLearn Adaptive Multimodal Learning Environment

Integrates:
- Person 1 (LOCKED): /src/models/baseline_pipeline.py
  Outputs: { "misconception_id", "confidence", "evidence", "rationale", "decision_source" }
- Person 2 (LOCKED): /src/models/intervention_engine.py
  Outputs: { "intervention", "followup", "resolution", "second_intervention" }

Workflow:
1. Student Input (Question + Correct Answer + Student Answer + Student Reasoning)
2. Person 1 diagnose_student() -> Authoritative Diagnosis
3. Person 2 run_person2() -> Targeted Intervention or Routing Response (NONE, INS, OOS)
4. Follow-up Question served (if M01-M08)
5. Student Follow-up Answer submitted
6. Person 2 evaluate_followup() -> RESOLVED or NOT_RESOLVED
7. If NOT_RESOLVED -> Person 2 generate_second_intervention()
"""

import json
import os
import sys
from typing import Dict, Any, Optional

# Ensure imports resolve from current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from baseline_pipeline import Person1MisconceptionClassifier
from intervention_engine import (
    run_person2,
    evaluate_followup,
    generate_second_intervention
)


def find_dataset_path(default_path: str = "src/data/dataset_160.json") -> str:
    """Resolve dataset path from workspace root or models directory."""
    if os.path.exists(default_path):
        return default_path
    alt_path = os.path.join(CURRENT_DIR, "..", "data", "dataset_160.json")
    if os.path.exists(alt_path):
        return os.path.abspath(alt_path)
    return default_path


TRACKED_MISCONCEPTION_IDS = {"M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08"}


class ReLearnPipeline:
    """
    Unified End-to-End Pipeline coordinating Person 1 diagnosis, Person 2 adaptive remediation,
    and lightweight in-memory Learner Model tracking across attempts.
    """

    def __init__(self, confidence_threshold: float = 0.40, dataset_path: Optional[str] = None):
        if dataset_path is None:
            dataset_path = find_dataset_path()

        with open(dataset_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        train_records = [r for r in records if r["split"] == "train"]

        # Initialize and fit Person 1 classifier on train split (87 records)
        self.classifier = Person1MisconceptionClassifier(confidence_threshold=confidence_threshold)
        self.classifier.train(train_records)

        # Initialize lightweight in-memory Learner Model state
        self.learner_state: Dict[str, Any] = {}
        self.reset_learner_state()

    def reset_learner_state(self) -> Dict[str, Any]:
        """Resets only the learner progress history for a fresh session/demo."""
        self.learner_state = {
            "attempts": 0,
            "misconception_detections": 0,
            "resolved_count": 0,
            "unresolved_count": 0,
            "last_status": "READY",
            "misconceptions": {},
        }
        return self.get_learner_state()

    def get_learner_state(self) -> Dict[str, Any]:
        """Returns a clean snapshot of the current learner state."""
        sorted_misconceptions = {
            k: dict(v)
            for k, v in sorted(self.learner_state["misconceptions"].items())
        }
        mastered_count = sum(
            1 for v in sorted_misconceptions.values() if v.get("current_status") == "MASTERED"
        )
        needs_practice_count = sum(
            1 for v in sorted_misconceptions.values() if v.get("current_status") == "NEEDS_PRACTICE"
        )
        awaiting_followup_count = sum(
            1 for v in sorted_misconceptions.values() if v.get("current_status") == "AWAITING_FOLLOWUP"
        )
        return {
            "attempts": self.learner_state["attempts"],
            "misconception_detections": self.learner_state["misconception_detections"],
            "resolved_count": self.learner_state["resolved_count"],
            "unresolved_count": self.learner_state["unresolved_count"],
            "mastered_count": mastered_count,
            "needs_practice_count": needs_practice_count,
            "awaiting_followup_count": awaiting_followup_count,
            "last_status": self.learner_state["last_status"],
            "misconceptions": sorted_misconceptions,
        }

    def _record_diagnosis_in_learner_model(self, misconception_id: str) -> None:
        """
        Updates learner state when a student response is diagnosed:
        - Increments total attempts for every diagnosis (M01-M08, NONE, INSUFFICIENT, OOS).
        - Only tracks M01-M08 inside misconceptions[id].detected and sets
          current_status = "AWAITING_FOLLOWUP" (even if previously MASTERED or NEEDS_PRACTICE).
        - Does NOT create misconception entries for NONE, INSUFFICIENT, or OOS.
        """
        self.learner_state["attempts"] += 1
        if misconception_id in TRACKED_MISCONCEPTION_IDS:
            if misconception_id not in self.learner_state["misconceptions"]:
                self.learner_state["misconceptions"][misconception_id] = {
                    "detected": 0,
                    "resolved": 0,
                    "unresolved": 0,
                    "current_status": "AWAITING_FOLLOWUP",
                }
            self.learner_state["misconceptions"][misconception_id]["detected"] += 1
            self.learner_state["misconceptions"][misconception_id]["current_status"] = "AWAITING_FOLLOWUP"
            self.learner_state["misconception_detections"] += 1
            self.learner_state["last_status"] = f"DETECTED:{misconception_id}"
        else:
            self.learner_state["last_status"] = misconception_id

    def _record_resolution_in_learner_model(
        self,
        original_misconception_id: str,
        resolution_status: str,
        resolution_type: Optional[str] = None,
    ) -> None:
        """
        Updates learner state when a follow-up response is evaluated against the
        ORIGINAL diagnosed misconception (M01-M08):
        - RESOLVED -> increments resolved & resolved_count, sets current_status = "MASTERED"
          (historical unresolved counter is preserved and never decremented).
        - NOT_RESOLVED (including CLARIFICATION_NEEDED) -> increments unresolved &
          unresolved_count, sets current_status = "NEEDS_PRACTICE".
        """
        if original_misconception_id not in TRACKED_MISCONCEPTION_IDS:
            return

        if original_misconception_id not in self.learner_state["misconceptions"]:
            self.learner_state["misconceptions"][original_misconception_id] = {
                "detected": 1,
                "resolved": 0,
                "unresolved": 0,
                "current_status": "AWAITING_FOLLOWUP",
            }

        entry = self.learner_state["misconceptions"][original_misconception_id]

        if resolution_status == "RESOLVED":
            entry["resolved"] += 1
            entry["current_status"] = "MASTERED"
            self.learner_state["resolved_count"] += 1
            self.learner_state["last_status"] = f"RESOLVED:{original_misconception_id}"
        elif resolution_status == "NOT_RESOLVED" or resolution_type == "CLARIFICATION_NEEDED":
            entry["unresolved"] += 1
            entry["current_status"] = "NEEDS_PRACTICE"
            self.learner_state["unresolved_count"] += 1
            self.learner_state["last_status"] = f"NOT_RESOLVED:{original_misconception_id}"

    def diagnose_and_intervene(
        self,
        question: str,
        correct_answer: str,
        student_answer: str,
        student_reasoning: str
    ) -> Dict[str, Any]:
        """
        Executes Stage 1:
        1. Calls Person 1 diagnose_student() for authoritative diagnosis.
        2. Updates Learner Model state for this attempt.
        3. Passes diagnosis and student context to Person 2 run_person2().
        4. Returns structured session state including learner_state.
        """
        q = str(question if question is not None else "")
        ca = str(correct_answer if correct_answer is not None else "")
        sa = str(student_answer if student_answer is not None else "")
        sr = str(student_reasoning if student_reasoning is not None else "")

        # Context preserving original student interaction
        context = {
            "question": q,
            "correct_answer": ca,
            "student_answer": sa,
            "student_reasoning": sr
        }

        try:
            # Step 1: Person 1 diagnosis
            diagnosis = self.classifier.diagnose_student(
                question=q,
                correct_answer=ca,
                student_answer=sa,
                student_reasoning=sr
            )
        except Exception as exc:
            diagnosis = {
                "misconception_id": "INSUFFICIENT",
                "confidence": 0.75,
                "evidence": f"Could not extract clear conceptual evidence from input ({str(exc)[:80]}).",
                "rationale": "Reasoning does not provide enough evidence to diagnose a specific misconception.",
                "decision_source": "abstention_rule",
                "misconception_variant": None,
                "secondary_misconception_ids": [],
                "error_type": "other",
                "answer_correct": False,
            }

        # Ensure all required diagnosis keys exist with safe defaults
        m_id = str(diagnosis.get("misconception_id") or "INSUFFICIENT")
        diagnosis["misconception_id"] = m_id
        diagnosis["confidence"] = float(diagnosis.get("confidence", 0.75) or 0.75)
        diagnosis["evidence"] = str(diagnosis.get("evidence") or "Reasoning evaluated.")
        diagnosis["rationale"] = str(diagnosis.get("rationale") or "Diagnosis completed.")
        diagnosis["decision_source"] = str(diagnosis.get("decision_source") or "abstention_rule")
        diagnosis.setdefault("misconception_variant", None)
        diagnosis.setdefault("secondary_misconception_ids", [])
        diagnosis.setdefault("error_type", "other")
        diagnosis["answer_correct"] = bool(diagnosis.get("answer_correct", False))

        # Update Learner Model
        self._record_diagnosis_in_learner_model(m_id)

        # Step 2: Person 2 intervention / routing
        try:
            p2_resp = run_person2(diagnosis, context)
        except Exception:
            p2_resp = {
                "diagnosis": diagnosis,
                "status": "NEED_MORE_EVIDENCE",
                "message": "Please explain the step-by-step rule you used to arrive at your answer.",
                "next_action": "PROMPT_FOR_EXPLANATION",
            }

        session = {
            "context": context,
            "diagnosis": diagnosis,
            "intervention": p2_resp.get("intervention"),
            "followup": p2_resp.get("followup"),
            "status": p2_resp.get("status"),
            "message": p2_resp.get("message"),
            "next_action": p2_resp.get("next_action"),
            "learner_state": self.get_learner_state()
        }

        return session

    def evaluate_resolution(
        self,
        session: Dict[str, Any],
        followup_answer: str,
        followup_reasoning: str
    ) -> Dict[str, Any]:
        """
        Executes Stage 2:
        1. Receives active session + student's follow-up submission.
        2. Calls Person 2 evaluate_followup() using the ORIGINAL session diagnosis.
        3. Updates Learner Model against the original diagnosed misconception.
        4. If RESOLVED -> marks COMPLETED.
        5. If NOT_RESOLVED -> triggers Person 2 generate_second_intervention().
        """
        safe_session = session if isinstance(session, dict) else {}
        diagnosis = safe_session.get("diagnosis") or {
            "misconception_id": "INSUFFICIENT",
            "confidence": 0.75,
            "evidence": "Missing active diagnosis session.",
            "rationale": "No active misconception session was found.",
            "decision_source": "abstention_rule",
            "misconception_variant": None,
            "secondary_misconception_ids": [],
            "error_type": "other",
            "answer_correct": False,
        }
        followup_obj = safe_session.get("followup")
        fa = str(followup_answer if followup_answer is not None else "")
        fr = str(followup_reasoning if followup_reasoning is not None else "")

        # If no follow-up was generated (NONE, INSUFFICIENT, OOS)
        if not followup_obj:
            return {
                "diagnosis": diagnosis,
                "status": safe_session.get("status", "NEED_MORE_EVIDENCE"),
                "message": safe_session.get("message", "No follow-up check is required for this diagnosis."),
                "next_action": safe_session.get("next_action", "PROMPT_FOR_EXPLANATION"),
                "learner_state": self.get_learner_state()
            }

        # Step 3: Person 2 follow-up evaluation
        try:
            resolution = evaluate_followup(
                diagnosis=diagnosis,
                followup_question=followup_obj,
                followup_answer=fa,
                followup_reasoning=fr
            )
        except Exception:
            resolution = {
                "status": "NOT_RESOLVED",
                "resolution_type": "CLARIFICATION_NEEDED",
                "confidence": 0.75,
                "evidence": f"Follow-up response ({repr(fr[:60])}) could not be verified.",
                "rationale": "Please provide both the expected output and a clear explanation of the Python rule.",
                "persistent_misconception": False,
            }

        # Update Learner Model for the ORIGINAL diagnosed misconception
        orig_m_id = diagnosis.get("misconception_id", "")
        self._record_resolution_in_learner_model(
            orig_m_id,
            resolution["status"],
            resolution.get("resolution_type"),
        )

        out = {
            "context": safe_session.get("context"),
            "diagnosis": diagnosis,
            "intervention": safe_session.get("intervention"),
            "followup": followup_obj,
            "student_followup_submission": {
                "answer": fa,
                "reasoning": fr
            },
            "resolution": resolution
        }

        if resolution["status"] == "RESOLVED":
            out["next_action"] = "COMPLETED"
        else:
            out["next_action"] = "SECOND_INTERVENTION"
            out["second_intervention"] = generate_second_intervention(
                diagnosis=diagnosis,
                context=safe_session.get("context", {}),
                previous_intervention=safe_session.get("intervention"),
                resolution=resolution,
            )

        out["learner_state"] = self.get_learner_state()
        return out


# =============================================================================
# LEARNER MODEL STATE-TRANSITION TEST SUITE (TESTS 1-9)
# =============================================================================

def run_learner_model_tests(pipeline: ReLearnPipeline) -> None:
    """
    Executes the 9 explicit Learner Model state-transition tests:
    - TEST 1: Diagnose M06 -> current_status = AWAITING_FOLLOWUP
    - TEST 2: M06 follow-up is NOT_RESOLVED -> current_status = NEEDS_PRACTICE, unresolved += 1
    - TEST 3: Same M06 is successfully resolved -> current_status = MASTERED, resolved += 1, unresolved unchanged
    - TEST 4: Previously MASTERED M06 is diagnosed again -> current_status = AWAITING_FOLLOWUP
    - TEST 5: The new follow-up fails (including CLARIFICATION_NEEDED check) -> current_status = NEEDS_PRACTICE
    - TEST 6: The retry succeeds -> current_status = MASTERED
    - TEST 7: NONE diagnosis -> attempts increments, no misconception entry created
    - TEST 8: INSUFFICIENT diagnosis -> attempts increments, no misconception entry created
    - TEST 9: OOS diagnosis -> attempts increments, no misconception entry created
    """
    print("\n" + "=" * 65)
    print("LEARNER MODEL MASTERY & STATE-TRANSITION SUITE (TESTS 1-9)")
    print("=" * 65)

    pipeline.reset_learner_state()

    # TEST 1: Diagnose M06 -> current_status = AWAITING_FOLLOWUP
    s1 = pipeline.diagnose_and_intervene(
        "for x in range(1, 5):\n    print(x)",
        "1 2 3 4",
        "1 2 3 4 5",
        "range(1, 5) includes 5 because the stop value is inclusive.",
    )
    ls1 = s1["learner_state"]
    m06_1 = ls1["misconceptions"]["M06"]
    print(f"\n[TEST 1 — Diagnose M06] {m06_1}")
    assert m06_1["current_status"] == "AWAITING_FOLLOWUP"
    assert m06_1["detected"] == 1 and m06_1["resolved"] == 0 and m06_1["unresolved"] == 0
    assert ls1["last_status"] == "DETECTED:M06"

    # TEST 2: M06 follow-up is NOT_RESOLVED -> current_status = NEEDS_PRACTICE, unresolved += 1
    r2 = pipeline.evaluate_resolution(
        s1,
        "[2, 3, 4, 5]",
        "range includes 5 because the stop value is inclusive.",
    )
    ls2 = r2["learner_state"]
    m06_2 = ls2["misconceptions"]["M06"]
    print(f"[TEST 2 — M06 Follow-up NOT_RESOLVED] {m06_2}")
    assert m06_2["current_status"] == "NEEDS_PRACTICE"
    assert m06_2["detected"] == 1 and m06_2["resolved"] == 0 and m06_2["unresolved"] == 1
    assert ls2["unresolved_count"] == 1
    assert ls2["last_status"] == "NOT_RESOLVED:M06"

    # TEST 3: Same M06 is diagnosed again and successfully resolved -> current_status = MASTERED,
    # resolved += 1, historical unresolved count remains 1.
    s3 = pipeline.diagnose_and_intervene(
        "print(range(1, 5))",
        "1 2 3 4",
        "1 2 3 4 5",
        "range includes both the starting and ending values.",
    )
    assert s3["learner_state"]["misconceptions"]["M06"]["current_status"] == "AWAITING_FOLLOWUP"
    r3 = pipeline.evaluate_resolution(
        s3,
        "[2, 3, 4]",
        "The stop boundary 5 is exclusive, so range(2, 5) stops before 5 at 4.",
    )
    ls3 = r3["learner_state"]
    m06_3 = ls3["misconceptions"]["M06"]
    print(f"[TEST 3 — M06 Resolved After Prior Unresolved] {m06_3}")
    assert m06_3 == {
        "detected": 2,
        "resolved": 1,
        "unresolved": 1,
        "current_status": "MASTERED",
    }
    assert ls3["resolved_count"] == 1 and ls3["unresolved_count"] == 1
    assert ls3["last_status"] == "RESOLVED:M06"

    # TEST 4: Previously MASTERED M06 is diagnosed again -> current_status = AWAITING_FOLLOWUP
    s4 = pipeline.diagnose_and_intervene(
        "for i in range(0, 4):\n    print(i)",
        "0 1 2 3",
        "0 1 2 3 4",
        "range(0, 4) includes 4 because the stop value is inclusive.",
    )
    ls4 = s4["learner_state"]
    m06_4 = ls4["misconceptions"]["M06"]
    print(f"[TEST 4 — Previously MASTERED M06 Diagnosed Again] {m06_4}")
    assert m06_4["current_status"] == "AWAITING_FOLLOWUP"
    assert m06_4["detected"] == 3 and m06_4["resolved"] == 1 and m06_4["unresolved"] == 1

    # TEST 5: The new follow-up fails (specifically testing CLARIFICATION_NEEDED bare answer restatement)
    # -> current_status = NEEDS_PRACTICE (must NOT mark MASTERED)
    expected_fq_ans = s4["followup"]["expected_answer"]
    r5 = pipeline.evaluate_resolution(
        s4,
        expected_fq_ans,
        f"The answer is {expected_fq_ans}.",
    )
    ls5 = r5["learner_state"]
    m06_5 = ls5["misconceptions"]["M06"]
    print(f"[TEST 5 — New Follow-up Fails (CLARIFICATION_NEEDED)] {m06_5}")
    assert r5["resolution"]["status"] == "NOT_RESOLVED"
    assert r5["resolution"].get("resolution_type") == "CLARIFICATION_NEEDED"
    assert m06_5["current_status"] == "NEEDS_PRACTICE"
    assert m06_5["detected"] == 3 and m06_5["resolved"] == 1 and m06_5["unresolved"] == 2
    assert ls5["last_status"] == "NOT_RESOLVED:M06"

    # TEST 6: The retry succeeds -> NEEDS_PRACTICE transitions to MASTERED
    r6 = pipeline.evaluate_resolution(
        s4,
        expected_fq_ans,
        "The stop boundary in range is exclusive, so the loop stops one integer before the stop value.",
    )
    ls6 = r6["learner_state"]
    m06_6 = ls6["misconceptions"]["M06"]
    print(f"[TEST 6 — Retry Succeeds (NEEDS_PRACTICE -> MASTERED)] {m06_6}")
    assert r6["resolution"]["status"] == "RESOLVED"
    assert m06_6["current_status"] == "MASTERED"
    assert m06_6["detected"] == 3 and m06_6["resolved"] == 2 and m06_6["unresolved"] == 2
    assert ls6["last_status"] == "RESOLVED:M06"

    # TEST 7: NONE diagnosis -> attempts increments, no misconception entry created
    attempts_before_7 = ls6["attempts"]
    s7 = pipeline.diagnose_and_intervene(
        "total = 14 + 8\nprint(total)",
        "22",
        "21",
        "I meant to type 22 but hit the 1 key by accident; 14 + 8 is 22.",
    )
    ls7 = s7["learner_state"]
    print(f"[TEST 7 — NONE Diagnosis] attempts={ls7['attempts']}, keys={list(ls7['misconceptions'].keys())}")
    assert s7["diagnosis"]["misconception_id"] == "NONE"
    assert ls7["attempts"] == attempts_before_7 + 1
    assert "NONE" not in ls7["misconceptions"]
    assert set(ls7["misconceptions"].keys()) == {"M06"}

    # TEST 8: INSUFFICIENT diagnosis -> attempts increments, no misconception entry created
    attempts_before_8 = ls7["attempts"]
    s8 = pipeline.diagnose_and_intervene(
        "total = 0\nfor i in range(3):\n    total += i\nprint(total)",
        "3",
        "7",
        "I guessed 7.",
    )
    ls8 = s8["learner_state"]
    print(f"[TEST 8 — INSUFFICIENT Diagnosis] attempts={ls8['attempts']}, keys={list(ls8['misconceptions'].keys())}")
    assert s8["diagnosis"]["misconception_id"] == "INSUFFICIENT"
    assert ls8["attempts"] == attempts_before_8 + 1
    assert "INSUFFICIENT" not in ls8["misconceptions"]
    assert set(ls8["misconceptions"].keys()) == {"M06"}

    # TEST 9: OOS diagnosis -> attempts increments, no misconception entry created
    attempts_before_9 = ls8["attempts"]
    s9 = pipeline.diagnose_and_intervene(
        "a = [1, 2]\nb = a\nb.append(3)\nprint(a)",
        "[1, 2, 3]",
        "[1, 2]",
        "Setting b = a creates a separate copy of the list, so appending to b does not touch list a because aliasing does not happen.",
    )
    ls9 = s9["learner_state"]
    print(f"[TEST 9 — OOS Diagnosis] attempts={ls9['attempts']}, keys={list(ls9['misconceptions'].keys())}")
    assert s9["diagnosis"]["misconception_id"] == "OOS"
    assert ls9["attempts"] == attempts_before_9 + 1
    assert "OOS" not in ls9["misconceptions"]
    assert set(ls9["misconceptions"].keys()) == {"M06"}

    # Verify reset behavior
    reset_after = pipeline.reset_learner_state()
    print(f"[RESET CHECK] {reset_after}")
    assert reset_after["attempts"] == 0
    assert reset_after["misconception_detections"] == 0
    assert reset_after["resolved_count"] == 0
    assert reset_after["unresolved_count"] == 0
    assert reset_after["last_status"] == "READY"
    assert reset_after["misconceptions"] == {}


# =============================================================================
# END-TO-END DEMO EXECUTION
# =============================================================================

def run_end_to_end_demo():
    """
    Demonstrates full end-to-end flow across 5 canonical scenarios:
    A. M01 -> Intervention -> Correct Follow-up -> RESOLVED
    B. M06 -> Intervention -> Incorrect Follow-up -> NOT_RESOLVED -> Second Intervention
    C. NONE -> Normal Feedback (CONTINUE_NORMAL_CURRICULUM)
    D. INSUFFICIENT -> Clarification Request (PROMPT_FOR_EXPLANATION)
    E. OOS -> Out-of-Scope Response (REFER_TO_DOCUMENTATION)
    F. Recurring M06 -> RESOLVED + Learner Model Check
    + Explicit 9-Test Learner Model Mastery & State-Transition Suite
    """
    print("=" * 65)
    print("RELEARN END-TO-END DEMO: PERSON 1 + PERSON 2 INTEGRATION")
    print("=" * 65)

    pipeline = ReLearnPipeline()

    # -------------------------------------------------------------
    # Scenario A: M01 -> Intervention -> Correct Follow-up -> RESOLVED
    # -------------------------------------------------------------
    print("\n" + "#" * 65)
    print("SCENARIO A: M01 (String vs Number Type Confusion) -> RESOLVED")
    print("#" * 65)

    q_a = 'x = "5"\ny = 2\nprint(x + y)'
    ca_a = "TypeError"
    sa_a = "7"
    sr_a = 'The quotes around "5" don\'t matter; Python treats it as integer 5 so adding gives 7.'

    print("\n[STEP 1: STUDENT SUBMISSION]")
    print(f"Question:        {q_a.replace(chr(10), ' ')}")
    print(f"Student Answer:  {sa_a}")
    print(f"Student Reason:  {sr_a}")

    session_a = pipeline.diagnose_and_intervene(q_a, ca_a, sa_a, sr_a)
    assert session_a["learner_state"]["misconceptions"]["M01"]["current_status"] == "AWAITING_FOLLOWUP"

    print("\n[STEP 2: PERSON 1 DIAGNOSIS]")
    diag_a = session_a["diagnosis"]
    print(f"Misconception:   {diag_a['misconception_id']}")
    print(f"Confidence:      {diag_a['confidence']}")
    print(f"Evidence:        {diag_a['evidence']}")
    print(f"Rationale:       {diag_a['rationale']}")
    print(f"Decision Source: {diag_a['decision_source']}")

    print("\n[STEP 3: PERSON 2 TARGETED INTERVENTION]")
    int_a = session_a["intervention"]
    print(f"Title:           {int_a['title']}")
    print(f"Explanation:     {int_a['short_explanation']}")
    print(f"Concrete Code:\n{int_a['concrete_example']}")
    print(f"Contrast Code:\n{int_a['contrast_example']}")
    print(f"Key Rule:        {int_a['key_rule']}")

    print("\n[STEP 4: PERSON 2 FOLLOW-UP QUESTION]")
    fol_a = session_a["followup"]
    print(fol_a["question"])
    print(f"Next Action:     {session_a['next_action']}")

    print("\n[STEP 5: STUDENT FOLLOW-UP SUBMISSION]")
    follow_ans_a = "34"
    follow_reason_a = "Both are strings in quotes, so the + operator concatenates them into '34'."
    print(f"Student Answer:  {follow_ans_a}")
    print(f"Student Reason:  {follow_reason_a}")

    res_a = pipeline.evaluate_resolution(session_a, follow_ans_a, follow_reason_a)
    resol_a = res_a["resolution"]

    print("\n[STEP 6: RESOLUTION EVALUATION]")
    print(f"Status:          {resol_a['status']}")
    print(f"Confidence:      {resol_a['confidence']}")
    print(f"Evidence:        {resol_a['evidence']}")
    print(f"Rationale:       {resol_a['rationale']}")
    print(f"Next Action:     {res_a['next_action']}")
    assert resol_a["status"] == "RESOLVED"
    assert res_a["next_action"] == "COMPLETED"
    assert res_a["learner_state"]["misconceptions"]["M01"]["current_status"] == "MASTERED"

    # -------------------------------------------------------------
    # Scenario B: M06 -> Intervention -> Incorrect Follow-up -> NOT_RESOLVED -> Second Intervention
    # -------------------------------------------------------------
    print("\n" + "#" * 65)
    print("SCENARIO B: M06 (Loop Boundaries) -> PERSISTENT -> SECOND INTERVENTION")
    print("#" * 65)

    q_b = "for x in range(1, 5):\n    print(x)"
    ca_b = "1 2 3 4"
    sa_b = "1 2 3 4 5"
    sr_b = "range(1, 5) includes 5 because the stop value is inclusive."

    print("\n[STEP 1: STUDENT SUBMISSION]")
    print(f"Question:        {q_b.replace(chr(10), ' ')}")
    print(f"Student Answer:  {sa_b}")
    print(f"Student Reason:  {sr_b}")

    session_b = pipeline.diagnose_and_intervene(q_b, ca_b, sa_b, sr_b)
    assert session_b["learner_state"]["misconceptions"]["M06"]["current_status"] == "AWAITING_FOLLOWUP"

    print("\n[STEP 2: PERSON 1 DIAGNOSIS]")
    diag_b = session_b["diagnosis"]
    print(f"Misconception:   {diag_b['misconception_id']}")
    print(f"Confidence:      {diag_b['confidence']}")
    print(f"Evidence:        {diag_b['evidence']}")
    print(f"Decision Source: {diag_b['decision_source']}")

    print("\n[STEP 3: PERSON 2 TARGETED INTERVENTION]")
    int_b = session_b["intervention"]
    print(f"Title:           {int_b['title']}")
    print(f"Explanation:     {int_b['short_explanation']}")
    print(f"Key Rule:        {int_b['key_rule']}")

    print("\n[STEP 4: PERSON 2 FOLLOW-UP QUESTION]")
    fol_b = session_b["followup"]
    print(fol_b["question"])

    print("\n[STEP 5: STUDENT INCORRECT FOLLOW-UP SUBMISSION]")
    follow_ans_b = "[2, 3, 4, 5]"
    follow_reason_b = "range includes 5 because the stop value is inclusive."
    print(f"Student Answer:  {follow_ans_b}")
    print(f"Student Reason:  {follow_reason_b}")

    res_b = pipeline.evaluate_resolution(session_b, follow_ans_b, follow_reason_b)
    resol_b = res_b["resolution"]

    print("\n[STEP 6: RESOLUTION EVALUATION]")
    print(f"Status:          {resol_b['status']}")
    print(f"Confidence:      {resol_b['confidence']}")
    print(f"Persistent Flag: {resol_b['persistent_misconception']}")
    print(f"Rationale:       {resol_b['rationale']}")
    print(f"Next Action:     {res_b['next_action']}")
    assert resol_b["status"] == "NOT_RESOLVED"
    assert res_b["next_action"] == "SECOND_INTERVENTION"
    assert res_b["learner_state"]["misconceptions"]["M06"]["current_status"] == "NEEDS_PRACTICE"

    print("\n[STEP 7: PERSON 2 SECOND INTERVENTION (REINFORCED TRACE)]")
    sec_b = res_b["second_intervention"]
    print(f"Title:           {sec_b['title']}")
    print(f"Approach:        {sec_b['pedagogical_approach']}")
    print(f"Explanation:     {sec_b['explanation']}")
    print(f"Step Trace:\n{sec_b['step_by_step_trace']}")
    print(f"Key Takeaway:    {sec_b['key_takeaway']}")

    # -------------------------------------------------------------
    # Scenario C: NONE -> Normal Feedback
    # -------------------------------------------------------------
    print("\n" + "#" * 65)
    print("SCENARIO C: NONE (Typo / Careless Slip) -> NORMAL FEEDBACK")
    print("#" * 65)

    q_c = "total = 14 + 8\nprint(total)"
    ca_c = "22"
    sa_c = "21"
    sr_c = "I meant to type 22 but hit the 1 key by accident; 14 + 8 is 22."

    session_c = pipeline.diagnose_and_intervene(q_c, ca_c, sa_c, sr_c)

    print(f"Question:        {q_c.replace(chr(10), ' ')}")
    print(f"Student Answer:  {sa_c}")
    print(f"Student Reason:  {sr_c}")
    print(f"Person 1 Result: {session_c['diagnosis']['misconception_id']}")
    print(f"Person 2 Status: {session_c['status']}")
    print(f"Person 2 Msg:    {session_c['message']}")
    print(f"Follow-up:       {session_c['followup']}")
    print(f"Next Action:     {session_c['next_action']}")
    assert session_c["diagnosis"]["misconception_id"] == "NONE"
    assert session_c["followup"] is None
    assert session_c["next_action"] == "CONTINUE_NORMAL_CURRICULUM"

    # -------------------------------------------------------------
    # Scenario D: INSUFFICIENT -> Clarification Request
    # -------------------------------------------------------------
    print("\n" + "#" * 65)
    print("SCENARIO D: INSUFFICIENT -> CLARIFICATION REQUEST")
    print("#" * 65)

    q_d = "total = 0\nfor i in range(3):\n    for j in range(2):\n        total += i * j\nprint(total)"
    ca_d = "3"
    sa_d = "7"
    sr_d = "I guessed 7."

    session_d = pipeline.diagnose_and_intervene(q_d, ca_d, sa_d, sr_d)

    print(f"Question:        {q_d.replace(chr(10), ' ')}")
    print(f"Student Answer:  {sa_d}")
    print(f"Student Reason:  {sr_d}")
    print(f"Person 1 Result: {session_d['diagnosis']['misconception_id']}")
    print(f"Person 2 Status: {session_d['status']}")
    print(f"Person 2 Msg:    {session_d['message']}")
    print(f"Follow-up:       {session_d['followup']}")
    print(f"Next Action:     {session_d['next_action']}")
    assert session_d["diagnosis"]["misconception_id"] == "INSUFFICIENT"
    assert session_d["followup"] is None
    assert session_d["next_action"] == "PROMPT_FOR_EXPLANATION"

    # -------------------------------------------------------------
    # Scenario E: OOS -> Out-of-Scope Response
    # -------------------------------------------------------------
    print("\n" + "#" * 65)
    print("SCENARIO E: OOS (Object Aliasing) -> OUT-OF-SCOPE RESPONSE")
    print("#" * 65)

    q_e = "a = [1, 2]\nb = a\nb.append(3)\nprint(a)"
    ca_e = "[1, 2, 3]"
    sa_e = "[1, 2]"
    sr_e = "Setting b = a creates a separate copy of the list, so appending to b does not touch list a because aliasing does not happen."

    session_e = pipeline.diagnose_and_intervene(q_e, ca_e, sa_e, sr_e)

    print(f"Question:        {q_e.replace(chr(10), ' ')}")
    print(f"Student Answer:  {sa_e}")
    print(f"Student Reason:  {sr_e}")
    print(f"Person 1 Result: {session_e['diagnosis']['misconception_id']}")
    print(f"Person 2 Status: {session_e['status']}")
    print(f"Person 2 Msg:    {session_e['message']}")
    print(f"Follow-up:       {session_e['followup']}")
    print(f"Next Action:     {session_e['next_action']}")
    assert session_e["diagnosis"]["misconception_id"] == "OOS"
    assert session_e["followup"] is None
    assert session_e["next_action"] == "REFER_TO_DOCUMENTATION"

    # -------------------------------------------------------------
    # Scenario F: Recurring M06 -> RESOLVED + Learner Model Check
    # -------------------------------------------------------------
    print("\n" + "#" * 65)
    print("SCENARIO F: RECURRING M06 -> RESOLVED & LEARNER MODEL PROGRESSION")
    print("#" * 65)

    session_f = pipeline.diagnose_and_intervene(
        "print(range(1, 5))",
        "1 2 3 4",
        "1 2 3 4 5",
        "range includes both the starting and ending values.",
    )
    assert session_f["learner_state"]["misconceptions"]["M06"]["current_status"] == "AWAITING_FOLLOWUP"
    res_f = pipeline.evaluate_resolution(
        session_f,
        "[2, 3, 4]",
        "The stop boundary 5 is exclusive, so range(2, 5) stops before 5 at 4.",
    )
    learner = res_f["learner_state"]
    print(f"Learner State: {json.dumps(learner, indent=2)}")
    assert learner["attempts"] == 6
    assert set(learner["misconceptions"].keys()) == {"M01", "M06"}
    assert learner["misconceptions"]["M01"] == {
        "detected": 1,
        "resolved": 1,
        "unresolved": 0,
        "current_status": "MASTERED",
    }
    assert learner["misconceptions"]["M06"] == {
        "detected": 2,
        "resolved": 1,
        "unresolved": 1,
        "current_status": "MASTERED",
    }

    reset_state = pipeline.reset_learner_state()
    assert reset_state["attempts"] == 0 and reset_state["misconceptions"] == {}

    # Run the 9 explicit Learner Model state-transition tests
    run_learner_model_tests(pipeline)

    # Run the comprehensive Adversarial & End-to-End Evaluation Suite
    run_adversarial_end_to_end_evaluation(pipeline)

    # Run the 75-case Broad Website Robustness Matrix + 10 Website Scenarios
    run_broad_robustness_matrix(pipeline)

    print("\n" + "=" * 65)
    print("ALL RELEARN END-TO-END DEMO SCENARIOS COMPLETED SUCCESSFULLY!")
    print("=" * 65)


def run_adversarial_end_to_end_evaluation(pipeline: ReLearnPipeline):
    """
    Comprehensive adversarial & end-to-end validation suite covering:
    1. Person 1 Diagnosis Cases (A-G) across M02, M04, M05, M06 + normalized answer formats
    2. Person 2 Intervention Tests (1-5: misconception match, variant-awareness, question context, follow-up gen, deterministic rotation)
    3. Person 2 Resolution Tests (Cases A-E + normalized follow-up answers)
    4. Full End-to-End Scenario: Correct final answer + incorrect conceptual reasoning ->
       misconception detected -> targeted variant-aware intervention -> follow-up ->
       CLARIFICATION_NEEDED on bare answer (NEEDS_PRACTICE) -> RESOLVED only after
       demonstrated conceptual understanding (MASTERED).
    """
    print("\n" + "#" * 65)
    print("ADVERSARIAL & END-TO-END EVALUATION SUITE")
    print("#" * 65)

    pipeline.reset_learner_state()

    # =========================================================================
    # PART 1: PERSON 1 DIAGNOSIS CASES (A - G)
    # =========================================================================
    print("\n--- PART 1: PERSON 1 DIAGNOSIS CASES (A-G) ---")

    # Case A: Correct answer + sound conceptual reasoning -> NONE (across M02, M04, M05, M06 + normalized formats)
    case_a_inputs = [
        (
            "M02-Sound",
            "print(7 / 2)",
            "3.5",
            "3.5",
            "In Python 3, single slash / performs true floating-point division, returning 3.5.",
        ),
        (
            "M02-FloorDiv-Sound",
            "print(7 // 2)",
            "3",
            "3",
            "The // operator performs floor division. 7 divided by 2 is 3.5, and floor division returns the largest integer less than or equal to 3.5, which is 3.",
        ),
        (
            "M04-Sound",
            "print(2 + 3 * 4)",
            "14",
            " 14 ",
            "Multiplication * has higher operator precedence than addition +, so 3 * 4 = 12 is evaluated before adding 2.",
        ),
        (
            "M05-Sound-QuotedNormalized",
            's = "hello"\nprint(s[1])',
            "e",
            "'e'",
            "Python uses 0-based indexing, so index 0 is h and index 1 is the second character e.",
        ),
        (
            "M06-Sound-NewlineNormalized",
            "for i in range(1, 5):\n    print(i)",
            "1 2 3 4",
            "1\n2\n3\n4",
            "range(1, 5) starts at 1 and stops before the exclusive stop boundary 5, yielding 1, 2, 3, 4.",
        ),
    ]
    for label, q, ca, sa, sr in case_a_inputs:
        res = pipeline.diagnose_and_intervene(q, ca, sa, sr)
        diag = res["diagnosis"]
        print(f"[P1 Case A - {label}] answer_correct={diag['answer_correct']} -> {diag['misconception_id']}")
        assert diag["answer_correct"] is True, f"Expected answer_correct=True for {label}"
        assert diag["misconception_id"] == "NONE", f"Expected NONE for {label}, got {diag['misconception_id']}"

    # Case B: Correct answer + explicit wrong conceptual reasoning -> misconception detected (M01-M08)
    case_b_inputs = [
        (
            "M02-RightAns-WrongReason",
            "print(15 // 4)",
            "3",
            "3",
            "The // operator gives the remainder when 15 is divided by 4, and the remainder is 3.",
            "M02",
            "believes_double_slash_returns_remainder",
        ),
        (
            "M04-RightAns-WrongReason",
            "print(2 * 3 + 4)",
            "10",
            "10",
            "Python always evaluates arithmetic strictly left-to-right regardless of operator precedence, so 2 * 3 is 6 and + 4 is 10.",
            "M04",
            "left_to_right_evaluation_override",
        ),
        (
            "M05-RightAns-WrongReason",
            's = "aba"\nprint(s[2])',
            "a",
            "a",
            "Indexing starts at 1, so s[1] is b and s[2] is the second character a.",
            "M05",
            "one_based_indexing_belief",
        ),
        (
            "M06-RightAns-WrongReason",
            "for i in range(1, 4):\n    print(i)",
            "1 2 3",
            "[1, 2, 3]",
            "range includes the stop value, except I stopped at 3 here because range includes the end boundary.",
            "M06",
            "range_stop_inclusive_belief",
        ),
    ]
    for label, q, ca, sa, sr, expected_mid, expected_var in case_b_inputs:
        res = pipeline.diagnose_and_intervene(q, ca, sa, sr)
        diag = res["diagnosis"]
        print(
            f"[P1 Case B - {label}] answer_correct={diag['answer_correct']} -> "
            f"{diag['misconception_id']} ({diag['misconception_variant']})"
        )
        assert diag["answer_correct"] is True, f"Expected answer_correct=True for {label}"
        assert diag["misconception_id"] == expected_mid, f"Expected {expected_mid} for {label}, got {diag['misconception_id']}"
        assert diag["misconception_variant"] == expected_var, (
            f"Expected {expected_var} for {label}, got {diag['misconception_variant']}"
        )

    # Case C: Wrong answer + explicit misconception reasoning -> correct M01-M08 (across M02, M04, M05, M06)
    case_c_inputs = [
        (
            "M02-SlashInt",
            "print(7 / 2)",
            "3.5",
            "3",
            "Both 7 and 2 are integers, so / does integer division and drops the decimal part.",
            "M02",
            "believes_slash_returns_int",
        ),
        (
            "M04-LeftToRight",
            "print(2 + 3 * 4)",
            "14",
            "20",
            "Expressions evaluate strictly from left to right: 2 + 3 is 5, then 5 * 4 is 20.",
            "M04",
            "left_to_right_evaluation_override",
        ),
        (
            "M05-OneBased",
            'msg = "hello"\nprint(msg[1])',
            "e",
            "h",
            "Index 1 gets the first character of the string because indexing starts at 1.",
            "M05",
            "one_based_indexing_belief",
        ),
        (
            "M06-StopInclusive",
            "for i in range(1, 5):\n    print(i)",
            "1 2 3 4",
            "1 2 3 4 5",
            "range(1, 5) goes from 1 up to and including the stop value 5.",
            "M06",
            "range_stop_inclusive_belief",
        ),
    ]
    for label, q, ca, sa, sr, expected_mid, expected_var in case_c_inputs:
        res = pipeline.diagnose_and_intervene(q, ca, sa, sr)
        diag = res["diagnosis"]
        print(
            f"[P1 Case C - {label}] answer_correct={diag['answer_correct']} -> "
            f"{diag['misconception_id']} ({diag['misconception_variant']})"
        )
        assert diag["answer_correct"] is False
        assert diag["misconception_id"] == expected_mid
        assert diag["misconception_variant"] == expected_var

    # Case D: Wrong answer + trace/calculation mistake without conceptual misconception -> INSUFFICIENT
    res_d = pipeline.diagnose_and_intervene(
        "print(2 + 3 * 4)",
        "14",
        "13",
        "Multiplication goes first because of operator precedence, so I did 3 * 4 = 11 and then 2 + 11 = 13.",
    )
    diag_d = res_d["diagnosis"]
    print(f"[P1 Case D - Trace/Calculation Slip] -> {diag_d['misconception_id']} (error_type={diag_d['error_type']})")
    assert diag_d["misconception_id"] == "INSUFFICIENT"
    assert diag_d["error_type"] == "trace_error"

    res_d_m02 = pipeline.diagnose_and_intervene(
        "print(7 // 2)",
        "3",
        "4",
        "The // operator performs floor division and should return the integer quotient, but I calculated 7 divided by 2 incorrectly and got 4.",
    )
    diag_d_m02 = res_d_m02["diagnosis"]
    print(f"[P1 Case D2 - M02 Sound + Calculation Slip] -> {diag_d_m02['misconception_id']} (error_type={diag_d_m02['error_type']})")
    assert diag_d_m02["misconception_id"] == "INSUFFICIENT"
    assert diag_d_m02["error_type"] == "trace_error"

    # Case E: Wrong answer + missing/very thin reasoning -> INSUFFICIENT
    res_e = pipeline.diagnose_and_intervene(
        "print(7 / 2)",
        "3.5",
        "3",
        "idk",
    )
    diag_e = res_e["diagnosis"]
    print(f"[P1 Case E - Thin/Missing Reasoning] -> {diag_e['misconception_id']}")
    assert diag_e["misconception_id"] == "INSUFFICIENT"

    # Case F: Out-of-scope conceptual belief -> OOS
    res_f = pipeline.diagnose_and_intervene(
        "x = [1, 2]\ny = x\ny.append(3)\nprint(x)",
        "[1, 2, 3]",
        "[1, 2]",
        "Assigning y = x copies the list into a new memory space, so mutating y does not affect x.",
    )
    diag_f = res_f["diagnosis"]
    print(f"[P1 Case F - Out-of-Scope Belief] -> {diag_f['misconception_id']}")
    assert diag_f["misconception_id"] == "OOS"

    res_f_cross = pipeline.diagnose_and_intervene(
        "print(7 // 2)",
        "3",
        "3",
        "I thought that if x = y, Python automatically creates a separate copy of the list, so changing one list does not affect the other.",
    )
    diag_f_cross = res_f_cross["diagnosis"]
    print(f"[P1 Case F2 - OOS List Copy on Unrelated Question] -> {diag_f_cross['misconception_id']}")
    assert diag_f_cross["misconception_id"] == "OOS"

    # Case G: Ambiguous evidence (bare restatement of code without conceptual mechanism) -> INSUFFICIENT
    res_g = pipeline.diagnose_and_intervene(
        "print(7 / 2)",
        "3.5",
        "3",
        "The code runs print(7 / 2) and outputs 3.",
    )
    diag_g = res_g["diagnosis"]
    print(f"[P1 Case G - Ambiguous Evidence] -> {diag_g['misconception_id']}")
    assert diag_g["misconception_id"] == "INSUFFICIENT"

    # =========================================================================
    # PART 2: PERSON 2 INTERVENTION TESTS (1 - 5)
    # =========================================================================
    print("\n--- PART 2: PERSON 2 INTERVENTION TESTS (1-5) ---")

    # Test 1 & Test 2 & Test 3: Misconception match, Variant-awareness, and Question contextualization
    m02_slash = pipeline.diagnose_and_intervene(
        "print(7 / 2)",
        "3.5",
        "3",
        "7 and 2 are ints so / returns an integer.",
    )
    m02_rem = pipeline.diagnose_and_intervene(
        "print(7 // 2)",
        "3",
        "1",
        "// gives the remainder of 7 divided by 2.",
    )
    assert m02_slash["diagnosis"]["misconception_id"] == "M02"
    assert m02_rem["diagnosis"]["misconception_id"] == "M02"
    assert m02_slash["intervention"]["title"] != m02_rem["intervention"]["title"]
    assert "7 / 2" in m02_slash["intervention"]["concrete_example"]
    assert "7 // 2" in m02_rem["intervention"]["concrete_example"]
    print(
        f"[P2 Tests 1-3] M02 variant titles: '{m02_slash['intervention']['title']}' vs "
        f"'{m02_rem['intervention']['title']}' (contextualized with learner's expression)"
    )

    # Test 4 & Test 5: Follow-up generation and deterministic follow-up rotation across repeated attempts
    f_q1 = m02_slash["followup"]["question"]
    m02_slash_second = pipeline.diagnose_and_intervene(
        "print(9 / 2)",
        "4.5",
        "4",
        "Since 9 and 2 are integers, / drops the decimal.",
    )
    f_q2 = m02_slash_second["followup"]["question"]
    assert f_q1 != f_q2, "Expected deterministic rotation to pick a distinct follow-up question on next attempt"
    print(f"[P2 Tests 4-5] Follow-up rotation verified: Q1 != Q2")

    # =========================================================================
    # PART 3: PERSON 2 RESOLUTION TESTS (Cases A - E)
    # =========================================================================
    print("\n--- PART 3: PERSON 2 RESOLUTION TESTS (A-E) ---")

    # Prepare an M06 session whose follow-up expects "[2, 3, 4]"
    pipeline.reset_learner_state()
    session_m06 = pipeline.diagnose_and_intervene(
        "for i in range(1, 5):\n    print(i)",
        "1 2 3 4",
        "1 2 3 4 5",
        "range(1, 5) includes the stop value 5.",
    )
    expected_m06_fq = session_m06["followup"]["expected_answer"]
    assert expected_m06_fq == "[2, 3, 4]"

    # Case B: Correct follow-up answer (using newline-separated normalized format!) + bare answer only -> CLARIFICATION_NEEDED
    res_case_b = pipeline.evaluate_resolution(session_m06, "2\n3\n4", "The output is 2 3 4.")
    print(
        f"[P2 Resolution Case B - Bare Answer] status={res_case_b['resolution']['status']}, "
        f"type={res_case_b['resolution'].get('resolution_type')}"
    )
    assert res_case_b["resolution"]["status"] == "NOT_RESOLVED"
    assert res_case_b["resolution"].get("resolution_type") == "CLARIFICATION_NEEDED"

    # Case C: Correct follow-up answer + explicit wrong conceptual reasoning -> NOT_RESOLVED (persistent_misconception)
    res_case_c = pipeline.evaluate_resolution(
        session_m06,
        "[2, 3, 4]",
        "range includes the stop value, so it stops at 4.",
    )
    print(
        f"[P2 Resolution Case C - Wrong Concept] status={res_case_c['resolution']['status']}, "
        f"persistent={res_case_c['resolution']['persistent_misconception']}"
    )
    assert res_case_c["resolution"]["status"] == "NOT_RESOLVED"
    assert res_case_c["resolution"]["persistent_misconception"] is True

    # Case D: Incorrect follow-up answer -> NOT_RESOLVED
    res_case_d = pipeline.evaluate_resolution(
        session_m06,
        "[2, 3, 4, 5]",
        "The stop value 5 is excluded so it stops at 4.",
    )
    print(f"[P2 Resolution Case D - Wrong Answer] status={res_case_d['resolution']['status']}")
    assert res_case_d["resolution"]["status"] == "NOT_RESOLVED"

    # Case E: Guessing / insufficient follow-up reasoning -> NOT_RESOLVED
    res_case_e = pipeline.evaluate_resolution(
        session_m06,
        "[2, 3, 4]",
        "I just guessed this list.",
    )
    print(f"[P2 Resolution Case E - Guessing] status={res_case_e['resolution']['status']}")
    assert res_case_e["resolution"]["status"] == "NOT_RESOLVED"

    # Case A: Correct follow-up answer (in normalized format "2, 3, 4") + explicit conceptual reasoning -> RESOLVED
    res_case_a = pipeline.evaluate_resolution(
        session_m06,
        "2, 3, 4",
        "The stop boundary 5 is exclusive in range(2, 5), so it stops before 5 at 4.",
    )
    print(f"[P2 Resolution Case A - Conceptual Mastery] status={res_case_a['resolution']['status']}")
    assert res_case_a["resolution"]["status"] == "RESOLVED"

    # =========================================================================
    # PART 4: FULL END-TO-END RIGHT-ANSWER WRONG-REASONING SCENARIO
    # =========================================================================
    print("\n--- PART 4: END-TO-END RIGHT-ANSWER + WRONG-REASONING FLOW ---")
    pipeline.reset_learner_state()

    # Step 1: Student submits the RIGHT final answer ("3") for `print(15 // 4)`
    # but with WRONG conceptual reasoning ("// gives the remainder of 15 divided by 4, which is 3").
    e2e_session = pipeline.diagnose_and_intervene(
        "print(15 // 4)",
        "3",
        "3",
        "The // operator calculates the remainder of 15 divided by 4, which is 3.",
    )
    e2e_diag = e2e_session["diagnosis"]
    e2e_int = e2e_session["intervention"]
    e2e_fq = e2e_session["followup"]
    e2e_ls1 = e2e_session["learner_state"]
    print(
        f"[E2E Step 1 - Diagnosis] answer_correct={e2e_diag['answer_correct']}, "
        f"misconception={e2e_diag['misconception_id']} ({e2e_diag['misconception_variant']}), "
        f"learner_status={e2e_ls1['misconceptions']['M02']['current_status']}"
    )
    assert e2e_diag["answer_correct"] is True
    assert e2e_diag["misconception_id"] == "M02"
    assert e2e_diag["misconception_variant"] == "believes_double_slash_returns_remainder"
    assert "15 // 4" in e2e_int["concrete_example"]
    assert e2e_fq is not None
    assert e2e_ls1["misconceptions"]["M02"]["current_status"] == "AWAITING_FOLLOWUP"

    # Step 2: Student answers the follow-up question with a bare answer restatement -> NOT_RESOLVED (CLARIFICATION_NEEDED)
    fq_expected = e2e_fq["expected_answer"]
    e2e_res_bare = pipeline.evaluate_resolution(
        e2e_session,
        fq_expected,
        f"The answer is {fq_expected}.",
    )
    print(
        f"[E2E Step 2 - Bare Follow-up] status={e2e_res_bare['resolution']['status']}, "
        f"type={e2e_res_bare['resolution'].get('resolution_type')}, "
        f"learner_status={e2e_res_bare['learner_state']['misconceptions']['M02']['current_status']}"
    )
    assert e2e_res_bare["resolution"]["status"] == "NOT_RESOLVED"
    assert e2e_res_bare["resolution"].get("resolution_type") == "CLARIFICATION_NEEDED"
    assert e2e_res_bare["learner_state"]["misconceptions"]["M02"]["current_status"] == "NEEDS_PRACTICE"

    # Step 3: Student retries follow-up with genuine conceptual understanding -> RESOLVED & MASTERED
    e2e_res_mastered = pipeline.evaluate_resolution(
        e2e_session,
        fq_expected,
        "The // floor division operator returns the integer quotient and truncates the decimal part, whereas % gives the remainder.",
    )
    print(
        f"[E2E Step 3 - Conceptual Follow-up] status={e2e_res_mastered['resolution']['status']}, "
        f"learner_status={e2e_res_mastered['learner_state']['misconceptions']['M02']['current_status']}"
    )
    assert e2e_res_mastered["resolution"]["status"] == "RESOLVED"
    assert e2e_res_mastered["learner_state"]["misconceptions"]["M02"]["current_status"] == "MASTERED"

    pipeline.reset_learner_state()
    print("--- ALL ADVERSARIAL & END-TO-END EVALUATIONS PASSED ---")


def run_broad_robustness_matrix(pipeline: ReLearnPipeline) -> None:
    """
    Executes the 75-case Broad Website Robustness Matrix across all 10 required categories:
      1. 10 Correct answer + sound reasoning cases -> NONE
      2. 10 Correct answer + wrong reasoning cases -> M01-M08
      3. 10 Wrong answer + misconception cases -> M01-M08
      4. 10 Wrong answer + trace/calculation error cases -> INSUFFICIENT (trace_error)
      5. 10 Missing/thin/bare reasoning cases (both wrong & correct answers) -> INSUFFICIENT
      6. 5 Out-of-scope (OOS) cases -> OOS
      7. 5 Ambiguous cases -> INSUFFICIENT
      8. 5 Natural-language paraphrase cases -> M01-M08
      9. 5 Unusual formatting cases -> normalized answer_correct=True
     10. 5 Empty / short / guessing / very long input cases -> INSUFFICIENT (safe, no crash)
    Plus the 10 Website Verification Scenarios (Scenarios 1-10).
    """
    print("\n" + "#" * 72)
    print("BROAD WEBSITE ROBUSTNESS MATRIX (75 CASES + 10 WEBSITE SCENARIOS)")
    print("#" * 72)

    pipeline.reset_learner_state()

    matrix_cases = [
        # --- GROUP 1: 10 Correct Answer + Sound Reasoning -> NONE ---
        ("G1-01 M01 Sound Concat", 'print("3" + "2")', "32", "32", "Both are strings enclosed in quotes, so + concatenates them into '32'.", "NONE", True, "correct"),
        ("G1-02 M01 Sound Repeat", 'print("4" * 2)', "44", "44", "Multiplying a string by 2 repeats the string twice to produce '44'.", "NONE", True, "correct"),
        ("G1-03 M02 Sound Floor", "print(7 // 2)", "3", "3", "The // operator performs floor division and returns the largest integer less than or equal to 3.5, which is 3.", "NONE", True, "correct"),
        ("G1-04 M02 Sound Float", "print(7 / 2)", "3.5", "3.5", "Single slash / always performs true floating-point division and returns a float 3.5.", "NONE", True, "correct"),
        ("G1-05 M03 Sound Equality", "x = 5\nprint(x == 5)", "True", "True", "Single equals = assigns 5 to x, and double equals == checks whether x is equal to 5.", "NONE", True, "correct"),
        ("G1-06 M04 Sound Precedence", "print(2 + 3 * 4)", "14", "14", "Multiplication * has higher precedence than addition +, so 3 * 4 evaluates first to 12, then 2 + 12 gives 14.", "NONE", True, "correct"),
        ("G1-07 M04 Sound Parentheses", "print((2 + 3) * 4)", "20", "20", "Parentheses have the highest precedence and are evaluated first before multiplication.", "NONE", True, "correct"),
        ("G1-08 M05 Sound Zero-Based", 's = "hello"\nprint(s[1])', "e", "e", "Python uses 0-based indexing, so index 0 is 'h' and index 1 is the second character 'e'.", "NONE", True, "correct"),
        ("G1-09 M06 Sound Range Exclusive", "for i in range(1, 5):\n    print(i)", "1 2 3 4", "1 2 3 4", "In range(1, 5), the stop boundary 5 is exclusive, so it stops before 5 at 4.", "NONE", True, "correct"),
        ("G1-10 M07/M08 Sound Base Case", "def f(n):\n    if n == 0: return 1\n    return f(n-1)\nprint(f(2))", "1", "1", "When n reaches 0, the base case triggers and returns immediately, stopping the recursion.", "NONE", True, "correct"),

        # --- GROUP 2: 10 Correct Answer + Wrong Reasoning -> M01-M08 ---
        ("G2-01 M01 RightAns WrongReason", 'print("2" * 2)', "22", "22", "Python converts '2' to a number and puts two 2s together because numeric strings act like integers.", "M01", True, "conceptual"),
        ("G2-02 M02 RightAns Remainder", "print(15 // 4)", "3", "3", "The // operator gives the remainder when 15 is divided by 4, which is 3.", "M02", True, "conceptual"),
        ("G2-03 M02 RightAns NormalDivRemoveDec", "print(7 // 2)", "3", "3", "The // operator does normal division and removes the decimal part.", "M02", True, "conceptual"),
        ("G2-04 M03 RightAns DoubleEqAssign", "x = 5\nprint(x == 5)", "True", "True", "Double equals == assigns 5 to x and returns True when the assignment succeeds.", "M03", True, "conceptual"),
        ("G2-05 M04 RightAns LeftToRight", "print(2 * 3 + 4)", "10", "10", "Python evaluates arithmetic strictly from left to right without operator precedence, so 2 * 3 is 6 plus 4 is 10.", "M04", True, "conceptual"),
        ("G2-06 M05 RightAns OneBasedPalindrome", 's = "aba"\nprint(s[1])', "b", "b", "Indexing starts at 1, and I counted from the right so position 1 is b.", "M05", True, "conceptual"),
        ("G2-07 M05 RightAns SliceInclusive", 's = "aaa"\nprint(s[0:2])', "aa", "aa", "String slicing includes the end index, and positions 1 and 2 give aa.", "M05", True, "conceptual"),
        ("G2-08 M06 RightAns StopInclusive", "for i in range(1, 4):\n    print(i)", "1 2 3", "1 2 3", "range includes the stop value, and here the stop value is 3.", "M06", True, "conceptual"),
        ("G2-09 M07 RightAns ArgNameMatch", "def add(x, y):\n    return x + y\nx = 2\ny = 3\nprint(add(y, x))", "5", "5", "Python matches arguments to parameters with the same variable name regardless of order.", "M07", True, "conceptual"),
        ("G2-10 M08 RightAns AutoStopZero", "def f(n):\n    if n == 0: return 0\n    return f(n-1)\nprint(f(2))", "0", "0", "Recursive functions automatically stop when the parameter reaches 0 even without a base case.", "M08", True, "conceptual"),

        # --- GROUP 3: 10 Wrong Answer + Misconception -> M01-M08 ---
        ("G3-01 M01 StrPlusStrAdd", 'print("3" + "2")', "32", "5", "Since '3' and '2' contain digits, Python adds them mathematically to get 5.", "M01", False, "conceptual"),
        ("G3-02 M01 StrPlusIntCoerce", 'print("Age: " + 5)', "TypeError", "Age: 5", "Python automatically converts the integer 5 to a string when using + with text.", "M01", False, "conceptual"),
        ("G3-03 M02 SlashReturnsInt", "print(7 / 2)", "3.5", "3", "Both 7 and 2 are integers, so single slash / returns an integer and drops the decimal.", "M02", False, "conceptual"),
        ("G3-04 M02 FloorDivReturnsFloat", "print(9 // 2)", "4", "4.5", "The // operator divides 9 by 2 and keeps the decimal 4.5.", "M02", False, "conceptual"),
        ("G3-05 M03 SingleEqComparison", "if x = 5:\n    print('yes')", "SyntaxError", "yes", "In an if condition, a single equals sign = checks whether x is equal to 5.", "M03", False, "conceptual"),
        ("G3-06 M04 PlusBeforeMultiply", "print(10 - 2 * 3)", "4", "24", "You do subtraction first from left to right: 10 - 2 is 8, and 8 * 3 is 24.", "M04", False, "conceptual"),
        ("G3-07 M05 OneBasedIndexing", 's = "python"\nprint(s[1])', "y", "p", "Index 1 refers to the first character of the string because indexing starts at 1.", "M05", False, "conceptual"),
        ("G3-08 M06 RangeStartsAtOne", "for i in range(3):\n    print(i)", "0 1 2", "1 2 3", "range(3) starts at 1 by default and counts 1, 2, 3.", "M06", False, "conceptual"),
        ("G3-09 M07ReversePositional", "def sub(a, b):\n    return a - b\nprint(sub(10, 3))", "7", "-7", "Positional arguments bind from right to left so 10 goes to b and 3 goes to a.", "M07", False, "conceptual"),
        ("G3-10 M08 BaseCaseIncludedAfter", "def countdown(n):\n    if n == 0:\n        return\n    print(n)\n    countdown(n - 1)\ncountdown(3)", "3 2 1", "3 2 1 0", "The recursive function prints 3, 2, 1, 0 down to and including the base case 0.", "M08", False, "conceptual"),

        # --- GROUP 4: 10 Wrong Answer + Trace/Calculation Error -> INSUFFICIENT (trace_error) ---
        ("G4-01 M02 Sound Floor + Calc Slip", "print(7 // 2)", "3", "4", "The // operator performs floor division and returns the integer quotient, but I miscalculated 7 divided by 2 as 4.", "INSUFFICIENT", False, "trace_error"),
        ("G4-02 M02 Sound Float + Calc Slip", "print(9 / 2)", "4.5", "3.5", "Single slash / performs true floating-point division, and 9 divided by 2 is 3.5.", "INSUFFICIENT", False, "trace_error"),
        ("G4-03 M04 Sound Precedence + Mult Slip", "print(2 + 3 * 4)", "14", "13", "Multiplication * has higher precedence than +, so 3 * 4 is evaluated first to get 11, plus 2 is 13.", "INSUFFICIENT", False, "trace_error"),
        ("G4-04 M04 Sound Parentheses + Add Slip", "print((2 + 3) * 4)", "20", "24", "Parentheses are evaluated first before multiplication, so 2 + 3 is 6 and 6 * 4 is 24.", "INSUFFICIENT", False, "trace_error"),
        ("G4-05 M05 Sound ZeroBased + Lookup Slip", 's = "hello"\nprint(s[1])', "e", "l", "Python uses 0-based indexing where index 0 is the first character and index 1 is the second character, which I read as l.", "INSUFFICIENT", False, "trace_error"),
        ("G4-06 M06 Sound Exclusive + Sum Slip", "s = 0\nfor i in range(1, 5):\n    s += i\nprint(s)", "10", "9", "range(1, 5) excludes the stop boundary 5 so it visits 1, 2, 3, 4, and 1 + 2 + 3 + 4 is 9.", "INSUFFICIENT", False, "trace_error"),
        ("G4-07 M01 Sound Concat + Copy Slip", 'print("12" + "34")', "1234", "1243", "Both are strings in quotes so + concatenates them, and I wrote 1243 when copying the digits.", "INSUFFICIENT", False, "trace_error"),
        ("G4-08 M07 Sound Positional + Sub Slip", "def diff(a, b):\n    return a - b\nprint(diff(15, 6))", "9", "8", "Positional arguments bind in order from left to right so a is 15 and b is 6, and 15 - 6 is 8.", "INSUFFICIENT", False, "trace_error"),
        ("G4-09 Mechanical Loop Trace Narration", "x = 1\nfor i in range(3):\n    x += i\nprint(x)", "4", "5", "I traced through the loop step by step in my head and ended up with 5.", "INSUFFICIENT", False, "trace_error"),
        ("G4-10 Pure Arithmetic Steps Without Rule", "print(2 + 3 * 4 - 1)", "13", "15", "3 times 4 is 12, and 12 plus 2 is 15.", "INSUFFICIENT", False, "trace_error"),

        # --- GROUP 5: 10 Missing/Thin/Bare Reasoning Cases (Wrong & Correct Answers) -> INSUFFICIENT ---
        ("G5-01 WrongAns + Bare Output", "print(7 // 2)", "3", "4", "The answer is 4.", "INSUFFICIENT", False, "other"),
        ("G5-02 WrongAns + Code Restatement", "print(7 // 2)", "3", "3.5", "7 // 2 divides 7 by 2.", "INSUFFICIENT", False, "other"),
        ("G5-03 WrongAns + Loop Restatement", "for i in range(1, 5):\n    print(i)", "1 2 3 4", "1 2 3 4 5", "The loop prints i for range(1, 5).", "INSUFFICIENT", False, "other"),
        ("G5-04 WrongAns + Precedence Restatement", "print(2 + 3 * 4)", "14", "20", "It calculates 2 + 3 * 4 and prints 20.", "INSUFFICIENT", False, "other"),
        ("G5-05 WrongAns + Indexing Restatement", 's = "hello"\nprint(s[1])', "e", "h", "It prints s[1] from hello.", "INSUFFICIENT", False, "other"),
        ("G5-06 CorrectAns + Bare Answer Restatement", "print(7 // 2)", "3", "3", "The answer is 3.", "INSUFFICIENT", True, "other"),
        ("G5-07 CorrectAns + Bare Because Restatement", "print(7 // 2)", "3", "3", "Because 7 // 2 is 3.", "INSUFFICIENT", True, "other"),
        ("G5-08 CorrectAns + Bare Output Claim", "print(2 + 3 * 4)", "14", "14", "The output will be 14.", "INSUFFICIENT", True, "other"),
        ("G5-09 CorrectAns + Single Number Reason", "print(7 // 2)", "3", "3", "3", "INSUFFICIENT", True, "other"),
        ("G5-10 CorrectAns + Empty Reasoning", "print(7 // 2)", "3", "3", "", "INSUFFICIENT", True, "other"),

        # --- GROUP 6: 5 Out-of-Scope (OOS) Cases -> OOS ---
        ("G6-01 OOS List Aliasing Copy", "x = [1, 2]\ny = x\ny.append(3)\nprint(x)", "[1, 2, 3]", "[1, 2]", "Assigning y = x automatically creates a separate copy of the list.", "OOS", False, "conceptual"),
        ("G6-02 OOS List Sort Return Value", "nums = [3, 1, 2]\nres = nums.sort()\nprint(res)", "None", "[1, 2, 3]", "Calling .sort() returns a new sorted list and assigns it to res.", "OOS", False, "conceptual"),
        ("G6-03 OOS Mutable Default Arg", "def f(a=[]):\n    a.append(1)\n    return a\nprint(f(), f())", "[1] [1, 1]", "[1] [1]", "Default list arguments are re-created fresh on every function call.", "OOS", False, "conceptual"),
        ("G6-04 OOS Dict Missing Key None", "d = {'a': 1}\nprint(d['b'])", "KeyError", "None", "Accessing a missing key in a dictionary with brackets returns None instead of raising KeyError.", "OOS", False, "conceptual"),
        ("G6-05 OOS Set Bracket Indexing", "s = {10, 20, 30}\nprint(s[0])", "TypeError", "10", "Sets maintain insertion order and support bracket indexing just like lists.", "OOS", False, "conceptual"),

        # --- GROUP 7: 5 Ambiguous Cases -> INSUFFICIENT ---
        ("G7-01 Ambiguous Vague Feel", "print(7 // 2)", "3", "3.5", "I think Python just does regular math on these two numbers here.", "INSUFFICIENT", False, "other"),
        ("G7-02 Ambiguous Noncommittal", "print(2 + 3 * 4)", "14", "20", "I computed the numbers in the expression and got 20 as the result.", "INSUFFICIENT", False, "other"),
        ("G7-03 Ambiguous String Mention", 's = "hello"\nprint(s[1])', "e", "h", "The variable s holds the text hello and we print a character from it.", "INSUFFICIENT", False, "other"),
        ("G7-04 Ambiguous Loop Mention", "for i in range(1, 5):\n    print(i)", "1 2 3 4", "1 2 3 4 5", "The for loop iterates over the numbers in range and prints each one.", "INSUFFICIENT", False, "other"),
        ("G7-05 Ambiguous Function Mention", "def f(a, b):\n    return a - b\nprint(f(5, 2))", "3", "-3", "The function takes a and b and subtracts them to return the answer.", "INSUFFICIENT", False, "other"),

        # --- GROUP 8: 5 Natural-Language Paraphrase Cases (Section 4) -> M01-M08 ---
        ("G8-01 Paraphrase M02 Keeps Decimal", "print(7 // 2)", "3", "3.5", "I think // keeps the decimal", "M02", False, "conceptual"),
        ("G8-02 Paraphrase M04 Plus First", "print(2 + 3 * 4)", "14", "20", "Since + comes first in the expression, I added before multiplying", "M04", False, "conceptual"),
        ("G8-03 Paraphrase M05 Pos 1 First Char", 's = "hello"\nprint(s[1])', "e", "h", "Position 1 is the first character", "M05", False, "conceptual"),
        ("G8-04 Paraphrase M06 Stops At 5 Included", "for i in range(1, 5):\n    print(i)", "1 2 3 4", "1 2 3 4 5", "range stops at 5 so 5 is included", "M06", False, "conceptual"),
        ("G8-05 Paraphrase M01 Looks Like Number", 'print("3" + "2")', "32", "5", "Since '3' looks like a number, Python adds 3 and 2", "M01", False, "conceptual"),

        # --- GROUP 9: 5 Unusual Formatting Cases -> Normalized answer_correct=True ---
        ("G9-01 Whitespace Padded Answer", "print(7 // 2)", "3", "   3   ", "The // operator performs floor division and drops the decimal part to return 3.", "NONE", True, "correct"),
        ("G9-02 Newline vs Space Sequence", "for i in range(1, 5):\n    print(i)", "1 2 3 4", "1\n2\n3\n4", "In range(1, 5), the stop value 5 is exclusive, so it outputs 1, 2, 3, 4.", "NONE", True, "correct"),
        ("G9-03 Comma-Separated Sequence", "for i in range(1, 5):\n    print(i)", "1 2 3 4", "1, 2, 3, 4", "In range(1, 5), the stop value 5 is exclusive, so it outputs 1, 2, 3, 4.", "NONE", True, "correct"),
        ("G9-04 Bracketed List Sequence", "for i in range(1, 5):\n    print(i)", "1 2 3 4", "[1, 2, 3, 4]", "In range(1, 5), the stop value 5 is exclusive, so it outputs 1, 2, 3, 4.", "NONE", True, "correct"),
        ("G9-05 Quoted Character Output", 's = "hello"\nprint(s[1])', "e", "'e'", "Python uses 0-based indexing so index 1 is the second character 'e'.", "NONE", True, "correct"),

        # --- GROUP 10: 5 Empty / Short / Guessing / Very Long Inputs -> Safe INSUFFICIENT ---
        ("G10-01 Completely Empty Inputs", "print(7 // 2)", "3", "", "", "INSUFFICIENT", False, "other"),
        ("G10-02 Whitespace Only Inputs", "print(7 // 2)", "3", "   ", "   ", "INSUFFICIENT", False, "other"),
        ("G10-03 Question Mark Reasoning", "print(7 // 2)", "3", "4", "???", "INSUFFICIENT", False, "other"),
        ("G10-04 Explicit Guessing", "print(7 // 2)", "3", "4", "I don't know, I just guessed 4.", "INSUFFICIENT", False, "other"),
        ("G10-05 Very Long Rambling Input", "print(7 // 2)", "3", "4", ("I looked at the numbers in the question and tried to figure out what happens. " * 40), "INSUFFICIENT", False, "other"),
    ]

    passed = 0
    for label, q, ca, sa, sr, exp_mid, exp_ans_corr, exp_err_type in matrix_cases:
        session = pipeline.diagnose_and_intervene(q, ca, sa, sr)
        diag = session["diagnosis"]
        assert diag["misconception_id"] == exp_mid, (
            f"[{label}] Expected misconception_id={exp_mid}, got {diag['misconception_id']} | diag={diag}"
        )
        assert diag["answer_correct"] is exp_ans_corr, (
            f"[{label}] Expected answer_correct={exp_ans_corr}, got {diag['answer_correct']} | diag={diag}"
        )
        assert diag["error_type"] == exp_err_type, (
            f"[{label}] Expected error_type={exp_err_type}, got {diag['error_type']} | diag={diag}"
        )
        # Verify response contract safety for frontend rendering
        assert isinstance(diag.get("confidence"), float)
        assert isinstance(diag.get("evidence"), str) and len(diag["evidence"]) > 0
        assert isinstance(diag.get("rationale"), str) and len(diag["rationale"]) > 0
        if exp_mid in TRACKED_MISCONCEPTION_IDS:
            assert session.get("intervention") is not None, f"[{label}] Missing intervention for {exp_mid}"
            assert session.get("followup") is not None, f"[{label}] Missing followup for {exp_mid}"
        else:
            assert isinstance(session.get("message"), str) and len(session["message"]) > 0, (
                f"[{label}] Missing non-M message for {exp_mid}"
            )
        passed += 1

    print(f"Broad Robustness Matrix Result: {passed}/{len(matrix_cases)} passed.")

    # Verify the 10 explicit Website Test Scenarios from Section 11
    print("\n--- VERIFYING 10 EXPLICIT WEBSITE SCENARIOS (SCENARIOS 1-10) ---")
    website_scenarios = [
        ("Website Scenario 1", "print(7 // 2)", "3", "3", "// performs floor division and gives the integer part, which is 3", "NONE", True, "correct"),
        ("Website Scenario 2", "print(7 // 2)", "3", "3", "// does normal division and removes the decimal part", "M02", True, "conceptual"),
        ("Website Scenario 3", "print(7 // 2)", "3", "1", "// gives the remainder when dividing", "M02", False, "conceptual"),
        ("Website Scenario 4", "print(7 // 2)", "3", "4", "// performs floor division, and I miscalculated 7 divided by 2 as 4", "INSUFFICIENT", False, "trace_error"),
        ("Website Scenario 5", "print(7 // 2)", "3", "4", "I don't know", "INSUFFICIENT", False, "other"),
        ("Website Scenario 6", "print(7 // 2)", "3", "3", "The answer is 3", "INSUFFICIENT", True, "other"),
        ("Website Scenario 7", "print(7 // 2)", "3", "3", "I thought x = y creates a separate copy of a list", "OOS", True, "conceptual"),
        ("Website Scenario 8", "print(2 + 3 * 4)", "14", "20", "Python evaluates from left to right so 2 + 3 is 5 and 5 * 4 is 20", "M04", False, "conceptual"),
        ("Website Scenario 9", 's = "hello"\nprint(s[1])', "e", "h", "Position 1 is the first character", "M05", False, "conceptual"),
        ("Website Scenario 10", "for i in range(1, 5):\n    print(i)", "1 2 3 4", "1 2 3 4 5", "range includes the stop value 5", "M06", False, "conceptual"),
    ]
    for w_label, q, ca, sa, sr, exp_mid, exp_ans_corr, exp_err_type in website_scenarios:
        w_res = pipeline.diagnose_and_intervene(q, ca, sa, sr)
        w_diag = w_res["diagnosis"]
        print(
            f"  [PASS] {w_label}: Pred={w_diag['misconception_id']} (Exp={exp_mid}) | "
            f"AnsCorr={w_diag['answer_correct']} | ErrType={w_diag['error_type']} | Conf={w_diag['confidence']}"
        )
        assert w_diag["misconception_id"] == exp_mid, f"{w_label} failed: expected {exp_mid}, got {w_diag['misconception_id']}"
        assert w_diag["answer_correct"] is exp_ans_corr, f"{w_label} failed: expected answer_correct={exp_ans_corr}"
        assert w_diag["error_type"] == exp_err_type, f"{w_label} failed: expected error_type={exp_err_type}"

    pipeline.reset_learner_state()
    print("--- ALL 75 MATRIX CASES + 10 WEBSITE SCENARIOS PASSED ---")


if __name__ == "__main__":
    run_end_to_end_demo()



