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


class ReLearnPipeline:
    """
    Unified End-to-End Pipeline coordinating Person 1 diagnosis and Person 2 adaptive remediation.
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
        2. Passes diagnosis and student context to Person 2 run_person2().
        3. Returns structured session state.
        """
        # Context preserving original student interaction
        context = {
            "question": question,
            "correct_answer": correct_answer,
            "student_answer": student_answer,
            "student_reasoning": student_reasoning
        }

        # Step 1: Person 1 diagnosis
        diagnosis = self.classifier.diagnose_student(
            question=question,
            correct_answer=correct_answer,
            student_answer=student_answer,
            student_reasoning=student_reasoning
        )

        # Step 2: Person 2 intervention / routing
        p2_resp = run_person2(diagnosis, context)

        session = {
            "context": context,
            "diagnosis": diagnosis,
            "intervention": p2_resp.get("intervention"),
            "followup": p2_resp.get("followup"),
            "status": p2_resp.get("status"),
            "message": p2_resp.get("message"),
            "next_action": p2_resp.get("next_action")
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
        2. Calls Person 2 evaluate_followup().
        3. If RESOLVED -> marks COMPLETED.
        4. If NOT_RESOLVED -> triggers Person 2 generate_second_intervention().
        """
        diagnosis = session["diagnosis"]
        followup_obj = session.get("followup")

        # If no follow-up was generated (NONE, INSUFFICIENT, OOS)
        if not followup_obj:
            return {
                "diagnosis": diagnosis,
                "status": session.get("status"),
                "message": session.get("message"),
                "next_action": session.get("next_action")
            }

        # Step 3: Person 2 follow-up evaluation
        resolution = evaluate_followup(
            diagnosis=diagnosis,
            followup_question=followup_obj,
            followup_answer=followup_answer,
            followup_reasoning=followup_reasoning
        )

        out = {
            "context": session.get("context"),
            "diagnosis": diagnosis,
            "intervention": session.get("intervention"),
            "followup": followup_obj,
            "student_followup_submission": {
                "answer": followup_answer,
                "reasoning": followup_reasoning
            },
            "resolution": resolution
        }

        if resolution["status"] == "RESOLVED":
            out["next_action"] = "COMPLETED"
        else:
            out["next_action"] = "SECOND_INTERVENTION"
            out["second_intervention"] = generate_second_intervention(
                diagnosis=diagnosis,
                context=session.get("context", {}),
                previous_intervention=session.get("intervention")
            )

        return out


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

    print("\n" + "=" * 65)
    print("ALL RELEARN END-TO-END DEMO SCENARIOS COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_end_to_end_demo()
