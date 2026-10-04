#!/usr/bin/env python3
"""
ReLearn Python Bridge Service
Provides a fast stdio JSON-RPC interface for the ReLearnPipeline.
Reads single-line JSON requests from stdin, processes using ReLearnPipeline,
and outputs single-line JSON responses to stdout.
"""

import sys
import os
import json
import traceback

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from relearn_pipeline import ReLearnPipeline

def main():
    try:
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        pipeline = ReLearnPipeline()
        # Notify host that pipeline is initialized and ready
        sys.stdout.write(json.dumps({"status": "READY"}) + "\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(json.dumps({"status": "INIT_ERROR", "error": str(e), "trace": traceback.format_exc()}) + "\n")
        sys.stdout.flush()
        return

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            action = req.get("action")
            req_id = req.get("id")

            if action == "diagnose":
                q_val = req.get("question")
                ca_val = req.get("correct_answer")
                sa_val = req.get("student_answer")
                sr_val = req.get("student_reasoning")
                result = pipeline.diagnose_and_intervene(
                    question="" if q_val is None else str(q_val),
                    correct_answer="" if ca_val is None else str(ca_val),
                    student_answer="" if sa_val is None else str(sa_val),
                    student_reasoning="" if sr_val is None else str(sr_val),
                )
                response = {"id": req_id, "success": True, "data": result}
            elif action == "evaluate":
                session = req.get("session") or {}
                fa_val = req.get("followup_answer")
                fr_val = req.get("followup_reasoning")
                result = pipeline.evaluate_resolution(
                    session=session if isinstance(session, dict) else {},
                    followup_answer="" if fa_val is None else str(fa_val),
                    followup_reasoning="" if fr_val is None else str(fr_val),
                )
                response = {"id": req_id, "success": True, "data": result}
            elif action == "get_learner_state":
                response = {"id": req_id, "success": True, "data": pipeline.get_learner_state()}
            elif action == "reset_learner_state":
                response = {"id": req_id, "success": True, "data": pipeline.reset_learner_state()}
            elif action == "ping":
                response = {"id": req_id, "success": True, "message": "pong"}
            else:
                response = {"id": req_id, "success": False, "error": f"Unknown action: {action}"}

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({
                "id": req.get("id") if "req" in locals() and isinstance(req, dict) else None,
                "success": False,
                "error": str(e),
                "trace": traceback.format_exc()
            }) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
