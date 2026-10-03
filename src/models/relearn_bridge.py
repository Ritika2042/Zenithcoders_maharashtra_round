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
                result = pipeline.diagnose_and_intervene(
                    question=req.get("question", ""),
                    correct_answer=req.get("correct_answer", ""),
                    student_answer=req.get("student_answer", ""),
                    student_reasoning=req.get("student_reasoning", "")
                )
                response = {"id": req_id, "success": True, "data": result}
            elif action == "evaluate":
                session = req.get("session", {})
                followup_answer = req.get("followup_answer", "")
                followup_reasoning = req.get("followup_reasoning", "")
                result = pipeline.evaluate_resolution(
                    session=session,
                    followup_answer=followup_answer,
                    followup_reasoning=followup_reasoning
                )
                response = {"id": req_id, "success": True, "data": result}
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
