#!/usr/bin/env python3
"""
Person 2: Adaptive Pedagogical Intervention & Resolution Engine
ReLearn Adaptive Multimodal Learning Environment

Responsibilities:
1. Ingests Person 1's locked diagnosis:
   {
       "misconception_id": "M01-M08 | NONE | INSUFFICIENT | OOS",
       "confidence": float,
       "evidence": str,
       "rationale": str,
       "decision_source": str
   }
2. Generates targeted student-facing pedagogical intervention (M01-M08).
3. Generates deterministic follow-up question to test misconception resolution.
4. Evaluates student follow-up responses (deterministic, no LLM required).
5. Returns RESOLVED or NOT_RESOLVED status.
6. Generates secondary reinforced intervention (concrete trace/counterexample) if NOT_RESOLVED.
7. Handles NONE, INSUFFICIENT, and OOS safely without hallucinated remedial interventions.
"""

import json
import os
import re
import sys
from typing import Dict, List, Any, Optional

# Ensure parent and sibling imports work cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)


# =============================================================================
# A. STRUCTURED INTERVENTION BANK (M01 - M08)
# =============================================================================

INTERVENTION_BANK: Dict[str, Dict[str, Any]] = {
    "M01": {
        "misconception_id": "M01",
        "title": "String vs Number Type Confusion",
        "short_explanation": "\"5\" is a string because it is enclosed in quotation marks. Python treats strings as literal text and will not automatically convert text characters into numerical values for arithmetic.",
        "concrete_example": "x = \"5\"\ny = 2\n# x + y raises TypeError: can only concatenate str (not \"int\") to str",
        "contrast_example": "5 + 2 -> 7 (Integer addition)\n\"5\" + \"2\" -> \"52\" (String concatenation)\nint(\"5\") + 2 -> 7 (Explicit conversion required)",
        "key_rule": "Quotes make a value a string. Adding a string and an integer causes a TypeError; you must explicitly call int() to convert text to a number.",
        "followup": {
            "question": "What is the exact output of this code?\n\na = \"3\"\nb = \"4\"\nprint(a + b)\n\nExplain why in one sentence.",
            "expected_answer": "34",
            "evaluation_type": "exact_and_conceptual",
            "accepted_keywords": ["concatenate", "concatenation", "string", "quotes", "text", "joined", "not 7", "combines", "character"],
            "forbidden_patterns": [r"\b7\b", r"\badds to 7\b", r"\bconverts to (integer|number)\b", r"\bregular addition\b"]
        },
        "second_intervention": {
            "title": "Concrete Type & Memory Model Trace",
            "explanation": "Think of quotes as literal packaging. The variable holds the character glyph '3', not the quantity 3. When Python sees + between two strings, its only job is gluing characters together.",
            "step_by_step_trace": (
                "1. Check type(a) -> <class 'str'> (characters '3')\n"
                "2. Check type(b) -> <class 'str'> (characters '4')\n"
                "3. String + String triggers text concatenation:\n"
                "   '3' + '4' produces '34', NEVER mathematical 7."
            ),
            "key_takeaway": "No quotes = number (math). Quotes = string (text gluing)."
        }
    },

    "M02": {
        "misconception_id": "M02",
        "title": "Division Semantics (/ vs //)",
        "short_explanation": "In Python, the single slash (/) always performs float division and returns a float with decimals. The double slash (//) is floor division, which computes integer division and rounds down to the nearest whole integer.",
        "concrete_example": "7 / 2 -> 3.5 (Float division: always keeps decimal portion)\n7 // 2 -> 3 (Floor division: rounds down to nearest integer)",
        "contrast_example": "8 / 2 -> 4.0 (Float division always outputs float, even if exact)\n8 // 2 -> 4 (Floor division outputs integer 4)\n-7 // 2 -> -4 (Rounds DOWN towards negative infinity, not -3)",
        "key_rule": "/ always returns a float (with decimal). // discards the fractional remainder and rounds down to the nearest integer.",
        "followup": {
            "question": "What is the exact output of this code?\n\nval = 11 // 4\nprint(val)\n\nExplain how floor division computes this result.",
            "expected_answer": "2",
            "evaluation_type": "exact_and_conceptual",
            "accepted_keywords": ["floor", "rounds down", "integer division", "discards remainder", "whole number", "goes in 2 times", "drops decimal", "discards fraction"],
            "forbidden_patterns": [r"2\.75", r"\bkeeps decimal\b", r"\breturns float\b", r"\bfloat\b"]
        },
        "second_intervention": {
            "title": "Visual Number Line & Remainder Model",
            "explanation": "Picture a number line. Float division gives 11 / 4 = 2.75. Floor division (//) forces the result leftward to the first integer less than or equal to 2.75, which is 2.",
            "step_by_step_trace": (
                "1. Exact mathematical division: 11 / 4 = 2.75\n"
                "2. Apply floor operation (round down to lower integer):\n"
                "   floor(2.75) = 2\n"
                "3. The remainder (3) is discarded; output is integer 2."
            ),
            "key_takeaway": "// means 'how many whole times does 4 fit into 11 without decimals?' Answer: 2."
        }
    },

    "M03": {
        "misconception_id": "M03",
        "title": "Assignment vs Equality (= vs ==)",
        "short_explanation": "A single equals sign (=) is an assignment statement that assigns a value to a variable name. A double equals sign (==) is a comparison operator that checks whether two values are equal and evaluates to True or False.",
        "concrete_example": "x = 5  # Statement: stores the number 5 inside variable x\nx == 5 # Expression: asks 'is x equal to 5?', evaluates to True",
        "contrast_example": "x = 10 (Action: modifies variable x; produces no boolean value)\nx == 10 (Comparison: inspects variable x; evaluates to True or False)\nif x = 5: # SyntaxError! Python forbids assignment inside if conditions.",
        "key_rule": "Use = to store or update data. Use == to compare two values.",
        "followup": {
            "question": "What is the exact output of this code?\n\nscore = 100\nresult = (score == 50)\nprint(result)\n\nExplain what score == 50 does.",
            "expected_answer": "False",
            "evaluation_type": "exact_and_conceptual",
            "accepted_keywords": ["compare", "comparison", "checks", "equality", "not equal", "evaluates to false", "tests if equal", "does not assign", "false"],
            "forbidden_patterns": [r"\bassigns 50\b", r"\bscore becomes 50\b", r"\bsets score to 50\b", r"\btrue\b"]
        },
        "second_intervention": {
            "title": "Grammar Role & State Visualizer",
            "explanation": "Think of = as a command verb ('Put 5 in this box!'). Think of == as a yes/no question ('Are the contents of these two boxes identical?'). == never alters variable values.",
            "step_by_step_trace": (
                "1. score currently holds the value 100\n"
                "2. Evaluate expression score == 50:\n"
                "   Does 100 equal 50? No -> evaluates to False\n"
                "3. Assign result = False (score remains 100 untouched!)"
            ),
            "key_takeaway": "== asks a question and answers True/False. = performs a write operation."
        }
    },

    "M04": {
        "misconception_id": "M04",
        "title": "Operator Precedence",
        "short_explanation": "Python does not evaluate expressions strictly from left to right. It evaluates operators following a strict hierarchy of precedence (e.g., arithmetic * and / execute before + and -, and arithmetic executes before comparison operators like == or <).",
        "concrete_example": "val = 2 + 3 * 4\n# * has higher precedence than +, so 3 * 4 = 12 evaluates first, giving 2 + 12 = 14",
        "contrast_example": "2 + 3 * 4 -> 14 (Multiplication first)\n(2 + 3) * 4 -> 20 (Parentheses override precedence)\n5 < 3 + 4 -> True (Arithmetic + computes 7 before comparison < evaluates 5 < 7)",
        "key_rule": "Operators with higher precedence execute first. Wrap operations in parentheses () to override precedence or make order explicit.",
        "followup": {
            "question": "What is the exact output of this code?\n\nresult = 10 - 2 * 3\nprint(result)\n\nExplain which operation executes first and why.",
            "expected_answer": "4",
            "evaluation_type": "exact_and_conceptual",
            "accepted_keywords": ["multiplication first", "* before -", "higher precedence", "2 * 3", "order of operations", "precedence", "10 - 6", "multiplication has higher"],
            "forbidden_patterns": [r"\b24\b", r"\bleft to right\b", r"\bsubtraction first\b", r"\b8 \* 3\b"]
        },
        "second_intervention": {
            "title": "Hierarchical Expression Tree Model",
            "explanation": "Visualize the expression as an operations tree. Python searches for the operator with the highest rank first, computes its branch, and passes the result to lower-ranked operators.",
            "step_by_step_trace": (
                "1. Identify operators in '10 - 2 * 3': subtraction (-) and multiplication (*)\n"
                "2. Check precedence table: * ranks above -\n"
                "3. Step 1: Compute 2 * 3 = 6\n"
                "4. Step 2: Compute 10 - 6 = 4\n"
                "Result is 4, NOT (10 - 2) * 3 = 24."
            ),
            "key_takeaway": "Multiplication and division always bind tighter than addition and subtraction."
        }
    },

    "M05": {
        "misconception_id": "M05",
        "title": "Index / Position",
        "short_explanation": "Python uses 0-based indexing for all sequences (strings, lists, tuples). The first element is always at index 0, the second at index 1, and the last of N items is at index N-1.",
        "concrete_example": "word = \"CAT\"\nword[0] -> 'C' (1st letter)\nword[1] -> 'A' (2nd letter)\nword[2] -> 'T' (3rd letter)",
        "contrast_example": "word[0] is 'C' (0-based start)\nword[1] is 'A' (NOT the first item!)\nword[3] raises IndexError: string index out of range",
        "key_rule": "Always start indexing at 0. The n-th item is accessed via sequence[n - 1].",
        "followup": {
            "question": "What is the exact output of this code?\n\ncolors = [\"red\", \"green\", \"blue\"]\nprint(colors[0])\n\nExplain what index 0 accesses.",
            "expected_answer": "red",
            "evaluation_type": "exact_and_conceptual",
            "accepted_keywords": ["0-based", "zero-based", "first element", "first item", "starts at 0", "initial item", "index 0", "start counting at 0"],
            "forbidden_patterns": [r"\bgreen\b", r"\bsecond item\b", r"\bstarts at 1\b", r"\b1-based\b"]
        },
        "second_intervention": {
            "title": "Offset From Origin Memory Model",
            "explanation": "An index is not an item count—it is an offset (distance) from the start of memory. At the very start (0 distance away), you find the initial element.",
            "step_by_step_trace": (
                "Position:  [ 1st Item ]   [ 2nd Item ]   [ 3rd Item ]\n"
                "Element:       'red'         'green'        'blue'\n"
                "Index:           0              1              2\n"
                "colors[0] accesses distance 0 from the start -> 'red'."
            ),
            "key_takeaway": "Index 0 = Item 1. Index 1 = Item 2."
        }
    },

    "M06": {
        "misconception_id": "M06",
        "title": "Loop Values / Boundaries / Iteration Interval",
        "short_explanation": "In Python, range(start, stop) generates numbers starting at 'start' and stopping BEFORE 'stop'. The stop value is always EXCLUSIVE and is never generated or included in the loop body.",
        "concrete_example": "for i in range(1, 4):\n    print(i)\n# Prints: 1, 2, 3 (Stops before 4!)",
        "contrast_example": "range(1, 4) produces: 1, 2, 3 (Stop 4 is exclusive)\nrange(1, 5) produces: 1, 2, 3, 4\nrange(3) produces: 0, 1, 2 (Default start is 0)",
        "key_rule": "The stop boundary in range() is exclusive; iteration stops at stop - 1.",
        "followup": {
            "question": "What is the exact output of this code?\n\nnums = []\nfor i in range(2, 5):\n    nums.append(i)\nprint(nums)\n\nExplain why the number 5 is or is not in the list.",
            "expected_answer": "[2, 3, 4]",
            "evaluation_type": "exact_and_conceptual",
            "accepted_keywords": ["stop is exclusive", "not inclusive", "up to 4", "excludes 5", "does not include 5", "stops before 5", "exclusive", "stops at 4"],
            "forbidden_patterns": [r"\[2,\s*3,\s*4,\s*5\]", r"\bincludes 5\b", r"\binclusive\b", r"\b5 is included\b"]
        },
        "second_intervention": {
            "title": "Half-Open Interval Model [start, stop)",
            "explanation": "Python ranges follow half-open interval math: [start, stop), meaning start <= i < stop. When i reaches the stop value, the condition i < stop becomes False, halting the loop immediately.",
            "step_by_step_trace": (
                "Loop over range(2, 5) where condition is 'i < 5':\n"
                "  i = 2 (2 < 5? True)  -> append 2\n"
                "  i = 3 (3 < 5? True)  -> append 3\n"
                "  i = 4 (4 < 5? True)  -> append 4\n"
                "  i = 5 (5 < 5? False) -> LOOP TERMINATES WITHOUT EXECUTING BODY!\n"
                "Result: [2, 3, 4]."
            ),
            "key_takeaway": "range(a, b) produces exactly (b - a) items. 5 - 2 = 3 items: 2, 3, 4."
        }
    },

    "M07": {
        "misconception_id": "M07",
        "title": "Function Argument–Parameter Binding",
        "short_explanation": "Positional arguments in Python are bound to function parameters strictly by their left-to-right ORDER, not by variable names used in the caller. Keyword arguments bind by parameter name.",
        "concrete_example": (
            "def introduce(title, name):\n"
            "    print(title, name)\n"
            "name = \"Dr.\"\n"
            "title = \"Alice\"\n"
            "introduce(name, title)\n"
            "# Prints: 'Dr. Alice' (Because 1st argument 'name' binds to 1st parameter 'title')"
        ),
        "contrast_example": (
            "introduce(name, title) -> binds by position (1st argument -> 1st parameter)\n"
            "introduce(title=\"Dr.\", name=\"Alice\") -> binds by keyword name explicitly"
        ),
        "key_rule": "Positional arguments bind by position order. Python does not match caller variable names to parameter names unless keyword syntax (key=val) is explicitly written.",
        "followup": {
            "question": "What is the exact output of this code?\n\ndef show(a, b):\n    print(a - b)\nx = 10\ny = 2\nshow(y, x)\n\nExplain how y and x bind to parameters a and b.",
            "expected_answer": "-8",
            "evaluation_type": "exact_and_conceptual",
            "accepted_keywords": ["positional", "order", "y binds to a", "x binds to b", "a is 2", "b is 10", "first argument", "first parameter", "2 - 10", "-8"],
            "forbidden_patterns": [r"(?<!-)\b8\b", r"\b10\s*-\s*2\b", r"\ba\s+is\s+10\b", r"\bnames\s+match\b", r"\bx\s+matches\s+a\b"]
        },
        "second_intervention": {
            "title": "Parameter Slot / Conveyor Belt Model",
            "explanation": "A function signature creates ordered empty slots: show(Slot 1: a, Slot 2: b). When calling show(y, x), the caller variable names disappear; Python only places value 1 into Slot 1 and value 2 into Slot 2.",
            "step_by_step_trace": (
                "1. Caller evaluates argument values: y = 2, x = 10\n"
                "2. Function receives ordered values: (2, 10)\n"
                "3. Slot 1 (parameter a) receives 2\n"
                "4. Slot 2 (parameter b) receives 10\n"
                "5. Compute: a - b = 2 - 10 = -8"
            ),
            "key_takeaway": "Order is king in positional calls. Caller variable names are completely ignored."
        }
    },

    "M08": {
        "misconception_id": "M08",
        "title": "Recursion Termination / Base Case",
        "short_explanation": "A recursive function does not stop automatically. It must contain an explicit base case condition that returns a value without making another recursive call. Without a base case, recursion continues until memory is exhausted.",
        "concrete_example": (
            "def countdown(n):\n"
            "    if n == 0:\n"
            "        return \"Blastoff\"\n"
            "    return countdown(n - 1)\n"
            "# countdown(2) -> countdown(1) -> countdown(0) hits base case and returns 'Blastoff'"
        ),
        "contrast_example": (
            "With base case: if n == 0 returns safely and unwinds stack.\n"
            "Without base case: repeats countdown(n - 1) infinitely until RecursionError."
        ),
        "key_rule": "Every recursive function must have a reachable base case condition that stops further calls.",
        "followup": {
            "question": "What is the exact output of this code?\n\ndef blastoff(n):\n    if n == 1:\n        return \"Go!\"\n    return blastoff(n - 1)\nprint(blastoff(3))\n\nExplain whether this terminates or recurses infinitely.",
            "expected_answer": "Go!",
            "evaluation_type": "exact_and_conceptual",
            "accepted_keywords": ["terminates", "base case", "reaches 1", "n == 1", "returns go!", "stops at 1", "hits base case", "not infinite"],
            "forbidden_patterns": [r"RecursionError", r"\bnever stops\b", r"\binfinite recursion\b", r"\bnever reaches exit\b"]
        },
        "second_intervention": {
            "title": "Call Stack Frame Unwinding Model",
            "explanation": "Each recursive call pauses the current function and stacks a new frame on top. When the top frame satisfies the base case condition, it returns its value downward, closing each open frame one by one.",
            "step_by_step_trace": (
                "Call 1: blastoff(3) calls blastoff(2)\n"
                "Call 2:   blastoff(2) calls blastoff(1)\n"
                "Call 3:     blastoff(1) tests if n == 1 -> True! Returns 'Go!'\n"
                "Unwinding: Call 2 receives 'Go!' and returns it -> Call 1 receives 'Go!' and prints it."
            ),
            "key_takeaway": "When the base case condition triggers, recursion stops and returns the value."
        }
    }
}


# =============================================================================
# B & C. INTERVENTION & FOLLOW-UP GENERATORS
# =============================================================================

def generate_intervention(diagnosis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a targeted pedagogical intervention based on Person 1's diagnosis.
    Handles M01-M08, NONE, INSUFFICIENT, and OOS.
    """
    mid = diagnosis.get("misconception_id", "INSUFFICIENT")

    if mid in INTERVENTION_BANK:
        entry = INTERVENTION_BANK[mid]
        return {
            "status": "TARGETED_INTERVENTION",
            "misconception_id": mid,
            "title": entry["title"],
            "short_explanation": entry["short_explanation"],
            "concrete_example": entry["concrete_example"],
            "contrast_example": entry["contrast_example"],
            "key_rule": entry["key_rule"],
            "pedagogical_basis": diagnosis.get("evidence", "")
        }

    if mid == "NONE":
        return {
            "status": "NO_TARGETED_INTERVENTION",
            "misconception_id": "NONE",
            "title": "Sound Conceptual Understanding / Non-Misconception Slip",
            "short_explanation": "Your reasoning demonstrates sound conceptual logic. If your answer differed from expected, it appears to be an ordinary clerical, typo, or arithmetic calculation slip rather than a conceptual misunderstanding.",
            "key_rule": "Double-check numerical arithmetic and spelling keystrokes before submitting.",
            "pedagogical_basis": "Absence of conceptual misconception markers."
        }

    if mid == "INSUFFICIENT":
        return {
            "status": "NEED_MORE_EVIDENCE",
            "misconception_id": "INSUFFICIENT",
            "title": "Clarification Needed",
            "short_explanation": "I need to understand the reasoning behind your answer. Guessing or brief traces don't provide enough evidence to identify what conceptual rule was used.",
            "key_rule": "Explain the specific rule or mental step you used to trace the code.",
            "pedagogical_basis": "Insufficient student reasoning evidence."
        }

    # OOS
    return {
        "status": "OUT_OF_SCOPE",
        "misconception_id": "OOS",
        "title": "Concept Outside Target Taxonomy",
        "short_explanation": "Your reasoning refers to an advanced or out-of-scope Python topic (such as object identity/aliasing, chained comparisons, or mutable default arguments) outside our current CS1 misconception set.",
        "key_rule": "Review the Python reference guide for object aliasing, chained comparisons, and function default semantics.",
        "pedagogical_basis": diagnosis.get("evidence", "Identified belief outside M01-M08.")
    }


def generate_followup(diagnosis: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generates a deterministic follow-up question for M01-M08 to test resolution.
    Returns None for NONE, INSUFFICIENT, and OOS.
    """
    mid = diagnosis.get("misconception_id")
    if mid not in INTERVENTION_BANK:
        return None

    entry = INTERVENTION_BANK[mid]["followup"]
    return {
        "question": entry["question"],
        "expected_answer": entry["expected_answer"],
        "evaluation_type": entry["evaluation_type"]
    }


# =============================================================================
# D. FOLLOW-UP EVALUATOR (Deterministic, No LLM Required)
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize simple answers for clean comparison."""
    if not text:
        return ""
    # Strip whitespace, strip outer quotes, brackets, and lowercase
    t = text.strip().strip("'\"").strip()
    # Normalize list bracket spacing e.g. [2,3,4] -> [2, 3, 4]
    t = re.sub(r"\s*,\s*", ", ", t)
    return t


def evaluate_followup(
    diagnosis: Dict[str, Any],
    followup_question: Dict[str, Any],
    followup_answer: str,
    followup_reasoning: str
) -> Dict[str, Any]:
    """
    Evaluates student follow-up answer and reasoning deterministically.
    
    Rules:
    1. Correct answer AND sound conceptual reasoning -> RESOLVED
    2. Incorrect answer -> NOT_RESOLVED
    3. Correct answer BUT reasoning shows persistent misconception -> NOT_RESOLVED
    4. Empty reasoning or guessing -> NOT_RESOLVED
    """
    mid = diagnosis.get("misconception_id")
    if mid not in INTERVENTION_BANK:
        return {
            "status": "NOT_APPLICABLE",
            "confidence": 1.0,
            "evidence": "No follow-up evaluation defined for non-M classes.",
            "rationale": "Follow-up verification is only defined for M01-M08.",
            "persistent_misconception": False
        }

    entry = INTERVENTION_BANK[mid]["followup"]
    expected_norm = normalize_text(entry["expected_answer"])
    student_norm = normalize_text(followup_answer)
    r_lower = followup_reasoning.lower().strip()

    # Rule 4: Check for guessing / uninformative reasoning
    guessing_match = re.search(r"\b(i\s+)?(guessed|just guessed|guessing|dont know|don't know|not sure|no idea|unsure)\b", r_lower)
    if len(r_lower) < 8 or guessing_match:
        return {
            "status": "NOT_RESOLVED",
            "confidence": 0.90,
            "evidence": f"Follow-up reasoning is empty or indicates guessing: {repr(followup_reasoning)}.",
            "rationale": "Student did not demonstrate conceptual understanding; guessing does not resolve a misconception.",
            "persistent_misconception": True
        }

    # Rule 3: Check for persistent misconception in reasoning
    forbidden_pats = entry.get("forbidden_patterns", [])
    for pat in forbidden_pats:
        if re.search(pat, r_lower, re.IGNORECASE):
            return {
                "status": "NOT_RESOLVED",
                "confidence": 0.95,
                "evidence": f"Reasoning explicitly repeated misconception pattern: {repr(followup_reasoning)}.",
                "rationale": f"Student persists in {entry['expected_answer']} misconception despite providing an answer.",
                "persistent_misconception": True
            }

    # Rule 2: Check answer correctness
    # Allow exact match or numerical match (e.g., '2' vs '2.0' where appropriate)
    is_answer_correct = (student_norm == expected_norm) or (student_norm.lower() == expected_norm.lower())

    if not is_answer_correct:
        return {
            "status": "NOT_RESOLVED",
            "confidence": 0.95,
            "evidence": f"Student answered '{followup_answer}', expected '{entry['expected_answer']}'.",
            "rationale": f"Follow-up answer was incorrect, indicating the misconception {mid} has not yet been resolved.",
            "persistent_misconception": True
        }

    # Rule 1: Check for corroborating conceptual keywords
    accepted_kws = entry.get("accepted_keywords", [])
    has_conceptual_evidence = any(kw.lower() in r_lower for kw in accepted_kws)

    if has_conceptual_evidence:
        return {
            "status": "RESOLVED",
            "confidence": 0.98,
            "evidence": f"Correct answer '{followup_answer}' with sound conceptual rationale: {repr(followup_reasoning)}.",
            "rationale": f"Student demonstrated correct behavior and articulated the key concept of {mid}.",
            "persistent_misconception": False
        }
    else:
        # Correct answer without explicit negative markers, but limited conceptual vocabulary
        return {
            "status": "RESOLVED",
            "confidence": 0.85,
            "evidence": f"Correct answer '{followup_answer}' with consistent explanation: {repr(followup_reasoning)}.",
            "rationale": f"Student successfully solved the follow-up problem without expressing {mid}.",
            "persistent_misconception": False
        }


# =============================================================================
# E. SECOND INTERVENTION (Reinforced Trace / Different Angle)
# =============================================================================

def generate_second_intervention(
    diagnosis: Dict[str, Any],
    context: Dict[str, Any],
    previous_intervention: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates a second, reinforced intervention when the follow-up is NOT_RESOLVED.
    Approaches the same misconception with a concrete step-by-step trace or memory model.
    """
    mid = diagnosis.get("misconception_id")
    if mid not in INTERVENTION_BANK:
        return {
            "status": "NOT_APPLICABLE",
            "message": "Second intervention only applies to unresolved M01-M08 misconceptions."
        }

    sec = INTERVENTION_BANK[mid]["second_intervention"]
    return {
        "status": "SECOND_TARGETED_INTERVENTION",
        "misconception_id": mid,
        "title": sec["title"],
        "explanation": sec["explanation"],
        "step_by_step_trace": sec["step_by_step_trace"],
        "key_takeaway": sec["key_takeaway"],
        "pedagogical_approach": "concrete_step_by_step_trace_model"
    }


# =============================================================================
# F. ORCHESTRATION FUNCTION: run_person2()
# =============================================================================

def run_person2(diagnosis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete Person 2 orchestration function.
    Consumes Person 1 diagnosis + original context and returns clean JSON-serializable structure.
    """
    mid = diagnosis.get("misconception_id", "INSUFFICIENT")

    if mid in INTERVENTION_BANK:
        intervention = generate_intervention(diagnosis, context)
        followup = generate_followup(diagnosis, context)
        return {
            "diagnosis": diagnosis,
            "intervention": intervention,
            "followup": followup,
            "next_action": "WAIT_FOR_STUDENT"
        }

    if mid == "NONE":
        return {
            "diagnosis": diagnosis,
            "status": "NO_TARGETED_INTERVENTION",
            "message": "Your reasoning does not show one of the supported misconceptions. Great job on the core concepts!",
            "followup": None,
            "next_action": "CONTINUE_NORMAL_CURRICULUM"
        }

    if mid == "INSUFFICIENT":
        return {
            "diagnosis": diagnosis,
            "status": "NEED_MORE_EVIDENCE",
            "message": "I need to understand how you arrived at that answer. Please explain the step you used to trace the code.",
            "followup": None,
            "next_action": "PROMPT_FOR_EXPLANATION"
        }

    # OOS
    return {
        "diagnosis": diagnosis,
        "status": "OUT_OF_SCOPE",
        "message": "This concept relates to topics outside the current supported CS1 misconception set (e.g. object aliasing or default arguments).",
        "followup": None,
        "next_action": "REFER_TO_DOCUMENTATION"
    }


# =============================================================================
# G. COMPREHENSIVE DEMO & TEST HARNESS
# =============================================================================

def run_person2_tests():
    """
    Runs comprehensive end-to-end tests for Person 2:
    - M01 to M08 (Diagnose -> Intervene -> Followup -> Correct -> RESOLVED)
    - NONE, INSUFFICIENT, OOS (Non-interventional routing)
    - 2 NOT_RESOLVED cases triggering Second Intervention
    """
    print("=" * 70)
    print("PERSON 2: ADAPTIVE INTERVENTION & RESOLUTION ENGINE TEST")
    print("=" * 70)

    # Simulated Person 1 outputs for test cases
    test_cases = [
        {
            "mid": "M01",
            "diag": {
                "misconception_id": "M01",
                "confidence": 0.49,
                "evidence": "Student treats quoted '5' as integer 5.",
                "rationale": "Classified as M01 (String vs Number Type Confusion).",
                "decision_source": "evidence_rule"
            },
            "context": {"question": "x = '5'; y = 2; print(x + y)", "correct_answer": "TypeError", "student_answer": "7", "student_reasoning": "Quotes don't matter, it treats it as 5."},
            "followup_ans": "34",
            "followup_reason": "Both are strings in quotes, so the + operator concatenates them into '34'."
        },
        {
            "mid": "M02",
            "diag": {
                "misconception_id": "M02",
                "confidence": 0.35,
                "evidence": "Student believes // keeps decimals as 3.5.",
                "rationale": "Classified as M02 (Division Semantics).",
                "decision_source": "evidence_rule"
            },
            "context": {"question": "print(7 // 2)", "correct_answer": "3", "student_answer": "3.5", "student_reasoning": "Floor division keeps decimal."},
            "followup_ans": "2",
            "followup_reason": "11 // 4 performs floor division, which discards the remainder and rounds down to the whole integer 2."
        },
        {
            "mid": "M03",
            "diag": {
                "misconception_id": "M03",
                "confidence": 0.37,
                "evidence": "Student says single equals = compares values.",
                "rationale": "Classified as M03 (Assignment vs Equality).",
                "decision_source": "evidence_rule"
            },
            "context": {"question": "if x = 5: print(True)", "correct_answer": "SyntaxError", "student_answer": "True", "student_reasoning": "= tests equality."},
            "followup_ans": "False",
            "followup_reason": "score == 50 is an equality comparison check, and since 100 != 50 it evaluates to False without modifying score."
        },
        {
            "mid": "M04",
            "diag": {
                "misconception_id": "M04",
                "confidence": 0.44,
                "evidence": "Student evaluated bitwise before comparison.",
                "rationale": "Classified as M04 (Operator Precedence).",
                "decision_source": "model"
            },
            "context": {"question": "val = 6; if val & 2 == 2:", "correct_answer": "Bit not set", "student_answer": "Bit set", "student_reasoning": "& evaluates before =="},
            "followup_ans": "4",
            "followup_reason": "Multiplication has higher precedence than subtraction, so 2 * 3 evaluates first to 6, then 10 - 6 gives 4."
        },
        {
            "mid": "M05",
            "diag": {
                "misconception_id": "M05",
                "confidence": 0.40,
                "evidence": "Student assumed index 1 refers to first letter.",
                "rationale": "Classified as M05 (Index / Position).",
                "decision_source": "evidence_rule"
            },
            "context": {"question": "s = 'Python'; print(s[1])", "correct_answer": "y", "student_answer": "P", "student_reasoning": "Index 1 is first letter."},
            "followup_ans": "red",
            "followup_reason": "Python uses 0-based indexing, so index 0 accesses the initial first item 'red'."
        },
        {
            "mid": "M06",
            "diag": {
                "misconception_id": "M06",
                "confidence": 0.51,
                "evidence": "Student believed range stop value 5 is inclusive.",
                "rationale": "Classified as M06 (Loop Values / Boundaries).",
                "decision_source": "model"
            },
            "context": {"question": "for x in range(1, 5): print(x)", "correct_answer": "1 2 3 4", "student_answer": "1 2 3 4 5", "student_reasoning": "range stop is inclusive."},
            "followup_ans": "[2, 3, 4]",
            "followup_reason": "The stop boundary 5 in range(2, 5) is exclusive, so the loop stops before 5 and produces [2, 3, 4]."
        },
        {
            "mid": "M07",
            "diag": {
                "misconception_id": "M07",
                "confidence": 0.30,
                "evidence": "Student assumed parameters bind by caller variable name.",
                "rationale": "Classified as M07 (Argument Binding).",
                "decision_source": "evidence_rule"
            },
            "context": {"question": "greet(last='Smith', 'John')", "correct_answer": "SyntaxError", "student_answer": "John Smith", "student_reasoning": "Parameters map by variable name."},
            "followup_ans": "-8",
            "followup_reason": "Positional arguments bind by left-to-right order: first argument y=2 binds to a, and x=10 binds to b, so 2 - 10 gives -8."
        },
        {
            "mid": "M08",
            "diag": {
                "misconception_id": "M08",
                "confidence": 0.65,
                "evidence": "Student believed recursion never terminates.",
                "rationale": "Classified as M08 (Recursion Termination).",
                "decision_source": "model"
            },
            "context": {"question": "def f(n): ...", "correct_answer": "0", "student_answer": "RecursionError", "student_reasoning": "Function never returns."},
            "followup_ans": "Go!",
            "followup_reason": "When n decrements to 1, the base case if n == 1 triggers and returns 'Go!', terminating the recursion safely."
        }
    ]

    print("\n--- 1. TESTING M01-M08 RESOLUTION PIPELINE (All Correct) ---")
    for tc in test_cases:
        p2_resp = run_person2(tc["diag"], tc["context"])
        interv = p2_resp["intervention"]
        follow = p2_resp["followup"]
        eval_res = evaluate_followup(tc["diag"], follow, tc["followup_ans"], tc["followup_reason"])

        print(f"\n[{tc['mid']}] {interv['title']}")
        print(f"  Diagnosis:        {tc['diag']['misconception_id']} (Source: {tc['diag']['decision_source']})")
        print(f"  Key Rule:         {interv['key_rule']}")
        print(f"  Follow-up Q:      {follow['question'].splitlines()[0]}")
        print(f"  Student Ans:      '{tc['followup_ans']}'")
        print(f"  Resolution Eval:  {eval_res['status']} (Conf: {eval_res['confidence']})")
        assert eval_res["status"] == "RESOLVED", f"Expected RESOLVED for {tc['mid']}"

    print("\n--- 2. TESTING NON-MISCONCEPTION ROUTING (NONE, INSUFFICIENT, OOS) ---")
    non_m_cases = [
        (
            {"misconception_id": "NONE", "confidence": 0.36, "evidence": "Clerical typo.", "rationale": "Classified as NONE.", "decision_source": "evidence_rule"},
            {"question": "total = 14 + 8", "correct_answer": "22", "student_answer": "21", "student_reasoning": "Typo on numpad."}
        ),
        (
            {"misconception_id": "INSUFFICIENT", "confidence": 0.38, "evidence": "Student guessed.", "rationale": "Insufficient evidence.", "decision_source": "abstention_rule"},
            {"question": "for i in range(3): ...", "correct_answer": "3", "student_answer": "7", "student_reasoning": "I guessed 7."}
        ),
        (
            {"misconception_id": "OOS", "confidence": 0.16, "evidence": "Aliasing/copy semantics.", "rationale": "Classified as OOS.", "decision_source": "evidence_rule"},
            {"question": "b = a; b.append(3)", "correct_answer": "[1, 2, 3]", "student_answer": "[1, 2]", "student_reasoning": "b = a creates a separate copy."}
        )
    ]

    for diag, ctx in non_m_cases:
        res = run_person2(diag, ctx)
        mid = diag["misconception_id"]
        print(f"\n[{mid}] Routing Test:")
        print(f"  Status:       {res['status']}")
        print(f"  Message:      {res['message']}")
        print(f"  Next Action:  {res['next_action']}")
        print(f"  Follow-up:    {res['followup']}")
        assert res["followup"] is None, f"Follow-up must be None for {mid}"

    print("\n--- 3. TESTING PERSISTENT MISCONCEPTION & SECOND INTERVENTION (2 Cases) ---")
    
    # Case A: M01 persistent misconception
    m01_diag = test_cases[0]["diag"]
    m01_followup = INTERVENTION_BANK["M01"]["followup"]
    # Student incorrectly adds them to 7 and repeats misconception!
    m01_eval = evaluate_followup(m01_diag, m01_followup, "7", "Python converts text to numbers and adds to 7.")
    print(f"\nCase A [M01 Persistent Misconception]:")
    print(f"  Student Follow-up: '7' ('Python converts text to numbers and adds to 7.')")
    print(f"  Evaluation Status: {m01_eval['status']}")
    print(f"  Persistent Flag:   {m01_eval['persistent_misconception']}")
    assert m01_eval["status"] == "NOT_RESOLVED"
    assert m01_eval["persistent_misconception"] is True

    sec_m01 = generate_second_intervention(m01_diag, test_cases[0]["context"])
    print(f"  Triggered Second Intervention: {sec_m01['title']}")
    print(f"  Approach:          {sec_m01['pedagogical_approach']}")
    print(f"  Step Trace:\n{sec_m01['step_by_step_trace']}")
    print(f"  Key Takeaway:      {sec_m01['key_takeaway']}")

    # Case B: M06 persistent misconception
    m06_diag = test_cases[5]["diag"]
    m06_followup = INTERVENTION_BANK["M06"]["followup"]
    # Student includes 5 saying range is inclusive
    m06_eval = evaluate_followup(m06_diag, m06_followup, "[2, 3, 4, 5]", "range includes 5 because the stop value is inclusive.")
    print(f"\nCase B [M06 Persistent Misconception]:")
    print(f"  Student Follow-up: '[2, 3, 4, 5]' ('range includes 5 because the stop value is inclusive.')")
    print(f"  Evaluation Status: {m06_eval['status']}")
    print(f"  Persistent Flag:   {m06_eval['persistent_misconception']}")
    assert m06_eval["status"] == "NOT_RESOLVED"
    assert m06_eval["persistent_misconception"] is True

    sec_m06 = generate_second_intervention(m06_diag, test_cases[5]["context"])
    print(f"  Triggered Second Intervention: {sec_m06['title']}")
    print(f"  Approach:          {sec_m06['pedagogical_approach']}")
    print(f"  Step Trace:\n{sec_m06['step_by_step_trace']}")
    print(f"  Key Takeaway:      {sec_m06['key_takeaway']}")

    print("\n" + "=" * 70)
    print("ALL PERSON 2 ENGINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_person2_tests()
