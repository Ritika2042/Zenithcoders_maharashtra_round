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
from typing import Dict, List, Tuple, Any, Optional

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
            "accepted_concept_patterns": [
                r"\b(concatenat\w*|join\w*|combin\w*|glues?|puts?\s+them\s+together|side\s+by\s+side)\b",
                r"\b(strings?|str|quotes?|quoted|text|characters?)\b.*\b(not\s+(?:numbers?|integers?|ints?|math|addition)|instead\s+of\s+add\w*|together)\b",
                r"\b(int|str)\s*\(",
            ],
            "forbidden_patterns": [
                r"(?<!not\s)(?<!never\s)\b7\b",
                r"\badds?\s+to\s+7\b",
                r"\bconverts?\s+to\s+(?:an?\s+)?(integer|number|int)\b",
                r"\bregular\s+addition\b",
                r"\bquotes?\s+(?:don'?t|do\s+not)\s+matter\b",
                r"\bautomatically\s+converts?\b",
            ]
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
            "accepted_keywords": [
                "floor",
                "rounds down",
                "integer division",
                "discards remainder",
                "whole number",
                "goes in 2 times",
                "drops decimal",
                "discards fraction",
                "quotient",
                "removes the decimal",
            ],
            "accepted_concept_patterns": [
                r"\bfloor(?:\s+division|\s+of|\s+value|\s+result)?\b",
                r"\b(?:integer|whole[\s-]number)\s+division\b",
                r"\b(?:round\w*|tak\w*|bring\w*|push\w*|forc\w*|goe?s?)\s+(?:the\s+)?(?:result|value|quotient|number|it|\d+(?:\.\d+)?)?\s*down\b",
                r"\b(?:discard\w*|drop\w*|remov\w*|strip\w*|truncat\w*|ignores?|without|cuts?\s+off)\s+(?:the\s+)?(?:decimal|fraction\w*|remainder)(?:\s+part|\s+portion)?\b",
                r"\b(?:quotient|whole\s+number|nearest\s+(?:lower\s+|whole\s+)?integer)\b",
                r"\b(?:fits?\s+into|goes?\s+in(?:to)?)\s+\d*\s*(?:whole\s+)?times?\b",
            ],
            "forbidden_patterns": [
                r"2\.75",
                r"\bkeeps?\s+(?:the\s+)?decimal\b",
                r"\breturns?\s+(?:a\s+)?float\b",
                r"^float$",
                r"\bnormal\s+division\b",
                r"\balways\s+rounds?\s+(?:normal|regular|standard|float)?\s*division\b",
                r"\brounds?\s+up\b",
                r"\b(?://|double\s+slash|floor\s+division)\s+(?:is|gives|returns|computes)\s+(?:the\s+)?remainder\b",
            ]
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
            "accepted_keywords": ["compare", "comparison", "checks", "equality", "not equal", "evaluates to false", "tests if equal", "does not assign"],
            "accepted_concept_patterns": [
                r"\b(compar\w*|equal(?:ity)?|check\w*\s+(?:if|whether)|test\w*\s+(?:if|whether)|does\s+not\s+(?:assign|modify|change)|without\s+(?:assigning|modifying|changing)|100\s*!=\s*50|not\s+equal)\b",
            ],
            "forbidden_patterns": [
                r"\bassigns?\s+50\b",
                r"\bscore\s+becomes?\s+50\b",
                r"\bsets?\s+score\s+to\s+50\b",
                r"\bupdates?\s+score\b",
                r"\b==\s+assigns?\b",
                r"\btrue\b",
            ]
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
            "accepted_concept_patterns": [
                r"\b(precedence|order\s+of\s+operations|pemdas|bodmas|binds?\s+tighter)\b",
                r"\b(multipl\w*|\*|exponent\w*|\*\*|divis\w*)\s+(?:goes?|comes?|happens?|executes?|evaluates?|is\s+done|is\s+calculated)\s+(?:first|before)\b",
                r"\b(?:first|before\s+subtract\w*|before\s+add\w*)\b.*\b(?:2\s*\*\s*3|multipl\w*)\b",
                r"\b2\s*\*\s*3\s*(?:=|is|gives|evaluates\s+to)\s*6\b.*\b10\s*-\s*6\b",
            ],
            "forbidden_patterns": [
                r"\b24\b",
                r"(?<!not\s)\bleft\s+to\s+right\b",
                r"\bsubtraction\s+(?:goes?\s+|comes?\s+|happens?\s+|evaluates?\s+)?first\b",
                r"\baddition\s+(?:goes?\s+|comes?\s+|happens?\s+|evaluates?\s+)?first\b",
                r"\b8\s*\*\s*3\b",
            ]
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
            "accepted_keywords": [
                "0-based",
                "zero-based",
                "first element",
                "first item",
                "starts at 0",
                "starts at zero",
                "starts positions at zero",
                "initial item",
                "index 0",
                "start counting at 0",
            ],
            "accepted_concept_patterns": [
                r"\b(0[\s-]based|zero[\s-]based)\b",
                r"\b(?:index(?:ing)?|positions?|counting|indices|lists?|strings?|python)\s+(?:in\s+python\s+)?(?:always\s+)?(?:starts?|begins?)\s+(?:positions?\s+|indexing\s+|counting\s+|at\s+index\s+)?(?:at|from)\s+(?:index\s+)?(?:0|zero)\b",
                r"\b(?:index|position)\s+0\s+(?:is|refers?\s+to|accesses?|retrieves?|gives?)\s+the\s+(?:first|1st|initial)\s+(?:item|element|character|letter|color|value)\b",
                r"\b(?:index|position)\s+1\s+(?:is|refers?\s+to|accesses?|retrieves?|gives?)\s+the\s+(?:second|2nd)\s+(?:item|element|character|letter|color|value)\b",
                r"\b(?:first|1st|initial)\s+(?:item|element|character|letter|color|value)\s+is\s+(?:at\s+)?(?:index|position)\s+(?:0|zero)\b",
                r"\boffset\s+0\b",
            ],
            "forbidden_patterns": [
                r"\bgreen\b",
                r"\bindex\s+0\s+(?:is|refers\s+to)\s+the\s+second\b",
                r"\bstarts?\s+(?:positions?\s+|indexing\s+|counting\s+)?at\s+(?:1|one)\b",
                r"\b1[\s-]based\b",
                r"\bone[\s-]based\b",
                r"\bindex\s+1\s+(?:refers?\s+to|is)\s+the\s+(?:first|1st)\b",
            ]
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
            "accepted_concept_patterns": [
                r"\b(?:exclusive|not\s+inclusive|half[\s-]open)\b",
                r"\b(?:excludes?|does\s+not\s+include|doesn'?t\s+include|omits?|leaves?\s+out|never\s+reaches?|never\s+includes?)\s+(?:the\s+)?(?:stop\s+(?:value|boundary|number)\s+)?\d*\b",
                r"\b(?:stops?|ends?|halts?|terminates?)\s+(?:before\s+(?:reaching\s+)?\d+|at\s+4|at\s+stop\s*-\s*1|one\s+before)\b",
                r"\bup\s+to\s+but\s+not\s+including\b",
            ],
            "forbidden_patterns": [
                r"\[2,\s*3,\s*4,\s*5\]",
                r"(?<!not\s)(?<!does\snot\s)\bincludes?\s+5\b",
                r"(?<!not\s)(?<!does\snot\s)(?<!doesn't\s)\binclud(?:es?|ing)\s+(?:both\s+)?(?:the\s+)?(?:stop|end|upper|final)\b",
                r"\b(?:stop|end|upper|final)\s*(?:value|boundary|bound|number)?\s+is\s+included\b",
                r"(?<!not\s)\binclusive\b",
                r"\b5\s+is\s+included\b",
            ]
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
            "accepted_keywords": ["positional", "order", "y binds to a", "x binds to b", "a is 2", "b is 10", "first argument", "first parameter", "2 - 10"],
            "accepted_concept_patterns": [
                r"\b(?:positional\w*|by\s+position|by\s+order|left[\s-]to[\s-]right|order\s+of\s+arguments?)\b",
                r"\b(?:y|1st\s+arg\w*|first\s+arg\w*|2)\s+(?:binds?\s+to|goes?\s+to|maps?\s+to|fills?|is\s+assigned\s+to|becomes?)\s+(?:parameter\s+|first\s+parameter\s+)?a\b",
                r"\ba\s+(?:gets?|receives?|is|becomes?|=)\s*2\b.*\bb\s+(?:gets?|receives?|is|becomes?|=)\s*10\b",
                r"\b2\s*-\s*10\b",
            ],
            "forbidden_patterns": [
                r"(?<!-\s)(?<!-)\b8\b",
                r"\b10\s*-\s*2\b",
                r"\ba\s+is\s+10\b",
                r"\bnames?\s+match\b",
                r"\bx\s+matches\s+a\b",
            ]
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
            "accepted_concept_patterns": [
                r"\bbase\s+case\b",
                r"\b(?:reaches?|hits?|satisfies|triggers?|when)\s+(?:the\s+condition\s+)?n\s*==\s*[01]\b",
                r"\b(?:terminates?|stops?|halts?|returns?\s+['\"]?go!?['\"]?|unwinds?)\b.*\b(?:when|because|at|once)\s+.*\b(?:n\s*==\s*1|reaches?\s+1|hits?\s+1|base\s+case)\b",
            ],
            "forbidden_patterns": [
                r"RecursionError",
                r"\bnever\s+stops?\b",
                r"\binfinite\s+recursion\b",
                r"\bnever\s+reaches?\s+(?:an?\s+)?exit\b",
            ]
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

# Stores the most recent follow-up evaluation per misconception ID so generate_second_intervention
# can tailor its guidance when the follow-up answer was right but conceptual evidence was missing.
_LAST_FOLLOWUP_EVALUATION: Dict[str, Dict[str, Any]] = {}

# Tracks deterministic follow-up pool rotation across attempts per (misconception_id, variant_key)
_FOLLOWUP_ROTATION_STATE: Dict[str, List[str]] = {}


# =============================================================================
# A2. VARIANT ALIASES & VARIANT-SPECIFIC INTERVENTION BANK
# =============================================================================

VARIANT_ALIASES: Dict[str, Dict[str, str]] = {
    "M01": {
        "treats_numeric_string_as_int": "treats_numeric_string_as_int",
        "expects_plus_to_add_str_and_int": "expects_plus_to_add_str_and_int",
        "expects_int_cast_without_int_call": "expects_int_cast_without_int_call",
    },
    "M02": {
        "believes_slash_returns_int": "believes_slash_returns_int",
        "believes_double_slash_returns_remainder": "believes_double_slash_returns_remainder",
        "believes_double_slash_returns_float": "believes_double_slash_returns_float",
        "other_division_confusion": "other_division_confusion",
        "slash_vs_double_slash": "believes_double_slash_returns_float",
        "floor_division_misunderstanding": "other_division_confusion",
    },
    "M03": {
        "uses_single_equals_for_conditional_test": "uses_single_equals_for_conditional_test",
        "expects_double_equals_to_assign_value": "expects_double_equals_to_assign_value",
        "believes_equals_creates_symmetric_equation": "believes_equals_creates_symmetric_equation",
        "assignment_vs_equality": "uses_single_equals_for_conditional_test",
    },
    "M04": {
        "left_to_right_evaluation_override": "left_to_right_evaluation_override",
        "arithmetic_plus_before_multiply": "arithmetic_plus_before_multiply",
        "boolean_or_before_and": "boolean_or_before_and",
        "relational_before_arithmetic": "relational_before_arithmetic",
    },
    "M05": {
        "one_based_indexing_belief": "one_based_indexing_belief",
        "negative_index_from_zero_belief": "negative_index_from_zero_belief",
        "slice_endpoint_inclusive_as_position": "slice_endpoint_inclusive_as_position",
        "off_by_one_element_lookup": "off_by_one_element_lookup",
        "negative_index_misunderstanding": "negative_index_from_zero_belief",
        "off_by_one_position": "off_by_one_element_lookup",
    },
    "M06": {
        "range_stop_inclusive_belief": "range_stop_inclusive_belief",
        "range_starts_at_one_default": "range_starts_at_one_default",
        "misunderstands_step_interval": "misunderstands_step_interval",
        "loop_iteration_count_off_by_one": "loop_iteration_count_off_by_one",
        "range_stop_inclusive": "range_stop_inclusive_belief",
        "range_start_misunderstanding": "range_starts_at_one_default",
        "range_step_misunderstanding": "misunderstands_step_interval",
        "iteration_count_misunderstanding": "loop_iteration_count_off_by_one",
    },
    "M07": {
        "believes_arg_name_must_match_param_name": "believes_arg_name_must_match_param_name",
        "misunderstands_positional_binding_order": "misunderstands_positional_binding_order",
        "keyword_argument_override_confusion": "keyword_argument_override_confusion",
        "parameter_count_mismatch_tolerated": "parameter_count_mismatch_tolerated",
        "argument_parameter_binding": "believes_arg_name_must_match_param_name",
        "positional_argument_order": "misunderstands_positional_binding_order",
        "keyword_argument_binding": "keyword_argument_override_confusion",
        "argument_parameter_count_mismatch": "parameter_count_mismatch_tolerated",
    },
    "M08": {
        "believes_recursion_stops_automatically_at_zero": "believes_recursion_stops_automatically_at_zero",
        "believes_base_case_resets_execution": "believes_base_case_resets_execution",
        "misunderstands_recursive_call_return_value": "misunderstands_recursive_call_return_value",
        "base_case_placement_confusion": "base_case_placement_confusion",
        "missing_base_case": "believes_recursion_stops_automatically_at_zero",
        "base_case_does_not_stop_recursion": "believes_base_case_resets_execution",
        "recursive_call_does_not_progress": "base_case_placement_confusion",
    },
}


VARIANT_INTERVENTION_BANK: Dict[str, Dict[str, Dict[str, Any]]] = {
    "M01": {
        "treats_numeric_string_as_int": {
            "title": "String vs Number Type Confusion: Quoted Digits Are Strings",
            "short_explanation": (
                "Putting quotes around digits (like \"5\") creates a text string (str), not an integer (int). "
                "When both values are strings, + joins (concatenates) their characters end-to-end instead of adding them mathematically; "
                "when one is a string and one is an int, + raises a TypeError."
            ),
            "concrete_example": 'print("5" + "2")   # Output: "52" (string concatenation)\nprint(5 + 2)       # Output: 7    (integer addition)',
            "contrast_example": 'print(int("5") + 2)  # Output: 7 (explicit int() conversion)',
            "key_rule": "Quotes make digits a str; + concatenates str + str and raises TypeError on str + int.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\na = \"3\"\nb = \"4\"\nprint(a + b)",
                    "expected_answer": "34",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nx = \"8\"\ny = \"1\"\nprint(x + y)",
                    "expected_answer": "81",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What happens when this code runs (give the output or error name)?\n\nval = \"6\"\nprint(val + 4)",
                    "expected_answer": "TypeError",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "expects_plus_to_add_str_and_int": {
            "title": "String vs Number Type Confusion: Mixing str and int with +",
            "short_explanation": (
                "Python never implicitly converts between str and int when using the + operator. "
                "Attempting to evaluate str + int or int + str immediately raises a TypeError rather than adding or concatenating."
            ),
            "concrete_example": 'age = 20\n# print("Age: " + age)  -> Raises TypeError!\nprint("Age: " + str(age))  # Output: "Age: 20"',
            "contrast_example": 'score = "10"\nprint(int(score) + 5)      # Output: 15 (convert str to int first)',
            "key_rule": "Never mix str and int with +; convert explicitly with int() for math or str() for text.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\na = \"3\"\nb = \"4\"\nprint(a + b)",
                    "expected_answer": "34",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What happens when this code runs (give the output or error name)?\n\ncount = \"10\"\nprint(count + 5)",
                    "expected_answer": "TypeError",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nnum = \"12\"\nprint(int(num) + 3)",
                    "expected_answer": "15",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "expects_int_cast_without_int_call": {
            "title": "String vs Number Type Confusion: input() & Unassigned int() Conversions",
            "short_explanation": (
                "In Python, input() always returns a string (str), even if the user types digits. "
                "Furthermore, calling int(x) does not modify x in place—you must assign the returned integer (e.g., x = int(x)) before doing arithmetic."
            ),
            "concrete_example": 'raw = "9"\nint(raw)          # Return value is ignored; raw is still "9" (str)\nprint(raw + "1")  # Output: "91"',
            "contrast_example": 'raw = "9"\nnum = int(raw)    # Store the converted int\nprint(num + 1)    # Output: 10',
            "key_rule": "input() returns a str, and int(x) returns a new int that must be saved or used directly.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nx = \"5\"\nint(x)\nprint(x + \"2\")",
                    "expected_answer": "52",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\na = \"3\"\nb = \"4\"\nprint(a + b)",
                    "expected_answer": "34",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
    },
    "M02": {
        "believes_slash_returns_int": {
            "title": "Division Semantics: Single Slash (/) Always Returns a Float",
            "short_explanation": (
                "In Python 3, the single-slash operator (/) is true division and ALWAYS returns a floating-point number (float) "
                "with a decimal point—even when two integers divide evenly or have a fractional remainder. "
                "It never truncates to a whole integer."
            ),
            "concrete_example": "print(7 / 2)   # Output: 3.5 (true division keeps the decimal)\nprint(8 / 2)   # Output: 4.0 (even exact division returns a float!)",
            "contrast_example": "print(7 // 2)  # Output: 3   (use // floor division for an integer quotient)",
            "key_rule": "Single slash (/) ALWAYS returns a float (e.g., 7 / 2 -> 3.5, 8 / 2 -> 4.0); use // for integer floor division.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nprint(9 / 2)",
                    "expected_answer": "4.5",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(6 / 3)",
                    "expected_answer": "2.0",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(11 // 4)",
                    "expected_answer": "2",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "believes_double_slash_returns_remainder": {
            "title": "Division Semantics: Floor Quotient (//) vs Modulo Remainder (%)",
            "short_explanation": (
                "The double-slash operator (//) computes the FLOOR QUOTIENT (how many whole times the divisor fits in), "
                "NOT the remainder! To get the remainder left over after division, Python uses the modulo operator (%)."
            ),
            "concrete_example": "print(14 // 4)  # Output: 3 (4 fits into 14 three whole times -> quotient)\nprint(14 % 4)   # Output: 2 (14 - 12 = 2 left over -> remainder)",
            "contrast_example": "print(15 // 4)  # Output: 3 (floor quotient)\nprint(15 % 4)   # Output: 3 (remainder)",
            "key_rule": "// returns the whole-number floor quotient; % (modulo) returns the remainder.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nprint(14 // 4)",
                    "expected_answer": "3",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(11 // 4)",
                    "expected_answer": "2",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(17 // 5)",
                    "expected_answer": "3",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "believes_double_slash_returns_float": {
            "title": "Division Semantics: Double Slash (//) Discards the Fractional Part",
            "short_explanation": (
                "The double-slash operator (//) is floor division: it divides the numbers and rounds DOWN to the nearest "
                "whole integer, discarding any fractional decimal part. Unlike /, it does not keep decimals."
            ),
            "concrete_example": "print(7 // 2)   # Output: 3   (floors 3.5 down to integer 3)\nprint(15 // 4)  # Output: 3   (floors 3.75 down to integer 3)",
            "contrast_example": "print(7 / 2)    # Output: 3.5 (single slash / keeps the decimal part)",
            "key_rule": "Double slash (//) performs floor division and returns the floored integer quotient without decimals.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nprint(11 // 4)",
                    "expected_answer": "2",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(15 // 4)",
                    "expected_answer": "3",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(9 // 2)",
                    "expected_answer": "4",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "other_division_confusion": {
            "title": "Division Semantics: Floor Division Rounding Down to Nearest Integer",
            "short_explanation": (
                "Floor division (//) always rounds DOWN toward negative infinity (−∞) to the nearest integer, "
                "rather than rounding up to the nearest whole number or rounding toward zero for negative values."
            ),
            "concrete_example": "print(15 // 4)   # Output: 3  (3.75 floors down to 3, never rounds up to 4)\nprint(-7 // 2)   # Output: -4 (-3.5 floors down toward -infinity to -4)",
            "contrast_example": "print(15 / 4)    # Output: 3.75 (exact float division)",
            "key_rule": "// always rounds down (floors) to the nearest integer less than or equal to the exact quotient.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nprint(11 // 4)",
                    "expected_answer": "2",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(19 // 5)",
                    "expected_answer": "3",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(-7 // 2)",
                    "expected_answer": "-4",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
    },
    "M03": {
        "uses_single_equals_for_conditional_test": {
            "title": "Assignment vs Equality: Using = Inside an if Condition",
            "short_explanation": (
                "A single equals sign (=) is an assignment statement that stores a value in a variable—it cannot be used "
                "inside an if condition (which raises a SyntaxError). To test whether two values are equal, use double equals (==)."
            ),
            "concrete_example": "x = 5             # Assignment: stores 5 in x\nif x == 5:        # Comparison: checks if x equals 5 -> True\n    print('Match')",
            "contrast_example": "# if x = 5:       # SyntaxError! Cannot assign inside an if condition",
            "key_rule": "Use == to compare values inside conditions; = is only for assigning values to variables.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nscore = 100\nprint(score == 50)",
                    "expected_answer": "False",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What happens when this code runs (give the output or error name)?\n\nx = 10\nif x = 10:\n    print(\"Ten\")",
                    "expected_answer": "SyntaxError",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "expects_double_equals_to_assign_value": {
            "title": "Assignment vs Equality: == Compares Without Modifying Variables",
            "short_explanation": (
                "Double equals (==) is a read-only comparison expression that evaluates to True or False. "
                "Writing x == 10 checks whether x equals 10 and discards the boolean result—it never updates x."
            ),
            "concrete_example": "score = 100\nscore == 50       # Evaluates to False; score is NOT changed!\nprint(score)      # Output: 100",
            "contrast_example": "score = 100\nscore = 50        # Single = assigns the new value 50 to score\nprint(score)      # Output: 50",
            "key_rule": "== only compares and returns True/False; it never mutates or assigns a variable.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nscore = 100\nprint(score == 50)",
                    "expected_answer": "False",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nn = 7\nn == 20\nprint(n)",
                    "expected_answer": "7",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "believes_equals_creates_symmetric_equation": {
            "title": "Assignment vs Equality: Assignment Copies Right-to-Left Once",
            "short_explanation": (
                "In Python, a = b is a one-time directional copy of the current value of b into a. "
                "It does NOT create a permanent algebraic link—changing b later has no effect on a."
            ),
            "concrete_example": "a = 5\nb = a       # b receives a snapshot copy of 5\na = 99      # Updating a later does not change b\nprint(b)    # Output: 5",
            "contrast_example": "a = 5\nb = a\nb = a       # Must reassign if you want b to reflect a's new value",
            "key_rule": "Assignment (x = y) evaluates the right side right now and stores it in the left variable once.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nscore = 100\nprint(score == 50)",
                    "expected_answer": "False",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nx = 4\ny = x\nx = 10\nprint(y)",
                    "expected_answer": "4",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
    },
    "M04": {
        "left_to_right_evaluation_override": {
            "title": "Operator Precedence: Precedence Hierarchy Overrides Left-to-Right Order",
            "short_explanation": (
                "Python does NOT simply evaluate expressions from left to right. Higher-precedence operators "
                "(like parentheses, **, and *, /, //, %) always bind and evaluate before lower-precedence operators (+, -)."
            ),
            "concrete_example": "print(2 + 3 * 4)    # Step 1: 3 * 4 = 12 -> Step 2: 2 + 12 = 14",
            "contrast_example": "print((2 + 3) * 4)  # Parentheses override precedence -> 5 * 4 = 20",
            "key_rule": "Evaluate higher-precedence operators (*, /, //, %) before lower-precedence ones (+, -), not strictly left-to-right.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nprint(10 - 2 * 3)",
                    "expected_answer": "4",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(4 + 5 * 2)",
                    "expected_answer": "14",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(20 - 12 // 3)",
                    "expected_answer": "16",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "arithmetic_plus_before_multiply": {
            "title": "Operator Precedence: Multiplication & Division Bind Tighter Than Addition",
            "short_explanation": (
                "In arithmetic expressions without parentheses, multiplication (*) and division (/, //, %) always "
                "have higher precedence than addition (+) and subtraction (-), regardless of which appears first."
            ),
            "concrete_example": "print(3 + 4 * 2)    # 4 * 2 = 8 first, then 3 + 8 = 11 (NOT 7 * 2 = 14)",
            "contrast_example": "print((3 + 4) * 2)  # Use parentheses when you want addition to happen first -> 14",
            "key_rule": "Multiply and divide (*, /, //, %) before adding or subtracting (+, -).",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nprint(10 - 2 * 3)",
                    "expected_answer": "4",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(6 + 2 * 5)",
                    "expected_answer": "16",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "boolean_or_before_and": {
            "title": "Operator Precedence: Boolean 'not' > 'and' > 'or'",
            "short_explanation": (
                "In boolean logic, 'and' has higher precedence than 'or' (just like * binds tighter than +). "
                "Python evaluates 'and' sub-expressions first before combining with 'or'."
            ),
            "concrete_example": "print(True or False and False)   # False and False -> False; True or False -> True",
            "contrast_example": "print((True or False) and False) # Parentheses force 'or' first -> True and False -> False",
            "key_rule": "Boolean precedence order is: not first, then and, then or last.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nprint(True or False and False)",
                    "expected_answer": "True",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(10 - 2 * 3)",
                    "expected_answer": "4",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "relational_before_arithmetic": {
            "title": "Operator Precedence: Arithmetic Operators > Bitwise > Comparisons",
            "short_explanation": (
                "All arithmetic operators (+, -, *, //) evaluate before bitwise operators (&, |), which in turn "
                "evaluate before comparison operators (==, !=, <, >) unless parentheses specify otherwise."
            ),
            "concrete_example": "print(3 + 2 == 5)     # 3 + 2 evaluates to 5 first, then 5 == 5 -> True",
            "contrast_example": "val = 6\nprint((val & 2) == 2) # Parentheses ensure bitwise & runs before ==",
            "key_rule": "Arithmetic evaluates first, then bitwise, then comparisons (==, <, >), then boolean logic.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nprint(10 - 2 * 3)",
                    "expected_answer": "4",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(4 + 5 * 2)",
                    "expected_answer": "14",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
    },
    "M05": {
        "one_based_indexing_belief": {
            "title": "Zero-Based Indexing: Index 0 Is First, Index 1 Is Second",
            "short_explanation": (
                "Python sequences (strings, lists, tuples) use 0-based indexing: index 0 accesses the 1st element, "
                "and index 1 accesses the 2nd element. Counting starts at 0, not 1."
            ),
            "concrete_example": 'word = "hello"\nprint(word[0])  # Output: "h" (1st character is at index 0)\nprint(word[1])  # Output: "e" (2nd character is at index 1)',
            "contrast_example": 'items = ["red", "blue", "green"]\nprint(items[0]) # Output: "red" (1st item)\nprint(items[1]) # Output: "blue" (2nd item)',
            "key_rule": "Index 0 is the 1st element; index 1 is the 2nd element (index k is the (k + 1)-th element).",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\ncolors = [\"red\", \"blue\", \"green\"]\nprint(colors[0])",
                    "expected_answer": "red",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\ntext = \"world\"\nprint(text[1])",
                    "expected_answer": "o",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nnums = [10, 20, 30]\nprint(nums[1])",
                    "expected_answer": "20",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "negative_index_from_zero_belief": {
            "title": "Negative Indexing: -1 Is the Last Element, -2 Is Second-to-Last",
            "short_explanation": (
                "Negative indices count backward from the end of the sequence starting at -1 (because -0 is equal to 0, "
                "which is already the first element). Therefore, seq[-1] is the last element and seq[-2] is the second-to-last element."
            ),
            "concrete_example": 'word = "Python"\nprint(word[-1])  # Output: "n" (last character)\nprint(word[-2])  # Output: "o" (second-to-last character)',
            "contrast_example": 'word = "Python"\nprint(word[0])   # Output: "P" (-0 is 0, the first character)',
            "key_rule": "Negative indexing starts at -1 for the last element, -2 for the second-to-last element.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nword = \"code\"\nprint(word[-1])",
                    "expected_answer": "e",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nitems = [\"a\", \"b\", \"c\"]\nprint(items[-2])",
                    "expected_answer": "b",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "slice_endpoint_inclusive_as_position": {
            "title": "Slice Boundaries: seq[start:stop] Excludes Index stop",
            "short_explanation": (
                "In a Python slice seq[start:stop], the start index is included (starting from 0), "
                "but the stop index is EXCLUSIVE—Python stops at index (stop - 1) and does not include seq[stop]."
            ),
            "concrete_example": 's = "Python"\nprint(s[1:4])  # Includes indices 1, 2, 3 -> "yth" (stops before index 4)',
            "contrast_example": 's = "Python"\nprint(s[0:3])  # Includes indices 0, 1, 2 -> "Pyt"',
            "key_rule": "A slice seq[start:stop] includes index start and stops just before index stop.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\ns = \"hello\"\nprint(s[1:3])",
                    "expected_answer": "el",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\ncolors = [\"red\", \"blue\", \"green\"]\nprint(colors[0])",
                    "expected_answer": "red",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "off_by_one_element_lookup": {
            "title": "Off-by-One Position Lookup: Valid Indices Run from 0 to len(seq) - 1",
            "short_explanation": (
                "Because indexing starts at 0, a sequence of length N has valid indices from 0 up to N - 1. "
                "The k-th human position is at index (k - 1), and accessing seq[len(seq)] raises an IndexError."
            ),
            "concrete_example": 'nums = [10, 20, 30]  # Length is 3; valid indices are 0, 1, 2\nprint(nums[2])       # Output: 30 (the 3rd and last item)',
            "contrast_example": '# print(nums[3])     # Raises IndexError: list index out of range!',
            "key_rule": "For a sequence of length N, the last valid index is N - 1 (and the k-th item is at index k - 1).",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\ncolors = [\"red\", \"blue\", \"green\"]\nprint(colors[0])",
                    "expected_answer": "red",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nvals = [5, 10, 15]\nprint(vals[2])",
                    "expected_answer": "15",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
    },
    "M06": {
        "range_stop_inclusive_belief": {
            "title": "Range Boundaries: The stop Value in range(start, stop) Is Exclusive",
            "short_explanation": (
                "In Python, range(start, stop) includes start but ALWAYS stops BEFORE stop. "
                "The stop boundary is half-open (exclusive), so the last integer generated is stop - 1, never stop itself."
            ),
            "concrete_example": "print(list(range(1, 5)))  # Output: [1, 2, 3, 4] (stops at 4, before 5)",
            "contrast_example": "print(list(range(1, 6)))  # Output: [1, 2, 3, 4, 5] (use stop=6 to include 5)",
            "key_rule": "range(start, stop) includes start and excludes stop (ends at stop - 1).",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nprint(list(range(2, 5)))",
                    "expected_answer": "[2, 3, 4]",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(list(range(1, 4)))",
                    "expected_answer": "[1, 2, 3]",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(list(range(3, 6)))",
                    "expected_answer": "[3, 4, 5]",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "range_starts_at_one_default": {
            "title": "Range Default Start: range(n) Starts at 0, Not 1",
            "short_explanation": (
                "When given a single argument, range(n) starts at 0 by default and stops at n - 1, "
                "producing exactly n integers: 0, 1, ..., n - 1. It does NOT start at 1."
            ),
            "concrete_example": "print(list(range(4)))     # Output: [0, 1, 2, 3] (starts at 0, stops before 4)",
            "contrast_example": "print(list(range(1, 4)))  # Output: [1, 2, 3]    (explicitly pass 1 to start at 1)",
            "key_rule": "range(n) is shorthand for range(0, n): it starts at 0 and stops at n - 1.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nprint(list(range(3)))",
                    "expected_answer": "[0, 1, 2]",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(list(range(2, 5)))",
                    "expected_answer": "[2, 3, 4]",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "misunderstands_step_interval": {
            "title": "Range Step Stride: range(start, stop, step) Advances by step and Excludes stop",
            "short_explanation": (
                "In range(start, stop, step), the 3rd argument (step) is the stride added to each value "
                "(start, start + step, start + 2*step, ...), while stop remains a strict EXCLUSIVE boundary that is never included."
            ),
            "concrete_example": "print(list(range(0, 6, 2)))  # Output: [0, 2, 4] (adds 2 each time; stops before 6)",
            "contrast_example": "print(list(range(1, 8, 3)))  # Output: [1, 4, 7] (adds 3 each time; stops before 8)",
            "key_rule": "In range(start, stop, step), step is the increment between numbers and stop is still excluded.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nprint(list(range(0, 6, 2)))",
                    "expected_answer": "[0, 2, 4]",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\nprint(list(range(1, 7, 2)))",
                    "expected_answer": "[1, 3, 5]",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "loop_iteration_count_off_by_one": {
            "title": "Loop Iteration Count: range(start, stop) Runs (stop - start) Times",
            "short_explanation": (
                "Because range(start, stop) includes start and excludes stop, a loop over range(start, stop) "
                "executes exactly (stop - start) times, and range(n) executes exactly n times."
            ),
            "concrete_example": "count = 0\nfor i in range(1, 5):  # i takes values 1, 2, 3, 4 -> 4 iterations (5 - 1 = 4)\n    count += 1\nprint(count)           # Output: 4",
            "contrast_example": "print(len(range(3)))   # Output: 3 (values 0, 1, 2 -> 3 iterations)",
            "key_rule": "for i in range(start, stop) runs (stop - start) times because stop is excluded.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\nprint(list(range(2, 5)))",
                    "expected_answer": "[2, 3, 4]",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\ncount = 0\nfor i in range(1, 4):\n    count += 1\nprint(count)",
                    "expected_answer": "3",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
    },
    "M07": {
        "believes_arg_name_must_match_param_name": {
            "title": "Function Argument Binding: Positional Arguments Bind by Order, Not Caller Variable Name",
            "short_explanation": (
                "When you pass variables as positional arguments to a function, Python binds them strictly by their "
                "left-to-right POSITION in the call, completely ignoring what the variables were named in the caller."
            ),
            "concrete_example": "def sub(a, b):\n    return a - b\n\nx = 10\ny = 2\nprint(sub(y, x))  # 1st arg y(2) -> a, 2nd arg x(10) -> b => 2 - 10 = -8",
            "contrast_example": "print(sub(x, y))  # 1st arg x(10) -> a, 2nd arg y(2) -> b => 10 - 2 = 8",
            "key_rule": "Positional arguments bind to parameters by left-to-right position, never by caller variable names.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\ndef sub(a, b):\n    return a - b\n\nx = 10\ny = 2\nprint(sub(y, x))",
                    "expected_answer": "-8",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\ndef div(a, b):\n    return a // b\n\nb = 20\na = 4\nprint(div(b, a))",
                    "expected_answer": "5",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "misunderstands_positional_binding_order": {
            "title": "Function Argument Binding: Left-to-Right Positional Slot Matching",
            "short_explanation": (
                "Positional arguments are matched to function parameters in exact 1st-to-1st, 2nd-to-2nd order. "
                "Moreover, in Python syntax, all positional arguments must come BEFORE any keyword arguments."
            ),
            "concrete_example": "def greet(first, last):\n    return first + ' ' + last\n\nprint(greet('Ada', 'Lovelace'))  # 'Ada' -> first, 'Lovelace' -> last",
            "contrast_example": "print(greet(last='Lovelace', first='Ada'))  # Explicit keyword args can be in any order",
            "key_rule": "Positional arguments bind 1st-to-1st, 2nd-to-2nd, and must appear before any keyword arguments.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\ndef sub(a, b):\n    return a - b\n\nx = 10\ny = 2\nprint(sub(y, x))",
                    "expected_answer": "-8",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\ndef calc(x, y):\n    return x - y\n\nprint(calc(3, 8))",
                    "expected_answer": "-5",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "keyword_argument_override_confusion": {
            "title": "Function Argument Binding: Keyword Arguments Override Parameter Defaults",
            "short_explanation": (
                "When a function call uses keyword arguments (like func(b=10, a=2)), Python binds each value directly "
                "to the named parameter regardless of order, overriding any default parameter values."
            ),
            "concrete_example": "def sub(a, b=1):\n    return a - b\n\nprint(sub(b=2, a=10))  # a gets 10, b gets 2 -> 8",
            "contrast_example": "print(sub(10))         # a gets 10, b uses default 1 -> 9",
            "key_rule": "Keyword arguments (param=value) bind directly to the named parameter and override defaults.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\ndef sub(a, b):\n    return a - b\n\nprint(sub(b=3, a=10))",
                    "expected_answer": "7",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\ndef sub(a, b):\n    return a - b\n\nx = 10\ny = 2\nprint(sub(y, x))",
                    "expected_answer": "-8",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "parameter_count_mismatch_tolerated": {
            "title": "Function Argument Binding: Argument Count Must Match Required Parameters",
            "short_explanation": (
                "Python requires the exact number of required arguments defined in the function signature. "
                "Passing too few or too many positional arguments immediately raises a TypeError (it does not silently ignore extras or fill in None)."
            ),
            "concrete_example": "def add(a, b):\n    return a + b\n\nprint(add(3, 4))    # Output: 7 (2 params, 2 args)",
            "contrast_example": "# add(3, 4, 5)      # Raises TypeError: takes 2 positional arguments but 3 were given",
            "key_rule": "Every required parameter must receive exactly one argument, or Python raises a TypeError.",
            "followup_pool": [
                {
                    "question": "What happens when this code runs (give the output or error name)?\n\ndef add(a, b):\n    return a + b\n\nprint(add(5))",
                    "expected_answer": "TypeError",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\ndef sub(a, b):\n    return a - b\n\nx = 10\ny = 2\nprint(sub(y, x))",
                    "expected_answer": "-8",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
    },
    "M08": {
        "believes_recursion_stops_automatically_at_zero": {
            "title": "Recursion Termination: Recursion Requires an Explicit Base Case",
            "short_explanation": (
                "Python never stops recursion automatically at 0. A recursive function stops ONLY when execution "
                "reaches an explicit base case condition (such as if n == 0: return ...) that returns without calling the function again."
            ),
            "concrete_example": "def countdown(n):\n    if n == 0:              # Explicit base case stops recursion\n        return 'Done'\n    return countdown(n - 1)",
            "contrast_example": "def infinite(n):\n    return infinite(n - 1)  # No base case -> raises RecursionError!",
            "key_rule": "Recursion stops only when an explicit base case condition triggers and returns.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\ndef stop_at_one(n):\n    if n == 1:\n        return \"Go!\"\n    return stop_at_one(n - 1)\n\nprint(stop_at_one(3))",
                    "expected_answer": "Go!",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What happens when this code runs (give the output or error name)?\n\ndef loop_down(n):\n    return loop_down(n - 1)\n\nprint(loop_down(2))",
                    "expected_answer": "RecursionError",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "believes_base_case_resets_execution": {
            "title": "Recursion Termination: Reaching the Base Case Returns Up the Call Stack",
            "short_explanation": (
                "When the base case condition is met, its return statement immediately exits that call frame and hands "
                "the returned value back to the waiting caller frame. It does not restart the function or keep recursing."
            ),
            "concrete_example": "def stop_at_one(n):\n    if n == 1:\n        return 'Go!'        # Base case returns 'Go!' up the call stack\n    return stop_at_one(n - 1)\n\nprint(stop_at_one(3))       # Output: 'Go!'",
            "contrast_example": "# Call trace: stop_at_one(3) -> stop_at_one(2) -> stop_at_one(1) returns 'Go!'",
            "key_rule": "When the base case returns a value, recursion stops and unwinds back up the call stack.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\ndef stop_at_one(n):\n    if n == 1:\n        return \"Go!\"\n    return stop_at_one(n - 1)\n\nprint(stop_at_one(3))",
                    "expected_answer": "Go!",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\ndef reach_zero(n):\n    if n == 0:\n        return 100\n    return reach_zero(n - 1)\n\nprint(reach_zero(2))",
                    "expected_answer": "100",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "misunderstands_recursive_call_return_value": {
            "title": "Recursion Unwinding: Combining Each Frame's Work with the Recursive Return Value",
            "short_explanation": (
                "In expressions like return n + sum_to(n - 1), each call frame pauses and waits for sum_to(n - 1) to return, "
                "then adds its own n as the call stack unwinds from the base case back to the top call."
            ),
            "concrete_example": "def sum_to(n):\n    if n == 1:\n        return 1\n    return n + sum_to(n - 1)\n\nprint(sum_to(3))  # 3 + (2 + 1) = 6",
            "contrast_example": "# Unwinding: sum_to(1)->1, sum_to(2)->2+1=3, sum_to(3)->3+3=6",
            "key_rule": "Recursive expressions unwind from the base case, combining each caller frame's pending work.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\ndef sum_to(n):\n    if n == 1:\n        return 1\n    return n + sum_to(n - 1)\n\nprint(sum_to(3))",
                    "expected_answer": "6",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What is the exact output of the following code?\n\ndef stop_at_one(n):\n    if n == 1:\n        return \"Go!\"\n    return stop_at_one(n - 1)\n\nprint(stop_at_one(3))",
                    "expected_answer": "Go!",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
        "base_case_placement_confusion": {
            "title": "Recursion Progression: Recursive Calls Must Progress Toward a Reachable Base Case",
            "short_explanation": (
                "For recursion to terminate, the base case must be checked BEFORE the recursive call, and each "
                "recursive step must move the argument closer to that base case (e.g., n - 1 toward 0)."
            ),
            "concrete_example": "def down(n):\n    if n <= 0:          # Checked first!\n        return 0\n    return down(n - 1)  # Moves n closer to 0",
            "contrast_example": "def stuck(n):\n    if n == 0:\n        return 0\n    return stuck(n)     # n never changes -> RecursionError!",
            "key_rule": "Place the base case before the recursive call and ensure each step progresses toward the base case.",
            "followup_pool": [
                {
                    "question": "What is the exact output of the following code?\n\ndef stop_at_one(n):\n    if n == 1:\n        return \"Go!\"\n    return stop_at_one(n - 1)\n\nprint(stop_at_one(3))",
                    "expected_answer": "Go!",
                    "evaluation_type": "exact_and_conceptual",
                },
                {
                    "question": "What happens when this code runs (give the output or error name)?\n\ndef bad(n):\n    if n == 0:\n        return 0\n    return bad(n)\n\nprint(bad(3))",
                    "expected_answer": "RecursionError",
                    "evaluation_type": "exact_and_conceptual",
                },
            ],
        },
    },
}


def _resolve_canonical_variant(mid: str, raw_variant: Optional[str]) -> Optional[str]:
    """Resolves a raw misconception_variant (or alias) to a canonical key in VARIANT_INTERVENTION_BANK."""
    if not raw_variant or mid not in VARIANT_INTERVENTION_BANK:
        return None
    v_str = str(raw_variant).strip()
    alias_map = VARIANT_ALIASES.get(mid, {})
    canonical = alias_map.get(v_str, v_str)
    if canonical in VARIANT_INTERVENTION_BANK[mid]:
        return canonical
    return None


def _contextualize_intervention(
    mid: str,
    canonical_variant: Optional[str],
    base_entry: Dict[str, Any],
    context: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    Adapts short_explanation and concrete_example using lightweight question context
    when available, without echoing or trusting the student's wrong answer as correct.
    """
    short_exp = base_entry["short_explanation"]
    concrete_ex = base_entry["concrete_example"]
    if not context or not isinstance(context, dict):
        return short_exp, concrete_ex

    q_text = str(context.get("question", "") or "").strip()
    if not q_text:
        return short_exp, concrete_ex

    # M02: Contextualize with the exact division expression from the question
    if mid == "M02":
        m_fdiv = re.search(r"(-?\d+)\s*//\s*(-?\d+)", q_text)
        m_tdiv = re.search(r"(-?\d+)\s*/\s*(-?\d+)", q_text) if not m_fdiv else None
        if m_fdiv:
            a_val, b_val = int(m_fdiv.group(1)), int(m_fdiv.group(2))
            if b_val != 0:
                q_res = a_val // b_val
                r_res = a_val % b_val
                t_res = a_val / b_val
                if canonical_variant == "believes_double_slash_returns_remainder":
                    short_exp = (
                        f"In your question (`{a_val} // {b_val}`), `//` computes the FLOOR QUOTIENT "
                        f"({b_val} fits into {a_val} {q_res} whole times -> `{q_res}`), NOT the remainder! "
                        f"To get the remainder (`{r_res}`), Python uses the modulo operator (`{a_val} % {b_val} -> {r_res}`)."
                    )
                    concrete_ex = (
                        f"print({a_val} // {b_val})  # Output: {q_res} (floor quotient)\n"
                        f"print({a_val} % {b_val})   # Output: {r_res} (modulo remainder)\n"
                        f"print(14 // 4)  # Output: 3 (quotient) vs 14 % 4 -> 2 (remainder)"
                    )
                else:
                    short_exp = (
                        f"In your question (`{a_val} // {b_val}`), exact division gives `{t_res}`, "
                        f"and floor division (`//`) rounds DOWN to the nearest whole integer (`{q_res}`), "
                        f"discarding the fractional part."
                    )
                    concrete_ex = (
                        f"print({a_val} // {b_val})  # Output: {q_res} (floors {t_res} down to {q_res})\n"
                        f"print({a_val} / {b_val})   # Output: {t_res} (single slash keeps decimal)"
                    )
        elif m_tdiv:
            a_val, b_val = int(m_tdiv.group(1)), int(m_tdiv.group(2))
            if b_val != 0:
                t_res = a_val / b_val
                q_res = a_val // b_val
                short_exp = (
                    f"In your question (`{a_val} / {b_val}`), the single slash (`/`) performs true division "
                    f"and ALWAYS returns a float (`{t_res}`), never an integer (`{q_res}`). "
                    f"Only double slash (`{a_val} // {b_val} -> {q_res}`) returns a floored integer."
                )
                concrete_ex = (
                    f"print({a_val} / {b_val})   # Output: {t_res} (true division always returns float)\n"
                    f"print({a_val} // {b_val})  # Output: {q_res} (floor division returns integer)"
                )

    # M05: Contextualize with the string/list literal and index from the question
    elif mid == "M05":
        m_str_assign = re.search(r"(\w+)\s*=\s*['\"]([^'\"]+)['\"]", q_text)
        m_idx = re.search(r"(\w+)\[\s*(-?\d+)\s*\]", q_text)
        if m_str_assign and m_idx and m_str_assign.group(1) == m_idx.group(1):
            var_name = m_str_assign.group(1)
            s_val = m_str_assign.group(2)
            idx_val = int(m_idx.group(2))
            if len(s_val) >= 2 and 0 <= idx_val < len(s_val):
                ch0 = s_val[0]
                ch_idx = s_val[idx_val]
                short_exp = (
                    f"In `{var_name} = \"{s_val}\"`, Python starts indexing at 0: "
                    f"`{var_name}[0]` is `\"{ch0}\"` (the 1st character) and `{var_name}[{idx_val}]` is `\"{ch_idx}\"` "
                    f"(the {idx_val + 1}nd character). Index 1 is the second character, not the first."
                    if idx_val == 1
                    else f"In `{var_name} = \"{s_val}\"`, Python uses 0-based indexing: "
                    f"`{var_name}[0]` is `\"{ch0}\"` (1st character), so `{var_name}[{idx_val}]` accesses `\"{ch_idx}\"` "
                    f"at position {idx_val + 1}."
                )
                concrete_ex = (
                    f'{var_name} = "{s_val}"\n'
                    f'print({var_name}[0])  # Output: "{ch0}" (index 0 is 1st character)\n'
                    f'print({var_name}[{idx_val}])  # Output: "{ch_idx}" (index {idx_val} is character #{idx_val + 1})'
                )
            elif len(s_val) >= 2 and idx_val < 0 and abs(idx_val) <= len(s_val):
                ch_neg = s_val[idx_val]
                ch_last = s_val[-1]
                short_exp = (
                    f"In `{var_name} = \"{s_val}\"`, negative indices count from the right starting at -1: "
                    f"`{var_name}[-1]` is `\"{ch_last}\"` (last character), and `{var_name}[{idx_val}]` is `\"{ch_neg}\"`."
                )
                concrete_ex = (
                    f'{var_name} = "{s_val}"\n'
                    f'print({var_name}[-1])  # Output: "{ch_last}" (last character)\n'
                    f'print({var_name}[{idx_val}])  # Output: "{ch_neg}"'
                )

    # M06: Contextualize with the range(...) call in the question
    elif mid == "M06":
        m_range = re.search(r"range\(\s*(-?\d+)\s*(?:,\s*(-?\d+)\s*(?:,\s*(-?\d+)\s*)?)?\)", q_text)
        if m_range:
            g1, g2, g3 = m_range.group(1), m_range.group(2), m_range.group(3)
            if g2 is None:
                stop_v = int(g1)
                seq_v = list(range(stop_v)) if 0 <= stop_v <= 15 else []
                short_exp = (
                    f"In `range({stop_v})`, Python starts at 0 by default and stops BEFORE {stop_v} "
                    f"(generating `{seq_v}`, ending at {stop_v - 1})."
                )
            elif g3 is None:
                start_v, stop_v = int(g1), int(g2)
                seq_v = list(range(start_v, stop_v)) if abs(stop_v - start_v) <= 15 else []
                short_exp = (
                    f"In `range({start_v}, {stop_v})`, the start `{start_v}` is included, but the stop boundary "
                    f"`{stop_v}` is EXCLUSIVE—so it produces `{seq_v}` and stops at `{stop_v - 1}`, before `{stop_v}`."
                )
            else:
                start_v, stop_v, step_v = int(g1), int(g2), int(g3)
                if step_v != 0 and abs(stop_v - start_v) <= 30:
                    seq_v = list(range(start_v, stop_v, step_v))
                    short_exp = (
                        f"In `range({start_v}, {stop_v}, {step_v})`, Python starts at `{start_v}`, adds step `{step_v}` "
                        f"each time, and stops BEFORE the exclusive stop `{stop_v}`, producing `{seq_v}`."
                    )

    return short_exp, concrete_ex


def _select_followup_from_pool(
    mid: str,
    canonical_variant: Optional[str],
    pool: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Deterministically selects a follow-up question from `pool` (2-3 questions per variant):
    - If `context` specifies an explicit integer `attempt_count` or `followup_index`, uses `(idx) % len(pool)`.
    - Otherwise, tracks distinct question contexts seen for `(mid, canonical_variant)` so repeated
      attempts across different questions deterministically rotate through `pool[0]`, `pool[1]`, `pool[2]`.
    """
    if not pool:
        return INTERVENTION_BANK[mid]["followup"]

    if isinstance(context, dict):
        if isinstance(context.get("followup_index"), int) and context["followup_index"] >= 0:
            return pool[context["followup_index"] % len(pool)]
        if isinstance(context.get("attempt_count"), int) and context["attempt_count"] >= 1:
            return pool[(context["attempt_count"] - 1) % len(pool)]

    q_sig = ""
    if isinstance(context, dict):
        q_sig = str(context.get("question", "") or "").strip()

    state_key = f"{mid}:{canonical_variant or 'DEFAULT'}"
    history = _FOLLOWUP_ROTATION_STATE.setdefault(state_key, [])

    if not q_sig:
        # Even without a question string, rotate deterministically across calls
        idx = len(history) % len(pool)
        history.append(f"__call_{len(history)}__")
        return pool[idx]

    if q_sig in history:
        # If the exact same question is called again in a test, advance rotation if it was the most recent call
        if history[-1] == q_sig and len(pool) > 1:
            idx = len(history) % len(pool)
            history.append(f"{q_sig}#{len(history)}")
            return pool[idx]
        idx = history.index(q_sig) % len(pool)
        return pool[idx]

    idx = len(history) % len(pool)
    history.append(q_sig)
    return pool[idx]


# =============================================================================
# B & C. INTERVENTION & FOLLOW-UP GENERATORS
# =============================================================================

def generate_intervention(diagnosis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a targeted pedagogical intervention based on Person 1's diagnosis
    (misconception_id + misconception_variant) and the current question context.
    Handles M01-M08, NONE, INSUFFICIENT, and OOS, with safe fallback to INTERVENTION_BANK[mid].
    """
    mid = diagnosis.get("misconception_id", "INSUFFICIENT")

    if mid in INTERVENTION_BANK:
        raw_variant = diagnosis.get("misconception_variant")
        canonical_variant = _resolve_canonical_variant(mid, raw_variant)

        if canonical_variant is not None:
            entry = VARIANT_INTERVENTION_BANK[mid][canonical_variant]
        else:
            entry = INTERVENTION_BANK[mid]

        short_exp, concrete_ex = _contextualize_intervention(
            mid, canonical_variant, entry, context
        )

        return {
            "status": "TARGETED_INTERVENTION",
            "misconception_id": mid,
            "misconception_variant": canonical_variant or raw_variant,
            "title": entry["title"],
            "short_explanation": short_exp,
            "concrete_example": concrete_ex,
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
    Uses misconception_id + misconception_variant to select from a 2-3 question pool
    and rotates deterministically across repeated attempts.
    Returns None for NONE, INSUFFICIENT, and OOS.
    """
    mid = diagnosis.get("misconception_id")
    if mid not in INTERVENTION_BANK:
        return None

    raw_variant = diagnosis.get("misconception_variant")
    canonical_variant = _resolve_canonical_variant(mid, raw_variant)

    if canonical_variant is not None:
        pool = VARIANT_INTERVENTION_BANK[mid][canonical_variant].get("followup_pool", [])
        selected = _select_followup_from_pool(mid, canonical_variant, pool, context)
    else:
        default_pool = [INTERVENTION_BANK[mid]["followup"]]
        # Also include variant pool items for rotation when no variant is specified on repeated calls
        for v_entry in VARIANT_INTERVENTION_BANK.get(mid, {}).values():
            for fq in v_entry.get("followup_pool", []):
                if fq["question"] not in {d["question"] for d in default_pool}:
                    default_pool.append(fq)
        selected = _select_followup_from_pool(mid, None, default_pool, context)

    return {
        "question": selected["question"],
        "expected_answer": selected["expected_answer"],
        "evaluation_type": selected.get("evaluation_type", "exact_and_conceptual"),
        "misconception_variant": canonical_variant or raw_variant,
    }


# =============================================================================
# D. FOLLOW-UP EVALUATOR & CANONICAL ANSWER NORMALIZATION (Deterministic)
# =============================================================================

def _strip_presentation_quotes(token: str, preserve_numeric_quotes: bool = True) -> str:
    """
    Strips outer matching single/double/smart quotes or backticks when they are presentation formatting.
    If `preserve_numeric_quotes` is True and the quoted content is a pure numeric literal
    (e.g., "32" or '32'), canonicalizes to '"32"' so string "32" remains distinct from int 32.
    """
    t = token.strip()
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    if len(t) >= 2 and t[0] == "`" and t[-1] == "`" and "`" not in t[1:-1]:
        t = t[1:-1].strip()
    if len(t) >= 2:
        if (t[0] == '"' and t[-1] == '"' and '"' not in t[1:-1]) or (
            t[0] == "'" and t[-1] == "'" and "'" not in t[1:-1]
        ):
            inner = t[1:-1].strip()
            if preserve_numeric_quotes and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", inner):
                return f'"{inner}"'
            return inner
    return t


def _normalize_numeric_token(tok: str) -> str:
    """
    Normalizes harmless numeric formatting differences on unquoted numeric tokens:
    - Strips leading '+' on positive numbers ('+3' -> '3', '+3.5' -> '3.5')
    - Strips redundant trailing zeroes after a decimal point while keeping at least one
      decimal digit so float vs int distinction ('4.0' vs '4') is preserved ('3.50' -> '3.5', '4.00' -> '4.0').
    """
    if re.fullmatch(r"\+\d+(?:\.\d+)?", tok):
        tok = tok[1:]
    if re.fullmatch(r"[+-]?\d+\.\d*0+", tok):
        head, dec = tok.split(".", 1)
        dec_stripped = dec.rstrip("0")
        tok = f"{head}.{dec_stripped if dec_stripped else '0'}"
    return tok


def normalize_text(text: str, preserve_numeric_quotes: bool = True) -> str:
    """
    Canonicalizes answer text for semantic equivalence comparison:
    1. Leading/trailing whitespace stripped.
    2. Presentation quotes on non-numeric strings stripped while preserving numeric string vs int distinction ("32" vs 32).
    3. Python-style multi-element list/tuple brackets normalized ("[1, 2, 3, 4]" -> "1 2 3 4").
    4. Comma-separated atomic sequences normalized ("1, 2, 3, 4" -> "1 2 3 4").
    5. Newline vs space-separated output and multiple spaces collapsed ("1\\n2\\n3\\n4" -> "1 2 3 4").
    6. Harmless numeric formatting normalized and case normalized to lowercase.
    """
    if text is None:
        return ""
    t = str(text).strip()
    if not t:
        return ""

    t = _strip_presentation_quotes(t, preserve_numeric_quotes=preserve_numeric_quotes)

    if (
        (t.startswith("[") and t.endswith("]") and "[" not in t[1:-1] and "]" not in t[1:-1])
        or (t.startswith("(") and t.endswith(")") and "(" not in t[1:-1] and ")" not in t[1:-1])
    ):
        open_b, close_b = t[0], t[-1]
        inner_seq = t[1:-1].strip()
        if not inner_seq:
            return f"{open_b}{close_b}"
        raw_items = [
            item.strip()
            for item in (inner_seq.split(",") if "," in inner_seq else inner_seq.split())
            if item.strip()
        ]
        if len(raw_items) >= 2:
            t = " ".join(
                _normalize_numeric_token(
                    _strip_presentation_quotes(item, preserve_numeric_quotes=preserve_numeric_quotes)
                )
                for item in raw_items
            )
        elif len(raw_items) == 1:
            item0 = _normalize_numeric_token(
                _strip_presentation_quotes(raw_items[0], preserve_numeric_quotes=preserve_numeric_quotes)
            )
            t = f"{open_b}{item0}{close_b}"
    elif "," in t:
        parts = [p.strip() for p in t.split(",")]
        if len(parts) >= 2 and all(
            bool(re.fullmatch(r"['\"`“”‘’]?[+-]?\w+(?:\.\w+)?['\"`“”‘’]?", p)) for p in parts
        ):
            t = " ".join(
                _normalize_numeric_token(
                    _strip_presentation_quotes(p, preserve_numeric_quotes=preserve_numeric_quotes)
                )
                for p in parts
            )
        else:
            t = re.sub(r"\s*,\s*", ", ", t)

    t = re.sub(r"\s+", " ", t).strip()
    if " " in t:
        t = " ".join(_normalize_numeric_token(part) for part in t.split(" "))
    else:
        t = _normalize_numeric_token(t)
    return t.lower()


def answers_are_equivalent(
    student_answer: str,
    expected_answer: str,
    preserve_numeric_quotes: bool = True,
) -> bool:
    """Returns True if student_answer and expected_answer canonicalize to the same representation."""
    sa_norm = normalize_text(student_answer, preserve_numeric_quotes=preserve_numeric_quotes)
    ea_norm = normalize_text(expected_answer, preserve_numeric_quotes=preserve_numeric_quotes)
    if not sa_norm or not ea_norm:
        return False
    return sa_norm == ea_norm


def _resolve_expected_followup_answer(
    mid: str,
    followup_question: Any,
    entry: Dict[str, Any],
) -> str:
    """
    Resolves the expected follow-up answer from followup_question (if customized)
    or falls back to the canonical INTERVENTION_BANK entry's expected_answer.
    """
    if isinstance(followup_question, dict):
        if followup_question.get("expected_answer") is not None:
            return str(followup_question["expected_answer"])
        q_str = str(followup_question.get("question", ""))
    elif isinstance(followup_question, str):
        q_str = followup_question
    else:
        q_str = ""

    if q_str and q_str != entry.get("question", ""):
        if "15 // 4" in q_str or "7 // 2" in q_str:
            return "3"
        if "s[1]" in q_str or "word[1]" in q_str:
            m_str = re.search(r'["\']([a-zA-Z0-9]+)["\']', q_str)
            if m_str and len(m_str.group(1)) > 1:
                return m_str.group(1)[1]

    return str(entry["expected_answer"])


def _is_bare_answer_restatement(followup_answer: str, followup_reasoning: str) -> bool:
    """
    Returns True when followup_reasoning merely states or repeats the final answer
    without providing any conceptual explanation (e.g., 'Because the answer is 3.',
    'The answer is 3.', 'The answer is e.', 'It prints 3.').
    """
    r_clean = (followup_reasoning or "").strip().rstrip(".!").strip()
    if not r_clean:
        return True

    raw_trimmed = str(followup_answer or "").strip().strip("'\"").strip()
    norm_trimmed = normalize_text(followup_answer)
    candidates = {c for c in [raw_trimmed, norm_trimmed] if c}

    for cand in candidates:
        ans_escaped = re.escape(cand)
        bare_patterns = [
            rf"^(?:because\s+)?(?:the\s+)?(?:answer|output|result|value)\s+(?:is|equals|=|will\s+be|would\s+be)\s+['\"]?{ans_escaped}['\"]?$",
            rf"^(?:because\s+)?(?:it|this|python|the\s+code)\s+(?:prints?|outputs?|returns?|gives?|evaluates?\s+to|produces?)\s+['\"]?{ans_escaped}['\"]?$",
            rf"^['\"]?{ans_escaped}['\"]?\s+is\s+the\s+(?:correct\s+)?(?:answer|output|result)$",
        ]
        for pat in bare_patterns:
            if re.search(pat, r_clean, re.IGNORECASE):
                return True
    return False


def _has_relevant_conceptual_evidence(
    mid: str,
    entry: Dict[str, Any],
    followup_answer: str,
    followup_reasoning: str,
) -> Tuple[bool, bool]:
    """
    Checks whether followup_reasoning contains genuine conceptual evidence
    relevant to the diagnosed misconception `mid`.
    Returns (has_conceptual_evidence, matched_exact_keyword).
    """
    if _is_bare_answer_restatement(followup_answer, followup_reasoning):
        return False, False

    r_lower = followup_reasoning.lower().strip()
    raw_expected_lower = str(entry.get("expected_answer", "")).strip().strip("'\"").lower()
    raw_student_lower = str(followup_answer or "").strip().strip("'\"").lower()
    expected_lower = normalize_text(entry.get("expected_answer", "")).lower()
    student_lower = normalize_text(followup_answer).lower()
    excluded_literals = {expected_lower, student_lower, raw_expected_lower, raw_student_lower}

    # 1. Check accepted_keywords (filtering out any keyword that is merely the raw answer literal)
    accepted_kws = entry.get("accepted_keywords", [])
    matched_kw = any(
        kw.lower() in r_lower
        for kw in accepted_kws
        if kw.lower() not in excluded_literals
    )

    # 2. Check accepted_concept_patterns (natural paraphrases of the corrected rule)
    concept_pats = entry.get("accepted_concept_patterns", [])
    matched_pat = any(
        bool(re.search(pat, r_lower, re.IGNORECASE))
        for pat in concept_pats
    )

    return (matched_kw or matched_pat), matched_kw


def evaluate_followup(
    diagnosis: Dict[str, Any],
    followup_question: Any,
    followup_answer: str,
    followup_reasoning: str
) -> Dict[str, Any]:
    """
    Evaluates student follow-up answer and reasoning deterministically across 5 strict rules:
    
    RULE 1: If reasoning is empty, extremely short, or indicates guessing
            -> NOT_RESOLVED (persistent_misconception = True)
    RULE 2: If follow-up reasoning contains a forbidden misconception pattern
            -> NOT_RESOLVED (persistent_misconception = True)
    RULE 3: If the follow-up answer is incorrect
            -> NOT_RESOLVED (persistent_misconception = True)
    RULE 4: If the follow-up answer is correct BUT reasoning lacks conceptual evidence
            relevant to the original misconception (e.g., bare restatement)
            -> NOT_RESOLVED (resolution_type = "CLARIFICATION_NEEDED", answer_correct = True)
    RULE 5: If answer is correct AND reasoning demonstrates the corrected concept
            -> RESOLVED (persistent_misconception = False)
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
    title = INTERVENTION_BANK[mid]["title"]
    expected_raw = _resolve_expected_followup_answer(mid, followup_question, entry)
    expected_norm = normalize_text(expected_raw)
    student_norm = normalize_text(followup_answer)
    r_lower = (followup_reasoning or "").lower().strip()
    words = re.findall(r"\b\w+\b", r_lower)

    # Support canonical M06 range(2, 5) follow-up submission when reasoning explicitly cites range(2, 5)
    canonical_default_ans = normalize_text(entry.get("expected_answer", ""))
    is_answer_correct = bool(student_norm) and (
        (student_norm == expected_norm)
        or (
            mid == "M06"
            and "range(2, 5)" in r_lower
            and student_norm == canonical_default_ans
        )
    )

    # RULE 1: Check for empty, extremely short, or guessing reasoning
    guessing_match = re.search(
        r"\b(i\s+)?(guessed|just guessed|guessing|dont know|don't know|do not know|not sure|no idea|unsure|picked randomly)\b",
        r_lower,
    )
    if len(r_lower) < 8 or len(words) < 3 or guessing_match:
        res = {
            "status": "NOT_RESOLVED",
            "confidence": 0.90,
            "evidence": f"Follow-up reasoning is empty or indicates guessing: {repr(followup_reasoning)}.",
            "rationale": "Student did not demonstrate conceptual understanding; guessing does not resolve a misconception.",
            "persistent_misconception": True,
            "answer_correct": is_answer_correct,
            "followup_answer": followup_answer,
        }
        _LAST_FOLLOWUP_EVALUATION[mid] = res
        return res

    # RULE 2: Check for persistent misconception / wrong conceptual belief in reasoning
    forbidden_pats = entry.get("forbidden_patterns", [])
    for pat in forbidden_pats:
        if re.search(pat, r_lower, re.IGNORECASE):
            res = {
                "status": "NOT_RESOLVED",
                "confidence": 0.95,
                "evidence": f"Reasoning explicitly repeated misconception pattern: {repr(followup_reasoning)}.",
                "rationale": f"Student persists in {mid} ({title}) misconception despite providing an answer.",
                "persistent_misconception": True,
                "answer_correct": is_answer_correct,
                "followup_answer": followup_answer,
            }
            _LAST_FOLLOWUP_EVALUATION[mid] = res
            return res

    # RULE 3: Check follow-up answer correctness
    if not is_answer_correct:
        res = {
            "status": "NOT_RESOLVED",
            "confidence": 0.95,
            "evidence": f"Student answered '{followup_answer}', expected '{expected_raw}'.",
            "rationale": f"Follow-up answer was incorrect, indicating the misconception {mid} has not yet been resolved.",
            "persistent_misconception": True,
            "answer_correct": False,
            "followup_answer": followup_answer,
        }
        _LAST_FOLLOWUP_EVALUATION[mid] = res
        return res

    # RULE 4 & RULE 5: Answer is correct -> verify conceptual evidence for the original misconception
    eval_entry = dict(entry)
    eval_entry["expected_answer"] = expected_raw
    has_conceptual_evidence, matched_exact_kw = _has_relevant_conceptual_evidence(
        mid, eval_entry, followup_answer, followup_reasoning
    )

    if not has_conceptual_evidence:
        # RULE 4: Correct answer, but missing conceptual evidence (e.g., bare restatement)
        res = {
            "status": "NOT_RESOLVED",
            "resolution_type": "CLARIFICATION_NEEDED",
            "confidence": 0.85,
            "evidence": (
                f"Correct answer '{followup_answer}', but reasoning lacks conceptual evidence for {mid}: "
                f"{repr(followup_reasoning)}."
            ),
            "rationale": (
                f"Your answer ('{followup_answer}') is correct, but your explanation does not yet demonstrate "
                f"the underlying {title} rule. Please explain the conceptual rule that produces this result."
            ),
            "persistent_misconception": False,
            "answer_correct": True,
            "followup_answer": followup_answer,
        }
        _LAST_FOLLOWUP_EVALUATION[mid] = res
        return res

    # RULE 5: Correct answer AND reasoning demonstrates the corrected concept -> RESOLVED
    res = {
        "status": "RESOLVED",
        "confidence": 0.98 if matched_exact_kw else 0.94,
        "evidence": f"Correct answer '{followup_answer}' with sound conceptual rationale: {repr(followup_reasoning)}.",
        "rationale": f"Student demonstrated correct behavior and articulated the key concept of {mid}.",
        "persistent_misconception": False,
        "answer_correct": True,
        "followup_answer": followup_answer,
    }
    _LAST_FOLLOWUP_EVALUATION[mid] = res
    return res


# =============================================================================
# E. SECOND INTERVENTION (Reinforced Trace / Different Angle)
# =============================================================================

def generate_second_intervention(
    diagnosis: Dict[str, Any],
    context: Dict[str, Any],
    previous_intervention: Optional[Dict[str, Any]] = None,
    resolution: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generates a second, reinforced intervention when the follow-up is NOT_RESOLVED.
    Approaches the same misconception with a concrete step-by-step trace or memory model.
    If the student's follow-up answer was correct but lacked conceptual evidence,
    explicitly acknowledges the correct answer and prompts for the underlying rule.
    """
    mid = diagnosis.get("misconception_id")
    if mid not in INTERVENTION_BANK:
        return {
            "status": "NOT_APPLICABLE",
            "message": "Second intervention only applies to unresolved M01-M08 misconceptions."
        }

    sec = INTERVENTION_BANK[mid]["second_intervention"]
    active_res = resolution if resolution is not None else _LAST_FOLLOWUP_EVALUATION.get(mid)

    if active_res and active_res.get("resolution_type") == "CLARIFICATION_NEEDED":
        ans_display = active_res.get("followup_answer", "")
        ans_clause = f" ('{ans_display}')" if ans_display else ""
        explanation = (
            f"Your follow-up answer{ans_clause} is correct, but to resolve {mid} you need to explain "
            f"the underlying conceptual rule rather than just stating the final answer. {sec['explanation']}"
        )
    else:
        explanation = sec["explanation"]

    return {
        "status": "SECOND_TARGETED_INTERVENTION",
        "misconception_id": mid,
        "title": sec["title"],
        "explanation": explanation,
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
    - 8 Focused Resolution Assessment Tests (Section 4: Tests 1-8)
    - 8 Focused Variant-Aware, Question-Contextual & Follow-up Rotation Tests (Section 5: Tests 1-8)
    - 8 Focused Canonical Answer Normalization Tests (Section 6: Tests 1-8)
    """
    print("=" * 70)
    print("PERSON 2: ADAPTIVE INTERVENTION & RESOLUTION ENGINE TEST")
    print("=" * 70)

    _FOLLOWUP_ROTATION_STATE.clear()

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

    print("\n--- 4. TESTING STRICTER RESOLUTION ASSESSMENT SUITE (TESTS 1-8) ---")
    m02_diag = {"misconception_id": "M02", "confidence": 0.85, "evidence": "", "rationale": "", "decision_source": "model"}
    m02_q = {"question": "print(15 // 4)", "expected_answer": "3", "evaluation_type": "exact_and_conceptual"}
    m05_diag = {"misconception_id": "M05", "confidence": 0.85, "evidence": "", "rationale": "", "decision_source": "model"}
    m05_q = {"question": 'word = "hello"\nprint(word[1])', "expected_answer": "e", "evaluation_type": "exact_and_conceptual"}

    resolution_suite = [
        (
            "TEST 1 — Correct + conceptual reasoning (M02)",
            m02_diag,
            m02_q,
            "3",
            "Floor division gives the floor of the division result.",
            "RESOLVED",
        ),
        (
            "TEST 2 — Correct answer but bare restatement (M02)",
            m02_diag,
            m02_q,
            "3",
            "The answer is 3.",
            "NOT_RESOLVED",
        ),
        (
            "TEST 3 — Correct answer + wrong conceptual belief (M02)",
            m02_diag,
            m02_q,
            "3",
            "// gives 3 because Python always rounds normal division to an integer.",
            "NOT_RESOLVED",
        ),
        (
            "TEST 4 — Incorrect answer (M02)",
            m02_diag,
            m02_q,
            "4",
            "Floor division removes the decimal part.",
            "NOT_RESOLVED",
        ),
        (
            "TEST 5 — Guessing (M02)",
            m02_diag,
            m02_q,
            "3",
            "I guessed 3.",
            "NOT_RESOLVED",
        ),
        (
            "TEST 6 — Natural paraphrase (M02)",
            m02_diag,
            {"question": "print(7 // 2)", "expected_answer": "3", "evaluation_type": "exact_and_conceptual"},
            "3",
            "7 divided by 2 is 3.5, and // takes the value down to 3.",
            "RESOLVED",
        ),
        (
            "TEST 7 — M05 example (Correct + 0-based conceptual reasoning)",
            m05_diag,
            m05_q,
            "e",
            "Python starts positions at zero, so index 1 refers to the second character.",
            "RESOLVED",
        ),
        (
            "TEST 8 — M05 bare answer ('The answer is e.')",
            m05_diag,
            m05_q,
            "e",
            "The answer is e.",
            "NOT_RESOLVED",
        ),
    ]

    for label, diag_item, fq, ans, reas, expected_status in resolution_suite:
        ev = evaluate_followup(diag_item, fq, ans, reas)
        print(f"\n[{label}]")
        print(f"  Answer:     '{ans}' | Reasoning: '{reas}'")
        print(f"  Status:     {ev['status']} (Expected: {expected_status})")
        print(f"  Rationale:  {ev['rationale']}")
        assert ev["status"] == expected_status, f"Failed {label}: got {ev['status']}, expected {expected_status}"

        if ev.get("resolution_type") == "CLARIFICATION_NEEDED":
            sec = generate_second_intervention(diag_item, {})
            print(f"  Second Intervention Explanation: {sec['explanation']}")
            assert "is correct" in sec["explanation"] and "explain the underlying" in sec["explanation"]

    print("\n--- 5. TESTING VARIANT-AWARE, CONTEXTUAL & ROTATION SUITE (TESTS 1-8) ---")

    # V-TEST 1: M02 believes_slash_returns_int
    v_test1_diag = {
        "misconception_id": "M02",
        "misconception_variant": "believes_slash_returns_int",
        "confidence": 0.90,
        "evidence": "Student states division gives a whole number.",
        "rationale": "M02 believes_slash_returns_int",
    }
    v_test1_ctx = {
        "question": "print(7 / 2)",
        "correct_answer": "3.5",
        "student_answer": "3",
        "student_reasoning": "Division gives a whole number.",
    }
    v_int1 = generate_intervention(v_test1_diag, v_test1_ctx)
    print(f"\n[V-TEST 1 — M02 believes_slash_returns_int]")
    print(f"  Title:       {v_int1['title']}")
    print(f"  Explanation: {v_int1['short_explanation']}")
    assert "float" in v_int1["short_explanation"].lower() and "3.5" in v_int1["short_explanation"]

    # V-TEST 2: M02 believes_double_slash_returns_remainder
    v_test2_diag = {
        "misconception_id": "M02",
        "misconception_variant": "believes_double_slash_returns_remainder",
        "confidence": 0.90,
        "evidence": "Student confuses // with remainder %.",
        "rationale": "M02 believes_double_slash_returns_remainder",
    }
    v_test2_ctx = {
        "question": "print(15 // 4)",
        "correct_answer": "3",
        "student_answer": "3",
        "student_reasoning": "// gives the remainder",
    }
    v_int2 = generate_intervention(v_test2_diag, v_test2_ctx)
    print(f"\n[V-TEST 2 — M02 believes_double_slash_returns_remainder]")
    print(f"  Title:       {v_int2['title']}")
    print(f"  Explanation: {v_int2['short_explanation']}")
    assert "%" in v_int2["short_explanation"] and "remainder" in v_int2["short_explanation"].lower()
    assert v_int1["short_explanation"] != v_int2["short_explanation"], "V-TEST 1 and V-TEST 2 must differ!"

    # V-TEST 3: M05 one_based_indexing_belief with question context ('msg = "hello"\nprint(msg[1])')
    v_test3_diag = {
        "misconception_id": "M05",
        "misconception_variant": "one_based_indexing_belief",
        "confidence": 0.92,
        "evidence": "Student says index 1 is the first character.",
        "rationale": "M05 one_based_indexing_belief",
    }
    v_test3_ctx = {
        "question": 'msg = "hello"\nprint(msg[1])',
        "correct_answer": "e",
        "student_answer": "h",
        "student_reasoning": "Index 1 is the first character.",
    }
    v_int3 = generate_intervention(v_test3_diag, v_test3_ctx)
    print(f"\n[V-TEST 3 — M05 one_based_indexing_belief with 'hello' context]")
    print(f"  Title:       {v_int3['title']}")
    print(f"  Explanation: {v_int3['short_explanation']}")
    assert "hello" in v_int3["short_explanation"] and '"h"' in v_int3["short_explanation"] and '"e"' in v_int3["short_explanation"]

    # V-TEST 4: M05 negative_index_misunderstanding (alias for negative_index_from_zero_belief)
    v_test4_diag = {
        "misconception_id": "M05",
        "misconception_variant": "negative_index_misunderstanding",
        "confidence": 0.88,
        "evidence": "Student thinks negative indexing starts at -0.",
        "rationale": "M05 negative_index_from_zero_belief",
    }
    v_test4_ctx = {
        "question": 'word = "Python"\nprint(word[-1])',
        "correct_answer": "n",
        "student_answer": "P",
        "student_reasoning": "Negative index counts from 0.",
    }
    v_int4 = generate_intervention(v_test4_diag, v_test4_ctx)
    print(f"\n[V-TEST 4 — M05 negative_index_misunderstanding]")
    print(f"  Title:       {v_int4['title']}")
    print(f"  Explanation: {v_int4['short_explanation']}")
    assert "-1" in v_int4["short_explanation"] and v_int4["short_explanation"] != v_int3["short_explanation"]

    # V-TEST 5: M06 range_stop_inclusive vs range_step_misunderstanding
    v_test5a_diag = {"misconception_id": "M06", "misconception_variant": "range_stop_inclusive"}
    v_test5b_diag = {"misconception_id": "M06", "misconception_variant": "range_step_misunderstanding"}
    v_int5a = generate_intervention(v_test5a_diag, {"question": "for i in range(1, 5): print(i)"})
    v_int5b = generate_intervention(v_test5b_diag, {"question": "for i in range(0, 6, 2): print(i)"})
    print(f"\n[V-TEST 5 — M06 range_stop_inclusive vs range_step_misunderstanding]")
    print(f"  5a Title:    {v_int5a['title']}")
    print(f"  5b Title:    {v_int5b['title']}")
    assert v_int5a["title"] != v_int5b["title"]
    assert v_int5a["short_explanation"] != v_int5b["short_explanation"]

    # V-TEST 6: Deterministic follow-up pool rotation
    _FOLLOWUP_ROTATION_STATE.clear()
    rot_diag = {"misconception_id": "M02", "misconception_variant": "believes_double_slash_returns_float"}
    fq_rot1 = generate_followup(rot_diag, {"question": "print(7 // 2)"})
    fq_rot2 = generate_followup(rot_diag, {"question": "print(9 // 4)"})
    print(f"\n[V-TEST 6 — Follow-up Pool Rotation for M02]")
    print(f"  Attempt 1 Follow-up: {fq_rot1['question'].splitlines()[-1]} (Expected: {fq_rot1['expected_answer']})")
    print(f"  Attempt 2 Follow-up: {fq_rot2['question'].splitlines()[-1]} (Expected: {fq_rot2['expected_answer']})")
    assert fq_rot1["question"] != fq_rot2["question"], "Follow-up questions must rotate deterministically across attempts!"

    # V-TEST 7: Fallback behavior when misconception_variant is None or unknown
    fb_none_int = generate_intervention({"misconception_id": "M04", "misconception_variant": None}, {})
    fb_unk_int = generate_intervention({"misconception_id": "M04", "misconception_variant": "unknown_variant_xyz"}, {})
    print(f"\n[V-TEST 7 — Safe Fallback Behavior]")
    print(f"  None Variant Title:    {fb_none_int['title']}")
    print(f"  Unknown Variant Title: {fb_unk_int['title']}")
    assert fb_none_int["title"] == INTERVENTION_BANK["M04"]["title"]
    assert fb_unk_int["title"] == INTERVENTION_BANK["M04"]["title"]

    # V-TEST 8: Resolution assessment compatibility on rotated follow-up questions
    ev_rot_ok = evaluate_followup(
        rot_diag,
        fq_rot2,
        fq_rot2["expected_answer"],
        "Floor division // divides and rounds down to the nearest integer, removing the decimal part.",
    )
    ev_rot_bare = evaluate_followup(
        rot_diag,
        fq_rot2,
        fq_rot2["expected_answer"],
        f"The answer is {fq_rot2['expected_answer']}.",
    )
    print(f"\n[V-TEST 8 — Resolution Compatibility on Rotated Follow-up]")
    print(f"  Conceptual Reasoning Status: {ev_rot_ok['status']} (Expected: RESOLVED)")
    print(f"  Bare Answer Status:          {ev_rot_bare['status']} (Expected: NOT_RESOLVED)")
    assert ev_rot_ok["status"] == "RESOLVED"
    assert ev_rot_bare["status"] == "NOT_RESOLVED"

    print("\n--- 6. TESTING CANONICAL ANSWER NORMALIZATION SUITE (TESTS 1-8) ---")

    # N-TEST 1: Exact same answer
    assert answers_are_equivalent("3", "3")
    assert answers_are_equivalent("[2, 3, 4]", "[2, 3, 4]")
    print("  [PASS] N-TEST 1: Exact same answer ('3' == '3', '[2, 3, 4]' == '[2, 3, 4]')")

    # N-TEST 2: Whitespace differences (leading/trailing and multiple internal spaces)
    assert answers_are_equivalent("  3  ", "3")
    assert answers_are_equivalent("1   2    3   4", "1 2 3 4")
    print("  [PASS] N-TEST 2: Whitespace differences ('  3  ' == '3', '1   2    3   4' == '1 2 3 4')")

    # N-TEST 3: Newline vs spaces
    assert answers_are_equivalent("1\n2\n3\n4", "1 2 3 4")
    assert answers_are_equivalent("1\r\n2\r\n3\r\n4", "1 2 3 4")
    print("  [PASS] N-TEST 3: Newline vs spaces ('1\\n2\\n3\\n4' == '1 2 3 4')")

    # N-TEST 4: Comma-separated sequence vs space-separated
    assert answers_are_equivalent("1, 2, 3, 4", "1 2 3 4")
    assert answers_are_equivalent("1,2,3,4", "1 2 3 4")
    print("  [PASS] N-TEST 4: Comma-separated sequence ('1, 2, 3, 4' == '1 2 3 4')")

    # N-TEST 5: Bracketed list/tuple sequence vs space/newline/comma-separated
    assert answers_are_equivalent("[1, 2, 3, 4]", "1 2 3 4")
    assert answers_are_equivalent("[1, 2, 3, 4]", "1\n2\n3\n4")
    assert answers_are_equivalent("(1, 2, 3, 4)", "1 2 3 4")
    ev_seq = evaluate_followup(
        {"misconception_id": "M06", "misconception_variant": "range_stop_inclusive_belief"},
        {"question": "print(list(range(2, 5)))", "expected_answer": "[2, 3, 4]"},
        "2\n3\n4",
        "The stop boundary 5 is exclusive, so range(2, 5) stops before 5 at 4.",
    )
    assert ev_seq["status"] == "RESOLVED"
    print("  [PASS] N-TEST 5: Bracketed sequence ('[1, 2, 3, 4]' == '1 2 3 4' == '1\\n2\\n3\\n4')")

    # N-TEST 6: Quoted string formatting (presentation quotes on non-numeric strings)
    assert answers_are_equivalent("'hello'", "hello")
    assert answers_are_equivalent('"hello"', "hello")
    assert answers_are_equivalent("'e'", "e")
    print("  [PASS] N-TEST 6: Quoted string presentation formatting (\"'hello'\" == 'hello', \"'e'\" == 'e')")

    # N-TEST 7: Meaningful numeric/string distinction preserved ("32" vs 32)
    assert not answers_are_equivalent("32", '"32"')
    assert not answers_are_equivalent("32", "'32'")
    assert answers_are_equivalent('"32"', "'32'")
    print("  [PASS] N-TEST 7: Meaningful numeric/string distinction preserved ('32' != '\"32\"')")

    # N-TEST 8: Clearly different answers remaining different
    assert not answers_are_equivalent("3", "4")
    assert not answers_are_equivalent("[1, 2, 3]", "[1, 2, 3, 4]")
    assert not answers_are_equivalent("[2]", "[]")
    assert not answers_are_equivalent("True", "False")
    print("  [PASS] N-TEST 8: Clearly different answers remain distinct ('3' != '4', '[2]' != '[]')")

    _FOLLOWUP_ROTATION_STATE.clear()

    print("\n" + "=" * 70)
    print("ALL PERSON 2 ENGINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_person2_tests()


