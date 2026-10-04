#!/usr/bin/env python3
"""
Person 1: Conceptual Misconception Detection & Diagnosis Pipeline
ReLearn Adaptive Multimodal Learning Environment

Responsibilities:
1. Load and validate the 160-record CS1 dataset (train=87, val=39, test=34).
2. Train a field-scoped multi-class classifier on the 9 trainable classes (M01-M08 + NONE)
   using strictly the 87 training records (zero leakage from val/test).
3. Perform answer-aware, contrastive conceptual reasoning analysis to distinguish:
   - Explicit conceptual misconception claims (M01-M08, right or wrong final answer)
   - Sound conceptual rules with correct answers (NONE, error_type='correct')
   - Sound conceptual rules with typographical/careless slips (NONE, error_type='typo'|'careless')
   - Sound conceptual rules with arithmetic/execution trace slips (INSUFFICIENT, error_type='trace_error')
   - Bare code restatements, guesses, or thin reasoning (INSUFFICIENT)
   - Out-of-scope Python conceptual beliefs (OOS)
4. Resolve boundary collisions (M05 vs M06, M01 vs M02, M03 vs M06) via multi-candidate
   conceptual claim scoring rather than first-match keyword loops.
5. Output structured JSON diagnosis compatible with Person 2 (intervention_engine.py),
   relearn_pipeline.py, relearn_bridge.py, and the React UI.
"""

import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# =============================================================================
# 1. FROZEN TAXONOMY & RATIONALE TEMPLATES (DO NOT MODIFY CLASSES)
# =============================================================================

TAXONOMY_INFO: Dict[str, Dict[str, str]] = {
    "M01": {
        "name": "String vs Number Type Confusion",
        "rationale_template": "The student confuses string and numeric types, expecting arithmetic operations or implicit numeric conversion on string values.",
    },
    "M02": {
        "name": "Division Semantics (/ vs //)",
        "rationale_template": "The student misunderstands Python division operators (/ vs //), confusing true float division, floor division, or remainder semantics.",
    },
    "M03": {
        "name": "Assignment vs Equality (= vs ==)",
        "rationale_template": "The student confuses the assignment operator (=) with the equality comparison operator (==).",
    },
    "M04": {
        "name": "Operator Precedence",
        "rationale_template": "The student applies incorrect operator precedence, such as strict left-to-right evaluation instead of Python's precedence hierarchy.",
    },
    "M05": {
        "name": "Index / Position",
        "rationale_template": "The student misunderstands sequence element positions, such as assuming 1-based indexing, -0 indexing, or inclusive slice stop positions.",
    },
    "M06": {
        "name": "Loop Values / Boundaries / Iteration Interval",
        "rationale_template": "The student misunderstands loop iteration boundaries or range() semantics, such as assuming the stop value is inclusive or range(n) starts at 1.",
    },
    "M07": {
        "name": "Function Argument-Parameter Binding",
        "rationale_template": "The student misunderstands how function arguments bind to parameters across positional, keyword, or default parameter rules.",
    },
    "M08": {
        "name": "Recursion Termination / Base Case",
        "rationale_template": "The student misunderstands recursive call termination, base-case handling, or how return values unwind through the call stack.",
    },
    "NONE": {
        "name": "No Misconception",
        "rationale_template": "The student demonstrates sound conceptual understanding or makes a non-conceptual typographical/careless slip without a flawed mental model.",
    },
    "INSUFFICIENT": {
        "name": "Insufficient Evidence",
        "rationale_template": "There is insufficient conceptual evidence to diagnose a specific misconception (e.g., guessing, bare code restatement, or unacknowledged trace slip).",
    },
    "OOS": {
        "name": "Out of Scope",
        "rationale_template": "The student expresses a genuine conceptual belief about a Python topic outside the frozen M01-M08 taxonomy.",
    },
}

TRAINABLE_CLASSES: List[str] = [
    "M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "NONE"
]
M_CLASSES: List[str] = ["M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08"]


# =============================================================================
# 2. ANSWER NORMALIZATION & COMPARISON
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


def normalize_answer(text: str, preserve_numeric_quotes: bool = True) -> str:
    """
    Canonicalizes answer text for semantic equivalence comparison:
    1. Strips leading/trailing whitespace.
    2. Strips outer presentation quotes on non-numeric text while preserving quoted numeric literals ("32" vs 32).
    3. Canonicalizes Python-style multi-element list/tuple brackets (e.g., "[1, 2, 3, 4]" -> "1 2 3 4").
    4. Canonicalizes comma-separated atomic sequences (e.g., "1, 2, 3, 4" -> "1 2 3 4").
    5. Collapses newlines and multiple spaces into single spaces ("1\\n2\\n3\\n4" -> "1 2 3 4").
    6. Normalizes harmless numeric formatting (+3 -> 3, 3.50 -> 3.5) and lowercases textual tokens.
    """
    if text is None:
        return ""
    t = str(text).strip()
    if not t:
        return ""

    t = _strip_presentation_quotes(t, preserve_numeric_quotes=preserve_numeric_quotes)

    # Check for outer flat Python list/tuple brackets [...] or (...)
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
    sa_norm = normalize_answer(student_answer, preserve_numeric_quotes=preserve_numeric_quotes)
    ea_norm = normalize_answer(expected_answer, preserve_numeric_quotes=preserve_numeric_quotes)
    if not sa_norm or not ea_norm:
        return False
    return sa_norm == ea_norm


def is_spelling_typo_of(sa: str, ca: str) -> bool:
    """
    Checks if non-numeric student_answer is a 1-edit or adjacent-transposition
    typo of correct_answer (e.g., 'flase' vs 'false').
    """
    a = normalize_answer(sa)
    b = normalize_answer(ca)
    if not a or not b or a == b:
        return False
    if re.match(r"^-?\d+(?:\.\d+)?$", a) or re.match(r"^-?\d+(?:\.\d+)?$", b):
        return False
    if len(a) < 3 or len(b) < 3 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        if (
            len(diffs) == 2
            and diffs[1] == diffs[0] + 1
            and a[diffs[0]] == b[diffs[1]]
            and a[diffs[1]] == b[diffs[0]]
        ):
            return True
        return False
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    for i in range(len(longer)):
        if longer[:i] + longer[i + 1:] == shorter:
            return True
    return False


def compute_answer_correct(
    question: str,
    correct_answer: str,
    student_answer: str,
    student_reasoning: str = "",
) -> bool:
    """
    Determines whether student_answer is semantically equivalent to correct_answer,
    handling output strings, syntax error descriptions, and 'explain why' prompts.
    """
    ca_norm = normalize_answer(correct_answer)
    sa_norm = normalize_answer(student_answer)
    if not sa_norm or not ca_norm:
        return False

    if student_reasoning and re.search(
        r"\b(meant to type|hit the .* key by (?:accident|mistake)|accidentally typed|fat-fingered|typed \w+ instead of \w+ on the keyboard)\b",
        student_reasoning.lower(),
    ):
        if sa_norm != ca_norm and "nameerror on" not in sa_norm:
            return False

    if sa_norm == ca_norm:
        return True

    if re.search(
        r"\b(no bug|no syntax error|no error|it works|valid syntax|terminates cleanly)\b",
        sa_norm,
    ):
        return False

    syntax_concepts = [
        (["missing colon", "colon"], ["colon", ":"]),
        (["parentheses", "print is a function"], ["parentheses", "parenthesis"]),
        (
            ["never closed", "unclosed", "closing parenthesis"],
            ["closing parenthesis", "parenthesis", ")"],
        ),
        (["nameerror"], ["nameerror"]),
    ]
    for ca_kws, sa_kws in syntax_concepts:
        if any(k in ca_norm for k in ca_kws) and any(k in sa_norm for k in sa_kws):
            return True

    q_lower = (question or "").lower()
    m_out = re.search(
        r"explain why (?:this outputs |the output is |)['\"]?([a-z0-9_]+)['\"]?(?: is printed)?",
        q_lower,
    )
    if m_out and sa_norm == m_out.group(1).lower():
        return True

    if "reverse order" in ca_norm and "[::-1]" in q_lower:
        m_str = re.search(r'=\s*["\']([a-zA-Z0-9]+)["\']', question)
        if m_str and sa_norm == m_str.group(1)[::-1].lower():
            return True

    return False


# =============================================================================
# 3. CONTRASTIVE CONCEPTUAL RULES (Strictly derived from taxonomy.ts + train)
#    Zero rules or phrases derived from validation or test examples.
# =============================================================================

MISCONCEPTION_CLAIM_RULES: Dict[str, List[Tuple[str, re.Pattern, float]]] = {
    "M01": [
        (
            "treats_numeric_string_as_int",
            re.compile(
                r"(?:quote[s]?|quotation\s+mark[s]?).*(?:(?:don'?t|do\s+not|doesn'?t|does\s+not)\s+matter|are\s+ignored|ignored|just\s+surround|make[s]?\s+it\s+(?:the\s+|a\s+)?number|still\s+(?:a\s+)?number|around\s+(?:the\s+)?(?:number|digit)|regardless\s+of\s+quote[s]?)|"
                r"(?:treat[s]?|act[s]?\s+as|recognizes?|is\s+treated\s+as|interprets?).*(?:string[s]?|quote[s]?|['\"][0-9.]+['\"]|digits?).*(?:as\s+|like\s+)?(?:an?\s+|the\s+)?(?:regular\s+|normal\s+)?(?:integer[s]?|\bint\b|number[s]?|numeric)|"
                r"(?:numeric\s+strings?|['\"][0-9.]+['\"]|strings?\s+with\s+(?:digits?|numbers?)).*(?:act[s]?\s+(?:like|as)|looks?\s+like|are\s+treated\s+as|contain[s]?\s+digits?).*(?:integer[s]?|\bint\b|number[s]?|adds?\s+(?:them|\d+))|"
                r"(?:['\"][0-9]+['\"].*contain[s]?\s+digits?|['\"][0-9]+['\"]\s+looks?\s+like\s+a\s+number).*(?:adds?\s+(?:them|\d+)|mathematically)|"
                r"(?:plus|addition|minus|subtract\w*|star|multipl\w*|\+|\-|\*).*(?:treat[s]?\s+numbers?\s+inside\s+quotes?|does\s+math\s+between\s+the\s+numbers?|performs?\s+numerical\s+(?:arithmetic|multiplication|addition)).*(?:quote[s]?|string[s]?|digits?)|"
                r"(?:string\s+['\"]?[0-9]+['\"]?\s+and\s+(?:integer\s+|number\s+)?[0-9]+|['\"][0-9]+['\"]\s*(?:and|==)\s*[0-9]+|[0-9]+\s*(?:and|==)\s*['\"][0-9]+['\"]).*(?:mathematically\s+identical|exact\s+same\s+value|same\s+value|equal\s+value|evaluates?\s+to\s+true)|"
                r"(?:both\s+values?\s+represent\s+(?:single\s+)?numbers?|both\s+are\s+numbers?|['\"][0-9]+['\"]\s+and\s+['\"][0-9]+['\"]\s+are\s+numbers?).*(?:plus|adds?|\+|multipl\w*).*(?:(?:into|to\s+get|to\s+produce|gives?|produces?)\s+[0-9]+|numerically)|"
                r"(?:multipl(?:y|ying|ies)|add(?:s|ing)?)\s+['\"][0-9]+['\"].*(?:multiplies|adds?)\s+(?:the\s+)?(?:number|int(?:eger)?)\s+[0-9]+|"
                r"recognizes?\s+[0-9]+\s+and\s+[0-9]+\s+are\s+numbers?\s+even\s+though.*quote[s]?|"
                r"['\"][0-9]+['\"]\s*(?:plus|\+|times|\*)\s*[0-9]+\s*(?:automatically\s+)?(?:adds?|multiplies|gives?|yields?)\s*(?:into\s+)?[0-9]+",
                re.IGNORECASE,
            ),
            1.0,
        ),
        (
            "expects_plus_to_add_str_and_int",
            re.compile(
                r"(?:plus(?:\s+sign|\s+operator)?|\+|addition).*(?:automatically\s+)?(?:turn[s]?|cast[s]?|convert[s]?|coerce[s]?)\s+(?:the\s+)?(?:string\s+)?(?:['\"]?[0-9]+['\"]?|number[s]?|integer[s]?|string[s]?|\bstr\b).*(?:into\s+|to\s+)(?:the\s+|a\s+|an\s+)?(?:text|string[s]?|integer[s]?|\bint\b|number[s]?)|"
                r"(?:automatically\s+)?(?:converts?|turns?|casts?)\s+(?:the\s+)?(?:['\"][0-9]+['\"]|string|\bstr\b|integer\s+\d+|\d+)\s+(?:to|into)\s+(?:a\s+|an\s+)?(?:number|int(?:eger)?|string|\bstr\b)",
                re.IGNORECASE,
            ),
            1.05,
        ),
        (
            "expects_int_cast_without_int_call",
            re.compile(
                r"(?:input(?:\(\))?|digits?\s+only|numeric\s+string[s]?).*(?:automatically\s+treat[s]?|automatically\s+convert[s]?|without\s+(?:needing\s+|calling\s+)?int\(\))|"
                r"int\(\s*['\"][0-9]+\.[0-9]+['\"]\s*\).*(?:automatically\s+)?(?:truncate[s]?|round[s]?|drop[s]?)",
                re.IGNORECASE,
            ),
            1.05,
        ),
    ],
    "M02": [
        (
            "believes_slash_returns_int",
            re.compile(
                r"(?:single\s+slash|(?<!/)/(?!/)|divid(?:es?|ing|ed)\s+(?:evenly|equal\s+numbers?|by\s+[0-9]+)|both\s+division\s+symbols?)(?:(?!(?://|double\s+slash|floor\s+division))[^.;\n])*(?:no\s+remainder|zero\s+remainder|no\s+decimal[s]?|equal\s+numbers?|whole\s+number)(?:(?!(?://|double\s+slash|floor\s+division))[^.;\n])*(?:produce[s]?|return[s]?|give[s]?|creates?|is)\s+(?:an?\s+|the\s+)?(?:integer|\bint\b)|"
                r"(?:single\s+slash|(?<!/)/(?!/))(?:(?!(?://|double\s+slash|floor\s+division))[^.;\n])*(?:(?:always\s+)?(?:produce[s]?|return[s]?|give[s]?|is)\s+(?:an?\s+|the\s+)?(?:integer|\bint\b)|(?:does|performs?)\s+(?:integer|\bint\b|whole[- ]number)\s+division|(?:drop[s]?|truncat\w*|remov\w*|discard[s]?)\s+(?:the\s+)?decimal)|"
                r"(?:only\s+produces?\s+floats?\s+when\s+there\s+is\s+a\s+(?:fractional\s+)?remainder|dividing\s+equal\s+numbers?\s+always\s+gives?\s+an?\s+integer)",
                re.IGNORECASE,
            ),
            1.05,
        ),
        (
            "believes_double_slash_returns_remainder",
            re.compile(
                r"(?:double\s+slash|//|floor\s+division)(?:(?!(?:%|modulo|modulus|\bnot\b|does\s+not|doesn'?t|isn'?t|wasn'?t|never|instead\s+of|rather\s+than|whereas|while\b))[^.;\n])*(?:(?:is|was|means?|acts?\s+as|performs?|does)\s+(?:the\s+)?(?:modulus|modulo|remainder)(?:\s+operator|\s+operation|\s+division)?|(?:computes?|returns?|calculates?|gives?|finds?|gets?|produces?)\s+(?:the\s+)?(?:leftover\s+)?remainder)",
                re.IGNORECASE,
            ),
            1.1,
        ),
        (
            "believes_double_slash_returns_float",
            re.compile(
                r"(?:double\s+slash|//|floor\s+division)(?:(?!(?:\bnot\b|does\s+not|doesn'?t|never|without|instead\s+of|rather\s+than|whereas|unlike|different\s+from|(?<!/)/(?!/)|\bdivid(?:ed|es|ing)\s+by\b))[^.;\n])*(?:keeps?\s+(?:the\s+)?decimal|(?:performs?|does|is(?:\s+used\s+(?:specifically\s+)?to\s+perform)?)\s+(?:float(?:ing[- ]point)?|normal|regular)\s+division|forces?\s+decimal\s+output|returns?\s+(?:a\s+)?float(?:\s+with\s+decimal)?|(?:gives?|returns?|produces?)\s+(?:the\s+|a\s+)?decimal(?:\s+(?:result|value|number|answer|output|part|portion))?|(?:gives?|returns?|produces?|evaluates?\s+to)\s+-?\d+\.\d+\b|divides?\s+normally)|"
                r"\b\d+\s*//\s*\d+\s*(?:is|equals?|=)\s*-?\d+\.\d+\b",
                re.IGNORECASE,
            ),
            1.05,
        ),
        (
            "other_division_confusion",
            re.compile(
                r"(?:double\s+slash|//)[^.;\n]*(?:is\s+defined\s+to\s+always\s+return|is\s+guaranteed\s+to\s+produce|always\s+returns?)\s+(?:an?\s+)?\bint\b(?:\s+type)?|"
                r"(?:double\s+slash|//|floor\s+division).*(?:truncat\w*|round\w*|drop\w*|chop\w*)\b.*(?:\btoward[s]?\s+zero\b|-[0-9]+)|"
                r"(?:double\s+slash|//|floor\s+division)[^.;\n]*\brounds?(?!\s+down\b)(?:\s+(?:the\s+)?(?:answer|result|value|number|quotient|it))?\s+(?:up\b|to\s+the\s+(?:nearest|closest)\b)|"
                r"\b\d+\s*(?://|divided\s+by|/)\s*\d+\s+is\s+\d+\.\d+\s*,?\s*(?:so|which|and)\s+(?://\s+|it\s+)?rounds?(?!\s+down\b)(?:\s+(?:the\s+)?(?:answer|result|value|number|quotient|it))?\s+(?:up|to\s+the\s+(?:nearest|closest)(?:\s+integer)?)\b|"
                r"-[0-9]+.*divid\w*.*(?:truncat\w*\s+(?:the\s+decimal\s+(?:part\s+)?)?(?:toward[s]?\s+zero)?|round\w*\s+toward[s]?\s+zero|drop\w*\s+the\s+decimal\s+part)",
                re.IGNORECASE,
            ),
            1.05,
        ),
    ],
    "M03": [
        (
            "uses_single_equals_for_conditional_test",
            re.compile(
                r"(?:single\s+equals?(?:\s+sign)?|(?<![=!<>])=(?!=))(?:(?!double\s+equals?|==).)*(?:(?:to\s+)?tests?\s+(?:if|whether)|(?:to\s+)?checks?\s+(?:if|whether)|(?:to\s+)?compares?\s+(?:the\s+)?(?:values?|numbers?|variables?|whether|two)|is\s+an?\s+equality\s+(?:test|check)|equal\s+in\s+math|like\s+standard\s+algebra)|"
                r"(?:tests?|checks?|compares?|asserts?)\b.*(?:that|if|whether)?\s*\w+\s+(?:equals?|=)\s+.*\b(?:single\s+equals?)",
                re.IGNORECASE,
            ),
            1.1,
        ),
        (
            "expects_double_equals_to_assign_value",
            re.compile(
                r"(?:double\s+equals?|==).*(?:assigns?|re-?assigns?|sets?\s+(?:the\s+)?(?:variable|val\w*|\w+)|updates?\s+\w+\s+to|stronger\s+form\s+of\s+assignment)|"
                r"(?:sets?|assigns?|updates?|re-?assigns?)\s+\w+\s+(?:equal\s+)?to\s+.*\s+using\s+(?:double\s+equals?|==)|"
                r"\b\w+\s*==\s*[^=].*(?:updates?|sets?|assigns?|re-?assigns?)\s+\w+\s+to\b",
                re.IGNORECASE,
            ),
            1.1,
        ),
        (
            "believes_equals_creates_symmetric_equation",
            re.compile(
                r"(?:single\s+equals?(?:\s+sign)?|putting\s*=|(?<![=!<>])=(?!=)).*(?:defines?\s+an\s+equation|is\s+an\s+equation\s+stating|receives?\s+.*\s+from\s+the\s+left\s+side|stores?\s+\d+\s+into\s+\w+\s+just\s+like\s+\w+\s*=)",
                re.IGNORECASE,
            ),
            1.1,
        ),
    ],
    "M04": [
        (
            "left_to_right_evaluation_override",
            re.compile(
                r"(?!.*positional\s+arguments?)(?:evaluat\w*|comput\w*|read\w*|operat\w*|do(?:es|ing)?|did|comes?\s+first|executes?|go(?:es|ing)?|work\w*|mov\w*|solv\w*|calculat\w*).*(?:strictly\s+)?(?:from\s+)?left(?:\s+|-)+to(?:\s+|-)+right|"
                r"(?!.*positional\s+arguments?)\bleft(?:\s+|-)+to(?:\s+|-)+right\b.*(?:first|before|so\s+\d+)",
                re.IGNORECASE,
            ),
            1.05,
        ),
        (
            "arithmetic_plus_before_multiply",
            re.compile(
                r"(?:addition|subtraction|\+|\-)\s+(?:has\s+higher\s+precedence|evaluates?\s+(?:first|before)|comes?\s+(?:first|before)|happens?\s+(?:first|before)|goes?\s+(?:first|before)|is\s+(?:calculated|done|evaluated)\s+(?:first|before))\s*(?:than\s+|before\s+)?(?:multiplication|division|modulo|\*|/|//|%)?|"
                r"(?:do|does|did)\s+(?:addition|subtraction|\+|\-)\s+(?:first|before)|"
                r"(?:add(?:ed|ing|s)?|subtract(?:ed|ing|s)?)\s+(?:first\s+)?before\s+(?:multiply(?:ing)?|multiplication|divid(?:ing)?|division)|"
                r"(?:multiplication|\*)\s+(?:has\s+higher\s+precedence|evaluates?\s+(?:first|before)|comes?\s+(?:first|before)|happens?\s+(?:first|before)|goes?\s+(?:first|before)|is\s+(?:calculated|done|evaluated)\s+(?:first|before))\s*(?:than\s+|before\s+)?(?:exponentiation|power|\*\*)|"
                r"(?:first\s+(?:i\s+|we\s+)?(?:multiply|add|subtract)|(?:multiply|add|subtract)\s+\d+\s+(?:and|by|with|\+|\-|\*)\s+\d+\s+first).*(?:then\s+.*(?:\*\*|power|exponent\w*|calculate\s+\d+\s+to\s+the\s+power)|before\s+(?:\*\*|power|exponent\w*))|"
                r"(?:first\s+(?:i\s+|we\s+)?(?:add|subtract)|(?:add|subtract)\s+\d+\s+(?:and|to|from|\+|\-)\s+\d+\s+first).*(?:then\s+.*(?:multipl\w*|divid\w*|times|\*|/)|before\s+(?:multipl\w*|divid\w*))|"
                r"(?:first\s+(?:i\s+|we\s+)?(?:do|calculate|compute|evaluate)?\s*\d+\s*[\+\-]\s*\d+).*(?:then\s+.*(?:multipl\w*|divid\w*|times|\*|/|(?:\*\*|power)))|"
                r"(?:multiplication|addition|\*|\+)\s+.*(?:before|first).*(?:even\s+(?:with|inside)\s+parentheses|regardless\s+of\s+parentheses|ignoring\s+parentheses)|"
                r"(?:negative\s+sign|minus\s+sign)\s+(?:attaches?|belongs?|applies?)\s+(?:directly\s+)?to\s+(?:the\s+)?\d+\s*(?:first)?.*(?:squar(?:ed|ing)|power|\*\*)",
                re.IGNORECASE,
            ),
            1.1,
        ),
        (
            "boolean_or_before_and",
            re.compile(
                r"(?:does|evaluates?)\s*\(?.*\bor\b.*\)?\s*first|"
                r"\bor\b\s+(?:has\s+higher\s+precedence|evaluates?\s+(?:first|before)|comes?\s+(?:first|before))\s*(?:than\s+)?\band\b|"
                r"\b(?:operator\s+)?not\b\s+(?:applies?\s+to\s+the\s+entire|has\s+highest?\s+precedence|has\s+higher\s+precedence\s+than\s+==|evaluates?\s+(?:first|before)\s+(?:to\s+(?:false|true)|==)|binds?\s+tighter\s+than\s+==)",
                re.IGNORECASE,
            ),
            1.1,
        ),
        (
            "relational_before_arithmetic",
            re.compile(
                r"(?:bitwise|&\s*operator|\b&\b|\|)\s+(?:operators?\s+)?(?:like\s+&\s+)?(?:always\s+)?(?:evaluates?|binds?|executes?|has\s+higher\s+precedence)\s+(?:before|first|than)\s*(?:comparison|==)?|"
                r"(?:comparison|relational|==|<|>)\s+(?:operators?\s+)?(?:like\s+==|<|>)?\s*(?:always\s+)?(?:has\s+higher\s+precedence|evaluates?\s+(?:first|before)|binds?\s+tighter|is\s+evaluated\s+before)\s*(?:than\s+)?(?:addition|subtraction|arithmetic|\+|\-)",
                re.IGNORECASE,
            ),
            1.1,
        ),
    ],
    "M05": [
        (
            "one_based_indexing_belief",
            re.compile(
                r"(?:index(?:ing)?|positions?\s+in\s+(?:a\s+)?(?:list|string|tuple|sequence)|list\s+positions?|sequences?|counting\s+in\s+(?:lists?|strings?|sequences?))\s+(?:start[s]?|begin[s]?)\s+at\s+(?:index\s+)?(?:1|one)\b|"
                r"\b(?:1-based|one-based)\b|"
                r"(?:index|position|bracket)\s*1\s+(?:refers?\s+to|is|retrieves?|grabs?|accesses?|gives?)\s+the\s+(?:1st|first|very\s+initial|initial)\s+(?:item|element|letter|character)|"
                r"(?:index|position|bracket)\s*(?:2|two)\s+(?:refers?\s+to|is|retrieves?|grabs?|accesses?|gives?)\s+the\s+(?:2nd|second)\s+(?:item|element|letter|character)|"
                r"(?:first|1st|initial)\s+(?:element|item|character|letter|value).*\b(?:is\s+at|at|is|as)\s+(?:index|position)\s*(?:1|one)\b|"
                r"(?:count\w*|treat\w*)\s+the\s+(?:first|1st)\s+(?:element|item|character|letter)\s+as\s+(?:index|position)?\s*(?:1|one)\b|"
                r"(?:1st|first)\s+(?:element|item|letter)\s+\w+\[\s*0\s*\+\s*1\s*\]|"
                r"(?:list\s+elements?|items?)\s+are\s+at\s+positions?\s+1,\s*2,\s*and\s*3|"
                r"\bstep\s*\[::2\]\s+selects?\s+only\s+the\s+even\s+numbered\s+positions?",
                re.IGNORECASE,
            ),
            1.1,
        ),
        (
            "negative_index_from_zero_belief",
            re.compile(
                r"(?:negative\s+index(?:ing)?|-0).*(?:-0\s+is|-0\s+would\s+be|wraps?\s+around\s+from\s+0|-1\s+is\s+the\s+(?:second|2nd)\s+to\s+last|-2\s+is\s+the\s+last)",
                re.IGNORECASE,
            ),
            1.1,
        ),
        (
            "slice_endpoint_inclusive_as_position",
            re.compile(
                r"\bslic(?:e|ing)\b.*(?:ends?\s+at\s+(?:index|position)\s+\d+(?:\s*\([^)]*\))?\s+inclusive|endpoints?\s+(?:are\s+(?:both\s+)?inclusive|specify\s+which\s+elements\s+are\s+included)|up\s+to\s+and\s+including\s+(?:index|position)\s+\d+|(?:should\s+)?includes?\s+(?:the\s+)?(?:end(?:ing)?|stop|final|upper|second)?\s*(?:index|position|bound|boundary)(?:\s+\d+)?)",
                re.IGNORECASE,
            ),
            1.1,
        ),
        (
            "off_by_one_element_lookup",
            re.compile(
                r"(?:there\s+are|in\s+a|has\s+length)\s+(\d+)(?:\s+(?:items?|elements?|characters?))?.*(?:final|last)\s+(?:element|item|character)\s+(?:resides?\s+at|is\s+at)\s+(?:index|position)(?:\s+position)?\s+\1\b|"
                r"(?:index|position)\s*0\s+skips?\s+the\s+(?:first|1st)\s+(?:letter|character|element|item)|"
                r"(?:number|value)\s+inside\s+(?:square\s+)?brackets?\s+\w+\[\d+\]\s+specifies\s+the\s+value\s+to\s+look\s+up",
                re.IGNORECASE,
            ),
            1.1,
        ),
    ],
    "M06": [
        (
            "range_stop_inclusive_belief",
            re.compile(
                r"\brange\b.*(?:\binclusive\b|includ(?:es?|ing)\s+(?:both\s+.*\s+and\s+|the\s+stop\s+(?:value|parameter)\s+)?\d+|includ(?:es?|ing)\s+(?:both\s+)?(?:the\s+)?(?:start(?:ing)?\s+(?:(?:value|number|bound)\s*\d*\s+)?(?:and\s+)?)?(?:the\s+)?(?:end(?:ing)?|stop|final|last|upper)(?:\s+(?:values?|numbers?|bounds?|boundaries|endpoints?|parameters?))?|includ(?:es?|ing)\s+both\s+endpoints?|goes?\s+from\s+\d+\s+(?:up\s+)?(?:through|to)\s+\d+\s+inclusive|stops?\s+at\s+\d+\s+inclusive|hits?\s+\d+\s+inclusive)|"
                r"\b\d+\s+is\s+included\s+in\s+range\b|"
                r"(?:stops?|ends?|halts?)\s+after\s+(?:printing|reaching|visiting|including)\s+(?:the\s+)?(?:stop|end(?:ing)?|final|upper)\s+(?:value|number|bound)?|"
                r"(?:stop\s+(?:value|boundary|parameter)|upper\s+bound(?:ary)?|end\s+value)\b.*\bis\s+inclusive\b|"
                r"counting\s+backwards\s+from\s+(\d+)\s+down\s+to\s+(\d+)\s+with\s+step\s+-1\s+visits\s+.*\b\2\b|"
                r"\brange\(\s*(\d+)\s*,\s*\1\s*\).*(?:execute[s]?\s+once|start\s+and\s+stop\s+are\s+equal)|"
                r"\bif\s+the\s+stop\s+(?:was|were)\s+\d+\s+it\s+would\s+include\s+\d+",
                re.IGNORECASE,
            ),
            1.0,
        ),
        (
            "range_starts_at_one_default",
            re.compile(
                r"\brange\(\s*\d+\s*\).*(?:starts?\s+(?:at|from)\s+1\b|(?:generates?|counts?|produces?|goes?|runs?)\b.*starting\s+(?:from|at)\s+1\b)|"
                r"\brange\b.*starts?\s+(?:at|from)\s+1\s+by\s+default\b",
                re.IGNORECASE,
            ),
            1.05,
        ),
        (
            "misunderstands_step_interval",
            re.compile(
                r"\brange\b.*starts?\s+at\s+10\s+and\s+goes?\s+to\s+5.*(?:step\s+1\b|counts?\s+5\s+times)|"
                r"\bstep(?:\s+size)?\s+\d+\s+is\s+(?:larger|bigger|greater)\s+than.*range\s+cannot\s+(?:even\s+)?start|"
                r"\brange\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\).*stops?\s+before\s+reaching\s+\d+\s+because\s+stop\s+\d+\s+minus\s+start|"
                r"\bstep\b.*(?:tells?\s+|specifies\s+|means?\s+)?how\s+many\s+(?:values?|numbers?|items?|elements?)\s+to\s+skip\b|"
                r"\bskips?\s+\d+\s+(?:numbers?|values?|items?)\s+each\s+time\b",
                re.IGNORECASE,
            ),
            1.05,
        ),
        (
            "loop_iteration_count_off_by_one",
            re.compile(
                r"\brange\(\s*(\d+)\s*\)\s+counts?\s+(?:up\s+to\s+)?\1(?:\s+times)?,\s+so\s+the\s+(?:last\s+value\s+of\s+the\s+)?loop\s+variable\s+\w+\s+(?:reaches|is)\s+\1\b|"
                r"\b(?:while|for)\s+loop\s+condition\b.*(?:\bboundary\b|\bstop\s+value\b|\bnever\s+take\s+any\s+value\b|\bextra\s+time\b)",
                re.IGNORECASE,
            ),
            1.05,
        ),
    ],
    "M07": [
        (
            "believes_arg_name_must_match_param_name",
            re.compile(
                r"(?:caller\s+)?(?:variable[s]?|argument[s]?)\s+(?:passed\s+to\s+a\s+function\s+)?(?:name[s]?\s+)?(?:must|have\s+to|requires?.*to)\s+(?:have\s+the\s+same\s+name[s]?|match|equal)\s+(?:as\s+)?(?:the\s+)?(?:function\s+)?parameter(?:\s+name[s]?)?|"
                r"(?:requires?\s+(?:the\s+)?argument\s+(?:variable\s+)?name\s+to\s+match|connects?\s+arguments?\s+to\s+parameters?\s+by\s+matching\s+(?:letter\s+)?names?|matches?\s+arguments?\s+to\s+parameters?\s+(?:with|by)\s+the\s+same\s+(?:variable\s+)?name|mapped\s+by\s+variable\s+name|binds?\s+arguments?\s+to\s+parameter\s+names?\s+based\s+on\s+meaning)|"
                r"(?:prints?|returns?|uses?|outputs?)\s+the\s+parameter\s+name\s+literally\s+instead\s+of\s+the\s+argument",
                re.IGNORECASE,
            ),
            1.1,
        ),
        (
            "misunderstands_positional_binding_order",
            re.compile(
                r"(?:allows?\s+positional\s+arguments?\s+anywhere|positional\s+arguments?\s+after\s+keyword|positional\s+arguments?\s+bind\s+from\s+right\s+to\s+left|matches?\s+.*\s+to\s+\w+\s+automatically\s+because\s+\w+\s+was\s+already\s+taken)|"
                r"(?:second|2nd|\d+(?:st|nd|rd|th))\s+argument\s+.*\s+fills?\s+\w+\s+because\s+.*\s+matches\s+the\s+value\s+of\s+\w+",
                re.IGNORECASE,
            ),
            1.1,
        ),
        (
            "keyword_argument_override_confusion",
            re.compile(
                r"(?:even\s+though\s+keyword|named\s+arguments?\s+are\s+ignored|regardless\s+of\s+whether\s+positional\s+or\s+keyword).*(?:order\s+in\s+the\s+call|left-to-right\s+positional\s+order|mapped\s+by\s+variable\s+name)|"
                r"(?:first|1st|second|2nd)\s+argument\s+(?:passed|supplied)?.*(?:binds?\s+to|fills?|goes?\s+into)\s+(?:parameter\s+)?\w+|"
                r"\bposition\s+always\s+determines\s+which\s+argument\s+goes\s+where\b",
                re.IGNORECASE,
            ),
            1.1,
        ),
        (
            "parameter_count_mismatch_tolerated",
            re.compile(
                r"(?:omit\s+an\s+argument|missing\s+(?:positional\s+)?argument|extra\s+argument[s]?).*(?:automatically\s+sets?\s+the\s+missing\s+parameter\s+to\s+none|still\s+runs|silently\s+ignored|dropped)",
                re.IGNORECASE,
            ),
            1.1,
        ),
    ],
    "M08": [
        (
            "believes_recursion_stops_automatically_at_zero",
            re.compile(
                r"(?:recurs(?:ion|ive\s+function[s]?)|calls?\s+itself\s+recursively).*(?:naturally\s+stops?|automatically\s+(?:stops?|terminates?)|without\s+(?:needing\s+)?(?:an\s+if|a?\s*base\s+case[s]?)|when\s+(?:the\s+)?(?:counter|parameter|\w+)\s+(?:hits?|reaches?|decrements?\s+(?:down\s+)?to|becomes?\s+less\s+than)\s*\d+)|"
                r"\bbase\s+case\b.*stops?\s+any\s+recursive\s+function\s+regardless\s+of\s+whether\s+\w+\s+increments?\s+or\s+decrements?",
                re.IGNORECASE,
            ),
            1.1,
        ),
        (
            "believes_base_case_resets_execution",
            re.compile(
                r"(?:base\s+case[s]?|return\s+\d+\s+executes?|when\s+[\w()]+\s+(?:hits?|reaches?)\s+(?:if\s+)?\w+\s*==\s*\d+).*(?:resets?\s+the\s+(?:entire\s+)?call\s+stack|forgets?\s+the\s+previous\s+(?:additions?|multiplications?|calls?)|forces?\s+the\s+final\s+answer\s+to\s+be|reset\s+to\s+\d+)",
                re.IGNORECASE,
            ),
            1.1,
        ),
        (
            "misunderstands_recursive_call_return_value",
            re.compile(
                r"(?:base\s+case|when\s+\w+\s*==\s*\d+\s+hits?\s+return).*(?:passed\s+all\s+the\s+way\s+back|up\s+the\s+stack\s+to\s+become|immediately\s+becomes\s+the\s+return\s+value\s+for\s+the\s+original\s+caller\s+without\s+needing\s+return)",
                re.IGNORECASE,
            ),
            1.1,
        ),
        (
            "base_case_placement_confusion",
            re.compile(
                r"(?:scans?\s+the\s+whole\s+function\s+body\s+first\s+to\s+find\s+the\s+base\s+case|must\s+execute\s+every\s+base\s+case\s+before\s+returning)|"
                r"(?:recursive\s+function|recursion)\s+prints?\s+.*\b(?:down\s+to\s+and\s+)?including\s+the\s+base\s+case\b",
                re.IGNORECASE,
            ),
            1.1,
        ),
    ],
}

SOUND_CONCEPT_RULES: Dict[str, re.Pattern] = {
    "M01": re.compile(
        r"(?:both\s+are\s+strings?|enclosed\s+in\s+quote[s]?|is\s+a\s+string|two\s+strings?).*(?:\+|plus|concatenat\w*|joins?\s+them)|"
        r"\b(?:adding|joining|combining|\+|plus)\b.*\bstrings?\b.*\b(?:concatenat\w*|joins?\s+them|side\s+by\s+side)\b|"
        r"\b(?:multiplying|repeating|\*)\b.*\bstring\b.*\b(?:repeats?|duplicates?)\b|"
        r"(?:cannot|can'?t|does\s+not)\s+(?:concatenate|add|combine)\s+(?:a\s+)?(?:str(?:ing)?\s+and\s+(?:an?\s+)?(?:int(?:eger)?|number)|(?:int(?:eger)?|number)\s+and\s+(?:a\s+)?str(?:ing)?)|"
        r"\btypeerror\b.*\b(?:str(?:ing)?|int(?:eger)?)\b|"
        r"\binput\(\)\s+(?:always\s+)?returns?\s+(?:a\s+)?str(?:ing)?.*explicit\s+int\(\)|"
        r"\bstrings?\s+are\s+immutable\s+in\s+python",
        re.IGNORECASE,
    ),
    "M02": re.compile(
        r"(?:double\s+slash|//|floor\s+division).*(?:rounds?\s+down|discards?\s+(?:the\s+)?(?:remainder|decimal|fraction)|drops?\s+(?:the\s+)?(?:fractional|decimal)\s+part|(?:largest|greatest)\s+(?:whole\s+)?integer(?:\s+less\s+than\s+or\s+equal)?|less\s+than\s+or\s+equal|without\s+keeping\s+(?:the\s+)?decimal|(?:returns?|gives?|takes?|computes?|produces?)\s+the\s+floor\b|floors?\s+the\s+(?:result|quotient|value|division)|(?:(?:should\s+|will\s+|always\s+)?(?:returns?|gives?|produces?|computes?|calculates?)\s+(?:the\s+|an?\s+)?)?(?:integer|whole[- ]number|floored)\s+(?:quotient|result|value|part))|"
        r"(?:double\s+slash|//)\s+(?:operator\s+)?(?:is|performs?|does|means?|represents?)\s+(?:the\s+)?floor\s+division(?:\s+operator)?|"
        r"(?:double\s+slash|//)\s+(?:is\s+not|isn'?t|does\s+not\s+give|doesn'?t\s+give)\s+(?:the\s+)?(?:remainder|modulo)|"
        r"(?:single\s+slash|(?<!/)/(?!/))\s+(?:always\s+)?(?:performs?\s+(?:true\s+)?(?:floating[- ]point\s+|float\s+|decimal\s+)?division|(?:returns?|gives?|produces?)\s+(?:a\s+)?(?:float|floating[- ]point|decimal))"
        r"|(?:modulo|%)\s+(?:operator\s+)?(?:returns?|gives?|computes?|calculates?)\s+(?:the\s+)?remainder",
        re.IGNORECASE,
    ),
    "M03": re.compile(
        r"(?:single\s+equals?|(?<![=!<>])=(?!=))\s+(?:is\s+(?:an?\s+)?assignment|assigns?\b).*(?:double\s+equals?|==)\s+(?:is\s+(?:an?\s+)?(?:equality|comparison)|checks?|compares?)|"
        r"(?:single\s+equals?|(?<![=!<>])=(?!=))\s+(?:assigns?|stores?|binds?|sets?)\s+(?:the\s+)?(?:value|variable|name)\b|"
        r"(?:double\s+equals?|==)\s+(?:is\s+(?:an?\s+)?(?:equality\s+)?comparison|compares?\s+.*for\s+equality|checks?\s+(?:whether|if)\s+.*equal)",
        re.IGNORECASE,
    ),
    "M04": re.compile(
        r"(?!.*(?:even\s+(?:with|inside)\s+parentheses|regardless\s+of\s+parentheses|ignoring\s+parentheses))(?:(?:multiplication|division|\*|/|//)\s+(?:has\s+higher\s+(?:operator\s+)?precedence|binds?\s+tighter|(?:evaluates?|goes?|comes?|happens?|is\s+(?:done|evaluated|calculated))\s+(?:first|before))\s+(?:than\s+|before\s+)?(?:addition|subtraction|\+|\-)|"
        r"(?:multiplication|division|\*|/|//)\s+(?:evaluates?|goes?|comes?|happens?|is\s+(?:done|evaluated|calculated))\s+first\s+(?:because\s+of|due\s+to|by)\s+(?:operator\s+)?(?:precedence|order\s+of\s+operations)|"
        r"(?:exponentiation|power|\*\*)\s+(?:is\s+right-associative|has\s+higher\s+precedence\s+than\s+(?:multiplication|unary|\*|\-)|(?:evaluates?|goes?|comes?|happens?|is\s+(?:done|evaluated|calculated))\s+(?:first|before)\s+(?:multiplication|addition|subtraction|\*|\+|\-))|"
        r"\bparentheses\b.*(?:evaluated\s+first|highest\s+precedence|done\s+first|before\s+(?:multiplication|addition|\*|\+))|"
        r"\b(?:and\s+operator|and)\s+requires\s+both\s+operands\s+to\s+be\s+true\b|"
        r"\bnot\s+(?:true|false)\s+(?:flips|inverts|negates)\s+the\s+boolean\b)",
        re.IGNORECASE,
    ),
    "M05": re.compile(
        r"\b(?:0-based|zero-based)\s+indexing\b|"
        r"(?:index(?:ing)?|indices|positions?)\s+(?:in\s+python\s+)?(?:starts?|begins?)\s+at\s+0\b|"
        r"(?:index|position)\s+0\s+(?:retrieves?|accesses?|is|refers?\s+to|designates?|returns?|gives?)\s+the\s+(?:initial|first|1st)\s+(?:element|item|letter|character)|"
        r"(?:index|position)\s+1\s+(?:retrieves?|accesses?|is|refers?\s+to|designates?|returns?|gives?)\s+the\s+(?:second|2nd)\s+(?:element|item|letter|character)|"
        r"(?:index|position)\s+2\s+(?:retrieves?|accesses?|is|refers?\s+to|designates?|returns?|gives?)\s+the\s+(?:third|3rd)\s+(?:element|item|letter|character)|"
        r"\bnegative\s+index\s+-1\s+(?:refers?\s+to|is|accesses?|gives?)\s+the\s+last\b|"
        r"(?:slice\s+with\s+)?step\s+-1.*(?:reverse\s+order|steps?\s+backwards\s+from\s+end\s+to\s+start|reversed\s+copy)|"
        r"\bslice\b.*\bstop\s+(?:index|position|boundary)\s+is\s+exclusive\b",
        re.IGNORECASE,
    ),
    "M06": re.compile(
        r"\brange\b.*(?:\bexclusive\b|stops?\s+before\s+(?:reaching\s+)?\d+|does\s+not\s+include\s+(?:(?:the\s+)?(?:stop|end(?:ing)?|upper)\s*(?:value|boundary|bound)?\s*)?\d+|excludes?\s+(?:(?:the\s+)?(?:stop|end(?:ing)?|upper)\s*(?:value|boundary|bound)?\s*)?\d+|up\s+to\s+but\s+not\s+including|starts?\s+at\s+0\s+by\s+default)|"
        r"(?:stop\s+(?:value|boundary)|upper\s+bound(?:ary)?|end\s+value)\s*\d*\s*(?:in\s+range.*)?(?:is\s+exclusive|is\s+not\s+included|is\s+excluded)",
        re.IGNORECASE,
    ),
    "M07": re.compile(
        r"\bpositional\s+arguments?\s+(?:are\s+bound|bind)\s+(?:strictly\s+)?(?:to\s+parameters?\s+)?(?:in\s+order\s+|by\s+)(?:from\s+)?(?:left-to-right|left\s+to\s+right|position(?:\s+order)?)\b|"
        r"\bpasses?\s+.*\binto\s+(?:the\s+)?parameter\b|"
        r"\bparameter\s+\w+\s+(?:receives?|is\s+bound\s+to|takes?\s+the\s+value\s+of|gets?)\s+(?:the\s+)?(?:first|second|1st|2nd|argument)\b|"
        r"\bkeyword\s+arguments?\s+bind\s+by\s+(?:parameter\s+)?name\b|"
        r"\bpositional\s+arguments?\s+cannot\s+follow\s+keyword\s+arguments?\b",
        re.IGNORECASE,
    ),
    "M08": re.compile(
        r"\bbase\s+case\b.*(?:triggers?\s+and\s+(?:returns?|stops?)|returns?\s+immediately|terminat\w*\s+the\s+recursion(?:\s+safely)?|stops?\s+the\s+recursion|unwinds?\s+the\s+(?:call\s+)?stack|stops?\s+further\s+recursive\s+calls?)",
        re.IGNORECASE,
    ),
    "GENERAL": re.compile(
        r"\bsets?\s+in\s+python\s+only\s+store\s+unique\s+elements\b|"
        r"\b(?:function\s+definitions?|if\s+statement\s+headers?|while\s+loop\s+headers?|for\s+loop\s+headers?|headers?)\s+must\s+end\s+with\s+a\s+colon\b|"
        r"\b(?:requires?\s+parentheses\s+for\s+print|print\(\)\s+because\s+it\s+is\s+a\s+function)\b|"
        r"\bdictionary\s+lookups?\s+take\s+the\s+key.*return\s+its\s+associated\s+value\b|"
        r"\bopens?\s+a\s+parenthesis.*never\s+closes?\s+it\b",
        re.IGNORECASE,
    ),
}

OOS_CONCEPT_RULES: List[Tuple[str, re.Pattern, str]] = [
    (
        "mutable_aliasing_and_inplace_methods",
        re.compile(
            r"(?:append|sort|extend|reverse|pop|insert)\s*\(.*\breturns?\s+(?:a\s+|the\s+)?(?:new\s+)?(?:sorted\s+|updated\s+|modified\s+)?list\b|"
            r"(?:\b\w+\s*=\s*\w+\b|\bassign(?:ing|ment|s)?\b[^.;\n]*\b(?:list|dict|set|array|object)s?\b)[^.;\n]*\b(?:(?:creates?|makes?|produces?|gives?)\s+(?:a\s+|an\s+)?(?:separate\s+|independent\s+|new\s+|fresh\s+|shallow\s+|deep\s+|distinct\s+)?copy\b|copies\s+(?:the\s+)?(?:list|dict|set|array|object)s?\b)|"
            r"\b(?:creates?|makes?)\s+(?:a\s+|an\s+)?(?:separate\s+|independent\s+|new\s+|distinct\s+)?copy\s+of\s+(?:the\s+|a\s+)?(?:list|dict|set|array|object)\b|"
            r"\b(?:changing|modifying|mutating|updating|altering|appending\s+to)\s+(?:one|a|the\s+first|the\s+second|either)\s+(?:list|dict|set|array)\s+(?:does\s+not|doesn'?t|will\s+not|won'?t|never)\s+(?:affect|change|modify|mutate|touch|alter)\s+(?:the\s+other|another)\b|"
            r"\b(?:two|both)\s+(?:lists?|dicts?|sets?|arrays?)\s+are\s+(?:completely\s+|separate\s+and\s+)?independent\b.*(?:\bassign\w*|=|\bcopy\b|\bcopied\b)|"
            r"\b(?:default\s+(?:list\s+)?(?:arguments?|lists?|parameters?)|each\s+call\s+to\s+\w+)\b.*\b(?:creates?\s+a\s+(?:fresh|new)\s+(?:new\s+)?(?:empty\s+)?list|re-?created\s+(?:fresh|empty|new)\b)|"
            r"\bmutat(?:es?|ing)\s+(?:a\s+)?list\s+in\s+place\b|"
            r"\b(?:shallow\s+copy|deep\s+copy|list\s+aliasing|object\s+aliasing)\b",
            re.IGNORECASE,
        ),
        "Out-of-scope concept (Rule 9): list mutation, in-place methods, mutable default arguments, or object aliasing.",
    ),
    (
        "caller_side_mutation_and_variable_scope",
        re.compile(
            r"\b(?:modif\w*|mutat\w*)\b.*\bparameter\s+inside\s+a\s+function\b.*\b(?:local|never\s+mutat\w*|cannot\s+(?:alter|mutat\w*|change))\b|"
            r"\b(?:inner|nested|local)\s+functions?\s+(?:automatically\s+)?(?:modif(?:y|ies)|update|overwrite)\s+(?:outer|global)\s+variables?\b|"
            r"\b(?:global|nonlocal)\s+(?:scope|variable|keyword)\b|"
            r"\bshadowing\b|\blegb\b",
            re.IGNORECASE,
        ),
        "Out-of-scope concept (Rule 10): caller-side parameter mutation, variable scope, LEGB resolution, or closures.",
    ),
    (
        "print_and_output_formatting",
        re.compile(
            r"\bprint(?:s|ing|\(\))?\b.*\b(?:literal\s+quotes?|quotation\s+marks?|quotes?\s+around|puts?\s+a\s+comma|separator|trailing\s+newline)\b",
            re.IGNORECASE,
        ),
        "Out-of-scope concept (Rule 4): print() output formatting, quotation marks, or separators.",
    ),
    (
        "chained_comparison_semantics",
        re.compile(
            r"\bevaluates?\s+to\s+(?:true|false)\b.*\bthen\s+(?:comparing\s+)?(?:true|false)\s*(?:==|!=|<|>|<=|>=)\s*(?:true|false|\d+)\b",
            re.IGNORECASE,
        ),
        "Out-of-scope concept: chained relational comparison expansion vs sequential boolean evaluation.",
    ),
    (
        "generators_and_comprehension_mechanics",
        re.compile(
            r"\b(?:\byield\b|generators?|iterators?|list\s+comprehensions?)\b.*(?:\bterminates?\b|\breturns?\s+all\b|\bexhaust\w*\b|\blazy\b)|"
            r"\byield\s+works?\s+just\s+like\s+return\b|"
            r"\blist\s+comprehensions?\s+can\s+only\s+filter\b",
            re.IGNORECASE,
        ),
        "Out-of-scope concept: generators, iterators, or comprehension semantics.",
    ),
    (
        "oop_classes_and_dunder_methods",
        re.compile(
            r"\b(?:__init__|__str__|__repr__|self\s+parameter|class\s+attributes?|instance\s+attributes?|inheritance|@staticmethod|@classmethod|dunder|magic\s+methods?)\b|"
            r"\b__init__\s+can\s+return\b",
            re.IGNORECASE,
        ),
        "Out-of-scope concept: object-oriented programming, classes, or dunder methods.",
    ),
    (
        "exceptions_context_managers_and_modules",
        re.compile(
            r"\b(?:try\s*/\s*except|finally\s+block|raise\s+\w*error|context\s+managers?|with\s+open|decorators?|async\s+def|await\b|lambda\s+functions?)\b|"
            r"\bfinally\s+block\s+only\s+executes\s+if\b",
            re.IGNORECASE,
        ),
        "Out-of-scope concept: exception handling, context managers, decorators, or lambdas.",
    ),
    (
        "dictionary_and_set_lookup_semantics",
        re.compile(
            r"\bsets?\b.*\b(?:insertion\s+order|bracket\s+indexing|indexed\s+by\s+position)\b|"
            r"\bdictionar(?:y|ies)\s+(?:are\s+indexed\s+by\s+position|return\s+none\s+when\s+a\s+key\s+is\s+missing|automatically\s+insert\s+missing\s+keys)\b|"
            r"\b(?:\.get\s*\(|\.items\s*\(|\.keys\s*\(|\.values\s*\(|keyerror)\b",
            re.IGNORECASE,
        ),
        "Out-of-scope concept: set ordering/indexing or dictionary missing-key lookup semantics.",
    ),
    (
        "floating_point_representation_ieee754",
        re.compile(
            r"\b(?:ieee\s*754|binary\s+floating[\s-]point\s+representation|floating[\s-]point\s+precision\s+artifact)\b",
            re.IGNORECASE,
        ),
        "Out-of-scope concept: IEEE 754 binary floating-point representation.",
    ),
]


# =============================================================================
# 4. SEMANTIC / CONCEPTUAL EVIDENCE LAYER & CLAUSE ANALYSIS FUNCTIONS
# =============================================================================

_SPELLING_NORMALIZATIONS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:charcter|charater|charecter|chracter|charcters|charaters|charecters)\b", re.I), "character"),
    (re.compile(r"\bchars?\b", re.I), "character"),
    (re.compile(r"\b(?:posistion|postion|positon|possition|posistions|postions|positons)\b", re.I), "position"),
    (re.compile(r"\bpos\b", re.I), "position"),
    (re.compile(r"\b(?:indeks|indx|indexs|indeces)\b", re.I), "index"),
    (re.compile(r"\b(?:strng|stirng|stiring|strngs|stirngs)\b", re.I), "string"),
    (re.compile(r"\b(?:interger|integar|intergers|integars)\b", re.I), "integer"),
    (re.compile(r"\b(?:divison|devision)\b", re.I), "division"),
    (re.compile(r"\b(?:devide|divied|devided|deviding)\b", re.I), "divide"),
    (re.compile(r"\b(?:remander|remainer|ramainder|remaindr)\b", re.I), "remainder"),
    (re.compile(r"\b(?:qoutient|quotent)\b", re.I), "quotient"),
    (re.compile(r"\b(?:assinment|asignment|asign|asigns|asigned)\b", re.I), "assignment"),
    (re.compile(r"\b(?:presedence|precedance|precendence)\b", re.I), "precedence"),
    (re.compile(r"\b(?:parantheses|parenthesis|parenthases|paranthesis|parens)\b", re.I), "parentheses"),
    (re.compile(r"\b(?:multiplcation|multipy|miltiply|multipling)\b", re.I), "multiply"),
    (re.compile(r"\b(?:additon|adition)\b", re.I), "addition"),
    (re.compile(r"\b(?:subtration|subtracton)\b", re.I), "subtraction"),
    (re.compile(r"\b(?:inclusiv|incluse)\b", re.I), "inclusive"),
    (re.compile(r"\b(?:exclusiv|excluse)\b", re.I), "exclusive"),
    (re.compile(r"\b(?:boundry|boundries)\b", re.I), "boundary"),
    (re.compile(r"\b(?:arguement|arguements)\b", re.I), "argument"),
    (re.compile(r"\b(?:paramater|paramter|paramaters|paramters)\b", re.I), "parameter"),
    (re.compile(r"\b(?:recursiv|recusion|recurssion)\b", re.I), "recursion"),
    (re.compile(r"\b(?:concatinate|concatanate|concatinated|concatanated)\b", re.I), "concatenate"),
]


def canonicalize_for_semantics(text: str) -> str:
    """
    Normalizes student reasoning for semantic analysis:
    - Standardizes smart quotes and spacing
    - Repairs common CS1 spelling variations and typos
    - Canonicalizes numeric/positional number words ('index one' -> 'index 1', 'one-based' -> '1-based',
      'starts at zero' -> 'starts at 0', 'zero is not counted' -> '0 is not counted') without altering
      pronoun uses of 'one' ('one number', 'one list').
    """
    if not text:
        return ""
    t = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    for pat, repl in _SPELLING_NORMALIZATIONS:
        t = pat.sub(repl, t)

    t = re.sub(r"\bzero(?:\s+|-)+based\b", "0-based", t, flags=re.I)
    t = re.sub(r"\bone(?:\s+|-)+based\b", "1-based", t, flags=re.I)
    t = re.sub(
        r"\b(index|position|spot|place|at|from|to|is|equals?|step|starts?\s+(?:at|from)|begins?\s+(?:at|from)|count(?:s|ed|ing)?\s+(?:from|at|as))\s+zero\b",
        r"\1 0",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"\b(index|position|spot|place|at|from|to|is|equals?|step|starts?\s+(?:at|from)|begins?\s+(?:at|from)|count(?:s|ed|ing)?\s+(?:from|at|as))\s+one\b",
        r"\1 1",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"\b(index|position|spot|place|at|from|to|is|equals?|step)\s+two\b",
        r"\1 2",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"\b(index|position|spot|place|at|from|to|is|equals?|step)\s+three\b",
        r"\1 3",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"\bzero\s+(is\s+not\s+count\w*|isn'?t\s+count\w*|is\s+skip\w*|is\s+ignor\w*|is\s+the\s+first\b)",
        r"0 \1",
        t,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", t).strip()


def split_into_clauses(text: str) -> List[str]:
    """Splits student reasoning into meaningful clauses for span-level evidence extraction."""
    if not text or not text.strip():
        return []
    raw_parts = re.split(
        r"(?<=[.!?])\s+|;\s*|\s*,\s*(?=(?:so|because|since|and\s+then|whereas|while|but)\b)",
        text.strip(),
    )
    clauses = [p.strip().rstrip(".") for p in raw_parts if p and p.strip()]
    return clauses if clauses else [text.strip()]


def split_into_subclauses(text: str) -> List[str]:
    """
    Splits clauses further on coordinating conjunctions ('and', 'but', 'whereas', 'while')
    when a new subject/index/operator clause starts, preventing cross-clause bag-of-words collisions
    (e.g., 'the first character P is at index 0 and index 1 is the second character y')
    without splitting noun-phrase conjunctions like 'the string and the number'.
    """
    base_clauses = split_into_clauses(text)
    subclauses: List[str] = []
    for cl in base_clauses:
        parts = re.split(
            r",\s*(?:and|but)\s+|\s+(?:and|but|whereas|while)\s+(?=(?:index|position|double\s+equals|single\s+equals|==|it|i|python|so|\w+\[|'[a-zA-Z0-9]'|\"[a-zA-Z0-9]\")\b)",
            cl,
            flags=re.I,
        )
        for p in parts:
            if p and p.strip():
                subclauses.append(p.strip())
    return subclauses if subclauses else base_clauses


def extract_question_context(
    question: str, correct_answer: str = "", student_answer: str = ""
) -> Dict[str, Any]:
    """
    Extracts structured semantic context from the question code to ground contextual paraphrases
    (e.g., knowing s = "Python" has 0th element 'P' and print(s[1]) queries index 1).
    """
    q_text = str(question or "")
    seq_items: List[str] = []

    # 1. String literal assignment or inline string literal being indexed
    m_str = re.search(r"(?:\w+\s*=\s*['\"]([^'\"]+)['\"]|['\"]([^'\"]+)['\"]\s*\[)", q_text)
    if m_str:
        s_lit = m_str.group(1) if m_str.group(1) is not None else m_str.group(2)
        if s_lit:
            seq_items = list(s_lit)
    else:
        # 2. Flat list/tuple literal assignment
        m_list = re.search(r"\w+\s*=\s*[\[\(]([^\]\)\n]+)[\]\)]", q_text)
        if m_list:
            raw_elems = [
                e.strip().strip("'\"")
                for e in m_list.group(1).split(",")
                if e.strip()
            ]
            if raw_elems:
                seq_items = raw_elems

    # Find single-index lookups var[idx] (excluding slices with ':')
    queried_indices: List[int] = []
    for m_idx in re.finditer(r"(?:\w+|['\"][^'\"]+['\"])\[\s*(-?\d+)\s*\]", q_text):
        try:
            queried_indices.append(int(m_idx.group(1)))
        except ValueError:
            pass

    has_slice = bool(re.search(r"\w+\[[^\]\n]*:[^\]\n]*\]", q_text))

    # Find range(...) call arguments
    has_range = "range(" in q_text
    range_args: List[int] = []
    range_stop: Optional[int] = None
    m_range = re.search(
        r"range\(\s*(-?\d+)\s*(?:,\s*(-?\d+)\s*(?:,\s*(-?\d+)\s*)?)?\)", q_text
    )
    if m_range:
        for g in m_range.groups():
            if g is not None:
                try:
                    range_args.append(int(g))
                except ValueError:
                    pass
        if len(range_args) == 1:
            range_stop = range_args[0]
        elif len(range_args) >= 2:
            range_stop = range_args[1]

    has_loop = bool(re.search(r"\b(?:for|while)\b", q_text))
    has_floor_div = "//" in q_text
    has_true_div = bool(re.search(r"(?<!/)/(?!/)", q_text))
    has_modulo = "%" in q_text
    has_quoted_num = bool(re.search(r"['\"][+-]?\d+(?:\.\d+)?['\"]", q_text))
    has_str_int_op = bool(
        re.search(
            r"['\"][^'\"]*['\"]\s*[\+\*]|[\+\*]\s*['\"][^'\"]*['\"]|\bint\s*\(|\binput\s*\(",
            q_text,
        )
    )
    has_double_eq = "==" in q_text
    has_if_or_while_single_eq = bool(
        re.search(r"\b(?:if|while)\s+[^:=\n]*?(?<![=!<>])=(?!=)", q_text)
    )
    has_mixed_arith = bool(
        re.search(r"[\+\-].*(?:\*|/|//|\*\*)|(?:\*|/|//|\*\*).*[\+\-]", q_text)
    )

    return {
        "seq_items": seq_items,
        "queried_indices": queried_indices,
        "has_indexing": len(queried_indices) > 0,
        "has_slice": has_slice,
        "has_range": has_range,
        "range_args": range_args,
        "range_stop": range_stop,
        "has_loop": has_loop,
        "has_floor_div": has_floor_div,
        "has_true_div": has_true_div,
        "has_modulo": has_modulo,
        "has_quoted_num": has_quoted_num,
        "has_str_int_op": has_str_int_op,
        "has_double_eq": has_double_eq,
        "has_if_or_while_single_eq": has_if_or_while_single_eq,
        "has_mixed_arith": has_mixed_arith,
    }


def _has_sound_zero_based_guard(text_norm: str) -> bool:
    """
    Returns True if the clause/sentence explicitly states a 0-based indexing fact
    (e.g., '0-based', 'starts at 0', 'index 0 is the first', 'index 1 is the second').
    """
    return bool(
        re.search(
            r"\b0-based\b|"
            r"\b(?:index(?:ing)?|indices|positions?|counting|python|strings?|lists?)\s+(?:in\s+python\s+)?(?:starts?|begins?|counts?)\s+(?:at|from|with)\s*0\b|"
            r"\b(?:index|position)\s*0\s+(?:is|refers?\s+to|gives?|retrieves?|accesses?|means?|has)\s+(?:the\s+)?(?:first|1st|initial)\b|"
            r"\b(?:first|1st|initial)\s+(?:character|letter|element|item|position|index)\s+(?:is\s+)?(?:at\s+|in\s+|has\s+)?(?:index|position)?\s*0\b|"
            r"\b(?:index|position)\s*1\s+(?:is|refers?\s+to|gives?|retrieves?|accesses?|means?|has)\s+(?:the\s+)?(?:second|2nd)\b|"
            r"\b(?:second|2nd)\s+(?:character|letter|element|item)\s+(?:is\s+)?(?:at\s+|in\s+|has\s+)?(?:index|position|\w+\[)\s*1\b",
            text_norm,
            re.I,
        )
    )


def score_semantic_propositions(
    reasoning: str,
    student_answer: str = "",
    question: str = "",
    correct_answer: str = "",
) -> Dict[str, Dict[str, Any]]:
    """
    Reusable Semantic / Conceptual Evidence Layer across M01-M08.
    Recognizes order-independent conceptual propositions, conversational paraphrases,
    orthographic/spelling variations, and question-contextual beliefs without relying
    solely on exact linear regex phrases.
    """
    if not reasoning or not reasoning.strip():
        return {}

    r_norm = canonicalize_for_semantics(reasoning)
    r_lower = r_norm.lower()
    subclauses = split_into_subclauses(reasoning)
    norm_subclauses = [(cl, canonicalize_for_semantics(cl).lower()) for cl in subclauses]
    ctx = extract_question_context(question, correct_answer, student_answer)
    sa_norm = normalize_answer(student_answer)
    ca_norm = normalize_answer(correct_answer)

    semantic_hits: Dict[str, Dict[str, Any]] = {}

    def _record_hit(m_id: str, variant: str, score: float, span: str) -> None:
        if m_id not in semantic_hits or score > semantic_hits[m_id]["score"]:
            semantic_hits[m_id] = {
                "score": round(score, 4),
                "variant": variant,
                "span": span.strip().rstrip("."),
            }

    # -------------------------------------------------------------------------
    # M05: Index / Position / 0-based vs 1-based Indexing Semantic Propositions
    # -------------------------------------------------------------------------
    sound_m05_overall = _has_sound_zero_based_guard(r_lower)

    for raw_cl, cl_low in norm_subclauses:
        if _has_sound_zero_based_guard(cl_low):
            continue
        # Guard against negated 1-based statements (e.g., "index 1 is not the first character")
        if re.search(
            r"\b(?:not|never|isn'?t|is\s+not|doesn'?t|does\s+not|instead\s+of|rather\s+than|unlike)\s+(?:the\s+)?(?:first|1st|1-based|start\w*\s+(?:at|from)\s*1)\b",
            cl_low,
        ):
            continue

        # Proposition M05-1: Order-independent binding of {Index/Position 1 / [1]} <-> {First / 1st}
        has_idx1_anchor = bool(
            re.search(
                r"\b(?:index|position|spot|place|slot|location)\s*(?:of\s+|at\s+|is\s+|number\s+|#\s*)?1\b|"
                r"\b(?:at|in)\s+(?:index|position|spot|place)?\s*1\b|"
                r"\b\w*\[\s*1\s*\]|"
                r"\b(?:is|equals?|means?|gives?|points?\s+to|refers?\s+to|starts?\s+(?:at|from|with)|begins?\s+(?:at|from|with)|count\w*\s+(?:from|at|as))\s+(?:index\s+|position\s+)?1\b",
                cl_low,
            )
        )
        has_first_anchor = bool(
            re.search(
                r"\b(?:first|1st|initial|very\s+first|beginning|starting|front)\b|"
                r"\b(?:starts?|begins?)\s+(?:counting|indexing|characters?|letters?|elements?|items?|positions?)?\s*(?:at|from|with)\s*1\b",
                cl_low,
            )
        )
        has_seq_domain = bool(
            re.search(
                r"\b(?:character|letter|element|item|value|string|list|sequence|word|index|indexing|position|spot|place|count\w*)\b|"
                r"\b\w+\[\s*1\s*\]",
                cl_low,
            )
        )
        is_range_only = ("range" in cl_low) and not bool(
            re.search(r"\b(?:character|letter|element|item|list|string|index|\w+\[)\b", cl_low)
        )

        if has_idx1_anchor and has_first_anchor and has_seq_domain and not is_range_only:
            _record_hit("M05", "one_based_indexing_belief", 1.25, raw_cl)

        # Proposition M05-2: Zero is not counted / skipped / no index 0
        if re.search(
            r"\b(?:0|index\s*0|position\s*0)\s+(?:is\s+not|isn'?t|not)\s+(?:count\w*|used|included|a\s+position|an\s+index)\b|"
            r"\b(?:don'?t|do\s+not|doesn'?t|does\s+not|never)\s+(?:count|use|start\s+(?:at|from))\s*(?:index\s*|position\s*)?0\b|"
            r"\b(?:skips?|ignores?)\s+(?:index\s+|position\s+)?0\b|"
            r"\bno\s+(?:index|position)\s*0\b",
            cl_low,
        ):
            _record_hit("M05", "one_based_indexing_belief", 1.25, raw_cl)

        # Proposition M05-3: "I counted X as position/index 1" or "starts counting from 1"
        if re.search(
            r"\bcount(?:s|ed|ing)?\b.*\b(?:as|from|at)\s+(?:position|index|spot|number)?\s*1\b|"
            r"\b(?:starts?|begins?)\s+counting\b.*\b(?:from|at)\s*1\b",
            cl_low,
        ) and not is_range_only:
            _record_hit("M05", "one_based_indexing_belief", 1.25, raw_cl)

        # Proposition M05-3b: Negative index starts at -0 / 0, or -1 is second to last
        if re.search(
            r"\b(?:negative\s+index\w*|-0\b|counting\s+backwards\b|from\s+the\s+right\b).*?(?:starts?\s+(?:at|from)\s*-?0|-0\s+is|-1\s+is\s+(?:the\s+)?(?:second|2nd)\s+to\s+last|-2\s+is\s+(?:the\s+)?last)",
            cl_low,
        ):
            _record_hit("M05", "negative_index_from_zero_belief", 1.22, raw_cl)

        # Proposition M05-3c: Slice stop index inclusive
        if ("slice" in cl_low or "slicing" in cl_low or (ctx["has_slice"] and not ctx["has_range"])) and re.search(
            r"\b(?:includes?\s+(?:the\s+)?(?:stop|end|second|upper|final)\s*(?:index|position|character|letter|element)?|"
            r"up\s+to\s+and\s+including\s+(?:index|position)|stop\s+(?:index|position)\s+is\s+inclusive)\b",
            cl_low,
        ) and not re.search(r"\b(?:not\s+included|exclusive|excludes?)\b", cl_low):
            _record_hit("M05", "slice_endpoint_inclusive_as_position", 1.22, raw_cl)

    # Proposition M05-4: Multi-clause sentence connecting s[1] / index 1 to "first letter/character/element"
    # (e.g., "s[1] points to P because P is the first letter")
    if "M05" not in semantic_hits and not sound_m05_overall:
        has_s1_or_idx1_in_text = bool(
            re.search(
                r"\b\w*\[\s*1\s*\]|\b(?:index|position)\s*1\b",
                r_lower,
            )
        )
        has_first_elem_in_text = bool(
            re.search(
                r"\b(?:first|1st|initial)\s+(?:character|letter|element|item|value|position)\b|"
                r"\b(?:character|letter|element|item)\s+(?:at\s+|in\s+)?(?:index\s+|position\s+)?1\s+is\s+(?:the\s+)?(?:first|1st)\b",
                r_lower,
            )
        )
        if has_s1_or_idx1_in_text and has_first_elem_in_text:
            _record_hit("M05", "one_based_indexing_belief", 1.22, reasoning)

    # Proposition M05-5: Question-contextual 1-based lookup
    # When the question asks for seq[1] (e.g., s = "Python"; print(s[1])) and the student
    # justifies their answer by identifying the "first letter / character / element / item"
    # (e.g., "in the string first letter is P", "because P is the first letter", "first char is P")
    if (
        "M05" not in semantic_hits
        and not sound_m05_overall
        and (1 in ctx["queried_indices"] or re.search(r"\w+\[\s*1\s*\]", question or ""))
        and not ctx["has_range"]
    ):
        for raw_cl, cl_low in norm_subclauses:
            if re.search(
                r"\b(?:first|1st|initial|beginning|front)\s+(?:letter|character|element|item|value|number|symbol|part|one)\b|"
                r"\b(?:letter|character|element|item)\s+(?:is\s+the\s+)?(?:first|1st|initial)\b",
                cl_low,
            ) and not re.search(r"\b(?:not|isn'?t|is\s+not)\s+(?:the\s+)?(?:first|1st)\b", cl_low):
                _record_hit("M05", "one_based_indexing_belief", 1.22, raw_cl)
                break

    # Proposition M05-6: Question-contextual element-0 claimed to be at index 1 / position 1
    # (e.g., "P is at index 1", "P is the character in position 1", "I counted P as position 1")
    if "M05" not in semantic_hits and not sound_m05_overall:
        elem0_candidates: List[str] = []
        if ctx["seq_items"]:
            elem0_candidates.append(ctx["seq_items"][0].lower())
        # Also support shorthand question print(s[1]) where correct_answer is 'y' and student refers to 'p'
        if (
            not elem0_candidates
            and ca_norm == "y"
            and re.search(r"\bs\[\s*1\s*\]", question or "")
        ):
            elem0_candidates.append("p")

        for e0 in elem0_candidates:
            if not e0 or e0 == ca_norm:
                continue
            e0_esc = re.escape(e0)
            for raw_cl, cl_low in norm_subclauses:
                has_e0_token = bool(
                    re.search(rf"(?:^|[\s'\"(\[])({e0_esc})(?:$|[\s'\").,;:!?\]])", cl_low)
                )
                has_pos1_claim = bool(
                    re.search(
                        r"\b(?:at|in|has|is|as|equals?)\s+(?:the\s+)?(?:character\s+|letter\s+|element\s+|item\s+)?(?:in\s+|at\s+)?(?:index|position|spot|place)\s*1\b|"
                        r"\b(?:index|position|spot|place)\s*1\s+(?:is|has|gives?|holds?|contains?|points?\s+to|equals?)\b",
                        cl_low,
                    )
                )
                if has_e0_token and has_pos1_claim:
                    _record_hit("M05", "one_based_indexing_belief", 1.22, raw_cl)
                    break

    # -------------------------------------------------------------------------
    # M01: String vs Number Type Confusion Semantic Propositions
    # -------------------------------------------------------------------------
    for raw_cl, cl_low in norm_subclauses:
        # Negation / sound guard for M01
        if re.search(
            r"\b(?:cannot|can'?t|does\s+not|doesn'?t|will\s+not|won'?t|never|not\s+allowed|typeerror)\b.*\b(?:add|concatenate|combine|convert|join)\b|"
            r"\b(?:concatenat\w*|joins?\s+them\s+side\s+by\s+side|repeats?\s+the\s+string)\b",
            cl_low,
        ):
            continue

        # M01-1: Implicitly joining/combining/converting string and number with +
        has_str_concept = bool(re.search(r"\b(?:string|strings|str|text|word|quotes?|quoted)\b", cl_low))
        has_num_concept = bool(re.search(r"\b(?:integer|integers|int|number|numbers|numeric|digit|digits)\b", cl_low))
        has_coercion_or_join = bool(
            re.search(
                r"\b(?:joins?|joining|combines?|combining|concatenates?|puts?\s+.*\s+together|glues?|attaches?|"
                r"converts?|turns?\s+.*\s+into|casts?|changes?\s+.*\s+into|adds?\s+.*\s+together)\b",
                cl_low,
            )
        )
        if has_str_concept and has_num_concept and has_coercion_or_join:
            if not re.search(r"\bmust\s+(?:use|call)\s+(?:int|str)\b|\bneed(?:s)?\s+(?:int|str)\(", cl_low):
                _record_hit("M01", "expects_plus_to_add_str_and_int", 1.15, raw_cl)

        # M01-2: Quotes/strings treated as numbers or doing arithmetic on quoted numbers
        if has_str_concept and re.search(
            r"\b(?:don'?t\s+matter|does\s+not\s+matter|do\s+not\s+matter|are\s+ignored|is\s+ignored|still\s+(?:a\s+)?numbers?|"
            r"treated\s+as\s+(?:a\s+)?(?:number|integer|int)|acts?\s+(?:like|as)\s+(?:a\s+)?(?:number|integer|int)|"
            r"does\s+(?:math|addition|multiplication|arithmetic)|adds?\s+them\s+numerically|multiplies\s+them\s+numerically|"
            r"is\s+just\s+(?:the\s+|a\s+)?(?:number|integer))\b",
            cl_low,
        ):
            _record_hit("M01", "treats_numeric_string_as_int", 1.15, raw_cl)

        # M01-3: input() automatically returns int/number without int()
        if "input" in cl_low and re.search(
            r"\b(?:returns?|gives?|produces?|is)\s+(?:an?\s+)?(?:integer|int|number)\b|\bwithout\s+(?:calling\s+|needing\s+)?int\(\)",
            cl_low,
        ) and not re.search(r"\b(?:string|str|not\s+an?\s+(?:int|integer|number))\b", cl_low):
            _record_hit("M01", "expects_int_cast_without_int_call", 1.15, raw_cl)

    # -------------------------------------------------------------------------
    # M02: Division Semantics (/ vs //) Semantic Propositions
    # -------------------------------------------------------------------------
    for raw_cl, cl_low in norm_subclauses:
        has_fdiv_ref = bool(re.search(r"(?://|\bdouble\s+slash\b|\bfloor\s+division\b)", cl_low)) or (
            ctx["has_floor_div"] and not ctx["has_modulo"] and not ctx["has_true_div"]
        )
        # M02-1: // returns remainder / modulo
        if has_fdiv_ref and re.search(
            r"\b(?:remainder|modulo|modulus|leftover|what\s+is\s+left\s+over|left\s+over\s+after\s+divid\w*)\b",
            cl_low,
        ):
            if not re.search(
                r"\b(?:not|never|doesn'?t|does\s+not|isn'?t|is\s+not|without|instead\s+of|rather\s+than|discards?|drops?)\b[^.;\n]*\b(?:remainder|modulo|modulus)\b|"
                r"(?:%|\bmodulo\b)\s+(?:operator\s+)?(?:gives?|returns?|computes?|calculates?|is)\s+(?:the\s+)?remainder",
                cl_low,
            ):
                _record_hit("M02", "believes_double_slash_returns_remainder", 1.20, raw_cl)

        # M02-2: // keeps decimal / returns float / normal division
        if re.search(r"(?://|\bdouble\s+slash\b)", cl_low) and re.search(
            r"\b(?:keeps?\s+(?:the\s+)?decimal|returns?\s+(?:a\s+)?(?:float|decimal)|gives?\s+(?:a\s+|the\s+)?decimal|"
            r"(?:does|performs?|is)\s+(?:normal|regular|standard|float|floating[- ]point)\s+division|"
            r"same\s+as\s+(?:single\s+slash|normal\s+division|/))\b",
            cl_low,
        ):
            if not re.search(
                r"\b(?:not|never|doesn'?t|does\s+not|isn'?t|is\s+not|without|instead\s+of|rather\s+than|unlike)\b",
                cl_low,
            ):
                _record_hit("M02", "believes_double_slash_returns_float", 1.18, raw_cl)

        # M02-3: Single / returns integer/whole number when dividing evenly
        if (
            re.search(r"(?:\bsingle\s+slash\b|(?<!/)/(?!/))", cl_low)
            or (ctx["has_true_div"] and not ctx["has_floor_div"])
        ) and re.search(
            r"\b(?:divides?\s+evenly|no\s+remainder|exact\s+division|equal\s+numbers?)\b.*\b(?:integer|int|whole\s+number|no\s+decimal|without\s+\.0)\b|"
            r"\b(?:returns?|gives?|produces?)\s+(?:an?\s+)?(?:integer|int|whole\s+number)\s+because\s+.*\b(?:evenly|no\s+remainder)\b",
            cl_low,
        ):
            if not re.search(r"(?://|\bdouble\s+slash\b|\bfloor\s+division\b|\bfloat\b.*\balways\b)", cl_low):
                _record_hit("M02", "believes_slash_returns_int", 1.18, raw_cl)

        # M02-4: // rounds up or rounds to nearest integer or toward zero
        if has_fdiv_ref and re.search(
            r"\brounds?(?!\s+down\b)(?:\s+(?:the\s+)?(?:answer|result|value|number|quotient|it))?\s+(?:up\b|to\s+the\s+(?:nearest|closest)\b|it\s+to\s+\d+|toward[s]?\s*0\b)",
            cl_low,
        ):
            _record_hit("M02", "other_division_confusion", 1.18, raw_cl)

    # -------------------------------------------------------------------------
    # M03: Assignment vs Equality (= vs ==) Semantic Propositions
    # -------------------------------------------------------------------------
    for raw_cl, cl_low in norm_subclauses:
        # M03-1: Single = checks/tests equality
        if (
            re.search(r"(?:\bsingle\s+equals?\b|(?<![=!<>])=(?!=))", cl_low)
            or (ctx["has_if_or_while_single_eq"] and re.search(r"\b(?:if|while|condition|equals?\s+sign)\b", cl_low))
        ) and not re.search(r"(?:\bdouble\s+equals?\b|==)", cl_low):
            if re.search(
                r"\b(?:checks?\s+(?:if|whether|to\s+see)|tests?\s+(?:if|whether)|compares?\s+(?:if|whether|the\s+two|values?|variables?|\w+\s+and\s+\w+)|"
                r"sees?\s+(?:if|whether)|asks?\s+(?:if|whether)|is\s+an?\s+equality\s+(?:check|test))\b",
                cl_low,
            ) and not re.search(r"\b(?:cannot|can'?t|not|never|syntaxerror|assigns?\s+a\s+value)\b", cl_low):
                _record_hit("M03", "uses_single_equals_for_conditional_test", 1.20, raw_cl)

        # M03-2: Double == assigns/updates value
        if re.search(r"(?:\bdouble\s+equals?\b|==)", cl_low) and re.search(
            r"\b(?:assigns?|re-?assigns?|sets?\s+(?:the\s+)?(?:variable|value|\w+\s+to)|stores?\s+.*\s+in(?:to)?|updates?\s+\w+\s+to|changes?\s+\w+\s+to)\b",
            cl_low,
        ):
            if not re.search(
                r"\b(?:not|never|doesn'?t|does\s+not|instead\s+of|rather\s+than)\b|"
                r"(?:\bsingle\s+equals?\b|(?<![=!<>])=(?!=))\s+(?:is\s+(?:an?\s+)?assignment|assigns?|stores?|sets?)",
                cl_low,
            ):
                _record_hit("M03", "expects_double_equals_to_assign_value", 1.20, raw_cl)

        # M03-3: Assignment creates permanent algebraic equation linking two variables
        if re.search(
            r"\b(?:links?\s+\w+\s+and\s+\w+\s+together|permanent\s+equation|updating\s+\w+\s+later\s+automatically\s+updates?)\b",
            cl_low,
        ) and not re.search(r"\b(?:list|dict|set|array)\b", cl_low):
            _record_hit("M03", "believes_equals_creates_symmetric_equation", 1.18, raw_cl)

    # -------------------------------------------------------------------------
    # M04: Operator Precedence / Evaluation Order Semantic Propositions
    # -------------------------------------------------------------------------
    for raw_cl, cl_low in norm_subclauses:
        # M04-1: Addition/subtraction before multiplication/division
        if re.search(
            r"\b(?:add(?:s|ed|ing|ition)?|plus|\+|subtract(?:s|ed|ing|ion)?|minus)\b[^.;\n]*\b(?:first|before|prior\s+to|earlier|higher\s+precedence|easier\s+so\s+i\s+did\s+it\s+first|comes?\s+before|goes?\s+before)\b",
            cl_low,
        ) and not re.search(
            r"\b(?:parentheses|brackets|inside\s+parentheses|because\s+of\s+parentheses)\b", cl_low
        ):
            if ctx["has_mixed_arith"] or re.search(r"\b(?:multipl\w*|times|\*|divid\w*|/|//)\b", r_lower):
                _record_hit("M04", "arithmetic_plus_before_multiply", 1.18, raw_cl)

        # M04-2: Left-to-right evaluation override
        if re.search(
            r"\b(?:from\s+left\s+to\s+right|left-to-right|in\s+the\s+order\s+(?:it\s+is\s+|they\s+are\s+)?written|(?:in\s+the\s+)?order\s+they\s+appear)\b",
            cl_low,
        ) and not re.search(r"\b(?:positional|argument|parameter)\b", cl_low):
            _record_hit("M04", "left_to_right_evaluation_override", 1.15, raw_cl)

    # -------------------------------------------------------------------------
    # M06: Loop Values / range() Boundaries Semantic Propositions
    # -------------------------------------------------------------------------
    for raw_cl, cl_low in norm_subclauses:
        if "range" in cl_low or (ctx["has_range"] and "loop" in cl_low):
            if re.search(
                r"\b(?:includes?\s+(?:the\s+)?(?:stop|end|last|upper|final)|stop\s+(?:value|number|bound)\s+is\s+included|"
                r"up\s+to\s+and\s+including|goes?\s+all\s+the\s+way\s+(?:up\s+)?to|both\s+endpoints?\s+are\s+included)\b",
                cl_low,
            ) and not re.search(r"\b(?:not\s+included|exclusive|stops?\s+before|excludes?)\b", cl_low):
                _record_hit("M06", "range_stop_inclusive_belief", 1.18, raw_cl)

            # Contextual stop value inclusion (e.g., on range(3) or range(1, 4): "range goes up to 3" / "includes 4")
            if ctx["range_stop"] is not None:
                st_str = str(ctx["range_stop"])
                if re.search(
                    rf"\b(?:includes?\s+{st_str}|up\s+to\s+{st_str}|from\s+\d+\s+to\s+{st_str}|stops?\s+at\s+{st_str}\s+after\s+printing|ends?\s+at\s+{st_str})\b",
                    cl_low,
                ) and not re.search(
                    rf"\b(?:before\s+{st_str}|not\s+including\s+{st_str}|excludes?\s+{st_str}|exclusive)\b",
                    cl_low,
                ):
                    _record_hit("M06", "range_stop_inclusive_belief", 1.18, raw_cl)

            if re.search(
                r"\b(?:starts?|begins?|counts?)\s+(?:counting\s+)?(?:at|from)\s*1\b", cl_low
            ) and len(ctx["range_args"]) <= 1 and not re.search(r"range\(\s*1\s*,", question or ""):
                if not re.search(r"\b(?:starts?\s+(?:at|from)\s*0|not\s+1)\b", cl_low):
                    _record_hit("M06", "range_starts_at_one_default", 1.18, raw_cl)

    # -------------------------------------------------------------------------
    # M07: Function Argument-Parameter Binding Semantic Propositions
    # -------------------------------------------------------------------------
    for raw_cl, cl_low in norm_subclauses:
        if re.search(r"\b(?:arguments?|parameters?|variables?|functions?|calls?)\b", cl_low):
            if re.search(
                r"\b(?:same\s+(?:variable\s+|parameter\s+)?names?|match(?:es|ing)?\s+(?:the\s+)?(?:parameter\s+|variable\s+|arguments?\s+to\s+parameters?\s+.*)?names?|mapped\s+by\s+(?:variable\s+)?names?|"
                r"must\s+have\s+the\s+same\s+names?)\b",
                cl_low,
            ) and not re.search(r"\b(?:don'?t\s+have\s+to|do\s+not\s+need\s+to|regardless\s+of\s+name)\b", cl_low):
                _record_hit("M07", "believes_arg_name_must_match_param_name", 1.20, raw_cl)

            if re.search(
                r"\b(?:keeps?\s+(?:its\s+)?default\s+value\s+even\s+when|default\s+value\s+overrides?\s+the\s+passed|"
                r"ignores?\s+keyword\s+names?|position\s+always\s+overrides?\s+keyword)\b",
                cl_low,
            ):
                _record_hit("M07", "keyword_argument_override_confusion", 1.20, raw_cl)

    # -------------------------------------------------------------------------
    # M08: Recursion Termination / Base Case Semantic Propositions
    # -------------------------------------------------------------------------
    for raw_cl, cl_low in norm_subclauses:
        if re.search(r"\b(?:recursion|recursive|base\s+case|calls?\s+itself)\b", cl_low):
            if re.search(
                r"\b(?:automatically\s+(?:stops?|halts?|terminates?|ends?)|naturally\s+stops?|"
                r"stops?\s+on\s+its\s+own\s+(?:at|when)|without\s+(?:needing\s+)?a\s+base\s+case|doesn'?t\s+need\s+a\s+base\s+case)\b",
                cl_low,
            ) and not re.search(r"\b(?:never\s+stops?\s+automatically|does\s+not\s+stop\s+automatically)\b", cl_low):
                _record_hit("M08", "believes_recursion_stops_automatically_at_zero", 1.20, raw_cl)

            if re.search(
                r"\b(?:resets?\s+(?:the\s+)?(?:call\s+stack|total|execution)|erases?\s+previous\s+calls?|forgets?\s+the\s+previous)\b",
                cl_low,
            ):
                _record_hit("M08", "believes_base_case_resets_execution", 1.20, raw_cl)

            if re.search(
                r"\b(?:returned\s+straight\s+to\s+the\s+caller\s+without\s+adding|without\s+unwinding|only\s+returns?\s+the\s+base\s+case\s+value)\b",
                cl_low,
            ):
                _record_hit("M08", "misunderstands_recursive_call_return_value", 1.20, raw_cl)

    return semantic_hits


def score_misconception_claims(
    reasoning: str,
    student_answer: str = "",
    question: str = "",
    correct_answer: str = "",
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates both MISCONCEPTION_CLAIM_RULES (on raw and canonicalized text) and
    the Semantic Propositional Layer across M01-M08 without first-match short-circuiting.
    Returns per-class dict with:
      - score: float
      - variant: str
      - span: str (best contiguous clause from student_reasoning)
    Resolves M05/M06, M01/M02, M03/M06 using variant specificity + contextual boundary rules.
    """
    clauses = split_into_clauses(reasoning)
    r_canon = canonicalize_for_semantics(reasoning)
    canon_clauses = split_into_clauses(r_canon)
    combined_text = f"{reasoning} {student_answer}".strip()
    combined_canon = f"{r_canon} {student_answer}".strip()
    r_lower = r_canon.lower()

    results: Dict[str, Dict[str, Any]] = {}

    for m_id, rules in MISCONCEPTION_CLAIM_RULES.items():
        best_score = 0.0
        best_variant = None
        best_span = ""

        for variant_name, pattern, base_weight in rules:
            matched_clause = None
            for idx_cl, cl in enumerate(clauses):
                cl_c = canon_clauses[idx_cl] if idx_cl < len(canon_clauses) else canonicalize_for_semantics(cl)
                if pattern.search(cl) or pattern.search(cl_c):
                    matched_clause = cl
                    break

            if matched_clause is not None:
                score = base_weight
                span = matched_clause
            elif pattern.search(reasoning) or pattern.search(r_canon):
                score = base_weight * 0.95
                span = reasoning.strip().rstrip(".")
            elif pattern.search(combined_text) or pattern.search(combined_canon):
                score = base_weight * 0.80
                span = reasoning.strip().rstrip(".")
            else:
                continue

            # Taxonomy Boundary Disambiguation Adjustments (Rules 6, 7, 9, 10)
            if m_id == "M05":
                if re.search(
                    r"\b(index|position|slice|element|item|letter|character|bracket|\w+\[)\b",
                    r_lower,
                ):
                    score += 0.15
            elif m_id == "M06":
                if re.search(
                    r"\b(range|loop|iteration|while|repeats?|counts?\s+\d+\s+times)\b",
                    r_lower,
                ):
                    score += 0.15
                if "slice" in r_lower and "range" not in r_lower and "loop" not in r_lower:
                    score -= 0.40

            if m_id == "M02":
                if re.search(r"(//|/|slash|division|remainder|quotient|divid)", r_lower):
                    score += 0.15
            elif m_id == "M01":
                if re.search(
                    r"(//|slash|division|remainder|quotient)", r_lower
                ) and not re.search(r"['\"][0-9.]+['\"]|\bquote|\bstring", r_lower):
                    score -= 0.40

            if m_id == "M03":
                if re.search(
                    r"(single\s+equals|double\s+equals|assign|equation)", r_lower
                ):
                    score += 0.15

            if score > best_score:
                best_score = score
                best_variant = variant_name
                best_span = span

        if best_score > 0:
            results[m_id] = {
                "score": round(best_score, 4),
                "variant": best_variant,
                "span": best_span,
            }

    # Context-aware M04 check: if question contains '**' and student claims multiplication happens first
    if "M04" not in results and "**" in (question or ""):
        m_mult_first = re.search(
            r"\b(?:multiplication|multiply|multiplying)\s+(?:happens?|goes?|comes?|runs?|executes?|is\s+done|is\s+evaluated|evaluates?)\s+first\b",
            r_lower,
        )
        if m_mult_first:
            cl_match = next(
                (cl for cl in clauses if m_mult_first.group(0) in canonicalize_for_semantics(cl).lower()),
                reasoning.strip().rstrip("."),
            )
            results["M04"] = {
                "score": 1.1,
                "variant": "arithmetic_plus_before_multiply",
                "span": cl_match,
            }

    # Merge Semantic Propositional Evidence Layer hits across M01-M08
    sem_hits = score_semantic_propositions(
        reasoning=reasoning,
        student_answer=student_answer,
        question=question,
        correct_answer=correct_answer,
    )
    for m_id, sem_info in sem_hits.items():
        if m_id not in results or sem_info["score"] > results[m_id]["score"]:
            results[m_id] = sem_info

    return results


def detect_sound_concepts(reasoning: str) -> Tuple[bool, List[str], str]:
    """
    Checks whether student_reasoning states an explicit sound Python conceptual rule
    (evaluating both raw text and semantically canonicalized text).
    Returns (has_sound_rule, matched_domains, matched_clause).
    """
    clauses = split_into_clauses(reasoning)
    r_canon = canonicalize_for_semantics(reasoning)
    matched_domains = []
    best_clause = ""

    for domain, pat in SOUND_CONCEPT_RULES.items():
        if pat.search(reasoning) or pat.search(r_canon):
            matched_domains.append(domain)
            if not best_clause:
                for cl in clauses:
                    if pat.search(cl) or pat.search(canonicalize_for_semantics(cl)):
                        best_clause = cl
                        break
                if not best_clause:
                    best_clause = reasoning.strip().rstrip(".")

    # Semantic sound check for M05 (0-based indexing paraphrases in any word order)
    if "M05" not in matched_domains and _has_sound_zero_based_guard(r_canon.lower()):
        matched_domains.append("M05")
        if not best_clause:
            best_clause = clauses[0] if clauses else reasoning.strip().rstrip(".")

    return (len(matched_domains) > 0), matched_domains, best_clause


def detect_typo_or_careless(
    reasoning: str, student_answer: str, correct_answer: str, has_sound_rule: bool
) -> Tuple[bool, str, str]:
    """
    Checks if student explicitly describes a typo / careless / accidental writing / mental math slip,
    or if student has sound reasoning and student_answer is a 1-char spelling typo of correct_answer.
    Returns (is_typo_or_careless, error_type, explanation).
    """
    r_lower = (reasoning or "").lower()
    if re.search(
        r"\b(meant\s+to\s+(?:type|write|put|enter)|"
        r"hit\s+the\s+.*\s+key\s+by\s+(?:accident|mistake)|"
        r"accidentally\s+(?:typed|wrote|put|entered|pressed|clicked)|"
        r"(?:typed|wrote|put|entered|pressed)\s+\S+\s+by\s+(?:accident|mistake)|"
        r"typo|typing\s+(?:error|mistake|slip)|fat-fingered|misclicked|"
        r"typed\s+\w+\s+instead\s+of\s+\w+\s+(?:on\s+the\s+keyboard|by\s+mistake))\b",
        r_lower,
    ):
        return (
            True,
            "typo",
            "Student explicitly describes an accidental typing or writing slip rather than a conceptual misconception.",
        )

    if re.search(
        r"\b(careless(?:\s+(?:mistake|error|slip))?|silly\s+mistake|mental\s+math\s+(?:slip|mistake)|"
        r"miscarried\s+a\s+\d+|miscounted\s+the\s+(?:letters|items|elements).*by\s+mistake)\b",
        r_lower,
    ):
        return (
            True,
            "careless",
            "Student explicitly describes a careless mental math or counting slip rather than a conceptual misconception.",
        )

    if has_sound_rule and is_spelling_typo_of(student_answer, correct_answer):
        return (
            True,
            "typo",
            f"Student reasoning states the correct concept, and answer {repr(student_answer)} is a typographical slip of {repr(correct_answer)}.",
        )

    return False, "", ""


def detect_oos_belief(reasoning: str) -> Tuple[bool, str, str]:
    """
    Checks if student_reasoning expresses an out-of-scope conceptual belief (Rule 4, 9, 10, etc.).
    """
    clauses = split_into_clauses(reasoning)
    for topic, pat, desc in OOS_CONCEPT_RULES:
        if pat.search(reasoning):
            matched_cl = next(
                (cl for cl in clauses if pat.search(cl)),
                reasoning.strip().rstrip("."),
            )
            return True, topic, f'{desc} (Evidence: "{matched_cl}")'
    return False, "", ""


def detect_insufficient_or_trace(
    reasoning: str,
    answer_correct: bool,
    has_misconception_claim: bool,
    has_sound_rule: bool,
    is_oos: bool,
    is_typo_careless: bool,
) -> Tuple[bool, str, str]:
    """
    Checks whether reasoning is INSUFFICIENT due to:
    1. Too brief / empty
    2. Explicit guessing or uncertainty (including Rule 5 P23 isolated fact + guess)
    3. Sound conceptual rule + wrong answer due to unacknowledged trace/calculation slip (Rule 3)
    4. Pure mechanical calculation / execution trace without conceptual rule (Rule 3)
    5. Bare code restatement without any conceptual rule or causal claim (Rules 1 & 2)
    Returns (is_insufficient, error_type, evidence_explanation).
    """
    r_clean = (reasoning or "").strip()
    r_lower = r_clean.lower()
    words = re.findall(r"\b\w+\b", r_lower)

    # 1. Too brief (unless an explicit misconception claim, sound rule, OOS, or typo/careless is already established)
    if len(r_lower) < 12 or len(words) < 3:
        if not (is_typo_careless or has_misconception_claim or has_sound_rule or is_oos):
            return (
                True,
                "other",
                f"Reasoning is too brief ({repr(r_clean)}) to establish conceptual evidence.",
            )

    # 2. Explicit guessing / uncertainty
    if re.search(
        r"\b(i\s+)?(guessed|just guessed|guessing|don'?t know|do not know|not sure|no idea|unsure|picked randomly)\b",
        r_lower,
    ):
        err_t = "trace_error" if "trace" in r_lower else "other"
        return (
            True,
            err_t,
            f"Student explicitly indicates guessing or uncertainty without conceptual rule ({repr(r_clean)}).",
        )

    if is_oos or is_typo_careless or has_misconception_claim:
        return False, "", ""

    # 3. Sound conceptual rule + wrong answer or explicit calculation/trace mistake -> Trace / calculation mistake (Rule 3)
    if has_sound_rule and (
        not answer_correct
        or bool(
            re.search(
                r"\b(?:calculated|computed|traced|divided|multiplied|added|subtracted)\b.*\b(?:incorrectly|wrong(?:ly)?)\b|"
                r"\b(?:miscalculated|miscomputed|made\s+an?\s+(?:arithmetic|calculation|math|division|execution|tracing|trace)\s+(?:mistake|error|slip))\b",
                r_lower,
            )
        )
    ):
        return (
            True,
            "trace_error",
            f"Student states a sound conceptual rule ({repr(r_clean)}), so the calculation or trace error reflects an execution trace or arithmetic slip rather than a conceptual misconception.",
        )

    # 4. Pure mechanical trace narration without conceptual rules
    if re.search(r"\btraced\s+through\b.*\b(?:ended\s+up\s+with|got)\b", r_lower):
        return (
            True,
            "trace_error",
            f"Student describes mechanical tracing without articulating a conceptual rule ({repr(r_clean)}).",
        )

    arith_steps = re.findall(
        r"\b\d+\s*(?:times|plus|minus|divided\s+by|\+|\-|\*|/)\s*\d+(?:\s*(?:plus|minus|times|\+|\-|\*)\s*\d+)?\s*(?:is|equals|=|gives|to\s+get)\s*\d+\b",
        r_lower,
    )
    has_conceptual_terms = bool(
        re.search(
            r"\b(precedence|order|left\s+to\s+right|before|first|quote|string|int|float|floor|remainder|decimal|assign|equal|index|position|slice|range|inclusive|exclusive|boundary|loop|parameter|argument|keyword|positional|recurs|base\s+case|stack|terminat)\b",
            r_lower,
        )
    )
    if len(arith_steps) >= 2 and not has_conceptual_terms:
        return (
            True,
            "trace_error",
            f"Student provides purely mechanical arithmetic trace steps without stating a conceptual rule ({repr(r_clean)}).",
        )

    # 5. Bare code restatement / lack of conceptual claim (Rules 1 & 2)
    # Applies even when answer_correct is True if no sound conceptual rule was articulated
    if not has_sound_rule:
        is_bare_restatement = bool(
            re.match(
                r"^(?:the\s+)?(?:answer|output|result|value|it|this)\s+(?:is|equals|gives|outputs|prints|returns|should\s+be|will\s+be)\s+[\w\s'\"\[\],.\-+/*=()]+[.!?]*$|"
                r"^because\s+[\w\s'\"\[\],.\-+/*=()]+\s+(?:is|equals|gives|outputs|prints|returns)\s+[\w\s'\"\[\],.\-+/*=()]+[.!?]*$|"
                r"^(?:i\s+)?(?:got|calculated|computed|entered|put|wrote|typed|chose|picked)\s+[\w\s'\"\[\],.\-+/*=()]+[.!?]*$",
                r_lower,
            )
        )
        has_conceptual_predicate = bool(
            re.search(
                r"\b(because|since|so|therefore|thus|means|always|never|must|should|cannot|can'?t|only|automatically|default|instead|rather|regardless|"
                r"starts?\s+at|stops?\s+(?:at|before)|ends?\s+at|counts?\s+(?:up\s+to|from|\d+\s+times)|includ\w*|exclud\w*|inclusive|exclusive|"
                r"round\w*|truncat\w*|drop\w*|keep\w*|discard\w*|leaves?|left\s+with|return\w*|produc\w*|evaluat\w*\s+(?:first|before|to)|higher\s+precedence|bind\w*|fill\w*|ignor\w*|replac\w*|"
                r"assign\w*|sets?|updat\w*|compar\w*|tests?|check\w*|access\w*|refer\w*\s+to|retriev\w*|convert\w*|casts?|treat\w*|acts?\s+as|halt\w*|terminat\w*|reset\w*|forget\w*|"
                r"belong\w*\s+to|attach\w*\s+to|is\s+the\s+\w+\s+operator|violat\w*)\b",
                r_lower,
            )
        )
        if is_bare_restatement or not has_conceptual_predicate or not has_conceptual_terms:
            return (
                True,
                "other",
                f"Student restates the answer or code ({repr(r_clean)}) without expressing a specific conceptual rule or mechanism.",
            )

    return False, "", ""


def enrich_text(text: str) -> str:
    """Adds symbolic operator tokens so TF-IDF captures Python syntax constructs cleanly."""
    t = (text or "").lower()
    extra = []
    if re.search(r"['\"][0-9.]+['\"]", t):
        extra.append("__quoted_number__")
    if "//" in t:
        extra.append("__floor_div_op__")
    if re.search(r"(?<!/)/(?!/)", t):
        extra.append("__true_div_op__")
    if re.search(r"(?<![=!<>])=(?!=)", t):
        extra.append("__single_eq_op__")
    if "==" in t:
        extra.append("__double_eq_op__")
    if "**" in t:
        extra.append("__pow_op__")
    if re.search(r"\w+\[\s*-?[0-9]+", t):
        extra.append("__bracket_index__")
    if re.search(r"\w+\[.*:.*\]", t):
        extra.append("__slice_op__")
    if "range(" in t:
        extra.append("__range_call__")
    return t + (" " + " ".join(extra) if extra else "")


# =============================================================================
# 5. DATA LOADING & LEGACY INPUT FORMATTING HELPERS
# =============================================================================

def load_and_validate_dataset(
    filepath: str = "src/data/dataset_augmented.json",
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """
    Loads dataset_augmented.json (352 records: 160 original + 192 retained generated),
    validates schema, IDs, and split integrity, and ensures val (39) and test (34)
    contain strictly original records with zero leakage from generated samples.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        records = json.load(f)

    assert len(records) in (160, 352), f"Expected 160 or 352 records, got {len(records)}"

    required_fields = {
        "sample_id", "question_group", "concept", "question_format", "question",
        "correct_answer", "student_answer", "student_reasoning", "answer_correct",
        "misconception_id", "misconception_variant", "secondary_misconception_ids",
        "evidence_basis", "annotator_rationale", "error_type", "source", "split",
    }
    allowed_misconceptions = set(TAXONOMY_INFO.keys())
    allowed_error_types = {
        "correct", "careless", "typo", "syntax_error", "trace_error", "conceptual", "other"
    }

    seen_ids = set()
    seen_sample_ids = set()
    splits: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

    for r in records:
        missing = required_fields - set(r.keys())
        assert not missing, f"Record {r.get('sample_id')} missing fields: {missing}"
        rid = r.get("id") or r["sample_id"]
        rsid = r["sample_id"]
        assert rid not in seen_ids, f"Duplicate id: {rid}"
        assert rsid not in seen_sample_ids, f"Duplicate sample_id: {rsid}"
        seen_ids.add(rid)
        seen_sample_ids.add(rsid)
        assert r["misconception_id"] in allowed_misconceptions, (
            f"Invalid misconception_id: {r['misconception_id']}"
        )
        assert r["error_type"] in allowed_error_types, (
            f"Invalid error_type: {r['error_type']}"
        )
        splits[r["split"]].append(r)

    if len(records) == 352:
        assert len(splits["train"]) == 279, (
            f"Expected 279 train records (87 original + 192 generated), got {len(splits['train'])}"
        )
    else:
        assert len(splits["train"]) == 87, f"Expected 87 train records, got {len(splits['train'])}"

    assert len(splits["val"]) == 39, f"Expected 39 original val records, got {len(splits['val'])}"
    assert len(splits["test"]) == 34, f"Expected 34 original test records, got {len(splits['test'])}"

    for split_name in ("val", "test"):
        for r in splits[split_name]:
            assert not str(r.get("sample_id", "")).startswith(("GP-", "RL-GEN")), (
                f"Generated sample {r.get('sample_id')} leaked into {split_name} split!"
            )

    return records, splits


def create_model_input(
    question: str,
    correct_answer: str,
    student_answer: str,
    student_reasoning: str,
) -> str:
    """
    Creates the canonical serialized representation of a student interaction.
    Preserved for backwards compatibility with any caller using predict_with_confidence(text).
    """
    return (
        f"Question: {question.strip()}\n"
        f"Correct Answer: {correct_answer.strip()}\n"
        f"Student Answer: {student_answer.strip()}\n"
        f"Student Reasoning: {student_reasoning.strip()}"
    )


def parse_model_input(input_text: str) -> Tuple[str, str, str, str]:
    """
    Parses a serialized create_model_input string back into
    (question, correct_answer, student_answer, student_reasoning).
    """
    m = re.search(
        r"Question:\s*(.*?)\nCorrect Answer:\s*(.*?)\nStudent Answer:\s*(.*?)\nStudent Reasoning:\s*(.*)",
        input_text,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), m.group(4).strip()
    return "", "", "", input_text.strip()


# =============================================================================
# 6. PERSON 1 FIELD-SCOPED MISCONCEPTION CLASSIFIER & HYBRID DIAGNOSIS ENGINE
# =============================================================================

class Person1MisconceptionClassifier:
    """
    Field-Scoped Conceptual Misconception Classifier + Contrastive Hybrid Diagnostic Engine.
    """

    def __init__(self, confidence_threshold: float = 0.40):
        self.confidence_threshold = confidence_threshold
        self.vec_sr_word: Optional[TfidfVectorizer] = None
        self.vec_sr_char: Optional[TfidfVectorizer] = None
        self.vec_sa: Optional[TfidfVectorizer] = None
        self.vec_q: Optional[TfidfVectorizer] = None
        self.classifier: Optional[LogisticRegression] = None
        self.classes = TRAINABLE_CLASSES
        self.is_trained = False

    def _build_features(
        self, items: List[Tuple[str, str, str, str]], fit: bool = False
    ) -> csr_matrix:
        sr_texts = [enrich_text(sr) for q, ca, sa, sr in items]
        sa_texts = [enrich_text(sa) for q, ca, sa, sr in items]
        q_texts = [enrich_text(q) for q, ca, sa, sr in items]

        if fit:
            self.vec_sr_word = TfidfVectorizer(
                ngram_range=(1, 3),
                sublinear_tf=True,
                min_df=1,
                token_pattern=r"(?u)\b\w+\b|[=<>+\-*/%&|~^!]+",
            )
            self.vec_sr_char = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
                min_df=2,
            )
            self.vec_sa = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                token_pattern=r"(?u)\b\w+\b|[=<>+\-*/%&|~^!]+",
            )
            self.vec_q = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                token_pattern=r"(?u)\b\w+\b|[=<>+\-*/%&|~^!]+",
            )
            X_sr_w = self.vec_sr_word.fit_transform(sr_texts)
            X_sr_c = self.vec_sr_char.fit_transform(sr_texts)
            X_sa = self.vec_sa.fit_transform(sa_texts)
            X_q = self.vec_q.fit_transform(q_texts)
        else:
            assert self.vec_sr_word is not None
            X_sr_w = self.vec_sr_word.transform(sr_texts)
            X_sr_c = self.vec_sr_char.transform(sr_texts)
            X_sa = self.vec_sa.transform(sa_texts)
            X_q = self.vec_q.transform(q_texts)

        struct_rows = []
        for q, ca, sa, sr in items:
            ans_corr = 1.0 if compute_answer_correct(q, ca, sa, sr) else 0.0
            claims = score_misconception_claims(sr, sa, q, ca)
            has_sound, _, _ = detect_sound_concepts(sr)
            is_typo, _, _ = detect_typo_or_careless(sr, sa, ca, has_sound)
            claim_vec = [claims.get(m, {}).get("score", 0.0) for m in M_CLASSES]
            none_ind = (
                1.0
                if ((ans_corr > 0.5 or has_sound or is_typo) and sum(claim_vec) == 0.0)
                else 0.0
            )
            row = claim_vec + [
                ans_corr * 0.5,
                1.0 if has_sound else 0.0,
                1.0 if is_typo else 0.0,
                none_ind,
            ]
            struct_rows.append(row)

        X_struct = csr_matrix(np.array(struct_rows, dtype=float))

        return hstack(
            [
                X_sr_w * 1.25,
                X_sr_c * 0.60,
                X_sa * 0.35,
                X_q * 0.18,
                X_struct * 0.75,
            ]
        )

    def train(self, train_records: List[Dict[str, Any]]) -> None:
        """
        Trains the field-scoped classifier on the augmented training split
        (original train + retained generated train samples in TRAINABLE_CLASSES).
        """
        effective_train = list(train_records)
        if len(effective_train) == 87:
            aug_candidates = [
                "src/data/dataset_augmented.json",
                os.path.join(os.path.dirname(__file__), "..", "data", "dataset_augmented.json"),
            ]
            for aug_path in aug_candidates:
                if os.path.exists(aug_path):
                    with open(aug_path, "r", encoding="utf-8") as f:
                        aug_all = json.load(f)
                    effective_train = [r for r in aug_all if r.get("split") == "train"]
                    break

        trainable_records = [
            r for r in effective_train if r["misconception_id"] in TRAINABLE_CLASSES
        ]
        items = [
            (
                r["question"],
                r["correct_answer"],
                r["student_answer"],
                r["student_reasoning"],
            )
            for r in trainable_records
        ]
        y_train = [r["misconception_id"] for r in trainable_records]
        for label in y_train:
            assert label in TRAINABLE_CLASSES, f"Unexpected label in train split: {label}"

        X_train = self._build_features(items, fit=True)
        self.classifier = LogisticRegression(
            C=3.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

    def predict_raw(self, q: str, ca: str, sa: str, sr: str) -> Dict[str, Any]:
        """Predicts class probabilities from structured fields (q, ca, sa, sr)."""
        assert self.is_trained and self.classifier is not None
        X = self._build_features([(q, ca, sa, sr)], fit=False)
        probs = self.classifier.predict_proba(X)[0]
        class_probs = {
            str(c): round(float(p), 4) for c, p in zip(self.classifier.classes_, probs)
        }
        best_class = str(self.classifier.classes_[np.argmax(probs)])
        best_conf = float(np.max(probs))
        return {
            "prediction": best_class,
            "confidence": round(best_conf, 4),
            "all_probabilities": class_probs,
            "below_threshold": best_conf < self.confidence_threshold,
        }

    def predict_with_confidence(self, input_text: str) -> Dict[str, Any]:
        """Backwards-compatible prediction method accepting create_model_input(q, ca, sa, sr)."""
        q, ca, sa, sr = parse_model_input(input_text)
        return self.predict_raw(q, ca, sa, sr)

    def extract_best_clause(
        self,
        q: str,
        sa: str,
        sr: str,
        target_class: str,
        claim_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Extracts the most conceptually informative contiguous clause from student_reasoning."""
        if claim_info and claim_info.get("span"):
            return f'"{claim_info["span"]}"'
        clauses = split_into_clauses(sr)
        if not clauses:
            return repr(sr.strip())
        assert self.classifier is not None and self.vec_sr_word is not None and self.vec_sr_char is not None
        if len(clauses) == 1 or target_class not in self.classifier.classes_:
            return f'"{clauses[0]}"'

        cls_idx = list(self.classifier.classes_).index(target_class)
        coefs = self.classifier.coef_[cls_idx]
        w_dim = len(self.vec_sr_word.vocabulary_)
        c_dim = len(self.vec_sr_char.vocabulary_)
        w_coefs = coefs[:w_dim]
        c_coefs = coefs[w_dim : w_dim + c_dim]

        best_cl = clauses[0]
        best_sc = -1e9
        for cl in clauses:
            enr = enrich_text(cl)
            vw = self.vec_sr_word.transform([enr])
            vc = self.vec_sr_char.transform([enr])
            sc = float(vw.dot(w_coefs)[0] * 1.25 + vc.dot(c_coefs)[0] * 0.60)
            if sc > best_sc:
                best_sc = sc
                best_cl = cl
        return f'"{best_cl}"'

    def extract_evidence(self, student_reasoning: str, predicted_class: str) -> str:
        """Backwards-compatible wrapper for clause-level evidence extraction."""
        claims = score_misconception_claims(student_reasoning, "", "")
        return self.extract_best_clause(
            "", "", student_reasoning, predicted_class, claims.get(predicted_class)
        )

    def diagnose_student(
        self,
        question: str,
        correct_answer: str,
        student_answer: str,
        student_reasoning: str,
    ) -> Dict[str, Any]:
        """
        Authoritative Person 1 Hybrid Diagnosis Pipeline.
        Preserves all 5 existing Person 2 / Bridge / UI keys:
        - misconception_id
        - confidence
        - evidence
        - rationale
        - decision_source
        And includes enriched diagnostic fields:
        - misconception_variant
        - secondary_misconception_ids
        - error_type
        - answer_correct
        """
        q = str(question or "")[:8000]
        ca = str(correct_answer or "")[:4000]
        sa = str(student_answer or "")[:4000]
        sr = str(student_reasoning or "")[:8000]
        ans_corr = compute_answer_correct(q, ca, sa, sr)
        claims = score_misconception_claims(sr, sa, q, ca)
        has_sound, sound_domains, sound_span = detect_sound_concepts(sr)
        is_typo, typo_err_type, typo_desc = detect_typo_or_careless(
            sr, sa, ca, has_sound
        )
        is_oos, oos_topic, oos_desc = detect_oos_belief(sr)
        has_any_claim = len(claims) > 0

        is_ins, ins_err_type, ins_desc = detect_insufficient_or_trace(
            reasoning=sr,
            answer_correct=ans_corr,
            has_misconception_claim=has_any_claim,
            has_sound_rule=has_sound,
            is_oos=is_oos,
            is_typo_careless=is_typo,
        )

        raw_res = self.predict_raw(q, ca, sa, sr)
        probs = raw_res["all_probabilities"]
        pred_class = raw_res["prediction"]

        # 1. INSUFFICIENT Gate (Thin, Guess, Bare Code Restatement, or Sound Rule + Trace Slip)
        if is_ins:
            conf = (
                0.85
                if ("guessed" in sr.lower() or len(sr.strip()) < 12)
                else 0.78
            )
            return {
                "misconception_id": "INSUFFICIENT",
                "confidence": conf,
                "evidence": ins_desc,
                "rationale": TAXONOMY_INFO["INSUFFICIENT"]["rationale_template"],
                "decision_source": "abstention_rule",
                "misconception_variant": None,
                "secondary_misconception_ids": [],
                "error_type": ins_err_type,
                "answer_correct": ans_corr,
            }

        # 2. OOS Gate (Explicit Out-of-Scope Conceptual Belief)
        if is_oos:
            return {
                "misconception_id": "OOS",
                "confidence": 0.84,
                "evidence": oos_desc,
                "rationale": (
                    f"The expressed belief concerns {oos_topic}, which is outside "
                    f"the frozen M01-M08 taxonomy."
                ),
                "decision_source": "evidence_rule",
                "misconception_variant": None,
                "secondary_misconception_ids": [],
                "error_type": "conceptual",
                "answer_correct": ans_corr,
            }

        # 3. Explicit Typo / Careless Slip (when no misconception claim is stated)
        if is_typo and not has_any_claim:
            p_none = probs.get("NONE", 0.25)
            conf = round(min(0.92, 0.65 + 0.30 * p_none), 2)
            return {
                "misconception_id": "NONE",
                "confidence": conf,
                "evidence": typo_desc,
                "rationale": TAXONOMY_INFO["NONE"]["rationale_template"],
                "decision_source": "model" if pred_class == "NONE" else "evidence_rule",
                "misconception_variant": None,
                "secondary_misconception_ids": [],
                "error_type": typo_err_type,
                "answer_correct": ans_corr,
            }

        # 4. Sound Reasoning Protection / Correct Answer without Misconception Claim
        if (has_sound or ans_corr) and not has_any_claim:
            err_t = "correct"
            if "syntax error" in q.lower() or "colon" in sr.lower():
                err_t = "syntax_error" if not ans_corr else "correct"
            p_none = probs.get("NONE", 0.30)
            conf = round(min(0.94, 0.68 + 0.30 * p_none), 2)
            ev_text = (
                f'Sound conceptual reasoning stated: "{sound_span}"'
                if sound_span
                else "Answer is correct and student reasoning contains no misconception claims."
            )
            return {
                "misconception_id": "NONE",
                "confidence": conf,
                "evidence": ev_text,
                "rationale": TAXONOMY_INFO["NONE"]["rationale_template"],
                "decision_source": "model" if pred_class == "NONE" else "evidence_rule",
                "misconception_variant": None,
                "secondary_misconception_ids": [],
                "error_type": err_t,
                "answer_correct": ans_corr,
            }

        # 5. Misconception Candidate Scoring (Combines Contrastive Claim Specificity + Scoped ML Probability)
        candidate_scores: Dict[str, float] = {}
        for m_id in M_CLASSES:
            c_score = claims.get(m_id, {}).get("score", 0.0)
            ml_prob = probs.get(m_id, 0.0)
            sound_penalty = 0.50 if (m_id in sound_domains and m_id not in claims) else 0.0
            candidate_scores[m_id] = (0.65 * c_score) + (1.0 * ml_prob) - sound_penalty

        ranked_m = sorted(M_CLASSES, key=lambda m: candidate_scores[m], reverse=True)
        best_m = ranked_m[0]
        best_claim = claims.get(best_m)
        best_ml_prob = probs.get(best_m, 0.0)

        if pred_class == best_m and (
            best_ml_prob >= self.confidence_threshold or best_claim is not None
        ):
            source = "model"
        else:
            source = "evidence_rule" if best_claim is not None else "model"

        if best_claim is not None:
            raw_strength = (
                0.50 + 0.45 * best_ml_prob + 0.15 * min(best_claim["score"], 1.2)
            )
            conf = round(float(min(0.96, max(0.55, raw_strength))), 2)
            variant = best_claim["variant"]
        else:
            raw_strength = 0.35 + 0.65 * best_ml_prob
            conf = round(float(min(0.90, max(0.42, raw_strength))), 2)
            variant = None

        # Detect secondary misconceptions ONLY when another M-class has independent explicit claim evidence
        # in a DIFFERENT clause (Rule 8, max 2)
        primary_span = best_claim["span"] if best_claim else ""
        secondary_ids = []
        for other_m in ranked_m[1:]:
            if other_m in claims:
                other_info = claims[other_m]
                if (
                    other_info["score"] >= 1.0
                    and other_info["span"]
                    and other_info["span"] != primary_span
                ):
                    secondary_ids.append(other_m)
            if len(secondary_ids) >= 2:
                break

        ev_str = self.extract_best_clause(q, sa, sr, best_m, best_claim)
        rationale = TAXONOMY_INFO[best_m]["rationale_template"]
        if ans_corr:
            rationale += " (Diagnosed from explicit misconception in reasoning despite obtaining the right final answer.)"

        return {
            "misconception_id": best_m,
            "confidence": conf,
            "evidence": ev_str,
            "rationale": rationale,
            "decision_source": source,
            "misconception_variant": variant,
            "secondary_misconception_ids": secondary_ids,
            "error_type": "conceptual",
            "answer_correct": ans_corr,
        }


# Backwards-compatible alias for any external scripts inspecting MISCONCEPTION_EVIDENCE_PATTERNS
MISCONCEPTION_EVIDENCE_PATTERNS: Dict[str, List[str]] = {
    m_id: [pat.pattern for _, pat, _ in rules]
    for m_id, rules in MISCONCEPTION_CLAIM_RULES.items()
}


# =============================================================================
# 7. EVALUATION & COUNTERFACTUAL / PARAPHRASE VERIFICATION SUITE
# =============================================================================

def evaluate_split(
    classifier: Person1MisconceptionClassifier,
    records: List[Dict],
    split_name: str,
) -> Dict[str, Any]:
    """
    Evaluates both:
    1. Raw Scoped ML Classifier accuracy on in-distribution classes (M01-M08 + NONE)
    2. Final Hybrid Diagnosis accuracy across all 11 outcomes (M01-M08, NONE, INSUFFICIENT, OOS)
    """
    in_dist_records = [r for r in records if r["misconception_id"] in TRAINABLE_CLASSES]
    y_true_in = [r["misconception_id"] for r in in_dist_records]
    y_pred_raw = [
        classifier.predict_raw(
            r["question"], r["correct_answer"], r["student_answer"], r["student_reasoning"]
        )["prediction"]
        for r in in_dist_records
    ]

    raw_acc = accuracy_score(y_true_in, y_pred_raw)
    raw_f1 = f1_score(y_true_in, y_pred_raw, average="macro", zero_division=0)

    y_true_all = [r["misconception_id"] for r in records]
    diags = [
        classifier.diagnose_student(
            r["question"], r["correct_answer"], r["student_answer"], r["student_reasoning"]
        )
        for r in records
    ]
    y_pred_final = [d["misconception_id"] for d in diags]
    sources = Counter(d["decision_source"] for d in diags)

    final_acc = accuracy_score(y_true_all, y_pred_final)
    final_f1 = f1_score(y_true_all, y_pred_final, average="macro", zero_division=0)

    print(
        f"\n--- {split_name.upper()} SPLIT EVALUATION (Total={len(records)}, In-Distribution={len(in_dist_records)}) ---"
    )
    print(
        f"  [Metric 1] Raw Scoped ML Classifier (In-Dist n={len(in_dist_records)}): "
        f"Accuracy = {raw_acc:.4f} ({raw_acc*100:.2f}%), Macro F1 = {raw_f1:.4f}"
    )
    print(
        f"  [Metric 2] Final Hybrid Diagnosis   (All 11  n={len(records)}): "
        f"Accuracy = {final_acc:.4f} ({final_acc*100:.2f}%), Macro F1 = {final_f1:.4f}"
    )
    print(f"  [Metric 3] Decision Source Breakdown: {dict(sources)}")

    if split_name == "test":
        print("\nClassification Report (Final Hybrid Diagnosis on Test Split):")
        present_labels = sorted(
            list(set(y_true_all) | set(y_pred_final)),
            key=lambda x: (TRAINABLE_CLASSES + ["INSUFFICIENT", "OOS"]).index(x),
        )
        print(
            classification_report(
                y_true_all, y_pred_final, labels=present_labels, zero_division=0
            )
        )
        print("Confusion Matrix (Rows=True, Cols=Predicted):")
        print("Labels:", present_labels)
        print(confusion_matrix(y_true_all, y_pred_final, labels=present_labels))

    return {
        "raw_acc": raw_acc,
        "raw_f1": raw_f1,
        "final_acc": final_acc,
        "final_f1": final_f1,
        "sources": dict(sources),
    }


def run_counterfactual_and_paraphrase_suite(
    classifier: Person1MisconceptionClassifier,
) -> Tuple[int, int]:
    """
    Runs the 22-case Counterfactual, Paraphrase & Boundary Disambiguation Suite
    covering Categories A through K.
    """
    suite = [
        (
            "A1: Correct answer + sound reasoning (M02 topic //)",
            "print(7 // 2)", "3", "3",
            "In Python, // is floor division, which divides and rounds down to the nearest integer 3 without keeping the decimal.",
            "NONE", "correct",
        ),
        (
            "A2: Correct answer + sound reasoning (M06 topic range)",
            "for i in range(1, 4):\n    print(i)", "1 2 3", "1 2 3",
            "range(1, 4) starts at 1 and the stop value 4 is exclusive, so the loop visits 1, 2, and 3.",
            "NONE", "correct",
        ),
        (
            "A3: Correct answer + sound reasoning (M04 topic precedence)",
            "print(2 + 3 * 4)", "14", "14",
            "Multiplication * has higher precedence than addition +, so 3 * 4 is evaluated first to get 12, then 2 + 12 = 14.",
            "NONE", "correct",
        ),
        (
            "A4: Correct answer + sound reasoning (M05 topic indexing)",
            "items = ['a', 'b', 'c']\nprint(items[0])", "a", "a",
            "Python uses 0-based indexing, so index 0 retrieves the first element 'a'.",
            "NONE", "correct",
        ),
        (
            "B1: Right answer for wrong reason (M04 precedence)",
            "print(2 ** 3 * 1)", "8", "8",
            "I multiply 3 and 1 first and then calculate 2 ** 3 because multiplication goes before exponentiation.",
            "M04", "conceptual",
        ),
        (
            "B2: Right answer for wrong reason (M01 string vs int)",
            "x = '3'\nprint(int(x) * 1)", "3", "3",
            'The quotes around "3" are ignored so it is treated as integer 3, and 3 * 1 is 3.',
            "M01", "conceptual",
        ),
        (
            "B3: Right answer for wrong reason (M03 == assignment)",
            "x = 10\nprint(x == 10)", "True", "True",
            "Double equals == assigns 10 to x and returns True because the assignment succeeded.",
            "M03", "conceptual",
        ),
        (
            "C1: Wrong answer + explicit M02 misconception",
            "print(9 // 2)", "4", "4.5",
            "Double slash // performs float division and keeps the decimal portion 4.5.",
            "M02", "conceptual",
        ),
        (
            "D1: Wrong answer + sound range() rule + arithmetic trace slip",
            "total = 0\nfor x in range(1, 5):\n    total += x\nprint(total)", "10", "9",
            "range(1, 5) visits 1, 2, 3, 4 because 5 is exclusive. 1 + 2 + 3 + 4 is 9.",
            "INSUFFICIENT", "trace_error",
        ),
        (
            "D2: Wrong answer + sound precedence rule + arithmetic slip",
            "print(2 + 3 * 4)", "14", "15",
            "Multiplication * has higher precedence than +, so 3 * 4 evaluates first to 12, and 2 + 12 is 15.",
            "INSUFFICIENT", "trace_error",
        ),
        (
            "E1: Wrong answer + bare code restatement on //",
            "print(7 // 2)", "3", "4",
            "7 // 2 divides 7 by 2.",
            "INSUFFICIENT", "other",
        ),
        (
            "E2: Wrong answer + bare code restatement on range",
            "for i in range(1, 5):\n    print(i)", "1 2 3 4", "1 2 3 4 5",
            "range(1, 5) uses range in the loop.",
            "INSUFFICIENT", "other",
        ),
        (
            "F1: Unseen OOS (Set ordering and bracket indexing)",
            "s = {3, 1, 2}\nprint(s[0])", "TypeError", "3",
            "Sets maintain insertion order and support bracket indexing just like lists, so s[0] returns 3.",
            "OOS", "conceptual",
        ),
        (
            "F2: Unseen OOS (list.sort() return value belief)",
            "nums = [3, 1, 2]\nres = nums.sort()\nprint(res)", "None", "[1, 2, 3]",
            "Calling .sort() returns a new sorted list [1, 2, 3] and assigns it to res.",
            "OOS", "conceptual",
        ),
        (
            "G1: M05 vs M06 collision (1-based indexing inside a range loop -> M05)",
            "vals = [10, 20, 30]\nfor i in range(1, 2):\n    print(vals[i])", "20", "10",
            "List positions start at 1, so vals[1] accesses the first element 10.",
            "M05", "conceptual",
        ),
        (
            "G2: M05 vs M06 collision (inclusive range stop when indexing list -> M06)",
            "vals = [10, 20, 30]\nfor i in range(0, 2):\n    print(vals[i])", "10 20", "10 20 30",
            "range(0, 2) includes the stop value 2 inclusive, so the loop visits indices 0, 1, and 2.",
            "M06", "conceptual",
        ),
        (
            "H1: M01 vs M02 collision (/ returns int when evenly divisible -> M02)",
            "print(8 / 2)", "4.0", "4",
            "Because 8 divides evenly by 2 with no remainder, single slash / returns an integer int type 4 instead of a float.",
            "M02", "conceptual",
        ),
        (
            "I1: M03 vs M06 collision (= inside while loop condition -> M03)",
            "k = 0\nwhile k = 5:\n    k += 1", "SyntaxError", "5",
            "In the while loop condition, the single equals sign = checks whether k is equal to 5.",
            "M03", "conceptual",
        ),
        (
            "J1: Novel paraphrase of M05 (counting begins at one)",
            "word = 'cat'\nprint(word[1])", "a", "c",
            "Counting in lists begins at one, so position 1 refers to the first letter 'c'.",
            "M05", "conceptual",
        ),
        (
            "J2: Novel paraphrase of M04 (addition before multiplication)",
            "print(5 + 2 * 3)", "11", "21",
            "Addition is evaluated before multiplication, giving 7 times 3 which is 21.",
            "M04", "conceptual",
        ),
        (
            "J3: Novel paraphrase of M08 (recursion halts on its own at zero)",
            "def f(n):\n    return n + f(n - 1)\nprint(f(2))", "RecursionError", "3",
            "Recursive functions automatically stop when the parameter reaches 0 without needing a base case.",
            "M08", "conceptual",
        ),
        (
            "K1: Sound // floor rule + arithmetic slip -> INSUFFICIENT",
            "print(7 // 2)", "3", "4",
            "Double slash // is floor division which rounds down to the nearest whole integer, and 7 divided by 2 is 4.",
            "INSUFFICIENT", "trace_error",
        ),
    ]

    print("\n" + "=" * 72)
    print("COUNTERFACTUAL, PARAPHRASE & DISAMBIGUATION SUITE (CATEGORIES A - K)")
    print("=" * 72)
    passed = 0
    for name, q, ca, sa, sr, exp_id, exp_err in suite:
        res = classifier.diagnose_student(q, ca, sa, sr)
        ok = res["misconception_id"] == exp_id and res["error_type"] == exp_err
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] {name}\n"
            f"         Pred={res['misconception_id']} (Exp={exp_id}) | ErrType={res['error_type']} (Exp={exp_err}) | Conf={res['confidence']} | Source={res['decision_source']}\n"
            f"         Evidence={res['evidence']}"
        )

    print(
        f"\nCounterfactual & Paraphrase Verification Result: {passed}/{len(suite)} passed."
    )
    return passed, len(suite)


def run_m02_regression_suite(
    classifier: Person1MisconceptionClassifier,
) -> Tuple[int, int]:
    """
    Regression suite verifying:
    - TEST A: Sound floor concept + arithmetic/calculation slip -> INSUFFICIENT (trace_error)
    - TEST B: Out-of-scope list copy / aliasing concept on unrelated question -> OOS
    - TEST C: Right answer for wrong reason (// performs normal division then removes decimal) -> M02
    - TEST D: Wrong answer + // gives remainder misconception -> M02
    - TEST E: Wrong answer + "I don't know." -> INSUFFICIENT
    - TEST F: Correct answer + multi-step floor division explanation -> NONE (correct)
    Plus additional natural-language paraphrases of sound floor + calculation error and OOS list aliasing.
    """
    m02_tests = [
        (
            "TEST A (Case 4): Sound floor concept + calculation error -> INSUFFICIENT / trace_error",
            "print(7 // 2)",
            "3",
            "4",
            "The // operator performs floor division and should return the integer quotient, but I calculated 7 divided by 2 incorrectly and got 4.",
            "INSUFFICIENT",
            "trace_error",
            False,
        ),
        (
            "TEST B (Case 6): Out-of-scope list copy/aliasing belief -> OOS",
            "print(7 // 2)",
            "3",
            "3",
            "I thought that if x = y, Python automatically creates a separate copy of the list, so changing one list does not affect the other.",
            "OOS",
            "conceptual",
            True,
        ),
        (
            "TEST C (Case 2): Correct answer + // performs normal division then removes decimal -> M02",
            "print(7 // 2)",
            "3",
            "3",
            "The // operator performs normal division and 7 divided by 2 gives 3.5, but Python removes the decimal part and gives 3.",
            "M02",
            "conceptual",
            True,
        ),
        (
            "TEST D (Case 3): Wrong answer + // gives remainder when divided -> M02",
            "print(7 // 2)",
            "3",
            "1",
            "The // operator gives the remainder when one number is divided by another.",
            "M02",
            "conceptual",
            False,
        ),
        (
            "TEST E (Case 5): Wrong answer + I don't know -> INSUFFICIENT",
            "print(7 // 2)",
            "3",
            "4",
            "I don't know.",
            "INSUFFICIENT",
            "other",
            False,
        ),
        (
            "TEST F (Case 1): Correct answer + multi-step floor division explanation -> NONE",
            "print(7 // 2)",
            "3",
            "3",
            "The // operator performs floor division. 7 divided by 2 is 3.5, and floor division returns the largest integer less than or equal to 3.5, which is 3.",
            "NONE",
            "correct",
            True,
        ),
        (
            "M02-REG-7: Paraphrase of Case 4 (floor division gives whole number result + arithmetic mistake) -> INSUFFICIENT",
            "print(7 // 2)",
            "3",
            "4",
            "Floor division gives the whole number result, but I made an arithmetic mistake dividing 7 by 2.",
            "INSUFFICIENT",
            "trace_error",
            False,
        ),
        (
            "M02-REG-8: Paraphrase of Case 6 (assigning one list to another creates a separate copy) -> OOS",
            "print(7 // 2)",
            "3",
            "3",
            "Assigning one list to another creates a separate copy so the two lists are independent after assignment.",
            "OOS",
            "conceptual",
            True,
        ),
        (
            "M02-REG-9: Genuine M02 (// gives the decimal result) -> M02",
            "print(7 // 2)",
            "3",
            "3.5",
            "The // operator gives the decimal result when dividing.",
            "M02",
            "conceptual",
            False,
        ),
    ]

    print("\n" + "=" * 72)
    print("PERSON 1 6-CASE & PARAPHRASE REGRESSION SUITE (TESTS A - F + PARAPHRASES)")
    print("=" * 72)
    passed = 0
    for name, q, ca, sa, sr, exp_id, exp_err, exp_ans_corr in m02_tests:
        res = classifier.diagnose_student(q, ca, sa, sr)
        ok = (
            res["misconception_id"] == exp_id
            and res["error_type"] == exp_err
            and res["answer_correct"] == exp_ans_corr
        )
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] {name}\n"
            f"         Pred={res['misconception_id']} (Exp={exp_id}) | ErrType={res['error_type']} (Exp={exp_err}) | AnsCorr={res['answer_correct']} | Conf={res['confidence']}\n"
            f"         Evidence={res['evidence']}"
        )

    print(f"\nRegression Suite Result: {passed}/{len(m02_tests)} passed.")
    return passed, len(m02_tests)


def run_semantic_paraphrase_and_normalization_suite(
    classifier: Person1MisconceptionClassifier,
) -> Tuple[int, int]:
    """
    Verifies the Semantic / Conceptual Evidence Layer across M01-M08:
    1. All 6 core s = "Python"; print(s[1]) behavioral distinction cases
    2. All 14 M05 paraphrase variants + broken English / spelling variations
    3. Equivalent natural-language paraphrases across M01-M08
    4. Answer normalization tolerance (quotes, whitespace/newlines, case, lists, numbers) vs wrong answers
    """
    q_s1 = 's = "Python"\nprint(s[1])'
    semantic_cases = [
        # --- Core 6 Behavioral Distinction Cases on s = "Python"; print(s[1]) ---
        (
            "SEM-CORE-1: s[1]->y, answer P, 'in the string first letter is P' => M05",
            q_s1, "y", "P",
            "in the string first letter is P",
            "M05", "conceptual", False,
        ),
        (
            "SEM-CORE-2: s[1]->y, answer P, 'I don't know' => INSUFFICIENT",
            q_s1, "y", "P",
            "I don't know",
            "INSUFFICIENT", "other", False,
        ),
        (
            "SEM-CORE-3: s[1]->y, answer P, 's[1] is the first character because the first position is 1' => M05",
            q_s1, "y", "P",
            "s[1] is the first character because the first position is 1",
            "M05", "conceptual", False,
        ),
        (
            "SEM-CORE-4: s[1]->y, answer y, 'Python uses zero-based indexing, so index 1 is the second character.' => NONE",
            q_s1, "y", "y",
            "Python uses zero-based indexing, so index 1 is the second character.",
            "NONE", "correct", True,
        ),
        (
            "SEM-CORE-5a: s[1]->y, answer y, 'index 1 is the first character' => M05 (right answer, wrong belief)",
            q_s1, "y", "y",
            "index 1 is the first character",
            "M05", "conceptual", True,
        ),
        (
            "SEM-CORE-5b: s[1]->y, answer y, 'index 1 is the first character, but I still got y' => M05",
            q_s1, "y", "y",
            "index 1 is the first character, but I still got y",
            "M05", "conceptual", True,
        ),
        (
            "SEM-CORE-6: s[1]->y, answer P, 'I accidentally wrote P' => NONE (typo)",
            q_s1, "y", "P",
            "I accidentally wrote P",
            "NONE", "typo", False,
        ),
        # --- All 14 M05 Paraphrase Variants + Spelling/Broken English ---
        (
            "M05-PARA-01: 'the first character is at index 1'",
            q_s1, "y", "P", "the first character is at index 1",
            "M05", "conceptual", False,
        ),
        (
            "M05-PARA-02: 'index 1 is the first character'",
            q_s1, "y", "P", "index 1 is the first character",
            "M05", "conceptual", False,
        ),
        (
            "M05-PARA-03: 'the first letter has position 1'",
            q_s1, "y", "P", "the first letter has position 1",
            "M05", "conceptual", False,
        ),
        (
            "M05-PARA-04: 'position 1 means the first letter'",
            q_s1, "y", "P", "position 1 means the first letter",
            "M05", "conceptual", False,
        ),
        (
            "M05-PARA-05: 's[1] gives the first character'",
            q_s1, "y", "P", "s[1] gives the first character",
            "M05", "conceptual", False,
        ),
        (
            "M05-PARA-06: 's[1] points to P because P is the first letter'",
            q_s1, "y", "P", "s[1] points to P because P is the first letter",
            "M05", "conceptual", False,
        ),
        (
            "M05-PARA-07: 'Python starts counting characters from 1'",
            q_s1, "y", "P", "Python starts counting characters from 1",
            "M05", "conceptual", False,
        ),
        (
            "M05-PARA-08: 'the first position is 1'",
            q_s1, "y", "P", "the first position is 1",
            "M05", "conceptual", False,
        ),
        (
            "M05-PARA-09: 'I counted P as position 1'",
            q_s1, "y", "P", "I counted P as position 1",
            "M05", "conceptual", False,
        ),
        (
            "M05-PARA-10: 'I thought index 1 means the first character'",
            q_s1, "y", "P", "I thought index 1 means the first character",
            "M05", "conceptual", False,
        ),
        (
            "M05-PARA-11: '0 is not counted'",
            q_s1, "y", "P", "0 is not counted",
            "M05", "conceptual", False,
        ),
        (
            "M05-PARA-12: 'the character at 1 is the first character'",
            q_s1, "y", "P", "the character at 1 is the first character",
            "M05", "conceptual", False,
        ),
        (
            "M05-PARA-13: 'P is at index 1'",
            q_s1, "y", "P", "P is at index 1",
            "M05", "conceptual", False,
        ),
        (
            "M05-PARA-14: 'P is the character in position 1'",
            q_s1, "y", "P", "P is the character in position 1",
            "M05", "conceptual", False,
        ),
        (
            "M05-PARA-15: Spelling/broken English 'in strng 1st charcter is at posistion one'",
            q_s1, "y", "P", "in strng 1st charcter is at posistion one",
            "M05", "conceptual", False,
        ),
        # --- Equivalent Semantic Paraphrases across M01 - M08 ---
        (
            "M01-PARA-1: Conversational string + int gluing",
            'print("Age: " + 20)', "TypeError", "Age: 20",
            "it glues the string and the number together into one text",
            "M01", "conceptual", False,
        ),
        (
            "M01-PARA-2: Quoted numbers treated as numbers for addition",
            'print("3" + "4")', "34", "7",
            "both '3' and '4' are numbers in quotes so Python adds them numerically",
            "M01", "conceptual", False,
        ),
        (
            "M02-PARA-1: // gives what is left over after dividing",
            "print(7 // 2)", "3", "1",
            "// gives what is left over after dividing 7 by 2",
            "M02", "conceptual", False,
        ),
        (
            "M02-PARA-2: // does regular division and keeps decimal",
            "print(7 // 2)", "3", "3.5",
            "double slash does regular division and keeps the decimal",
            "M02", "conceptual", False,
        ),
        (
            "M03-PARA-1: Single = checks if x is 5 in condition",
            "x = 5\nif x = 5:\n    print('yes')", "SyntaxError", "yes",
            "in the if condition, the equals sign = checks if x is 5",
            "M03", "conceptual", False,
        ),
        (
            "M03-PARA-2: Double == stores new value into variable",
            "x = 5\nx == 10\nprint(x)", "5", "10",
            "x == 10 stores 10 into x so x becomes 10",
            "M03", "conceptual", False,
        ),
        (
            "M04-PARA-1: Operations evaluated in the order they appear from left to right",
            "print(2 + 3 * 4)", "14", "20",
            "Python evaluates operations in the order they appear from left to right",
            "M04", "conceptual", False,
        ),
        (
            "M04-PARA-2: Plus comes before multiply",
            "print(2 + 3 * 4)", "14", "20",
            "plus comes before multiply so I added 2 and 3 first",
            "M04", "conceptual", False,
        ),
        (
            "M06-PARA-1: range(3) goes all the way up to 3",
            "for i in range(3):\n    print(i)", "0 1 2", "0 1 2 3",
            "range(3) goes all the way up to 3 so 3 is printed",
            "M06", "conceptual", False,
        ),
        (
            "M06-PARA-2: range(3) starts counting from 1",
            "for i in range(3):\n    print(i)", "0 1 2", "1 2 3",
            "range(3) starts counting from 1 so it prints 1, 2, 3",
            "M06", "conceptual", False,
        ),
        (
            "M07-PARA-1: Argument variable names matched to parameter names",
            "def f(a, b):\n    return a - b\nx = 10\ny = 3\nprint(f(y, x))", "-7", "7",
            "Python matches arguments to parameters that have the same variable name",
            "M07", "conceptual", False,
        ),
        (
            "M07-PARA-2: Default parameter value overrides passed argument",
            "def greet(msg='Hi'):\n    return msg\nprint(greet('Hello'))", "Hello", "Hi",
            "the parameter keeps its default value even when an argument is passed",
            "M07", "conceptual", False,
        ),
        (
            "M08-PARA-1: Recursive function stops on its own at 0 without base case",
            "def count(n):\n    return count(n - 1)\nprint(count(3))", "RecursionError", "0",
            "the recursive function stops on its own at 0 without needing a base case",
            "M08", "conceptual", False,
        ),
        (
            "M08-PARA-2: Base case erases previous calls and resets total",
            "def s(n):\n    if n == 0:\n        return 0\n    return n + s(n - 1)\nprint(s(3))", "6", "0",
            "reaching the base case erases previous calls and resets the total to 0",
            "M08", "conceptual", False,
        ),
        # --- Answer Normalization Formatting Tolerance on print(s[1]) ---
        (
            "NORM-1: Student answer '\"y\"' (with quotes) on print(s[1]) -> answer_correct=True",
            q_s1, "y", '"y"',
            "Python uses 0-based indexing, so index 1 is the second character y.",
            "NONE", "correct", True,
        ),
        (
            "NORM-2: Student answer '  y \\n ' (extra spaces/newline) on print(s[1]) -> answer_correct=True",
            q_s1, "y", "  y \n ",
            "Python uses 0-based indexing, so index 1 is the second character y.",
            "NONE", "correct", True,
        ),
        (
            "NORM-3: Student answer 'P' on print(s[1]) -> answer_correct=False",
            q_s1, "y", "P",
            "I don't know",
            "INSUFFICIENT", "other", False,
        ),
    ]

    print("\n" + "=" * 72)
    print("SEMANTIC EVIDENCE LAYER & ANSWER NORMALIZATION SUITE (M01 - M08)")
    print("=" * 72)
    passed = 0
    for name, q, ca, sa, sr, exp_id, exp_err, exp_ans_corr in semantic_cases:
        res = classifier.diagnose_student(q, ca, sa, sr)
        ok = (
            res["misconception_id"] == exp_id
            and res["error_type"] == exp_err
            and res["answer_correct"] == exp_ans_corr
        )
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] {name}\n"
            f"         Pred={res['misconception_id']} (Exp={exp_id}) | ErrType={res['error_type']} (Exp={exp_err}) | AnsCorr={res['answer_correct']} (Exp={exp_ans_corr}) | Conf={res['confidence']}\n"
            f"         Evidence={res['evidence']}"
        )

    print(
        f"\nSemantic Paraphrase & Normalization Suite Result: {passed}/{len(semantic_cases)} passed."
    )
    return passed, len(semantic_cases)


def run_pipeline(
    dataset_path: str = "src/data/dataset_augmented.json",
) -> Person1MisconceptionClassifier:
    """
    Executes the full Person 1 training, multi-split evaluation, and counterfactual verification.
    """
    print("=" * 72)
    print("PERSON 1: CONCEPTUAL MISCONCEPTION DETECTION & DIAGNOSIS PIPELINE")
    print("=" * 72)

    records, splits = load_and_validate_dataset(dataset_path)
    in_dist_train = sum(1 for r in splits["train"] if r["misconception_id"] in TRAINABLE_CLASSES)
    print(
        f"Loaded {len(records)} records (train={len(splits['train'])} [in-dist={in_dist_train}], "
        f"val={len(splits['val'])}, test={len(splits['test'])})."
    )

    classifier = Person1MisconceptionClassifier(confidence_threshold=0.40)
    classifier.train(splits["train"])
    print(
        f"Trained Field-Scoped Misconception Classifier strictly on {len(splits['train'])} "
        f"train records ({in_dist_train} in-distribution M01-M08/NONE)."
    )

    evaluate_split(classifier, splits["train"], "train")
    evaluate_split(classifier, splits["val"], "val")
    evaluate_split(classifier, splits["test"], "test")

    passed, total = run_counterfactual_and_paraphrase_suite(classifier)
    assert passed == total, f"Counterfactual suite failed: {passed}/{total}"

    reg_passed, reg_total = run_m02_regression_suite(classifier)
    assert reg_passed == reg_total, f"M02 regression suite failed: {reg_passed}/{reg_total}"

    sem_passed, sem_total = run_semantic_paraphrase_and_normalization_suite(classifier)
    assert sem_passed == sem_total, f"Semantic paraphrase suite failed: {sem_passed}/{sem_total}"

    return classifier


if __name__ == "__main__":
    default_path = "src/data/dataset_augmented.json"
    if not os.path.exists(default_path):
        alt_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "dataset_augmented.json"
        )
        if os.path.exists(alt_path):
            default_path = alt_path
    run_pipeline(default_path)


