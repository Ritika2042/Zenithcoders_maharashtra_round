"""Canonical misconception taxonomy (initial prototype).

Implementation-level mapping of misconception ID -> canonical name.
To change the taxonomy, edit MISCONCEPTIONS only; the diagnosis
interface reads from it and does not need to change.
"""

MISCONCEPTIONS = {
    "M01": "Data Type Confusion",
    "M02": "Integer vs Floating-Point Division",
    "M03": "Assignment (=) vs Equality (==)",
    "M04": "Operator Precedence",
    "M05": "Zero-Based Indexing",
    "M06": "Loop Boundary / Off-by-One Error",
    "M07": "Function Parameter / Argument Confusion",
    "M08": "Recursion / Base-Case Confusion",
}
