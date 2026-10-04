#!/usr/bin/env python3
"""
Build generated_all / generated_clean / generated_rejected / merge_report from the 4
pasted synthetic batches. Never reads-for-write or modifies dataset_160.json (read-only
collision check only, if the file is present). No model training, no pipeline changes.

Run from the repo root:  python build_generated_dataset.py [--data-dir src/data]
"""
import argparse, copy, json, re, sys
from collections import Counter, OrderedDict
from difflib import SequenceMatcher
from pathlib import Path

REQUIRED = ["id", "sample_id", "question_group", "concept", "question_format", "question",
            "correct_answer", "student_answer", "student_reasoning", "answer_correct",
            "misconception_id", "misconception_variant", "secondary_misconception_ids",
            "evidence_basis", "annotator_rationale", "error_type", "source", "split"]
STR_NONEMPTY = ["id", "sample_id", "question_group", "concept", "question_format", "question",
                "correct_answer", "student_answer", "misconception_variant", "evidence_basis",
                "annotator_rationale", "error_type", "source", "split"]
STR_ALLOW_EMPTY = ["student_reasoning"]   # bare-answer samples legitimately have no reasoning
MIS_OK = {"M01","M02","M03","M04","M05","M06","M07","M08","NONE","INSUFFICIENT","OOS"}
SEC_OK = {"M01","M02","M03","M04","M05","M06","M07","M08"}
ERR_OK = {"correct","careless","typo","syntax_error","trace_error","conceptual","other"}
ERR_MAP = {"arithmetic":"careless","syntax":"syntax_error","no_evidence":"other","none":"correct"}

def norm_q(q):  return re.sub(r"\s+", " ", q).strip().lower()
def norm_t(t):  return re.sub(r"\s+", " ", str(t)).strip().lower()
def code_sig(q):
    parts = q.split("\n\n", 1)
    return norm_q(parts[1] if len(parts) > 1 else q)
def sig_match(a, b):
    return a == b or (min(len(a), len(b)) >= 12 and (a in b or b in a))

def validate(s):
    probs = []
    if not isinstance(s, dict): return ["not a JSON object"]
    for k in REQUIRED:
        if k not in s: probs.append(f"missing field '{k}'")
    if probs: return probs
    for k in STR_NONEMPTY:
        if not isinstance(s[k], str) or not s[k].strip(): probs.append(f"field '{k}' empty or not a string")
    for k in STR_ALLOW_EMPTY:
        if not isinstance(s[k], str): probs.append(f"field '{k}' not a string")
    if not isinstance(s["answer_correct"], bool): probs.append("answer_correct is not a boolean")
    if s["misconception_id"] not in MIS_OK: probs.append(f"invalid misconception_id '{s['misconception_id']}'")
    sec = s["secondary_misconception_ids"]
    if not isinstance(sec, list) or any(x not in SEC_OK for x in sec): probs.append("secondary_misconception_ids not a list of M01-M08")
    et = s["error_type"]
    if et not in ERR_OK and et not in ERR_MAP: probs.append(f"error_type '{et}' not allowed and has no mapping")
    return probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="src/data")
    a = ap.parse_args()
    d = Path(a.data_dir); raw = d / "raw_batches"

    batches = OrderedDict()
    parse_status = OrderedDict()
    for n in range(1, 5):
        name = f"batch{n}"
        try:
            obj = json.loads((raw / f"{name}.json").read_text(encoding="utf-8"))
            batches[name] = obj["samples"]; parse_status[name] = "ok"
        except Exception as e:
            batches[name] = []; parse_status[name] = f"FAILED: {e}"

    all_samples, batch_of = [], []
    for name, lst in batches.items():
        for s in lst: all_samples.append(s); batch_of.append(name)

    # read-only collision check against the original dataset, if present
    orig_path = d / "dataset_160.json"
    orig_ids, orig_sids, orig_qs, orig_status = set(), set(), set(), "not found (collision check skipped)"
    if orig_path.exists():
        try:
            o = json.loads(orig_path.read_text(encoding="utf-8"))
            rows = o["samples"] if isinstance(o, dict) and "samples" in o else o
            for r in rows:
                if isinstance(r, dict):
                    if "id" in r: orig_ids.add(r["id"])
                    if "sample_id" in r: orig_sids.add(r["sample_id"])
                    if "question" in r: orig_qs.add(norm_q(r["question"]))
            orig_status = f"read-only check against {len(rows)} original rows"
        except Exception as e:
            orig_status = f"present but unreadable ({e}); collision check skipped"

    clean, rejected, normalizations, near_warn, consistency = [], [], [], [], []
    seen_id, seen_sid, seen_q, kept_info = {}, {}, {}, []
    reason_counts, dup_counts = Counter(), Counter()

    for s, b in zip(all_samples, batch_of):
        reasons, dup_of = [], None
        probs = validate(s)
        if probs:
            reasons.append(("malformed", "; ".join(probs)))
        else:
            if s["id"] in orig_ids:  reasons.append(("collides_with_dataset_160_id", f"id '{s['id']}' already in dataset_160.json"))
            if s["sample_id"] in orig_sids: reasons.append(("collides_with_dataset_160_sample_id", f"sample_id '{s['sample_id']}' already in dataset_160.json"))
            if norm_q(s["question"]) in orig_qs: reasons.append(("collides_with_dataset_160_question", "identical question text already in dataset_160.json"))
            if s["id"] in seen_id:
                reasons.append(("duplicate_id", f"id '{s['id']}' already used by an earlier kept sample ({seen_id[s['id']]})")); dup_of = seen_id[s["id"]]
            if s["sample_id"] in seen_sid:
                reasons.append(("duplicate_sample_id", f"sample_id '{s['sample_id']}' already used by {seen_sid[s['sample_id']]}")); dup_of = dup_of or seen_sid[s["sample_id"]]
            nq = norm_q(s["question"])
            if nq in seen_q:
                o = seen_q[nq]
                reasons.append(("exact_duplicate_question", f"question text identical (ignoring whitespace/case) to {o['id']} ({o['sample_id']}, {o['batch']})"))
                dup_of = dup_of or o["id"]
            # near duplicates against kept samples
            if not reasons:
                sg = code_sig(s["question"])
                for k in kept_info:
                    if sig_match(sg, k["sig"]):
                        same_label = s["misconception_id"] == k["s"]["misconception_id"]
                        same_ans = norm_t(s["student_answer"]) == norm_t(k["s"]["student_answer"])
                        ratio = SequenceMatcher(None, norm_t(s["student_reasoning"]), norm_t(k["s"]["student_reasoning"])).ratio()
                        if same_label and same_ans and ratio >= 0.6:
                            reasons.append(("near_duplicate", f"same code/question as {k['s']['id']} ({k['s']['sample_id']}, {k['batch']}), same misconception_id, same student_answer, reasoning similarity {ratio:.2f}"))
                            dup_of = k["s"]["id"]; break
                        else:
                            near_warn.append({"kept_id": s["id"], "kept_sample_id": s["sample_id"], "batch": b,
                                              "similar_to_id": k["s"]["id"], "similar_to_sample_id": k["s"]["sample_id"],
                                              "same_misconception_id": same_label, "same_student_answer": same_ans,
                                              "reasoning_similarity": round(ratio, 2),
                                              "note": "same underlying code, different student reasoning/label/answer; KEPT"})
        if reasons:
            for code, _ in reasons: reason_counts[code] += 1
            first = reasons[0][0]
            if first in ("duplicate_id","duplicate_sample_id","exact_duplicate_question","near_duplicate"): dup_counts[first] += 1
            rej = {"batch": b, "reason_codes": [c for c, _ in reasons], "reason": " | ".join(t for _, t in reasons)}
            if dup_of: rej["duplicate_of_id"] = dup_of
            rej["sample"] = s
            rejected.append(rej); continue

        c = copy.deepcopy(s)
        if c["error_type"] in ERR_MAP:
            normalizations.append({"id": c["id"], "sample_id": c["sample_id"], "batch": b, "field": "error_type", "from": c["error_type"], "to": ERR_MAP[c["error_type"]]})
            c["error_type"] = ERR_MAP[c["error_type"]]
        if not c["answer_correct"] is False or norm_t(c["student_answer"]) != norm_t(c["correct_answer"]): pass
        else: consistency.append({"id": c["id"], "note": "answer_correct is false but student_answer equals correct_answer"})
        clean.append(c)
        seen_id[c["id"]] = c["id"]; seen_sid[c["sample_id"]] = c["id"]
        seen_q[norm_q(c["question"])] = {"id": c["id"], "sample_id": c["sample_id"], "batch": b}
        kept_info.append({"s": c, "sig": code_sig(c["question"]), "batch": b})

    # integrity: clean == raw except for error_type normalisation
    raw_by_id = {s["id"]: s for s in all_samples if isinstance(s, dict) and "id" in s}
    changed = []
    for c in clean:
        r = raw_by_id[c["id"]]
        diff = [k for k in c if c[k] != r[k]]
        if diff and diff != ["error_type"]: changed.append((c["id"], diff))
    assert not changed, f"content changed beyond error_type: {changed}"

    dist = lambda k: dict(sorted(Counter(x[k] for x in clean).items()))
    per_batch = {n: len(v) for n, v in batches.items()}
    rej_per_batch = Counter(r["batch"] for r in rejected)
    clean_ids = {c["id"] for c in clean}
    report = OrderedDict([
        ("parse_status", parse_status),
        ("samples_extracted_per_batch", per_batch),
        ("total_extracted", len(all_samples)),
        ("original_dataset_check", orig_status),
        ("duplicates", {"duplicate_id": reason_counts["duplicate_id"], "duplicate_sample_id": reason_counts["duplicate_sample_id"],
                        "exact_duplicate_question": reason_counts["exact_duplicate_question"], "near_duplicate": reason_counts["near_duplicate"]}),
        ("malformed_samples", reason_counts["malformed"]),
        ("rejected_samples", len(rejected)),
        ("rejected_per_batch", {n: rej_per_batch.get(n, 0) for n in batches}),
        ("rejected_by_reason", dict(reason_counts)),
        ("rejected_detail", [{"id": r["sample"].get("id"), "sample_id": r["sample"].get("sample_id"), "batch": r["batch"], "reason": r["reason"]} for r in rejected]),
        ("final_clean_count", len(clean)),
        ("clean_per_batch", {n: sum(1 for c, b in zip(all_samples, batch_of) if b == n and isinstance(c, dict) and c.get("id") in clean_ids and False) for n in batches}),
        ("distribution_by_misconception_id", dist("misconception_id")),
        ("distribution_by_split", dist("split")),
        ("distribution_by_error_type_after_normalization", dist("error_type")),
        ("error_type_distribution_before_normalization_all_extracted", dict(sorted(Counter(s.get("error_type") for s in all_samples).items()))),
        ("error_type_normalizations_applied", {"count": len(normalizations), "details": normalizations}),
        ("question_format_values_observed_not_changed", dict(sorted(Counter(s.get("question_format") for s in all_samples).items()))),
        ("near_duplicate_warnings_kept", {"count": len(near_warn), "details": near_warn}),
        ("consistency_warnings", consistency),
        ("content_integrity", "clean samples are identical to the extracted samples except error_type normalisation; verified by assertion"),
    ])
    # clean per batch (computed properly)
    bmap = {}
    for s, b in zip(all_samples, batch_of): bmap.setdefault(s.get("id"), b)
    report["clean_per_batch"] = {n: sum(1 for c in clean if bmap.get(c["id"]) == n) for n in batches}

    w = lambda name, obj: (d / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    w("generated_all.json", {"samples": all_samples})
    w("generated_clean.json", {"samples": clean})
    w("generated_rejected.json", {"rejected": rejected})
    w("merge_report.json", report)
    print(json.dumps({k: report[k] for k in ["parse_status","samples_extracted_per_batch","total_extracted","duplicates","malformed_samples","rejected_samples","final_clean_count"]}, indent=2))

if __name__ == "__main__":
    main()
