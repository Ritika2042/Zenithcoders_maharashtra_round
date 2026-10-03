import copy
import json
import os
import tempfile
import unittest

from relearn import (
    Intervention,
    InterventionError,
    InterventionNotFoundError,
    get_intervention,
)
from relearn.intervention import (
    DEFAULT_KB_PATH,
    load_interventions,
    parse_intervention,
)


def valid_entry(**overrides):
    e = {
        "misconception_id": "M02",
        "misconception": "Integer vs Floating-Point Division",
        "explanation": "explanation text",
        "example": "print(7 / 2)",
        "hint": "hint text",
        "common_error": "common error text",
        "follow_up_questions": ["q1?", "q2?"],
    }
    e.update(overrides)
    return e


class TestGetIntervention(unittest.TestCase):
    def setUp(self):
        self.m02 = get_intervention("M02")

    def test_m02_loads(self):
        self.assertIsInstance(self.m02, Intervention)

    def test_id_and_name(self):
        self.assertEqual(self.m02.misconception_id, "M02")
        self.assertEqual(self.m02.misconception, "Integer vs Floating-Point Division")

    def test_text_fields_exist(self):
        for field in ("explanation", "example", "hint", "common_error"):
            value = getattr(self.m02, field)
            self.assertIsInstance(value, str)
            self.assertTrue(value.strip(), field)

    def test_follow_up_questions_multiple(self):
        qs = self.m02.follow_up_questions
        self.assertGreater(len(qs), 1)
        self.assertTrue(all(isinstance(q, str) and q.strip() for q in qs))

    def test_immutable(self):
        with self.assertRaises(Exception):
            self.m02.hint = "changed"
        self.assertIsInstance(self.m02.follow_up_questions, tuple)

    def test_default_kb_file_is_valid(self):
        self.assertTrue(DEFAULT_KB_PATH.exists())
        self.assertIn("M02", load_interventions())

    def test_unknown_id_raises(self):
        for bad in ("M99", "m02", "", None, 2):
            with self.assertRaises(InterventionNotFoundError):
                get_intervention(bad)

    def test_valid_id_without_content_raises(self):
        with self.assertRaises(InterventionNotFoundError) as cm:
            get_intervention("M01")
        self.assertIn("M01", str(cm.exception))

    def test_not_found_is_intervention_error(self):
        self.assertTrue(issubclass(InterventionNotFoundError, InterventionError))
        self.assertTrue(issubclass(InterventionError, ValueError))


class TestParseIntervention(unittest.TestCase):
    def test_valid(self):
        i = parse_intervention(valid_entry())
        self.assertEqual(i.follow_up_questions, ("q1?", "q2?"))

    def test_extra_fields_ignored(self):
        i = parse_intervention(valid_entry(difficulty="easy"))
        self.assertEqual(i.misconception_id, "M02")

    def test_non_mapping_rejected(self):
        for bad in (None, [], "M02"):
            with self.assertRaises(InterventionError):
                parse_intervention(bad)

    def test_missing_fields_rejected(self):
        for key in valid_entry():
            e = valid_entry()
            del e[key]
            with self.assertRaises(InterventionError) as cm:
                parse_intervention(e)
            self.assertIn(key, str(cm.exception))

    def test_invalid_id_rejected(self):
        for bad in ("M09", "m02", None, 2):
            with self.assertRaises(InterventionError):
                parse_intervention(valid_entry(misconception_id=bad))

    def test_name_mismatch_rejected(self):
        with self.assertRaises(InterventionError):
            parse_intervention(valid_entry(misconception="Zero-Based Indexing"))

    def test_empty_or_bad_text_rejected(self):
        for field in ("explanation", "example", "hint", "common_error"):
            for bad in ("", "   ", None, 5, ["x"]):
                with self.assertRaises(InterventionError, msg=f"{field}={bad!r}"):
                    parse_intervention(valid_entry(**{field: bad}))

    def test_malformed_follow_up_questions_rejected(self):
        for bad in ([], "a question?", None, {"a": 1}, [""], ["ok?", "  "], ["ok?", 3], [None]):
            with self.assertRaises(InterventionError, msg=repr(bad)):
                parse_intervention(valid_entry(follow_up_questions=bad))


class TestLoadInterventions(unittest.TestCase):
    def write(self, content):
        fd, path = tempfile.mkstemp(suffix=".json")
        self.addCleanup(os.remove, path)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content if isinstance(content, str) else json.dumps(content))
        return path

    def test_custom_path_loads(self):
        path = self.write([valid_entry()])
        self.assertEqual(get_intervention("M02", path=path).hint, "hint text")

    def test_missing_file(self):
        with self.assertRaises(InterventionError):
            load_interventions("/nonexistent/interventions.json")

    def test_invalid_json(self):
        with self.assertRaises(InterventionError):
            load_interventions(self.write("{not json"))

    def test_top_level_not_list(self):
        with self.assertRaises(InterventionError):
            load_interventions(self.write({"M02": valid_entry()}))

    def test_malformed_entry_in_file_rejected(self):
        path = self.write([valid_entry(hint="")])
        with self.assertRaises(InterventionError) as cm:
            load_interventions(path)
        self.assertIn("entry 0", str(cm.exception))

    def test_name_mismatch_in_file_rejected(self):
        path = self.write([valid_entry(misconception="Operator Precedence")])
        with self.assertRaises(InterventionError):
            get_intervention("M02", path=path)

    def test_duplicate_ids_rejected(self):
        path = self.write([valid_entry(), copy.deepcopy(valid_entry())])
        with self.assertRaises(InterventionError):
            load_interventions(path)


if __name__ == "__main__":
    unittest.main()
