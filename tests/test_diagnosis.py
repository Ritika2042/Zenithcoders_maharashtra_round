import subprocess
import sys
import unittest

from relearn import MISCONCEPTIONS, Diagnosis, DiagnosisError, parse_diagnosis


def valid(**overrides):
    d = {
        "misconception_id": "M02",
        "misconception": "Integer vs Floating-Point Division",
        "confidence": 0.91,
    }
    d.update(overrides)
    return d


def without(key):
    d = valid()
    del d[key]
    return d


class TestDiagnosis(unittest.TestCase):
    def test_valid_diagnosis(self):
        d = parse_diagnosis(valid())
        self.assertEqual(
            d, Diagnosis("M02", "Integer vs Floating-Point Division", 0.91)
        )
        with self.assertRaises(Exception):
            d.confidence = 0.5  # immutable

    def test_all_ids_accepted(self):
        self.assertEqual(
            list(MISCONCEPTIONS), [f"M0{i}" for i in range(1, 9)]
        )
        for mid, name in MISCONCEPTIONS.items():
            d = parse_diagnosis(
                {"misconception_id": mid, "misconception": name, "confidence": 1}
            )
            self.assertEqual(d.misconception_id, mid)

    def test_unknown_id_rejected(self):
        for bad in ("M09", "m02", "", None, 2):
            with self.assertRaises(DiagnosisError):
                parse_diagnosis(valid(misconception_id=bad))

    def test_missing_fields_rejected(self):
        for key in ("misconception_id", "misconception", "confidence"):
            with self.assertRaises(DiagnosisError) as cm:
                parse_diagnosis(without(key))
            self.assertIn(key, str(cm.exception))

    def test_mismatched_name_rejected(self):
        with self.assertRaises(DiagnosisError):
            parse_diagnosis(valid(misconception="Zero-Based Indexing"))
        with self.assertRaises(DiagnosisError):
            parse_diagnosis(valid(misconception="integer vs floating-point division"))

    def test_confidence_below_zero_rejected(self):
        with self.assertRaises(DiagnosisError):
            parse_diagnosis(valid(confidence=-0.01))

    def test_confidence_above_one_rejected(self):
        with self.assertRaises(DiagnosisError):
            parse_diagnosis(valid(confidence=1.01))

    def test_confidence_bounds_inclusive(self):
        self.assertEqual(parse_diagnosis(valid(confidence=0)).confidence, 0.0)
        self.assertEqual(parse_diagnosis(valid(confidence=1.0)).confidence, 1.0)

    def test_string_confidence_rejected(self):
        with self.assertRaises(DiagnosisError):
            parse_diagnosis(valid(confidence="0.91"))

    def test_boolean_confidence_rejected(self):
        for b in (True, False):
            with self.assertRaises(DiagnosisError):
                parse_diagnosis(valid(confidence=b))

    def test_nan_rejected(self):
        with self.assertRaises(DiagnosisError):
            parse_diagnosis(valid(confidence=float("nan")))

    def test_infinity_rejected(self):
        for inf in (float("inf"), float("-inf")):
            with self.assertRaises(DiagnosisError):
                parse_diagnosis(valid(confidence=inf))

    def test_non_dict_rejected(self):
        for bad in (None, [], "M02", 3):
            with self.assertRaises(DiagnosisError):
                parse_diagnosis(bad)

    def test_additional_fields_accepted(self):
        d = parse_diagnosis(valid(model_version="v1", extra={"a": 1}))
        self.assertEqual(d.misconception_id, "M02")
        self.assertFalse(hasattr(d, "model_version"))

    def test_error_is_value_error(self):
        self.assertTrue(issubclass(DiagnosisError, ValueError))

    def test_no_ml_dependency(self):
        code = (
            "import sys, relearn\n"
            "ml = {'numpy','sklearn','scipy','torch','tensorflow','pandas',"
            "'pydantic','joblib'}\n"
            "loaded = ml & set(sys.modules)\n"
            "sys.exit(1 if loaded else 0)\n"
        )
        r = subprocess.run([sys.executable, "-c", code])
        self.assertEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
