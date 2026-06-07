import unittest
from pathlib import Path

from src.demo.bigfive_demo_validation import (
    check_demo_fallback,
    check_demo_paths,
    demo_validation_has_errors,
)


class TestBigFiveDemoValidation(unittest.TestCase):
    def test_check_demo_paths(self):
        checks = check_demo_paths(Path(__file__).resolve().parent.parent)

        self.assertEqual(checks[0].status, "ok")

    def test_check_demo_fallback(self):
        checks = check_demo_fallback(Path(__file__).resolve().parent.parent)

        self.assertEqual(checks[0].status, "ok")
        self.assertIn("scenario=", checks[0].details)

    def test_demo_validation_has_errors(self):
        self.assertFalse(demo_validation_has_errors([]))


if __name__ == "__main__":
    unittest.main()
