import csv
import tempfile
import unittest
from pathlib import Path

from src.data.bfi import (
    BFIItem,
    apply_bfi_text_overlay,
    load_bfi_items,
    load_bfi_text_overlay,
    normalize_likert_score,
    reverse_likert_score,
    score_bfi_responses,
)


class TestBFIScoring(unittest.TestCase):
    def test_reverse_likert_score(self):
        self.assertEqual(reverse_likert_score(1), 5)
        self.assertEqual(reverse_likert_score(5), 1)
        self.assertEqual(reverse_likert_score(3), 3)

    def test_normalize_likert_score(self):
        self.assertAlmostEqual(normalize_likert_score(1), 0.0)
        self.assertAlmostEqual(normalize_likert_score(3), 0.5)
        self.assertAlmostEqual(normalize_likert_score(5), 1.0)

    def test_score_bfi_responses_applies_reverse_scoring(self):
        items = [
            BFIItem("bfi_01", "extraversion", reverse_scored=False),
            BFIItem("bfi_02", "extraversion", reverse_scored=True),
        ]
        scores = score_bfi_responses(items, {"bfi_01": 5, "bfi_02": 1})
        self.assertAlmostEqual(scores["extraversion"], 1.0)

    def test_score_bfi_responses_requires_complete_responses(self):
        items = [BFIItem("bfi_01", "agreeableness")]
        with self.assertRaises(ValueError):
            score_bfi_responses(items, {})

    def test_load_bfi_items_from_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bfi.csv"
            with open(path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["item_id", "trait", "reverse_scored", "text"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "item_id": "bfi_01",
                        "trait": "openness",
                        "reverse_scored": "true",
                        "text": "",
                    }
                )
            items = load_bfi_items(path)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].trait, "openness")
        self.assertTrue(items[0].reverse_scored)

    def test_load_bfi_items_applies_text_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = root / "bfi.csv"
            overlay_path = root / "bfi_text.local.csv"
            metadata_path.write_text(
                "\n".join(
                    [
                        "item_id,trait,reverse_scored,text",
                        "bfi_01,openness,false,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            overlay_path.write_text(
                "item_id,text\nbfi_01,I like complex ideas.\n",
                encoding="utf-8",
            )

            items = load_bfi_items(metadata_path, text_path=overlay_path)

        self.assertEqual(items[0].text, "I like complex ideas.")

    def test_text_overlay_rejects_unknown_item_id(self):
        items = [BFIItem("bfi_01", "openness")]
        with self.assertRaises(ValueError):
            apply_bfi_text_overlay(items, {"bfi_99": "Unknown item"})

    def test_text_overlay_requires_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("item_id\nbfi_01\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_bfi_text_overlay(path)


if __name__ == "__main__":
    unittest.main()
