import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.data.trait_convert import (
    assign_split,
    build_split_index,
    convert_raw_trait_rows,
    load_raw_trait_rows,
    normalize_trait_name,
    write_split_index,
)


class TestTraitConvert(unittest.TestCase):
    def test_normalize_trait_name(self):
        self.assertEqual(normalize_trait_name("Agreeableness"), "agreeableness")
        self.assertEqual(normalize_trait_name("Openness to Experience"), "openness")
        self.assertIsNone(normalize_trait_name("Machiavellianism"))

    def test_convert_raw_trait_rows_filters_big_five_and_balances_options(self):
        rows = [
            self._raw_row("Agreeableness", "q1"),
            self._raw_row("Agreeableness", "q2"),
            self._raw_row("Machiavellianism", "q3"),
        ]
        scenarios = convert_raw_trait_rows(rows, split_ratios=(0.5, 0.25, 0.25))

        self.assertEqual(len(scenarios), 2)
        self.assertEqual(scenarios[0]["scenario_id"], "trait_agreeableness_00001")
        self.assertEqual(scenarios[0]["actions"][0]["trait_direction"], "high")
        self.assertEqual(scenarios[1]["actions"][0]["trait_direction"], "low")
        self.assertEqual({action["trait_direction"] for action in scenarios[0]["actions"]}, {"high", "low"})

    def test_assign_split(self):
        self.assertEqual(assign_split(0, 10, (0.7, 0.2, 0.1)), "train")
        self.assertEqual(assign_split(7, 10, (0.7, 0.2, 0.1)), "dev")
        self.assertEqual(assign_split(9, 10, (0.7, 0.2, 0.1)), "test")
        with self.assertRaises(ValueError):
            assign_split(0, 10, (0.5, 0.3, 0.3))

    def test_build_and_write_split_index(self):
        scenarios = convert_raw_trait_rows(
            [
                self._raw_row("Agreeableness", "q1"),
                self._raw_row("Agreeableness", "q2"),
                self._raw_row("Agreeableness", "q3"),
            ],
            split_ratios=(0.34, 0.33, 0.33),
        )
        index = build_split_index(scenarios)
        self.assertIn("agreeableness", index["train"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits.json"
            write_split_index(scenarios, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("trait_agreeableness_00001", text)

    def test_load_raw_trait_rows_jsonl_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            jsonl_path = tmp_path / "trait.jsonl"
            csv_path = tmp_path / "trait.csv"
            row = self._raw_row("Extraversion", "q1")
            jsonl_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)

            self.assertEqual(len(load_raw_trait_rows(jsonl_path)), 1)
            self.assertEqual(len(load_raw_trait_rows(csv_path)), 1)

    def test_load_raw_trait_rows_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first = tmp_path / "a.jsonl"
            second = tmp_path / "b.jsonl"
            first.write_text(json.dumps(self._raw_row("Agreeableness", "q1")) + "\n", encoding="utf-8")
            second.write_text(json.dumps(self._raw_row("Openness", "q2")) + "\n", encoding="utf-8")

            rows = load_raw_trait_rows(tmp_path)

        self.assertEqual(len(rows), 2)

    def test_script_converts_raw_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "raw.jsonl"
            out_path = tmp_path / "scenarios.jsonl"
            rows = [
                self._raw_row("Agreeableness", "q1"),
                self._raw_row("Agreeableness", "q2"),
                self._raw_row("Extraversion", "q3"),
            ]
            raw_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/11_convert_trait_bigfive.py",
                    "--input",
                    str(raw_path),
                    "--output",
                    str(out_path),
                    "--traits",
                    "agreeableness",
                    "--split_ratios",
                    "0.5,0.25,0.25",
                    "--splits_output",
                    str(tmp_path / "splits.json"),
                ],
                cwd=Path(__file__).resolve().parent.parent,
                check=True,
                capture_output=True,
                text=True,
            )

            output_rows = out_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(output_rows), 2)
        self.assertIn("Wrote 2 normalized Big Five scenarios", result.stdout)
        self.assertIn("Wrote split index", result.stdout)

    @staticmethod
    def _raw_row(personality, question):
        return {
            "personality": personality,
            "question": question,
            "response_high1": "High response one.",
            "response_high2": "High response two.",
            "response_low1": "Low response one.",
            "response_low2": "Low response two.",
        }


if __name__ == "__main__":
    unittest.main()
