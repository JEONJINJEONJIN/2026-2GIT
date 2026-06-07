import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.manifest import RunManifest, file_sha256, save_manifest


class TestManifest(unittest.TestCase):
    def test_save_manifest_writes_json(self):
        manifest = RunManifest(
            run_id="run_001",
            model_name="test-model",
            condition="baseline",
            trait="agreeableness",
            target_direction="high",
            prompt_style="none",
            uses_as=False,
            uses_pas=False,
            data_split="dev",
            seed=42,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            save_manifest(manifest, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["run_id"], "run_001")
        self.assertEqual(loaded["condition"], "baseline")

    def test_file_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.txt"
            path.write_text("abc", encoding="utf-8")
            self.assertEqual(
                file_sha256(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


if __name__ == "__main__":
    unittest.main()

