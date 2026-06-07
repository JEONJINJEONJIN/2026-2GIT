import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "12_run_bigfive_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_bigfive_pilot", SCRIPT_PATH)
run_bigfive_pilot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_bigfive_pilot)


class TestBigFivePilotScript(unittest.TestCase):
    def test_save_execution_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            model_config_path = root / "model.yaml"
            vectors_path = root / "vectors.pt"
            manifest_path = root / "run_manifest.json"
            config_path.write_text("experiment: {}\n", encoding="utf-8")
            model_config_path.write_text("model:\n  name: test\n", encoding="utf-8")
            vectors_path.write_bytes(b"vectors")

            run_bigfive_pilot.save_execution_manifest(
                manifest_path,
                experiment={"name": "v4_test"},
                model_name="test-model",
                conditions=[
                    SimpleNamespace(
                        name="baseline",
                        steering=False,
                        pas=False,
                        vector_mode=None,
                    ),
                    SimpleNamespace(
                        name="as_pas_only",
                        steering=True,
                        pas=True,
                        vector_mode="trait_contrast",
                    ),
                ],
                traits=["agreeableness"],
                target_directions=["high", "low"],
                split="dev",
                seed=42,
                alpha=1.0,
                layer_indices=[18, 21, 24],
                config_path=config_path,
                model_config_path=model_config_path,
                vectors_path=vectors_path,
                input_metadata={"trait_scenario_count": 50},
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["run_id"], "v4_test__seed_42")
        self.assertEqual(manifest["condition"], "baseline,as_pas_only")
        self.assertTrue(manifest["uses_as"])
        self.assertTrue(manifest["uses_pas"])
        self.assertEqual(len(manifest["vector_hash"]), 64)
        self.assertEqual(manifest["extra"]["input_metadata"]["trait_scenario_count"], 50)


if __name__ == "__main__":
    unittest.main()
