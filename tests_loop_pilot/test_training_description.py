from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from loop_pilot.report import training_description


class TrainingDescriptionTests(unittest.TestCase):
    def test_global_iteration_checkpoint_is_not_described_as_96_visit_training(self):
        text = training_description({"iteration_policy": "global", "max_visits_per_token": None})
        self.assertIn("누적 iteration", text)
        self.assertIn("강제 공급 없이", text)
        self.assertNotIn("96회", text)

    def test_fixed_budget_is_read_from_source_configuration(self):
        self.assertIn("48회", training_description({"max_visits_per_token": 48}))


if __name__ == "__main__":
    unittest.main()
