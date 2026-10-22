from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"


class MergeTests(unittest.TestCase):
    def run_merge(self, policies):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config = directory / "config.yaml"
            config.write_text("task: sudoku-easy\neval:\n  metric: exact\n", encoding="utf-8")
            for index, policy in enumerate(policies):
                part = {"idxs": [index], "hyps": [SOLUTION], "refs": [SOLUTION]}
                if policy is not None:
                    part["generation"] = dict(
                        adaptive=policy, max_levels=12, min_levels=1,
                        levels_used=[index + 1], verified=[True], seconds=0.25,
                    )
                with (directory / f"shard_{index}.pkl").open("wb") as output:
                    pickle.dump(part, output)
            return subprocess.run(
                [sys.executable, str(ROOT / "scripts/eval_merge.py"),
                 "--config", str(config), "--glob", str(directory / "shard_*.pkl")],
                cwd=ROOT, capture_output=True, text=True,
            )

    def test_adaptive_statistics_and_task_metric(self):
        result = self.run_merge(["sudoku", "sudoku"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mean=1.50, min=1, max=2", result.stdout)
        self.assertIn("verified Sudoku outputs: 2/2", result.stdout)
        self.assertIn("0.500", result.stdout)
        self.assertIn("100.00", result.stdout)

    def test_legacy_shards_still_score(self):
        result = self.run_merge([None, None])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("100.00", result.stdout)

    def test_reject_legacy_and_new_shard_mixture(self):
        result = self.run_merge([None, "sudoku"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot merge legacy shards", result.stderr)

    def test_reject_different_generation_policies(self):
        result = self.run_merge(["sudoku", "other-policy"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("different generation policies", result.stderr)


if __name__ == "__main__":
    unittest.main()
