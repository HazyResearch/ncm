import copy
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.metrics import sudoku_metrics
from evaluation.shards import load_shards


BOARD = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def save(self, name, part):
        path = self.directory / name
        path.write_bytes(pickle.dumps(part))
        return path

    def part(self, indices, **kwargs):
        return dict(idxs=indices, hyps=[BOARD] * len(indices), refs=[BOARD] * len(indices), **kwargs)

    def test_short_and_empty_predictions_do_not_receive_perfect_cell_scores(self):
        result = sudoku_metrics([BOARD[:3], ""], [BOARD, BOARD])
        self.assertAlmostEqual(result["cell_accuracy"], 3 / 162)
        self.assertEqual(result["solved"], 0)
        self.assertEqual(result["malformed"], 2)

    def test_normal_sudoku_scores_and_separators(self):
        separated = "|".join(BOARD[i:i + 9] for i in range(0, 81, 9))
        result = sudoku_metrics([separated], [separated])
        self.assertEqual(result, dict(n=1, solved=1, cell_accuracy=1.0, valid=1, malformed=0))
        invalid = sudoku_metrics(["1" * 81], [BOARD])
        self.assertEqual(invalid["valid"], 0)
        self.assertEqual(invalid["malformed"], 0)

    def test_bad_references_are_rejected(self):
        for refs in ([BOARD[:3]], [], ["0" * 81]):
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                sudoku_metrics([BOARD], refs)

    def test_duplicate_missing_and_misaligned_samples_are_rejected(self):
        examples = [
            [self.part([0]), self.part([0])],
            [self.part([0]), self.part([2])],
            [dict(idxs=[0, 1], hyps=[BOARD], refs=[BOARD])],
            [self.part([0, 0])],
            [self.part([])],
        ]
        for parts in examples:
            paths = [self.save(f"part{i}.pkl", p) for i, p in enumerate(parts)]
            with self.subTest(parts=parts), self.assertRaises(ValueError):
                load_shards(paths)

    def test_unconditional_local_indices_can_repeat_across_seed_shards(self):
        paths = [self.save(f"part{s}.pkl", self.part([0, 1], seeds=[s, s], toks=[[1], [2]], raw_ids=[[1], [2]])) for s in (42, 0)]
        self.assertEqual(len(load_shards(paths, unconditional=True)), 2)
        with self.assertRaisesRegex(ValueError, "seed"):
            load_shards([paths[0], paths[0]], unconditional=True)

    def test_provenance_and_expected_shards_are_checked(self):
        first = self.part([0], metadata=dict(run_id="new", shard=0, nshards=2, policy={"cfg": 0}))
        second = self.part([1], metadata=dict(run_id="new", shard=1, nshards=2, policy={"cfg": 0}))
        paths = [self.save("0.pkl", first), self.save("1.pkl", second)]
        self.assertEqual(len(load_shards(paths, run_id="new")), 2)
        with self.assertRaisesRegex(ValueError, "different run"):
            load_shards(paths, run_id="old")
        with self.assertRaisesRegex(ValueError, "missing shard"):
            load_shards(paths[:1], run_id="new")
        changed = copy.deepcopy(second)
        changed["metadata"]["policy"]["cfg"] = 1
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            load_shards([paths[0], self.save("changed.pkl", changed)])

    def test_merge_cli_uses_only_explicit_shards(self):
        cfg = self.directory / "config.yaml"
        cfg.write_text("task: sudoku-easy\neval:\n  metric: exact\n", encoding="utf-8")
        good = self.save("shard_0.pkl", self.part([0]))
        self.save("shard_9.pkl", self.part([0]))  # stale output from another run
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts/eval_merge.py"), "--config", str(cfg), "--shards", str(good),
        ], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("merged 1 shards", result.stdout)
        self.assertIn("malformed", result.stdout)

    def test_truncated_tail_is_rejected_even_when_indices_have_no_gaps(self):
        part = self.part([0], metadata=dict(run_id="run", shard=0, nshards=1, n_inputs=2))
        with self.assertRaisesRegex(ValueError, "missing conditional"):
            load_shards([self.save("truncated.pkl", part)])


if __name__ == "__main__":
    unittest.main()
