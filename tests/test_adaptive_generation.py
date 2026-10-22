import contextlib
import io
from types import SimpleNamespace
import unittest

import torch
from omegaconf import OmegaConf

from evaluation.sudoku import is_sudoku_solution
from models.next_concept import GenerationResult, NextConceptTransformer
from models.quant import MultiscaleResidualVectorQuantizer


def tiny_model(query_level0=True, use_rope=True, learned_sampling=False):
    torch.manual_seed(7)
    levels = OmegaConf.create({"use_manual_levels": True, "manual_levels": [1, 2, 4]})
    vq = MultiscaleResidualVectorQuantizer(
        [4, 4, 4], embed_dim=4, levels=levels,
        learned_l1_sampling=learned_sampling, learned_l1_cascade=learned_sampling,
    )
    tokenizer = SimpleNamespace(
        embed_dim=4, vq=vq, encoder=SimpleNamespace(wte=torch.nn.Embedding(12, 4)),
    )
    config = OmegaConf.create({
        "n_embd": 8, "n_head": 2, "n_layer": 1, "dropout": 0.0, "bias": True,
        "vocab_size": [4] * (3 if query_level0 else 2), "prefix_len": 3,
        "use_level_encoding": True, "query_level0": query_level0,
        "predict_level0": not query_level0, "raw_token_prefix": True,
        "use_rope": use_rope, "prefix_pad_token_id": 0,
    })
    with contextlib.redirect_stdout(io.StringIO()):
        model = NextConceptTransformer(tokenizer, config).eval()
    return model, vq


class AdaptiveGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model, self.vq = tiny_model()
        self.prefix = torch.tensor([[1, 2, 0], [3, 0, 0], [4, 5, 6]])

    def test_no_exit_matches_complete_generation_and_latents(self):
        codes = self.model.generate(None, self.prefix, temperature=0)
        result = self.model.generate(None, self.prefix, temperature=0, return_info=True)
        self.assertIsInstance(result, GenerationResult)
        self.assertTrue(torch.equal(codes, result.indices))
        torch.testing.assert_close(result.latents, self.vq.idx_to_z_q(codes))
        self.assertEqual(result.levels_used.tolist(), [3, 3, 3])
        self.assertEqual(result.stopped.tolist(), [False] * 3)

    def test_individual_exits_compact_batch_and_preserve_order(self):
        full = self.model.generate(None, self.prefix, temperature=0)
        seen = []

        def stop(level, latents, ids):
            seen.append((level, ids.tolist(), len(latents)))
            return ids == {1: 1, 2: 0, 3: 2}[level]

        result = self.model.generate(None, self.prefix, temperature=0, stop_fn=stop, return_info=True)
        self.assertEqual(seen, [(1, [0, 1, 2], 3), (2, [0, 2], 2), (3, [2], 1)])
        self.assertEqual(result.levels_used.tolist(), [2, 1, 3])
        self.assertTrue(result.stopped.all())
        for i, used in enumerate(result.levels_used.tolist()):
            width = sum(self.model.levels[:used])
            self.assertTrue(torch.equal(result.indices[i, :width], full[i, :width]))
            self.assertTrue((result.indices[i, width:] == -1).all())
            expected = self.vq.idx_to_z_q(full[i:i + 1, :width])
            torch.testing.assert_close(result.latents[i:i + 1], expected)

    def test_budget_and_minimum_levels(self):
        seen = []

        def stop(level, latents, ids):
            seen.append(level)
            return torch.ones_like(ids, dtype=torch.bool)

        result = self.model.generate(None, self.prefix, temperature=0, max_levels=2,
                                     min_levels=2, stop_fn=stop, return_info=True)
        self.assertEqual(seen, [2])
        self.assertEqual(result.levels_used.tolist(), [2] * 3)
        self.assertEqual(result.latents.shape, (3, 4, 4))
        capped = self.model.generate(None, self.prefix, temperature=0, max_levels=1)
        self.assertEqual(capped.shape, (3, 1))
        full = self.model.generate(None, self.prefix, temperature=0)
        self.assertTrue(torch.equal(capped, full[:, :1]))

    def test_budget_exhaustion_is_not_a_verified_exit(self):
        result = self.model.generate(None, self.prefix, temperature=0, max_levels=1,
                                     stop_fn=lambda level, z, ids: ids < 0, return_info=True)
        self.assertEqual(result.levels_used.tolist(), [1] * 3)
        self.assertFalse(result.stopped.any())

    def test_other_level0_and_position_modes(self):
        for query in (True, False):
            for rope in (True, False):
                with self.subTest(query=query, rope=rope):
                    model, vq = tiny_model(query, rope)
                    codes = model.generate(None, self.prefix, temperature=0)
                    result = model.generate(None, self.prefix, temperature=0, max_levels=2, return_info=True)
                    self.assertTrue(torch.equal(result.indices, codes[:, :3]))
                    torch.testing.assert_close(result.latents, vq.idx_to_z_q(codes[:, :3]))

    def test_explicit_level0_codes(self):
        idx = torch.tensor([[0], [1], [2]])
        result = self.model.generate(idx, self.prefix, temperature=0, max_levels=1, return_info=True)
        self.assertTrue(torch.equal(result.indices, idx))
        torch.testing.assert_close(result.latents, self.vq.idx_to_z_q(idx))

    def test_learned_upsampling_at_partial_exits(self):
        model, vq = tiny_model(learned_sampling=True)
        for budget in (1, 2, 3):
            with self.subTest(budget=budget):
                result = model.generate(None, self.prefix, temperature=0, max_levels=budget, return_info=True)
                torch.testing.assert_close(result.latents, vq.idx_to_z_q(result.indices))

    def test_callback_exception_is_cleared(self):
        def fail(*args):
            raise RuntimeError("verification failed")

        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            self.model.generate(None, self.prefix, temperature=0, stop_fn=fail, return_info=True)
        self.assertIsNone(self.model._cfg)
        self.assertEqual(self.model.generate(None, self.prefix, temperature=0).shape, (3, 7))

    def test_invalid_options_and_masks(self):
        options = [
            {"max_levels": 0}, {"max_levels": 4}, {"max_levels": True},
            {"max_levels": 1.5}, {"min_levels": 0}, {"max_levels": 1, "min_levels": 2},
            {"stop_fn": lambda *args: True}, {"return_info": True, "cfg_scale": 1},
            {"return_info": True, "use_kv_cache": True},
            {"return_info": True, "stop_fn": lambda *args: torch.ones(3)},
            {"return_info": True, "stop_fn": lambda *args: torch.ones(1, dtype=torch.bool)},
        ]
        for kwargs in options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model.generate(None, self.prefix, temperature=0, **kwargs)
        self.assertIsNone(self.model._cfg)
        # An exception in verification must not poison the next generation call.
        self.assertEqual(self.model.generate(None, self.prefix, temperature=0).shape, (3, 7))

    def test_partial_reconstruction_rejects_incomplete_levels(self):
        for shape in ((1, 0), (1, 2), (1, 8), (3,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.vq.idx_to_z_q(torch.zeros(shape, dtype=torch.long))

    def test_guided_budget_matches_full_prefix(self):
        full = self.model.generate(None, self.prefix, temperature=0, cfg_scale=1, cfg_null_eos=11)
        capped = self.model.generate(None, self.prefix, temperature=0, cfg_scale=1, cfg_null_eos=11, max_levels=2)
        self.assertEqual(capped.shape, (3, 3))
        self.assertTrue(torch.equal(capped, full[:, :3]))


class SudokuVerificationTests(unittest.TestCase):
    board = [int(c) for c in (
        "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
    )]

    def test_solution_and_clues(self):
        clues = [value if i % 3 == 0 else 0 for i, value in enumerate(self.board)]
        self.assertTrue(is_sudoku_solution(self.board, clues))
        clues[0] = 1
        self.assertFalse(is_sudoku_solution(self.board, clues))

    def test_row_separators_and_prompt_delimiters(self):
        separated = []
        for start in range(0, 81, 9):
            if start:
                separated.append(10)
            separated.extend(self.board[start:start + 9])
        self.assertTrue(is_sudoku_solution(separated, [11] + separated + [11]))
        malformed = separated.copy()
        malformed[9], malformed[10] = malformed[10], malformed[9]
        self.assertFalse(is_sudoku_solution(malformed, self.board))

    def test_malformed_invalid_and_incomplete_boards(self):
        invalid = self.board.copy()
        invalid[0] = invalid[1]
        for candidate in ([], self.board[:1], self.board[:-1], self.board + [1],
                          [0] * 81, [11] * 81, invalid):
            with self.subTest(candidate=candidate):
                self.assertFalse(is_sudoku_solution(candidate, self.board))
        self.assertFalse(is_sudoku_solution(self.board, [0]))


if __name__ == "__main__":
    unittest.main()
