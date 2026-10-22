import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from evaluation.gen_ppl import generation_perplexity


class Batch(dict):
    def to(self, device):
        return Batch({key: value.to(device) for key, value in self.items()})


class CharTokenizer:
    pad_token_id = 0

    def __init__(self, *, fast=True, padding="right"):
        self.is_fast = fast
        self.padding_side = padding

    def __call__(self, texts, *, max_length=None, return_offsets_mapping=False, **kwargs):
        texts = [text[:max_length] for text in texts]
        length = max(map(len, texts))
        ids, masks, offsets = [], [], []
        for text in texts:
            tokens = [1 + ord(c) % 6 for c in text]
            n = length - len(tokens)
            pad, real = [0] * n, [1] * len(tokens)
            spans = [(i, i + 1) for i in range(len(tokens))]
            if self.padding_side == "right":
                ids.append(tokens + pad); masks.append(real + pad); offsets.append(spans + [(0, 0)] * n)
            else:
                ids.append(pad + tokens); masks.append(pad + real); offsets.append([(0, 0)] * n + spans)
        result = Batch(input_ids=torch.tensor(ids, dtype=torch.long), attention_mask=torch.tensor(masks))
        if return_offsets_mapping:
            result["offset_mapping"] = torch.tensor(offsets)
        return result


class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.arange(49).view(7, 7).float().sin())
        self.calls = []

    def forward(self, input_ids, attention_mask, use_cache, position_ids=None):
        self.calls.append((use_cache, position_ids))
        return SimpleNamespace(logits=self.table[input_ids])


class GenPPLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_padding_side_fast_tokenizer_and_batch_size_do_not_change_scores(self):
        baseline = None
        for fast in (False, True):
            for padding in ("left", "right"):
                for batch in (1, 3):
                    with self.subTest(fast=fast, padding=padding, batch=batch):
                        model = BigramLM()
                        result = generation_perplexity(
                            model, CharTokenizer(fast=fast, padding=padding),
                            ["", "", ""], ["abcde", "ab", "abcd"], device="cpu", batch_size=batch,
                        )
                        self.assertEqual(result.token_count.tolist(), [4, 1, 3])
                        self.assertTrue(all(not cached for cached, _ in model.calls))
                        if padding == "left":
                            self.assertTrue(all(positions is not None for _, positions in model.calls))
                        if baseline is None:
                            baseline = result.nll_sum
                        torch.testing.assert_close(result.nll_sum, baseline)
                        self.assertAlmostEqual(result.mean_ppl, math.exp(float(baseline.sum()) / 8), places=5)

    def test_prompt_boundary_and_zero_token_examples_are_reported(self):
        result = generation_perplexity(BigramLM(), CharTokenizer(),
                                       ["ab", "", ""], ["cd", "", "x"], device="cpu")
        self.assertEqual(result.token_count.tolist(), [2, 0, 0])
        self.assertEqual(result.unscored_examples, 2)
        self.assertTrue(torch.isnan(result.ppl[1:]).all())

    def test_unscorable_or_invalid_corpora_fail_explicitly(self):
        for prompts, texts, batch in (([], [], 1), ([""], [""], 1), ([""], ["x"], 1), ([""], ["ab"], 0)):
            with self.subTest(texts=texts, batch=batch), self.assertRaises(ValueError):
                generation_perplexity(BigramLM(), CharTokenizer(), prompts, texts, device="cpu", batch_size=batch)

    def test_small_gpt2_left_padding_matches_individual_scoring(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = GPT2LMHeadModel(GPT2Config(
                vocab_size=7, n_embd=16, n_head=2, n_layer=1, n_positions=16,
                bos_token_id=1, eos_token_id=1, pad_token_id=0,
                resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
            )).eval()
        values = []
        for padding, batch in (("right", 1), ("right", 2), ("left", 2)):
            result = generation_perplexity(model, CharTokenizer(padding=padding),
                                           ["", ""], ["abcde", "ab"], device="cpu", batch_size=batch)
            values.append(result.nll_sum)
        for actual in values[1:]:
            torch.testing.assert_close(actual, values[0])


if __name__ == "__main__":
    unittest.main()
