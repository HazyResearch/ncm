import unittest

import torch

from tests.helpers import tiny_vqvae
from tokenization.tokenizer import Tokenizer
from tokenization.vqvae_tokenizer import VQVAETokenizer


class SymbolTokenizer(Tokenizer):
    pad_token_id = 16

    def encode(self, text):
        return [ord(c) - ord("a") for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(ord("a") + i) for i in ids)


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_cumulative_snapshots_are_independent(self):
        vq = tiny_vqvae(quant_resi=False).vq
        with torch.no_grad():
            for i, embedding in enumerate(vq.embeddings):
                embedding.weight.fill_(i + 1)
        indices = torch.zeros(2, sum(vq.levels), dtype=torch.long)
        snapshots, native = vq.idx_to_zhats(indices)
        for snapshot, expected in zip(snapshots, [1, 3, 6, 10]):
            torch.testing.assert_close(snapshot, torch.full_like(snapshot, expected))
        snapshots[-1].zero_()
        self.assertEqual(snapshots[0][0, 0, 0].item(), 1)
        self.assertEqual([x.shape[1] for x in native], list(vq.levels))

    def test_encoding_and_index_reconstruction_agree(self):
        for learned in (False, True):
            with self.subTest(learned=learned), torch.no_grad():
                model = tiny_vqvae(learned=learned)
                x = torch.arange(16).view(2, 8) % model.in_vocab_size
                z = model.encode_z(x)
                indices, encoded, native = model.vq.z_to_idx(z, True, True)
                decoded, decoded_native = model.vq.idx_to_zhats(indices)
                for a, b in zip(encoded, decoded):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                for a, b in zip(native, decoded_native):
                    # Area reductions can round differently for different strides.
                    torch.testing.assert_close(a, b)
                torch.testing.assert_close(model.vq.idx_to_z_q(indices), encoded[-1])
                self.assertTrue(torch.equal(model.vq.z_to_idx(z), indices))
                for kwargs, expected in (
                    ({"return_zhats": True}, encoded),
                    ({"return_zlevels": True}, native),
                ):
                    actual_indices, history = model.vq.z_to_idx(z, **kwargs)
                    self.assertTrue(torch.equal(actual_indices, indices))
                    for a, b in zip(history, expected):
                        torch.testing.assert_close(a, b)

    def test_parallel_and_incremental_ncm_inputs_agree(self):
        model = tiny_vqvae(learned=True)
        indices = model.encode(torch.arange(16).view(2, 8) % 17)
        expected = model.vq.idx_to_ncm_input(indices)
        cumulative = torch.zeros(2, 8, model.embed_dim)
        code_offset = target_offset = 0
        for i, length in enumerate(model.vq.levels[:-1]):
            contribution = model.vq.embeddings[i](indices[:, code_offset:code_offset + length])
            cumulative, next_input = model.vq.get_next_autoregressive_input(i, cumulative, contribution)
            end = target_offset + model.vq.levels[i + 1]
            torch.testing.assert_close(next_input, expected[:, target_offset:end], rtol=0, atol=0)
            code_offset += length
            target_offset = end

    def test_reconstruct_matches_forward_and_decode(self):
        model = tiny_vqvae(learned=True)
        x = torch.arange(16).view(2, 8) % 17
        with torch.no_grad():
            expected = model(x)[0]
        torch.testing.assert_close(model.reconstruct(x), expected)
        torch.testing.assert_close(model.reconstruct(x), model.decode(model.encode(x)))

    def test_multiscale_resolution_is_explicit_and_final_output_agrees(self):
        model = tiny_vqvae(learned=True, dec_positional=True)
        x = torch.arange(16).view(2, 8) % 17
        native = model.reconstruct_multiscales(x)
        full = model.reconstruct_multiscales(x, full_resolution=True)
        self.assertEqual([len(row) for row in native[0]], list(model.vq.levels))
        self.assertEqual([len(row) for row in full[0]], [8] * 4)
        final = model.decode(model.encode(x)).argmax(-1).tolist()
        self.assertEqual([row[-1] for row in full], final)
        self.assertEqual([row[-1] for row in native], final)

    def test_tokenizer_multiscale_methods_keep_level_and_block_order(self):
        model = tiny_vqvae(learned=True)
        tokenizer = VQVAETokenizer(SymbolTokenizer(), model)
        text = "abcdefghi"
        blocks = torch.tensor(tokenizer._adjust_length(tokenizer.pre_tokenizer.encode(text)))
        codes = model.encode(blocks)
        native = tokenizer.decode_multiscales(codes)
        full = tokenizer.decode_multiscales(codes, full_resolution=True)
        self.assertEqual([len(s) for s in native[0]], list(model.vq.levels))
        self.assertEqual([len(s) for s in full[0]], [8] * 4)
        combined = tokenizer.reconstruct_multiscales(text, full_resolution=True)
        self.assertEqual(len(combined), 1)
        for i in range(4):
            self.assertEqual(combined[0][i], (full[0][i] + full[1][i])[:len(text)])


if __name__ == "__main__":
    unittest.main()
