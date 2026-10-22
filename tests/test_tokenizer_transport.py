import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tokenization.tokenizer import Tokenizer
from tokenization.vqvae_tokenizer import VQVAETokenizer


class ByteTokenizer(Tokenizer):
    pad_token_id = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=False):
        return b"".join(
            b"<pad>" if i == 256 else bytes([i])
            for i in ids if not (skip_special_tokens and i == 256)
        ).decode("utf-8")


class IdentityVQVAE(nn.Module):
    """A lossless model isolates wrapper transport from learned reconstruction."""
    context_length = 4

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vq = type("VQ", (), {"levels": [4]})()

    def encode(self, ids):
        if ids.device != self.anchor.device:
            raise ValueError("wrong input device")
        return ids

    def decode(self, ids):
        return F.one_hot(ids[:, :4], num_classes=257).float()

    def reconstruct(self, ids):
        return self.decode(self.encode(ids))


class TokenizerTransportTests(unittest.TestCase):
    def setUp(self):
        self.pre = ByteTokenizer()
        self.model = IdentityVQVAE()
        self.tokenizer = VQVAETokenizer(self.pre, self.model)

    def test_decode_preserves_all_blocks_and_cross_block_utf8(self):
        # The emoji straddles a model-block boundary; decode bytes only after joining.
        text = "abc🙂defg"
        codes = self.tokenizer.encode(text)
        self.assertEqual(self.tokenizer.decode(codes, skip_special_tokens=True), text)

    def test_decode_rejects_incomplete_hierarchy(self):
        for ids in ([1], [1, 2, 3, 4, 5]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "complete"):
                self.tokenizer.decode(ids)

    def test_empty_input(self):
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")
        self.assertEqual(self.tokenizer.reconstruct(""), "")

    def test_reconstruct_removes_only_added_padding(self):
        for text in ("x", "é", "abc🙂defg", "abcde"):
            with self.subTest(text=text):
                self.assertEqual(self.tokenizer.reconstruct(text), text)
                self.assertEqual(self.tokenizer.reconstruct_text_from_ids(self.pre.encode(text)), text)

    def test_missing_pad_token_is_reported(self):
        self.pre.pad_token_id = None
        with self.assertRaisesRegex(ValueError, "pad_token_id"):
            self.tokenizer.encode("a")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_wrapper_uses_model_device(self):
        self.model.cuda()
        self.assertEqual(self.tokenizer.encode("abcd"), self.pre.encode("abcd"))
        self.assertEqual(self.tokenizer.reconstruct("abcde"), "abcde")

    def test_adjacent_byte_chunks_preserve_utf8(self):
        text = "aé🙂漢字\r\nz"
        data = text.encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_bytes(data)
            for cut in range(len(data) + 1):
                with self.subTest(cut=cut):
                    first = self.pre.encode_chunk(path, 0, cut)
                    second = self.pre.encode_chunk(path, cut, len(data))
                    self.assertEqual(self.pre.decode(first + second), text)
            chunks = [self.pre.encode_chunk(path, i, i + 1) for i in range(len(data))]
            self.assertEqual(self.pre.decode([i for chunk in chunks for i in chunk]), text)

    def test_chunks_do_not_silently_ignore_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.txt"
            path.write_bytes(b"a\xffb")
            with self.assertRaises(UnicodeDecodeError):
                self.pre.encode_chunk(path, 0, 3)

    def test_invalid_chunk_bounds(self):
        for start, end in ((-1, 2), (2, 1)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.pre.encode_chunk("unused", start, end)


if __name__ == "__main__":
    unittest.main()
