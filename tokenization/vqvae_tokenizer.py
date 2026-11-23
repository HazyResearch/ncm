from tokenization.tokenizer import Tokenizer
import hydra
from utils.registry import tokenizers
from models.vqvae import VQVAE
import torch 

class VQVAETokenizer(Tokenizer):
    def __init__(self, pre_tokenizer, vqvae):
        super().__init__()
        self.vqvae = vqvae
        self.pre_tokenizer = pre_tokenizer
        
    @classmethod
    def from_config(cls, config):
        Tokenizer = hydra.utils.get_class(tokenizers[config.pre_tokenizer.name])
        pre_tokenizer = Tokenizer.from_config(config.pre_tokenizer)
        vqvae = VQVAE(**config.vqvae)
        vqvae.load_state_dict(torch.load(config.vqvae.checkpoint_path, weights_only=True, map_location='cpu')['model'], strict=True)
        return VQVAETokenizer(pre_tokenizer, vqvae)
    
    def _adjust_length(self, ids: list[int]) -> list[list[int]]:
        batched_ids = []
        for i in range(0, len(ids), self.vqvae.context_length):
            if len(ids) < i + self.vqvae.context_length:
                batched_ids.append(ids[i:] + [self.pre_tokenizer.pad_token_id] * (i + self.vqvae.context_length - len(ids)))
            else:
                batched_ids.append(ids[i:i + self.vqvae.context_length])
        return batched_ids
    
    # Take a string and return a 1D list of token ids
    def encode(self, text: str):
        ids_pre_tokenized = self.pre_tokenizer.encode(text)            
        return self.encode_pre_tokenized(ids_pre_tokenized)
    
    # Take a 1D list of pre_tokenized token ids and return a 1D list of VQ-VAE token ids
    def encode_pre_tokenized(self, ids_pre_tokenized: list[int]):
        idx_pre_tokenized = torch.tensor(self._adjust_length(ids_pre_tokenized), dtype=torch.long)
        ids = self.vqvae.encode(idx_pre_tokenized).squeeze().view(-1).tolist()
        return ids
    
    # syntatic sugar, Take a 2D tensor with ids and return a 2D tensor with VQ-VAE token ids
    def encode_pre_tokenized_idx(self, idx_pre_tokenized: torch.Tensor):
        assert idx_pre_tokenized.shape[-1] == self.vqvae.context_length, f"Input tensor must have the last dimension equal to {self.vqvae.context_length}"
        ids = self.vqvae.encode(idx_pre_tokenized)
        return ids
     
    def decode(self, ids, skip_special_tokens: bool = False):
        idx = torch.tensor(ids, dtype=torch.long, device=next(self.vqvae.parameters()).device).unsqueeze(0)
        logits = self.vqvae.decode(idx).squeeze()
        x = torch.argmax(logits, dim=-1)
        x = self.pre_tokenizer.decode(x.tolist(), skip_special_tokens=skip_special_tokens)
        return x
    
    def reconstruct(self, text: str):
        idx_bytes = torch.tensor(self.pre_tokenizer.encode(text), dtype=torch.long).unsqueeze(0)
        logits = self.vqvae.reconstruct(idx_bytes)
        x = torch.argmax(logits, dim=-1)
        x = self.pre_tokenizer.decode(x.squeeze().tolist())
        return x
    
    def reconstruct_text_from_ids(self, ids: list[int]):
        idx_bytes = torch.tensor(ids, dtype=torch.long, device=next(self.vqvae.parameters()).device).unsqueeze(0)
        logits = self.vqvae.reconstruct(idx_bytes)
        x = torch.argmax(logits, dim=-1)
        x = self.pre_tokenizer.decode(x.squeeze().tolist())
        return x
    
    def reconstruct_multiscales(self, text: str):
        idx_pre_tokenized = torch.tensor(self.pre_tokenizer.encode(text), dtype=torch.long, device=next(self.vqvae.parameters()).device).unsqueeze(0)
        reconstructions = self.vqvae.reconstruct_multiscales(idx_pre_tokenized)
        reconstructions = [self.pre_tokenizer.decode_batch(x) for x in reconstructions]

        return reconstructions
    
    def decode_multiscales(self, idx, skip_special_tokens: bool = False):
        _, z_levels = self.vqvae.vq.idx_to_zhats(idx, return_zlevels=True)
        B, _ = idx.shape
        reconstructions = [[] for _ in range(B)]
        for z_level in z_levels:
            logits = self.vqvae.decoder(z_level)
            x = torch.argmax(logits, dim=-1)
            x = self.pre_tokenizer.decode_batch(x.tolist(), skip_special_tokens=skip_special_tokens)

            for i in range(B):
                reconstructions[i].append(x[i])
        return reconstructions