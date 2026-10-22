import torch.nn as nn
import torch.nn.functional as F
from models.vae import Encoder, Decoder
from utils.registry import vector_quantizers
import torch
import hydra 
from omegaconf import OmegaConf
from utils.misc import count_params

class VQVAE(nn.Module):
    def __init__(
        self,
        in_vocab_size,
        context_length,
        embed_dim,
        vqvae_vocab_size,
        n_layers,
        vector_quantizer_config,
        compression_factor=4,
        dropout=0.1,
        pre_quant_groupnorm=None,
        pre_quant_dropout=None,
        use_qk_norm=False,
        enc_layers=0,
        enc_attn_window=0,
        enc_conv_layers=0,
        enc_conv_kernel=3,
        dec_layers=1,
        pre_quant_norm_type='group',
        enc_positional=True,
        dec_positional=False,
        **kwargs
    ):
        super().__init__()
        _ = kwargs
        self.embed_dim = embed_dim
        self.vqvae_vocab_size = vqvae_vocab_size
        self.in_vocab_size = in_vocab_size
        self.context_length = context_length

        # enc_layers=0 / dec_layers=1 reproduce the original architecture: encoder = wte+wpe lookup, decoder = one Block.
        self.encoder = Encoder(
            in_vocab_size, context_length, embed_dim, n_layers, compression_factor=compression_factor,
            dropout=dropout, enc_layers=enc_layers, use_qk_norm=use_qk_norm,
            enc_attn_window=enc_attn_window,
            enc_conv_layers=enc_conv_layers, enc_conv_kernel=enc_conv_kernel,
            use_pos_emb=enc_positional,
        )
        
        vq = hydra.utils.get_class(vector_quantizers[vector_quantizer_config.clss])
        self.vq = vq.from_config({ **{'vqvae_vocab_size': vqvae_vocab_size, 'embed_dim': embed_dim}, **dict(vector_quantizer_config)})
                
        self.decoder = Decoder(
            in_vocab_size, embed_dim, n_layers, compression_factor=compression_factor,
            dropout=dropout, use_qk_norm=use_qk_norm, dec_layers=dec_layers,
            use_pos_emb=dec_positional, context_length=context_length,
        )

        # affine=False: ||z-e||^2 is not scale-invariant, so a learnable gamma would inflate without bound.
        if not pre_quant_groupnorm:
            self.pre_quant_norm = None
        elif str(pre_quant_norm_type) == 'layer':
            self.pre_quant_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
            self._pre_quant_is_layer = True
        else:
            self.pre_quant_norm = nn.GroupNorm(int(pre_quant_groupnorm), embed_dim, affine=False)
            self._pre_quant_is_layer = False

        self.pre_quant_drop = (
            nn.Dropout(float(pre_quant_dropout)) if pre_quant_dropout else None
        )

        self.context_length = context_length

        total_params, non_embd_params = count_params(self, exclude_params=['wte', 'wpe'])

        print("VQVAE, Total parameter count: %.2fM" % (total_params/1e6,))
        print("VQVAE, Non embedding parameter count: %.2fM" % (non_embd_params/1e6,))

    def _encode_pre_quant(self, x):
        z = self.encoder(x)
        if self.pre_quant_norm is not None:
            if getattr(self, '_pre_quant_is_layer', False):
                z = self.pre_quant_norm(z)
            else:
                z = self.pre_quant_norm(z.transpose(1, 2)).transpose(1, 2)
        if self.pre_quant_drop is not None:
            z = self.pre_quant_drop(z)
        return z

    def forward(self, x):
        z = self._encode_pre_quant(x)
        z_q, vq_loss, per_level_codebook_util, global_codebook_util, ppl = self.vq(z)
        x = self.decoder(z_q)
        return x, vq_loss, per_level_codebook_util, global_codebook_util, ppl

    @torch.no_grad()
    def encode(self, x):
        z = self._encode_pre_quant(x)
        idx = self.vq.z_to_idx(z)
        return idx

    @torch.no_grad()
    def encode_z(self, x):
        return self._encode_pre_quant(x)
    
    @torch.no_grad()
    def decode(self, idx):
        z_q = self.vq.idx_to_z_q(idx)
        x = self.decoder(z_q)
        return x
    
    @torch.no_grad()
    def reconstruct(self, x):
        z = self._encode_pre_quant(x)
        z_q, *_ = self.vq(z)
        x_recon = self.decoder(z_q)
        return x_recon
    
    @torch.no_grad()
    def get_ncm_inputs(self, pre_tokenized_ids: torch.Tensor):
        idx = self.encode(pre_tokenized_ids)
        x =  self.vq.idx_to_ncm_input(idx)
        return x, idx[:, self.vq.levels[0]:]
    
    @torch.no_grad()
    def reconstruct_multiscales(self, idx_pre_tokenized: torch.Tensor, *, full_resolution=False):
        """Decode each cumulative latent at native or final sequence resolution.

        Native resolution is retained by default for existing visualizations.
        Use full_resolution=True to compare the same text positions at each level.
        """
        B, _ = idx_pre_tokenized.shape
        z = self._encode_pre_quant(idx_pre_tokenized)
        _, multiscale_zqs = self.vq.z_to_idx(
            z, return_zhats=full_resolution, return_zlevels=not full_resolution,
        )
        reconstructions = []
        for z_q in multiscale_zqs:
            logits = self.decoder(z_q)
            x = torch.argmax(logits, dim=-1)
            reconstructions.append(x)
            
        reconstructions_reordered = [[] for _ in range(B)]
        for i in range(len(reconstructions)):
            for j in range(B):
                reconstructions_reordered[j].append(reconstructions[i][j].tolist())
        return reconstructions_reordered
    
    
if __name__ == "__main__":
    
    vqvae = VQVAE(
        in_vocab_size=1000,
        context_length=512,
        embed_dim=256,
        vqvae_vocab_size=512,
        n_layers=8,
        vector_quantizer_config= OmegaConf.create({
            'clss': 'multiscale_residual_vector_quantizer',
            'decay': 0.99,
            'commitment_cost': 0.25,
            'levels': [1, 2, 4, 8, 12, 16, 24, 32, 48, 56, 64, 72, 84, 96, 114, 128]
        })
    )

    print(vqvae)
    sample_input = torch.randint(0, 1000, (2, 512))
    output, vq_loss, _, _, ppl = vqvae(sample_input)
    print("Output shape:", output.shape)
    print("VQ Loss:", vq_loss.item())
