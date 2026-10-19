import math
from utils.misc import round_levels

tokenizers = {
    "bpe": "tokenization.bpe_tokenizer.BPETokenizer",
    "hf": "tokenization.hf_tokenizer.HFTokenizer",
    "sudoku": "tokenization.sudoku_tokenizer.SudokuTokenizer",
    "vqvae": "tokenization.vqvae_tokenizer.VQVAETokenizer",
}

vector_quantizers = {
    "multiscale_residual_vector_quantizer": "models.quant.MultiscaleResidualVectorQuantizer",
    "multiscale_residual_fsq": "models.fsq_quant.MultiscaleResidualFSQ",
}

level_schedulers = {
    "linear": lambda start, end, n: round_levels([start + (end - start) * (i-1)/(n-1) for i in range(1, n+1)], start, end),
    "cubic": lambda start, end, n: round_levels([start + (end - start) * (i-1)**3/(n-1)**3 for i in range(1, n+1)], start, end),
    "exp": lambda start, end, n: round_levels([start + (end - start) * (math.exp(i-1))/(math.exp(n-1)) for i in range(1, n+1)], start, end),
    "sqrt": lambda start, end, n: round_levels([start + (end - start) * math.sqrt(i-1)/math.sqrt(n-1) for i in range(1, n+1)], start, end),
    "sine": lambda start, end, n: round_levels([start + (end - start) * math.sin((math.pi/2) * ((i-1)/(n-1))) for i in range(1, n+1)], start, end),
    "log": lambda start, end, n: round_levels([start + (end - start) * math.log(i)/math.log(n) for i in range(1, n+1)], start, end),
    "quadratic": lambda start, end, n: round_levels([start + (end - start) * (i-1)**2/(n-2)**2 for i in range(1, n+1)], start, end),
}
