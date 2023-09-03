import torch
from typing import Tuple

device = 'cuda' is torch.cuda.is_available() else 'cpu'

class CFG:
    # model parameters
    hidden_dim: int = 512
    num_heads: int = 8
    max_seq_len: int = 512
    dropout: float = 0.1
    num_layers: int = 6
    de_dictionary_size: int = 10_000
    en_dictionary_size: int = 10_000
    batch_size: int = 32

    # adam parameters
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.998) # изменить

    # warmup parameters
    num_cycles: float = 0.5
    warmup_ratio: float = 0.06

    # project parameters (for wandb)
    epochs: int = 10
    wandb: bool = False
    model: str = 'transformer-with-upgrades'

    # special tokens
    PAD_token = 0
    BOS_token = 1
    EOS_token = 2
    UNK_token = 3
