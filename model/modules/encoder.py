import torch
import torch.nn as nn
from layernorm import LayerNorm
from attn import AttentionModule
from mlp import MLP
from config import device


class EncoderTransformerLayer(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int
    ):
        super().__init__()

        self.attention = AttentionModule(hidden_dim, num_heads)
        self.mlp = MLP(hidden_dim)

        self.norm_for_v = LayerNorm(hidden_dim)
        self.norm_for_k = LayerNorm(hidden_dim)
        self.norm_for_q = LayerNorm(hidden_dim)

        self.norm_for_attention = LayerNorm(hidden_dim)

    def forward(self, value, key, query, mask):
        attn_output = self.attention(self.norm_for_v(value),
                                     self.norm_for_k(key),
                                     self.norm_for_q(query),
                                     mask) + query
        attn_output = self.norm_for_attention(attn_output)

        mlp_output = self.mlp(attn_output)

        return mlp_output


class Encoder(nn.Module):
    def __init__(
        self,
        de_dictionary_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()

        self.word_embedding = nn.Embedding(de_dictionary_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.layers = nn.ModuleList(
            [
                EncoderTransformerLayer(
                    hidden_dim,
                    num_heads,
                    dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        batch_size, seq_len = inputs.shape
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(device)
        hidden_dim = self.dropout(self.word_embedding(inputs) + self.pos_embedding(positions))

        for layer in self.layers:
            hidden_dim = layer(hidden_dim, hidden_dim, hidden_dim, mask)

        return hidden_dim
