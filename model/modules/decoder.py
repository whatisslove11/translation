import torch
import torch.nn as nn
from encoder import (
    EncoderTransformerLayer,
    LayerNorm,
    AttentionModule,
    device
)


class DecoderTransformerLayer(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int
    ):
        super().__init__()

        self.self_attention = AttentionModule(hidden_dim, num_heads)  # Аттенш на то, что происходит в переводе
        self.out_attention = EncoderTransformerLayer(hidden_dim, num_heads)  # Аттенш на то, что происходит в оригинале

        self.norm_for_hidden = LayerNorm(hidden_dim)

        self.norm_for_attention = LayerNorm(hidden_dim)
        self.norm_for_encoder = LayerNorm(hidden_dim)

    def forward(self, hidden_state, encoder_layer_output, src_mask, trg_mask):
        encoder_layer_output = self.norm_for_encoder(encoder_layer_output)

        normalized_hidden_state = self.norm_for_hidden(hidden_state)
        self_attn_output = self.self_attention(
            normalized_hidden_state,
            normalized_hidden_state,
            normalized_hidden_state,
            trg_mask) + hidden_state

        self_attn_output = self.norm_for_attention(self_attn_output)

        output = self.out_attention(
            encoder_layer_output,
            encoder_layer_output,
            self_attn_output, src_mask
        )

        return output


class Decoder(torch.nn.Module):
    def __init__(
            self,
            en_dictionary_size: int,
            hidden_dim: int,
            num_layers: int,
            num_heads: int,
            dropout: float = 0.1,
            max_seq_len: int = 512
    ):
        super().__init__()

        self.word_embedding = nn.Embedding(en_dictionary_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.layers = nn.ModuleList(
            [
                DecoderTransformerLayer(hidden_dim, num_heads)
                for _ in range(num_layers)
            ]
        )

        self.lm_head = nn.Linear(hidden_dim, en_dictionary_size)
        self.dropout = nn.Dropout(dropout)

        # weight tying
        self.word_embedding.weight = self.lm_head.weight

    def forward(self, inputs, encoder_output, src_mask, trg_mask):
        batch_size, seq_len = inputs.shape
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(device)
        inputs = self.dropout(self.word_embedding(inputs) + self.pos_embedding(positions))

        for layer in self.layers:
            inputs = layer(inputs, encoder_output, src_mask, trg_mask)

        return self.lm_head(inputs)
