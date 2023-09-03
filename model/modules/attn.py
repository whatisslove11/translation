import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class AttentionModule(nn.Module):

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = 0.1

        self.flash = hasattr(F, 'scaled_dot_product_attention')

        self.out_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)  # c_proj
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, K, V, Q, mask):

        batch_size, hidden_dim = Q.size(0), Q.size(2)
        key_len, value_len, query_len = K.size(1), V.size(1), Q.size(1)

        assert hidden_dim % self.num_heads == 0, "Hidden_dim must be equal to num_heads * head_dim"

        K = K.reshape(batch_size, key_len, self.num_heads, -1).transpose(1,
                                                                         2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.reshape(batch_size, value_len, self.num_heads, -1).transpose(1,
                                                                           2)  # (batch_size, num_heads, seq_len, head_dim)
        Q = Q.reshape(batch_size, query_len, self.num_heads, -1).transpose(1,
                                                                           2)  # (batch_size, num_heads, seq_len, head_dim)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            with autocast():
                # print(Q.shape, K.shape, V.shape, mask.shape)
                y = F.scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=False
                )
        else:
            raise ImportError("PyTorch >= 2.0 must be installed for using Flash Attention")

        y = y.transpose(1, 2).contiguous().view(batch_size, query_len, hidden_dim)
        # y = self.resid_dropout(self.out_linear(y)) + y
        return self.resid_dropout(self.out_linear(y.to(torch.float32)))
