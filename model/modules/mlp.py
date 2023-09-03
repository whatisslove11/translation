import torch
import torch.nn as nn
from additional import gelu


class MLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.linear_0 = torch.nn.Linear(hidden_dim, 4 * hidden_dim)
        self.linear_1 = torch.nn.Linear(4 * hidden_dim, hidden_dim)

    def forward(self, hidden_state):
        return self.linear_1(gelu(self.linear_0(hidden_state))) + hidden_state
