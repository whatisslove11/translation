import torch
import torch.nn as nn
import math


@torch.jit.script
def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.linear_0 = torch.nn.Linear(hidden_dim, 4 * hidden_dim)
        self.linear_1 = torch.nn.Linear(4 * hidden_dim, hidden_dim)

    def forward(self, hidden_state):
        return self.linear_1(new_gelu(self.linear_0(hidden_state))) + hidden_state
