import torch
import torch.nn as nn
from model.layers.block import resconv


class UIBBlock(nn.Module):
    def __init__(self, dim, expand_ratio=4):
        super().__init__()
        hidden = int(dim * expand_ratio)
        self.dw      = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.expand  = nn.Conv2d(dim, hidden, 1)
        self.act     = nn.GELU()
        self.project = nn.Conv2d(hidden, dim, 1)
        self.norm    = nn.BatchNorm2d(dim)

    def forward(self, x):
        return x + self.project(self.act(self.expand(self.act(self.dw(self.norm(x))))))


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.reset_gate  = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)
        self.update_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)
        self.out_gate    = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        o = torch.tanh(self.out_gate(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * o


class MNv4Iter(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_uib_blocks=4, res_layers=4):
        super().__init__()
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, 1)
        self.uib_blocks = nn.Sequential(*[UIBBlock(hidden_dim) for _ in range(num_uib_blocks)])
        self.gru        = ConvGRUCell(hidden_dim, hidden_dim)
        self.res_convs  = nn.Sequential(*[resconv(hidden_dim, hidden_dim, k=3, s=1) for _ in range(res_layers)])
        self.out_proj   = nn.Conv2d(hidden_dim, input_dim, 1)
        self.output_dim = input_dim

    def forward(self, inp):
        x      = self.input_proj(inp)
        x      = self.uib_blocks(x)
        h      = torch.zeros_like(x)
        hidden = self.gru(x, h)
        hidden = self.res_convs(hidden)
        out    = self.out_proj(hidden)
        return out
