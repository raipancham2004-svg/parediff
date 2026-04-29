"""
Netlist GNN encoder.
Lightweight GraphGPS-style encoder: GIN message passing + global self-attention.
Outputs a per-macro embedding c_i ∈ R^d (used as conditioning in the denoiser).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GINConv, global_mean_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class GINBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if not PYG_AVAILABLE:
            self.conv = nn.Identity()
        else:
            mlp = nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
            )
            self.conv = GINConv(mlp)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, edge_index):
        if PYG_AVAILABLE:
            h = self.conv(x, edge_index)
        else:
            h = x  # smoke-test fallback
        return self.norm(x + h)


class GlobalAttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (N, D) — treat batch=1
        h = x.unsqueeze(0)
        a, _ = self.attn(h, h, h)
        h = self.norm(h + a)
        h = self.norm2(h + self.ff(h))
        return h.squeeze(0)


class NetlistGNN(nn.Module):
    """GraphGPS-style: alternate local (GIN) + global (attn)."""

    def __init__(self, in_dim: int = 3, hidden: int = 128, layers: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.gin_blocks = nn.ModuleList([GINBlock(hidden) for _ in range(layers)])
        self.attn_blocks = nn.ModuleList([GlobalAttnBlock(hidden) for _ in range(layers)])
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, x, edge_index):
        h = self.in_proj(x)
        for gin, attn in zip(self.gin_blocks, self.attn_blocks):
            h = gin(h, edge_index)
            h = attn(h)
        return self.out_proj(h)
