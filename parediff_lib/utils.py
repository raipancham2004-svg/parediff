"""Utility functions used across PareDiff."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch


# ============================================================
# CONSTANTS
# ============================================================
N_ORIENT = 8  # R0, R90, R180, R270, MX, MY, MXR90, MYR90
ORIENT_NAMES = ['R0', 'R90', 'R180', 'R270', 'MX', 'MY', 'MXR90', 'MYR90']


# ============================================================
# DIFFUSION SCHEDULE (cosine — Nichol & Dhariwal 2021)
# ============================================================
def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Cumulative product of (1 - β_t) under cosine schedule."""
    return torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2


def make_beta_schedule(T: int, schedule: str = 'cosine') -> torch.Tensor:
    """β_t schedule. Returns tensor of shape (T,)."""
    if schedule == 'linear':
        return torch.linspace(1e-4, 0.02, T)
    if schedule == 'cosine':
        steps = T + 1
        x = torch.linspace(0, T, steps) / T
        ab = cosine_alpha_bar(x)
        ab = ab / ab[0]
        betas = 1 - (ab[1:] / ab[:-1])
        return betas.clamp(0, 0.999)
    raise ValueError(f"Unknown schedule: {schedule}")


def precompute_diffusion_constants(betas: torch.Tensor) -> dict:
    """Pre-compute α_t, ᾱ_t and friends for fast indexing during training."""
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    alphas_bar_prev = torch.cat([torch.tensor([1.0]), alphas_bar[:-1]])
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_bar': alphas_bar,
        'alphas_bar_prev': alphas_bar_prev,
        'sqrt_alphas_bar': torch.sqrt(alphas_bar),
        'sqrt_one_minus_alphas_bar': torch.sqrt(1.0 - alphas_bar),
    }


# ============================================================
# HPWL — half-perimeter wirelength (differentiable)
# ============================================================
def hpwl(coords: torch.Tensor, hyperedges: list[list[int]]) -> torch.Tensor:
    """
    coords: (N, 2) tensor of (x, y) for each macro.
    hyperedges: list of lists; each inner list = macro indices on that net.
    Returns scalar HPWL summed over all nets.
    """
    total = coords.new_zeros(())
    for net in hyperedges:
        if len(net) < 2:
            continue
        sub = coords[net]
        bbox_w = sub[:, 0].max() - sub[:, 0].min()
        bbox_h = sub[:, 1].max() - sub[:, 1].min()
        total = total + bbox_w + bbox_h
    return total


# ============================================================
# OVERLAP PENALTY — differentiable, used in training loss
# ============================================================
def overlap_penalty(coords: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """
    coords: (N, 2) center positions (normalized to [0,1]).
    sizes:  (N, 2) (w, h) for each macro.
    Returns sum of overlap areas across all macro pairs (differentiable).
    """
    N = coords.shape[0]
    if N < 2:
        return coords.new_zeros(())
    # Pairwise center distances
    half = sizes / 2.0
    lo = coords - half  # (N, 2)
    hi = coords + half  # (N, 2)
    # For every pair (i, j):
    # overlap_x = max(0, min(hi_i.x, hi_j.x) - max(lo_i.x, lo_j.x))
    lo_exp_i = lo.unsqueeze(1)  # (N, 1, 2)
    lo_exp_j = lo.unsqueeze(0)  # (1, N, 2)
    hi_exp_i = hi.unsqueeze(1)
    hi_exp_j = hi.unsqueeze(0)
    overlap_xy = (
        torch.minimum(hi_exp_i, hi_exp_j) - torch.maximum(lo_exp_i, lo_exp_j)
    ).clamp(min=0)
    overlap_area = overlap_xy.prod(dim=-1)  # (N, N)
    # Zero out diagonal
    mask = 1.0 - torch.eye(N, device=coords.device)
    return (overlap_area * mask).sum() / 2.0  # divide by 2 for double-count


# ============================================================
# BOUNDARY VIOLATION
# ============================================================
def boundary_violation(coords: torch.Tensor, sizes: torch.Tensor,
                       die: Tuple[float, float] = (1.0, 1.0)) -> torch.Tensor:
    """How much each macro spills outside the [0, die_w] × [0, die_h] tile."""
    half = sizes / 2.0
    lo = coords - half
    hi = coords + half
    left   = (-lo[:, 0]).clamp(min=0)
    bottom = (-lo[:, 1]).clamp(min=0)
    right  = (hi[:, 0] - die[0]).clamp(min=0)
    top    = (hi[:, 1] - die[1]).clamp(min=0)
    return (left + right + bottom + top).sum()


# ============================================================
# ORIENTATION HELPERS
# ============================================================
def orient_to_onehot(orient_idx: torch.Tensor) -> torch.Tensor:
    """(N,) int → (N, N_ORIENT) one-hot."""
    return torch.nn.functional.one_hot(orient_idx, num_classes=N_ORIENT).float()


def onehot_to_orient(onehot: torch.Tensor) -> torch.Tensor:
    """(N, N_ORIENT) → (N,) argmax."""
    return onehot.argmax(dim=-1)


# ============================================================
# CONFIG DATACLASS
# ============================================================
@dataclass
class PareDiffConfig:
    # Model
    gnn_hidden: int = 128
    gnn_layers: int = 4
    denoiser_hidden: int = 256
    denoiser_layers: int = 6
    # Diffusion
    T: int = 1000
    sample_steps: int = 50
    schedule: str = 'cosine'
    # Training
    batch_size: int = 4
    lr: float = 1e-4
    epochs: int = 200
    grad_clip: float = 1.0
    # Loss weights
    eps_loss_weight: float = 1.0
    overlap_weight: float = 0.5
    boundary_weight: float = 0.5
    hpwl_weight: float = 0.1
    # Pareto sampling
    pareto_K: int = 8  # number of placements per inference
    guidance_scale_min: float = 0.0
    guidance_scale_max: float = 5.0
    # Logging
    project_name: str = "parediff"
    use_wandb: bool = False
