"""
PareDiff denoising network + DDPM training/sampling.

Coords (continuous) + Orientation (discrete 8-class) joint diffusion.
Continuous part: Gaussian noise + ε-prediction.
Discrete part: Multinomial diffusion (DiGress-style transition matrices).

For the 1-month sprint we use ε-prediction for coords and CROSS-ENTROPY
classification for orientation (treated as conditional generation), which is
simpler than full multinomial diffusion. We can upgrade to true mixed-modal
diffusion in v2 if time allows.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_encoder import NetlistGNN
from .utils import (
    N_ORIENT, PareDiffConfig, make_beta_schedule, precompute_diffusion_constants,
)


# ============================================================
# TIMESTEP EMBEDDING (sinusoidal — Vaswani-style)
# ============================================================
def sinusoidal_t_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# ============================================================
# DENOISING NETWORK (Transformer over macros, conditioned on GNN embedding)
# ============================================================
class DenoisingNetwork(nn.Module):
    def __init__(self, cfg: PareDiffConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.denoiser_hidden

        # Coord (2) + Orient one-hot (N_ORIENT) → D
        self.in_proj = nn.Linear(2 + N_ORIENT, D)
        # Per-macro conditioning from GNN
        self.cond_proj = nn.Linear(cfg.gnn_hidden, D)
        # Timestep
        self.t_proj = nn.Sequential(
            nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D)
        )
        # Stack of transformer layers
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=8, dim_feedforward=4 * D,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.denoiser_layers)
        # Output: predict ε for coords (2) + logits for orient (N_ORIENT)
        self.out_coord = nn.Linear(D, 2)
        self.out_orient = nn.Linear(D, N_ORIENT)

    def forward(self, x_t_coord, x_t_orient_onehot, t, c):
        """
        x_t_coord:           (N, 2)
        x_t_orient_onehot:   (N, N_ORIENT) — current noisy orientation
        t:                   (1,) timestep
        c:                   (N, gnn_hidden) — GNN per-macro conditioning
        """
        D = self.cfg.denoiser_hidden
        N = x_t_coord.shape[0]
        x = torch.cat([x_t_coord, x_t_orient_onehot], dim=-1)  # (N, 2+N_ORIENT)
        h = self.in_proj(x) + self.cond_proj(c)
        # Add timestep emb to every macro
        t_emb = self.t_proj(sinusoidal_t_emb(t, D))  # (1, D)
        h = h + t_emb  # broadcast
        # Transformer — treat macros as sequence
        h = self.transformer(h.unsqueeze(0)).squeeze(0)
        eps_coord = self.out_coord(h)        # predicted noise on coords
        orient_logits = self.out_orient(h)   # predicted clean orient logits
        return eps_coord, orient_logits


# ============================================================
# PareDiff WRAPPER
# ============================================================
class PareDiff(nn.Module):
    def __init__(self, cfg: PareDiffConfig, in_node_dim: int = 3):
        super().__init__()
        self.cfg = cfg
        self.gnn = NetlistGNN(
            in_dim=in_node_dim, hidden=cfg.gnn_hidden, layers=cfg.gnn_layers
        )
        # Project GNN output dim to denoiser conditioning dim if different
        self.denoiser = DenoisingNetwork(cfg)

        betas = make_beta_schedule(cfg.T, cfg.schedule)
        consts = precompute_diffusion_constants(betas)
        for k, v in consts.items():
            self.register_buffer(k, v)

    # ------------------------------------------------------------
    # Forward (training step) — predict ε on noisy coords + orient logits
    # ------------------------------------------------------------
    def training_step(self, batch):
        """
        batch is a single PyG Data (or dict) with:
          x, edge_index, target_pos (N,2), target_orient (N,)
        Returns dict of losses.
        """
        x = batch.x if hasattr(batch, 'x') else batch['x']
        edge_index = batch.edge_index if hasattr(batch, 'edge_index') else batch['edge_index']
        x_0 = batch.target_pos if hasattr(batch, 'target_pos') else batch['target_pos']
        o_0 = batch.target_orient if hasattr(batch, 'target_orient') else batch['target_orient']
        device = x.device

        # GNN conditioning
        c = self.gnn(x, edge_index)  # (N, gnn_hidden)

        # Sample timestep
        N = x_0.shape[0]
        t = torch.randint(0, self.cfg.T, (1,), device=device)

        # Forward noising on coords
        sqrt_ab = self.sqrt_alphas_bar[t]
        sqrt_one_minus_ab = self.sqrt_one_minus_alphas_bar[t]
        eps = torch.randn_like(x_0)
        x_t = sqrt_ab * x_0 + sqrt_one_minus_ab * eps

        # For orientation we use clean one-hot as input target;
        # noising = random uniform mix
        gamma = (1 - sqrt_ab).item()  # blend factor
        o_onehot = F.one_hot(o_0, num_classes=N_ORIENT).float()
        o_uniform = torch.full_like(o_onehot, 1.0 / N_ORIENT)
        o_t = (1 - gamma) * o_onehot + gamma * o_uniform

        # Predict
        eps_pred, orient_logits = self.denoiser(x_t, o_t, t, c)

        # Losses
        eps_loss = F.mse_loss(eps_pred, eps)
        orient_loss = F.cross_entropy(orient_logits, o_0)
        loss = eps_loss + 0.1 * orient_loss

        return {
            'loss': loss,
            'eps_loss': eps_loss.detach(),
            'orient_loss': orient_loss.detach(),
        }

    # ------------------------------------------------------------
    # Sampling — DDIM-style with optional classifier guidance
    # ------------------------------------------------------------
    @torch.no_grad()
    def sample(self, batch, guidance_fn=None, guidance_scale: float = 0.0,
               num_steps: int | None = None):
        """
        Generate macro placements.

        batch:           PyG Data with x, edge_index, sizes
        guidance_fn:     callable(x_t_coord) → scalar score to MAXIMIZE
                         (e.g., negative routability — we want low congestion)
        guidance_scale:  strength of guidance (0 = unguided)
        num_steps:       sampling steps (default cfg.sample_steps)
        """
        steps = num_steps or self.cfg.sample_steps
        x = batch.x if hasattr(batch, 'x') else batch['x']
        edge_index = batch.edge_index if hasattr(batch, 'edge_index') else batch['edge_index']
        device = x.device
        N = x.shape[0]

        c = self.gnn(x, edge_index)

        # Initial noise
        x_t = torch.randn(N, 2, device=device)
        o_t = torch.full((N, N_ORIENT), 1.0 / N_ORIENT, device=device)

        # Step indices (skip-step DDIM)
        ts = torch.linspace(self.cfg.T - 1, 0, steps, device=device).long()

        for i, t in enumerate(ts):
            t_b = t.unsqueeze(0)
            # Predict
            with torch.enable_grad() if guidance_scale > 0 else torch.no_grad():
                if guidance_scale > 0:
                    x_t = x_t.detach().requires_grad_(True)
                eps_pred, orient_logits = self.denoiser(x_t, o_t, t_b, c)

                if guidance_scale > 0 and guidance_fn is not None:
                    # Estimate x_0 from x_t and eps_pred
                    sqrt_ab = self.sqrt_alphas_bar[t]
                    sqrt_one_minus_ab = self.sqrt_one_minus_alphas_bar[t]
                    x_0_hat = (x_t - sqrt_one_minus_ab * eps_pred) / sqrt_ab
                    # Compute guidance gradient on predicted x_0
                    score = guidance_fn(x_0_hat)
                    grad = torch.autograd.grad(score.sum(), x_t)[0]
                    # Shift eps_pred by grad (classifier guidance)
                    eps_pred = eps_pred - guidance_scale * sqrt_one_minus_ab * grad

            # DDIM step
            sqrt_ab_t = self.sqrt_alphas_bar[t]
            sqrt_one_minus_ab_t = self.sqrt_one_minus_alphas_bar[t]
            x_0_pred = (x_t - sqrt_one_minus_ab_t * eps_pred) / sqrt_ab_t
            x_0_pred = x_0_pred.clamp(0, 1)  # legal region projection (soft)

            if i == len(ts) - 1:
                x_t = x_0_pred
            else:
                t_prev = ts[i + 1]
                ab_prev = self.alphas_bar[t_prev]
                noise = torch.randn_like(x_t)
                x_t = (
                    torch.sqrt(ab_prev) * x_0_pred
                    + torch.sqrt(1 - ab_prev) * noise
                )

            # Orient: take argmax of current logits as denoising
            o_t = F.softmax(orient_logits, dim=-1)

        coords = x_t.detach().clamp(0, 1)
        orient = o_t.argmax(dim=-1)
        return coords, orient

    # ------------------------------------------------------------
    # PARETO SAMPLING — generate K placements at K guidance scales
    # ------------------------------------------------------------
    @torch.no_grad()
    def pareto_sample(self, batch, guidance_fn, K: int | None = None):
        """Generate K placements spanning the (HPWL ↔ routability) Pareto front."""
        K = K or self.cfg.pareto_K
        scales = torch.linspace(
            self.cfg.guidance_scale_min, self.cfg.guidance_scale_max, K
        )
        results = []
        for s in scales:
            coords, orient = self.sample(batch, guidance_fn=guidance_fn,
                                         guidance_scale=s.item())
            results.append({'coords': coords, 'orient': orient,
                            'guidance_scale': s.item()})
        return results
