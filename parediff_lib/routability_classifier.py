"""
Routability classifier — predicts post-route routability score from a placement.

Used for two purposes:
  1. Standalone evaluation: given a placement, estimate routability without
     calling OpenROAD (much faster).
  2. Classifier guidance during diffusion sampling: provides ∇log p(routable | x).

Architecture: small CNN over a multi-channel rendered image of the placement.

Input channels:
  - cell_density   (occupied area per pixel)
  - macro_indicator (1 if pixel inside a macro, else 0)
  - pin_density    (pin count per pixel)
  - net_demand     (sum of net half-perimeters touching this pixel)

Output: single scalar in [0, 1] — predicted routing-overflow ratio.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RENDER PLACEMENT → MULTI-CHANNEL IMAGE
# ============================================================
def render_placement(coords, sizes, hyperedges, resolution: int = 64):
    """
    coords:     (N, 2) ∈ [0, 1]
    sizes:      (N, 2) — normalized macro sizes
    hyperedges: list of [macro_idx, ...]
    resolution: image resolution (square)
    Returns:    (4, R, R) tensor — channels above
    """
    R = resolution
    device = coords.device
    img = torch.zeros(4, R, R, device=device)
    half = sizes / 2.0
    lo = (coords - half) * R
    hi = (coords + half) * R

    for i in range(coords.shape[0]):
        x0 = max(0, int(lo[i, 0].item())); x1 = min(R, int(hi[i, 0].item()) + 1)
        y0 = max(0, int(lo[i, 1].item())); y1 = min(R, int(hi[i, 1].item()) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        # cell_density: pretend macros are very dense
        img[0, y0:y1, x0:x1] += 1.0
        # macro_indicator: 1 inside macro
        img[1, y0:y1, x0:x1] = 1.0
        # pin_density: rough proxy (1 per 4 pixels of macro)
        img[2, y0:y1, x0:x1] += 0.25

    # net_demand: per-net bbox addition
    for net in hyperedges:
        if len(net) < 2:
            continue
        sub = coords[net] * R
        x0 = max(0, int(sub[:, 0].min().item())); x1 = min(R, int(sub[:, 0].max().item()) + 1)
        y0 = max(0, int(sub[:, 1].min().item())); y1 = min(R, int(sub[:, 1].max().item()) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        bbox_perim = (x1 - x0) + (y1 - y0)
        img[3, y0:y1, x0:x1] += bbox_perim / R

    img[0] = img[0] / (img[0].max() + 1e-6)
    img[2] = img[2] / (img[2].max() + 1e-6)
    img[3] = img[3] / (img[3].max() + 1e-6)
    return img


# ============================================================
# RoutabilityNet — small CNN classifier
# ============================================================
class RoutabilityNet(nn.Module):
    def __init__(self, in_channels: int = 4, base: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1), nn.GELU(),
            nn.Conv2d(base, base, 3, padding=1, stride=2), nn.GELU(),  # 32
            nn.Conv2d(base, 2 * base, 3, padding=1), nn.GELU(),
            nn.Conv2d(2 * base, 2 * base, 3, padding=1, stride=2), nn.GELU(),  # 16
            nn.Conv2d(2 * base, 4 * base, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * base, 4 * base), nn.GELU(),
            nn.Linear(4 * base, 1),
        )

    def forward(self, x):
        # x: (B, 4, R, R)
        return torch.sigmoid(self.head(self.encoder(x))).squeeze(-1)


# ============================================================
# GUIDANCE FUNCTION — wraps RoutabilityNet for use in PareDiff.sample
# ============================================================
class RoutabilityGuidance:
    """
    Differentiable scoring function used as classifier guidance during
    diffusion sampling. We MINIMIZE predicted overflow — i.e., MAXIMIZE
    -overflow, hence return the negative.
    """
    def __init__(self, classifier: RoutabilityNet, hyperedges: list,
                 sizes: torch.Tensor, resolution: int = 64):
        self.classifier = classifier.eval()
        self.hyperedges = hyperedges
        self.sizes = sizes
        self.resolution = resolution

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        # coords is differentiable (requires_grad=True from PareDiff.sample)
        # We need a differentiable rendering — soft Gaussian splatting:
        img = self._soft_render(coords)
        score = self.classifier(img.unsqueeze(0))
        return -score  # negative because guidance pushes UP

    def _soft_render(self, coords: torch.Tensor):
        """Differentiable rendering using Gaussian splats per macro."""
        R = self.resolution
        device = coords.device
        # Build coordinate grid
        ys, xs = torch.meshgrid(
            torch.linspace(0, 1, R, device=device),
            torch.linspace(0, 1, R, device=device),
            indexing='ij'
        )
        img = torch.zeros(4, R, R, device=device)
        for i in range(coords.shape[0]):
            cx, cy = coords[i, 0], coords[i, 1]
            sw, sh = self.sizes[i, 0], self.sizes[i, 1]
            # Gaussian splat centered at (cx, cy)
            gauss = torch.exp(
                -(((xs - cx) / (sw + 1e-4)) ** 2 + ((ys - cy) / (sh + 1e-4)) ** 2)
            )
            img[0] = img[0] + gauss
            img[1] = img[1] + gauss
            img[2] = img[2] + 0.25 * gauss
        # net_demand channel — skip in soft mode for speed
        img = img / (img.max() + 1e-6)
        return img
