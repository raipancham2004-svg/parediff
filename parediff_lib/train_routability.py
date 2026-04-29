"""
Stage 2: Train the routability classifier on (placement, routability) pairs.

For the 1-month sprint we use a HEURISTIC routability proxy (density-based,
no OpenROAD needed):
    routability_score = clamp(sum_of_local_overflow / max_overflow, 0, 1)
    overflow per GCell = max(0, total_net_demand - track_supply)

Real-paper version replaces this with OpenROAD's detail-route DRC count.

Usage:
    python -m parediff_lib.train_routability \
        --diffusion_ckpt experiments/checkpoints/parediff_final.pt \
        --n_samples 500 \
        --epochs 20 \
        --hf_repo "$HF_USER/parediff-routability"
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from .diffusion_model import PareDiff
from .routability_classifier import RoutabilityNet, render_placement
from .synthetic_dataset import (
    random_netlist, expert_placer, random_orientations, to_pyg_data
)
from .utils import PareDiffConfig


# ============================================================
# HEURISTIC ROUTABILITY (proxy for OpenROAD detail-route DRCs)
# ============================================================
def heuristic_routability(coords, sizes, hyperedges, resolution=64,
                          track_supply: float = 4.0):
    """
    Estimate routing overflow as a density-based proxy.
    Returns a scalar in [0, 1] — higher = more congested = LESS routable.

    Method:
    1. Render placement to a coarse grid
    2. For each net, accumulate its bounding-box demand into the overlapped GCells
    3. Compute (demand / supply) per GCell, sum the overflow part
    4. Normalize to [0, 1]
    """
    R = resolution
    device = coords.device
    grid_demand = torch.zeros(R, R, device=device)

    # Net-bbox demand
    for net in hyperedges:
        if len(net) < 2:
            continue
        sub = coords[net] * R
        x0 = max(0, int(sub[:, 0].min().item()))
        x1 = min(R, int(sub[:, 0].max().item()) + 1)
        y0 = max(0, int(sub[:, 1].min().item()))
        y1 = min(R, int(sub[:, 1].max().item()) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        # Each GCell in the bbox sees this net's perimeter / area
        gcells_in_bbox = (x1 - x0) * (y1 - y0)
        per_gcell_demand = ((x1 - x0) + (y1 - y0)) / max(gcells_in_bbox, 1)
        grid_demand[y0:y1, x0:x1] += per_gcell_demand

    # Macro density adds to demand (cells block routing)
    half = sizes / 2.0
    lo = (coords - half) * R
    hi = (coords + half) * R
    for i in range(coords.shape[0]):
        x0 = max(0, int(lo[i, 0].item())); x1 = min(R, int(hi[i, 0].item()) + 1)
        y0 = max(0, int(lo[i, 1].item())); y1 = min(R, int(hi[i, 1].item()) + 1)
        if x1 > x0 and y1 > y0:
            grid_demand[y0:y1, x0:x1] += 0.5  # extra demand from macro presence

    # Overflow = max(0, demand - supply), summed
    overflow = (grid_demand - track_supply).clamp(min=0).sum()
    # Normalize to [0, 1] — empirical max for our scale
    max_overflow = float(R * R * 4.0)
    score = (overflow / max_overflow).clamp(0, 1).item()
    return score


# ============================================================
# DATASET — generate (placement, routability) pairs
# ============================================================
class RoutabilityDataset(Dataset):
    """
    Generates (placement_image, routability_score) training pairs.

    Mix of:
      - Random placements (high coverage of bad cases)
      - SA-expert placements (good cases)
      - PareDiff-generated placements (mid cases) — if model provided
    """
    def __init__(self, n_samples: int = 500,
                 macros_range=(10, 30), nets_range=(15, 60),
                 model: PareDiff | None = None,
                 device: torch.device = torch.device('cpu'),
                 resolution: int = 64,
                 seed: int = 42):
        self.n_samples = n_samples
        self.resolution = resolution
        self.samples = []
        self._generate(macros_range, nets_range, model, device, seed)

    def _generate(self, macros_range, nets_range, model, device, seed):
        import random
        rng = random.Random(seed)
        t0 = time.time()
        for i in range(self.n_samples):
            n_macros = rng.randint(*macros_range)
            n_nets = rng.randint(*nets_range)
            netlist = random_netlist(n_macros=n_macros, n_nets=n_nets,
                                     seed=seed + i)
            # 3 source modes mixed
            mode = i % 3
            if mode == 0:
                # Random placement (bad)
                coords = torch.rand(n_macros, 2)
            elif mode == 1:
                # SA placement (good)
                coords = expert_placer(netlist, steps=200, seed=seed + i + 1000)
            elif mode == 2 and model is not None:
                # Model-generated (medium)
                with torch.no_grad():
                    orient = random_orientations(n_macros, seed=seed + i)
                    batch = to_pyg_data(netlist, torch.rand(n_macros, 2),
                                        orient).to(device)
                    coords, _ = model.sample(batch, num_steps=25)
                    coords = coords.cpu()
            else:
                coords = torch.rand(n_macros, 2)

            # Compute routability
            score = heuristic_routability(
                coords, netlist['sizes'], netlist['hyperedges'],
                resolution=self.resolution
            )
            # Render image (multi-channel)
            img = render_placement(coords, netlist['sizes'],
                                   netlist['hyperedges'],
                                   resolution=self.resolution)
            self.samples.append((img.cpu(), score))

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  generated {i+1}/{self.n_samples} samples "
                      f"({elapsed:.1f}s)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, score = self.samples[idx]
        return img, torch.tensor(score, dtype=torch.float32)


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--diffusion_ckpt', type=str, default=None,
                   help='Optional: path to trained PareDiff for sample diversity')
    p.add_argument('--n_samples', type=int, default=500)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--resolution', type=int, default=64)
    p.add_argument('--ckpt_dir', type=str,
                   default='experiments/routability_checkpoints')
    p.add_argument('--hf_repo', type=str, default=None)
    p.add_argument('--use_wandb', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ----- Load diffusion model (optional) -----
    diff_model = None
    if args.diffusion_ckpt:
        print(f"Loading diffusion model from {args.diffusion_ckpt}")
        state = torch.load(args.diffusion_ckpt, map_location=device,
                           weights_only=False)
        cfg: PareDiffConfig = state['cfg']
        diff_model = PareDiff(cfg, in_node_dim=3).to(device)
        diff_model.load_state_dict(state['model'])
        diff_model.eval()

    # ----- Build dataset -----
    print(f"Generating {args.n_samples} (placement, routability) pairs...")
    dataset = RoutabilityDataset(
        n_samples=args.n_samples, model=diff_model, device=device,
        resolution=args.resolution,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0)
    scores_only = [s for _, s in dataset.samples]
    print(f"  Routability range: [{min(scores_only):.4f}, {max(scores_only):.4f}]"
          f"   mean={sum(scores_only)/len(scores_only):.4f}")

    # ----- Build classifier -----
    classifier = RoutabilityNet(in_channels=4, base=32).to(device)
    n_params = sum(p.numel() for p in classifier.parameters())
    print(f"RoutabilityNet: {n_params/1e6:.1f}M params")

    # ----- WandB -----
    if args.use_wandb:
        import wandb
        wandb.init(project='parediff-routability',
                   config=vars(args), resume='allow')

    # ----- Train -----
    opt = AdamW(classifier.parameters(), lr=args.lr, weight_decay=1e-5)
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        classifier.train()
        losses = []
        for img, target in loader:
            img = img.to(device); target = target.to(device)
            pred = classifier(img)
            loss = F.mse_loss(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        avg = sum(losses) / max(len(losses), 1)
        print(f"epoch {epoch:3d} | MSE {avg:.5f}")
        if args.use_wandb:
            import wandb
            wandb.log({'mse': avg, 'epoch': epoch})

    final = ckpt_dir / 'routability_final.pt'
    torch.save({'model': classifier.state_dict()}, final)
    print(f"\n✓ Saved {final}")

    # ----- Push to HF -----
    if args.hf_repo:
        try:
            from huggingface_hub import HfApi, create_repo
            create_repo(args.hf_repo, repo_type='model', exist_ok=True,
                        private=True)
            HfApi().upload_folder(
                folder_path=str(ckpt_dir), repo_id=args.hf_repo,
                repo_type='model',
            )
            print(f"  ✓ Pushed to https://huggingface.co/{args.hf_repo}")
        except Exception as e:
            print(f"  ⚠ HF push failed: {e}")


if __name__ == '__main__':
    main()
