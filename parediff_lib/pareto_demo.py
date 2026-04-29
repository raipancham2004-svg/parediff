"""
Pareto-front sampling demo — the headline contribution of PareDiff.

Workflow:
  1. Load trained PareDiff model from disk (or HF Hub if local missing)
  2. Train (or load) a routability classifier on-the-fly if missing
  3. For each test netlist, generate K placements at K guidance scales
  4. Compute HPWL + heuristic routability for each
  5. Save results as JSON + print summary table

Usage:
    python -m parediff_lib.pareto_demo \
        --diffusion_ckpt experiments/checkpoints/parediff_final.pt \
        --rout_ckpt experiments/routability_checkpoints/routability_final.pt \
        --K 8 \
        --n_test 5 \
        --out experiments/pareto_results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .diffusion_model import PareDiff
from .routability_classifier import RoutabilityNet, RoutabilityGuidance
from .synthetic_dataset import (
    random_netlist, expert_placer, random_orientations, to_pyg_data,
)
from .train_routability import (
    heuristic_routability, RoutabilityDataset,
)
from .utils import hpwl, PareDiffConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--diffusion_ckpt', type=str, required=True)
    p.add_argument('--rout_ckpt', type=str, default=None,
                   help='If absent, train a small classifier on the fly')
    p.add_argument('--K', type=int, default=8,
                   help='Placements per netlist (Pareto-front size)')
    p.add_argument('--n_test', type=int, default=5,
                   help='Number of test netlists')
    p.add_argument('--macros_min', type=int, default=15)
    p.add_argument('--macros_max', type=int, default=25)
    p.add_argument('--guidance_min', type=float, default=0.0)
    p.add_argument('--guidance_max', type=float, default=5.0)
    p.add_argument('--sample_steps', type=int, default=50)
    p.add_argument('--out', type=str, default='experiments/pareto_results.json')
    return p.parse_args()


def quick_train_classifier(diff_model, device, n_samples=200, epochs=15):
    """Bootstrap a routability classifier in <5 minutes."""
    print(f"Quick-training routability classifier ({n_samples} samples, {epochs} epochs)...")
    ds = RoutabilityDataset(n_samples=n_samples, model=diff_model, device=device,
                            resolution=64, seed=99)
    rnet = RoutabilityNet(in_channels=4, base=32).to(device)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(rnet.parameters(), lr=1e-3)
    for ep in range(epochs):
        rnet.train()
        losses = []
        for img, tgt in loader:
            img, tgt = img.to(device), tgt.to(device)
            loss = F.mse_loss(rnet(img), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"  ep{ep:02d} | mse {sum(losses)/len(losses):.6f}")
    return rnet


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Load diffusion model ----
    print(f"Loading PareDiff from {args.diffusion_ckpt}")
    state = torch.load(args.diffusion_ckpt, map_location=device, weights_only=False)
    cfg: PareDiffConfig = state['cfg']
    model = PareDiff(cfg, in_node_dim=3).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    print(f"  ✓ {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # ---- Load (or train) routability classifier ----
    rnet = None
    if args.rout_ckpt and Path(args.rout_ckpt).exists():
        print(f"Loading routability classifier from {args.rout_ckpt}")
        r_state = torch.load(args.rout_ckpt, map_location=device, weights_only=False)
        rnet = RoutabilityNet(in_channels=4, base=32).to(device)
        rnet.load_state_dict(r_state['model'])
        rnet.eval()
        print("  ✓ Classifier loaded")
    else:
        rnet = quick_train_classifier(model, device, n_samples=200, epochs=15)
        # Save it for next time
        out_dir = Path('experiments/routability_checkpoints')
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'model': rnet.state_dict()}, out_dir / 'routability_final.pt')
        print(f"  ✓ Saved classifier to {out_dir / 'routability_final.pt'}")

    # ---- Generate K placements per test netlist ----
    print(f"\nGenerating {args.K} placements per netlist for {args.n_test} test netlists...")
    print(f"Guidance scales: linspace({args.guidance_min}, {args.guidance_max}, {args.K})")

    scales = torch.linspace(args.guidance_min, args.guidance_max, args.K).tolist()
    all_results = []

    for ti in range(args.n_test):
        seed = 5000 + ti * 17
        n_macros = (args.macros_min + args.macros_max) // 2
        netlist = random_netlist(n_macros=n_macros, n_nets=int(n_macros * 2),
                                 seed=seed)
        orient = random_orientations(n_macros, seed=seed)
        coords_init = torch.rand(n_macros, 2)
        batch = to_pyg_data(netlist, coords_init, orient).to(device)

        guidance = RoutabilityGuidance(rnet, hyperedges=netlist['hyperedges'],
                                       sizes=netlist['sizes'].to(device),
                                       resolution=64)

        netlist_results = {'netlist_id': ti, 'seed': seed,
                           'n_macros': n_macros, 'placements': []}

        for k, s in enumerate(scales):
            t0 = time.time()
            coords, orient_pred = model.sample(batch, guidance_fn=guidance,
                                               guidance_scale=s,
                                               num_steps=args.sample_steps)
            elapsed = time.time() - t0

            hp = hpwl(coords, netlist['hyperedges']).item()
            rt = heuristic_routability(coords, netlist['sizes'].to(device),
                                       netlist['hyperedges'], resolution=64)
            std_x = coords[:, 0].std().item()
            std_y = coords[:, 1].std().item()

            netlist_results['placements'].append({
                'k': k, 'guidance_scale': s,
                'hpwl': hp, 'routability': rt,
                'coord_std': (std_x, std_y),
                'sample_time_sec': elapsed,
            })
            print(f"  net{ti} k={k} scale={s:.2f} | HPWL={hp:.3f} "
                  f"rout={rt:.4f} std=({std_x:.2f},{std_y:.2f}) "
                  f"time={elapsed:.1f}s")

        all_results.append(netlist_results)

    # ---- Save + summary ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved Pareto results to {out_path}")

    # ---- Print summary table ----
    print("\n" + "="*70)
    print("SUMMARY — does HPWL go UP and routability DOWN as guidance scales up?")
    print("="*70)
    for nr in all_results:
        ti = nr['netlist_id']
        print(f"\nNetlist {ti} ({nr['n_macros']} macros):")
        print(f"  {'k':>3} {'scale':>6} {'HPWL':>10} {'routability':>12} {'std_x':>7} {'std_y':>7}")
        for p in nr['placements']:
            print(f"  {p['k']:>3} {p['guidance_scale']:>6.2f} "
                  f"{p['hpwl']:>10.3f} {p['routability']:>12.4f} "
                  f"{p['coord_std'][0]:>7.3f} {p['coord_std'][1]:>7.3f}")
    print("\n→ Healthy Pareto: HPWL trends UP and routability trends DOWN as scale increases.")


if __name__ == '__main__':
    main()
