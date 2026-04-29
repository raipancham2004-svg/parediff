"""
Inference entry point — generate K Pareto-optimal placements per design.

Usage:
    python -m code.sample --ckpt experiments/checkpoints/parediff_final.pt \
                          --data data/ispd15 --design bigblue4 \
                          --K 8 --out experiments/sampled_placements/

Output:
    For each of K guidance scales, writes:
      <out>/<design>_scale{i}.json   ← coords + orient + scale
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .dataset import MacroPlacementDataset
from .diffusion_model import PareDiff
from .routability_classifier import RoutabilityNet, RoutabilityGuidance
from .utils import PareDiffConfig, ORIENT_NAMES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--suite', type=str, default='ispd15')
    p.add_argument('--design', type=str, required=True)
    p.add_argument('--K', type=int, default=8)
    p.add_argument('--rout_ckpt', type=str, default=None,
                   help='Optional routability classifier checkpoint')
    p.add_argument('--out', type=str, default='experiments/sampled_placements')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    state = torch.load(args.ckpt, map_location=device)
    cfg: PareDiffConfig = state['cfg']
    cfg.pareto_K = args.K
    model = PareDiff(cfg).to(device)
    model.load_state_dict(state['model'])
    model.eval()

    # Load design
    dataset = MacroPlacementDataset(root=args.data, suite=args.suite, split='test')
    design = next((d for d in dataset.designs if d.name == args.design), None)
    if design is None:
        raise ValueError(f"Design {args.design} not found in test set")
    design = design.to(device)

    # Optional routability guidance
    guidance_fn = None
    if args.rout_ckpt:
        rnet = RoutabilityNet().to(device)
        rnet.load_state_dict(torch.load(args.rout_ckpt, map_location=device))
        # Build hyperedges back from edge_index (PyG-compatible reconstruction)
        # For sprint: use a placeholder — the dataset has it stored elsewhere
        hyperedges = getattr(design, 'hyperedges', [])
        guidance_fn = RoutabilityGuidance(
            rnet, hyperedges=hyperedges, sizes=design.sizes
        )

    # Pareto sample
    print(f"Generating {args.K} placements for {args.design} ...")
    results = model.pareto_sample(design, guidance_fn=guidance_fn, K=args.K)

    # Write outputs
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(results):
        d = {
            'design': args.design,
            'guidance_scale': r['guidance_scale'],
            'inst_names': design.inst_names,
            'coords_normalized': r['coords'].cpu().tolist(),
            'orient_idx': r['orient'].cpu().tolist(),
            'orient_names': [ORIENT_NAMES[o] for o in r['orient'].cpu().tolist()],
        }
        path = out / f'{args.design}_scale{i:02d}.json'
        path.write_text(json.dumps(d, indent=2))
    print(f"Wrote {args.K} placements to {out}/")


if __name__ == '__main__':
    main()
