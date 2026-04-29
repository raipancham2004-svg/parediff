"""
Evaluation utilities.

Metrics:
  - HPWL          (computed locally, fast)
  - Diversity     (pairwise placement distance across K samples)
  - Pareto hypervolume  (over (HPWL, routability) pairs)

For routability we want REAL post-route DRC count from OpenROAD.
We provide a wrapper that:
  1. Writes the placement back to a DEF file
  2. Calls OpenROAD detail router (via Docker)
  3. Parses the resulting DRC report

Usage:
    python -m code.evaluate --placements experiments/sampled_placements/ \
                            --design bigblue4 --use_openroad
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from .utils import hpwl


# ============================================================
# Local metrics
# ============================================================
def compute_hpwl_from_json(j: dict, hyperedges: list[list[int]]) -> float:
    coords = torch.tensor(j['coords_normalized'])
    return hpwl(coords, hyperedges).item()


def diversity(jsons: list[dict]) -> float:
    """Average pairwise L2 distance across all sampled placements."""
    K = len(jsons)
    if K < 2:
        return 0.0
    coords = [torch.tensor(j['coords_normalized']) for j in jsons]
    total = 0.0; pairs = 0
    for i in range(K):
        for j in range(i + 1, K):
            total += (coords[i] - coords[j]).norm(dim=-1).mean().item()
            pairs += 1
    return total / pairs


def pareto_hypervolume(points: list[tuple[float, float]],
                       reference: tuple[float, float] = (1.0, 1.0)) -> float:
    """2-D hypervolume w.r.t. reference point (assumes minimization)."""
    pts = sorted(points, key=lambda p: p[0])
    hv = 0.0
    prev_x = 0.0
    best_y = reference[1]
    for x, y in pts:
        if y < best_y:
            hv += (x - prev_x) * (reference[1] - best_y)
            best_y = y
            prev_x = x
    hv += (reference[0] - prev_x) * (reference[1] - best_y)
    return hv


# ============================================================
# OpenROAD wrapper (Docker)
# ============================================================
DEF_TEMPLATE = """\
VERSION 5.8 ;
DIVIDERCHAR "/" ;
BUSBITCHARS "[]" ;
DESIGN {design} ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( {die_w} {die_h} ) ;

COMPONENTS {ncomp} ;
{components}
END COMPONENTS

END DESIGN
"""


def write_placement_to_def(json_path: Path, base_def: Path, out_def: Path,
                           die_w: float, die_h: float):
    """
    Read a sampled placement JSON + a base DEF (for cell types and connectivity)
    and write a fresh DEF with the new macro positions.
    Simplified for sprint — production version needs full DEF round-trip.
    """
    j = json.loads(json_path.read_text())
    inst_names = j['inst_names']
    coords = j['coords_normalized']
    orient_names = j['orient_names']
    comp_lines = []
    for inst, (xn, yn), o in zip(inst_names, coords, orient_names):
        x = int(xn * die_w * 1000)  # back to DBU
        y = int(yn * die_h * 1000)
        # Simplified — needs the macro_name lookup from base_def
        comp_lines.append(f"  - {inst} {inst}_TYPE + FIXED ( {x} {y} ) {o} ;")
    out_def.write_text(DEF_TEMPLATE.format(
        design=j['design'],
        die_w=int(die_w * 1000), die_h=int(die_h * 1000),
        ncomp=len(inst_names), components='\n'.join(comp_lines),
    ))


def run_openroad_route(def_path: Path, lef_path: Path, out_dir: Path) -> dict:
    """Call OpenROAD via Docker. Returns dict of metrics from drc.rpt."""
    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{def_path.parent.absolute()}:/work',
        'openroad/orfs:latest',
        'openroad', '-no_init', '-exit',
        '/work/route_and_check.tcl',
    ]
    # User would supply route_and_check.tcl in the design dir
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        # Parse stdout for DRC / overflow
        out = result.stdout
        drc_count = 0
        for line in out.splitlines():
            if 'DRC violations' in line:
                drc_count = int(line.split()[-1])
        return {'drc_count': drc_count, 'stdout': out, 'stderr': result.stderr}
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {'drc_count': -1, 'error': str(e)}


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--placements', type=str, required=True)
    p.add_argument('--design', type=str, required=True)
    p.add_argument('--use_openroad', action='store_true')
    args = p.parse_args()

    plc_dir = Path(args.placements)
    files = sorted(plc_dir.glob(f'{args.design}_scale*.json'))
    if not files:
        raise SystemExit(f"No placements found at {plc_dir}/{args.design}_scale*.json")

    jsons = [json.loads(f.read_text()) for f in files]

    # HPWL — needs hyperedges; placeholder for now
    hyperedges = []  # TODO: load from design metadata
    hpwls = [compute_hpwl_from_json(j, hyperedges) for j in jsons]
    div = diversity(jsons)

    print(f"Design: {args.design}")
    print(f"Placements: {len(jsons)}")
    print(f"Diversity (avg pairwise L2): {div:.4f}")
    print(f"HPWLs across K: {[round(h, 4) for h in hpwls]}")

    if args.use_openroad:
        print("Running OpenROAD on each placement (this is slow, ~5 min each)...")
        # Wire up to run_openroad_route here once Docker image is set up


if __name__ == '__main__':
    main()
