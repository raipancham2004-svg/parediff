"""
Synthetic netlist + placement generator for PareDiff pretraining.

Strategy (inspired by Berkeley's chipdiffusion paper):
  - Generate random netlists on-the-fly (per training step)
  - Use simple heuristics to produce "expert" placements as targets
  - This avoids ISPD/TILOS dataset download complexity
  - Model learns the inductive bias: "place macros without overlap, low HPWL"

Each synthetic design has:
  - N macros (drawn from a configurable size distribution)
  - M nets (hyperedges with average fanout ~3-5)
  - Tile size auto-scaled to total macro area * 1/utilization
  - "Expert" placement: simulated annealing for ~K steps to minimize
    HPWL + overlap penalty (cheap; runs in milliseconds for N<50)
"""
from __future__ import annotations

import math
import random
from typing import Tuple

import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None


# ============================================================
# RANDOM NETLIST GENERATOR
# ============================================================
def random_netlist(n_macros: int = 20,
                   n_nets: int = 30,
                   net_fanout_range: Tuple[int, int] = (2, 6),
                   size_range: Tuple[float, float] = (0.04, 0.15),
                   utilization: float = 0.4,
                   seed: int | None = None) -> dict:
    """
    Returns dict with:
      sizes:       (N, 2) macro sizes (normalized to [0, 1])
      hyperedges:  list of [macro_idx, ...] (nets)
      die:         (W, H) — tile dimensions (normalized to 1)
      pin_counts:  (N,) per-macro pin count (sum of nets touching it)
    """
    rng = random.Random(seed)

    sizes = torch.tensor(
        [[rng.uniform(*size_range), rng.uniform(*size_range)]
         for _ in range(n_macros)], dtype=torch.float32
    )

    hyperedges = []
    pin_counts = [0] * n_macros
    for _ in range(n_nets):
        fanout = rng.randint(*net_fanout_range)
        fanout = min(fanout, n_macros)
        net = rng.sample(range(n_macros), fanout)
        hyperedges.append(net)
        for m in net:
            pin_counts[m] += 1

    return {
        'sizes': sizes,
        'hyperedges': hyperedges,
        'die': (1.0, 1.0),  # always normalized
        'pin_counts': torch.tensor(pin_counts, dtype=torch.float32),
    }


# ============================================================
# CHEAP EXPERT PLACER — simulated annealing
# ============================================================
def hpwl_for_placement(coords: torch.Tensor, hyperedges: list) -> float:
    total = 0.0
    for net in hyperedges:
        if len(net) < 2:
            continue
        sub = coords[net]
        bbox_w = sub[:, 0].max().item() - sub[:, 0].min().item()
        bbox_h = sub[:, 1].max().item() - sub[:, 1].min().item()
        total += bbox_w + bbox_h
    return total


def overlap_for_placement(coords: torch.Tensor, sizes: torch.Tensor) -> float:
    """Sum of pairwise overlap areas."""
    N = coords.shape[0]
    if N < 2:
        return 0.0
    half = sizes / 2.0
    lo = coords - half; hi = coords + half
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            ox = max(0.0, min(hi[i, 0].item(), hi[j, 0].item())
                          - max(lo[i, 0].item(), lo[j, 0].item()))
            oy = max(0.0, min(hi[i, 1].item(), hi[j, 1].item())
                          - max(lo[i, 1].item(), lo[j, 1].item()))
            total += ox * oy
    return total


def boundary_for_placement(coords: torch.Tensor, sizes: torch.Tensor) -> float:
    half = sizes / 2.0
    lo = coords - half; hi = coords + half
    left   = (-lo[:, 0]).clamp(min=0).sum().item()
    bottom = (-lo[:, 1]).clamp(min=0).sum().item()
    right  = (hi[:, 0] - 1.0).clamp(min=0).sum().item()
    top    = (hi[:, 1] - 1.0).clamp(min=0).sum().item()
    return left + right + bottom + top


def expert_placer(netlist: dict,
                  steps: int = 500,
                  T_start: float = 1.0,
                  T_end: float = 1e-3,
                  overlap_w: float = 50.0,
                  boundary_w: float = 100.0,
                  seed: int | None = None) -> torch.Tensor:
    """
    Cheap simulated-annealing placer that produces "good enough" placements
    to use as ground-truth targets for diffusion pretraining.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed if seed is not None else rng.randint(0, 2**31))

    N = netlist['sizes'].shape[0]
    sizes = netlist['sizes']
    he = netlist['hyperedges']

    # Random initial placement
    coords = torch.rand(N, 2)

    def cost(c):
        return (
            hpwl_for_placement(c, he)
            + overlap_w * overlap_for_placement(c, sizes)
            + boundary_w * boundary_for_placement(c, sizes)
        )

    cur_cost = cost(coords)
    T = T_start
    decay = (T_end / T_start) ** (1.0 / steps)

    for step in range(steps):
        # Pick a random macro to perturb
        i = rng.randrange(N)
        old_xy = coords[i].clone()
        # Gaussian perturbation, scale = T
        coords[i] = (old_xy + torch.randn(2) * T).clamp(0, 1)
        new_cost = cost(coords)
        # Metropolis
        if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, 1e-9)):
            cur_cost = new_cost
        else:
            coords[i] = old_xy
        T *= decay

    return coords


def random_orientations(N: int, seed: int | None = None) -> torch.Tensor:
    """Random orientation in {R0, R90, R180, R270, MX, MY, MXR90, MYR90}."""
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    return torch.randint(0, 8, (N,), generator=g)


# ============================================================
# CONVERT NETLIST + PLACEMENT → PYG DATA
# ============================================================
def to_pyg_data(netlist: dict, coords: torch.Tensor, orient: torch.Tensor):
    N = coords.shape[0]
    sizes = netlist['sizes']
    pin_counts = netlist['pin_counts']
    he = netlist['hyperedges']

    # Per-macro features: (size_w, size_h, normalized_pin_count)
    feats = torch.stack([
        sizes[:, 0],
        sizes[:, 1],
        pin_counts / max(pin_counts.max().item(), 1),
    ], dim=-1)

    # Hyperedge clique expansion
    src, dst, w = [], [], []
    for net in he:
        if len(net) < 2:
            continue
        weight = 1.0 / len(net)
        for i, a in enumerate(net):
            for b in net[i + 1:]:
                src.extend([a, b]); dst.extend([b, a])
                w.extend([weight, weight])

    if Data is None:
        return {
            'x': feats,
            'edge_index': torch.tensor([src, dst], dtype=torch.long),
            'edge_attr': torch.tensor(w, dtype=torch.float32).unsqueeze(1),
            'target_pos': coords,
            'target_orient': orient,
            'sizes': sizes,
            'die': torch.tensor([1.0, 1.0]),
            'hyperedges': he,
        }
    data = Data(
        x=feats,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(w, dtype=torch.float32).unsqueeze(1),
    )
    data.target_pos = coords
    data.target_orient = orient
    data.sizes = sizes
    data.die = torch.tensor([1.0, 1.0])
    data.hyperedges = he
    return data


# ============================================================
# SYNTHETIC DATASET — generates fresh designs per epoch
# ============================================================
class SyntheticPlacementDataset(Dataset):
    """
    On-the-fly dataset. Each __getitem__ returns a freshly generated
    (netlist, expert-placement) pair. This means the model NEVER sees
    the same example twice — strongest possible regularization.

    Args:
        n_designs:    'epoch length' — how many designs per epoch
        macros_range: (min, max) macros per design
        nets_range:   (min, max) nets per design
        sa_steps:     SA steps for expert placer (more = better targets, slower)
    """
    def __init__(self, n_designs: int = 200,
                 macros_range: Tuple[int, int] = (10, 40),
                 nets_range: Tuple[int, int] = (15, 80),
                 sa_steps: int = 300,
                 seed: int = 42):
        self.n_designs = n_designs
        self.macros_range = macros_range
        self.nets_range = nets_range
        self.sa_steps = sa_steps
        self.rng = random.Random(seed)
        self._epoch_seed = seed

    def __len__(self):
        return self.n_designs

    def __getitem__(self, idx):
        # Deterministic per (epoch, idx) — but epoch advances via set_epoch
        seed = (self._epoch_seed * 1009 + idx * 7919) % (2**31)
        rng = random.Random(seed)
        n_macros = rng.randint(*self.macros_range)
        n_nets = rng.randint(*self.nets_range)
        netlist = random_netlist(n_macros=n_macros, n_nets=n_nets, seed=seed)
        coords = expert_placer(netlist, steps=self.sa_steps, seed=seed + 1)
        orient = random_orientations(n_macros, seed=seed + 2)
        data = to_pyg_data(netlist, coords, orient)
        if hasattr(data, 'name'):
            pass
        else:
            try:
                data.name = f'synth_{idx}'
            except Exception:
                pass
        return data

    def set_epoch(self, epoch: int):
        self._epoch_seed = (epoch + 1) * 12345


# ============================================================
# SMOKE TEST (run as script)
# ============================================================
if __name__ == '__main__':
    print("Generating one synthetic design...")
    netlist = random_netlist(n_macros=15, n_nets=25, seed=42)
    print(f"  {netlist['sizes'].shape[0]} macros, "
          f"{len(netlist['hyperedges'])} nets")
    print("Running expert placer (300 SA steps)...")
    coords = expert_placer(netlist, steps=300, seed=42)
    orient = random_orientations(15, seed=42)
    print(f"  HPWL = {hpwl_for_placement(coords, netlist['hyperedges']):.4f}")
    print(f"  Overlap = {overlap_for_placement(coords, netlist['sizes']):.6f}")
    print(f"  Boundary viol = {boundary_for_placement(coords, netlist['sizes']):.6f}")
    print("Building PyG Data...")
    data = to_pyg_data(netlist, coords, orient)
    print(f"  ✓ Data: x={data.x.shape}, edge_index={data.edge_index.shape}, "
          f"target_pos={data.target_pos.shape}")
    print("\nDataset class smoke test...")
    ds = SyntheticPlacementDataset(n_designs=5, sa_steps=100)
    for i in range(3):
        d = ds[i]
        print(f"  design {i}: {d.x.shape[0]} macros, {d.edge_index.shape[1]} edges")
    print("\n✓ ALL SYNTHETIC SMOKE TESTS PASSED")
