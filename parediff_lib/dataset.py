"""
Dataset module — loads ISPD'15 / TILOS-AI MacroPlacement designs, converts
each design's netlist into a PyTorch Geometric Data object, and exposes
the macro-coordinate ground truth (when available, for training the model
to MATCH a known good placement during pretraining).

Two modes:
  1. SUPERVISED — pretraining on (netlist, expert_placement) pairs from
     analytical placers (TritonMacroPlace / AutoDMP outputs).
  2. SELF-SUPERVISED — score-matching pretraining where x_0 comes from
     uniform random valid placements, used as conditional augmentation.

Expected on-disk layout (after running scripts/download_data.sh):

    data/
      ispd15/
        bigblue1/
          design.def              ← parsed for tile boundary + macro list
          design.lef              ← parsed for macro sizes
          design.v                ← parsed for hyperedge connectivity
          baseline_placement.def  ← TritonMacroPlace ground truth
        bigblue2/ ...
      asap7/
        ariane/
          ...
        nvdla/
          ...
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None  # PyG not installed yet — placeholder


# ============================================================
# DESIGN LOADERS (LEF/DEF parsing)
# ============================================================
def parse_lef_macros(lef_path: Path) -> dict[str, dict]:
    """
    Lightweight LEF parser — extracts macro names and (size_x, size_y).
    Real LEF parsing is non-trivial; we use a regex-based subset for our designs.
    Returns: {macro_name: {'size': (w, h), 'pins': [...] }}
    """
    macros = {}
    current = None
    text = lef_path.read_text()
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith('MACRO '):
            current = line.split()[1]
            macros[current] = {'size': None, 'pins': []}
        elif current and line.startswith('SIZE '):
            # SIZE <w> BY <h> ;
            parts = line.split()
            w = float(parts[1]); h = float(parts[3])
            macros[current]['size'] = (w, h)
        elif line.startswith('END ') and current and line.split()[1] == current:
            current = None
    return macros


def parse_def_components(def_path: Path) -> tuple[list, tuple, list]:
    """
    DEF parser — extracts:
      - components: [(inst_name, macro_name, x, y, orient, fixed), ...]
      - die_area: (w, h)
      - nets: [(net_name, [inst_pin, ...]), ...]
    """
    text = def_path.read_text()
    components = []
    die = None
    nets = []
    in_components = in_nets = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith('DIEAREA'):
            # DIEAREA ( 0 0 ) ( w h ) ;
            parts = line.replace('(', '').replace(')', '').split()
            w = float(parts[3]); h = float(parts[4])
            die = (w, h)
        if line.startswith('COMPONENTS '):
            in_components = True; continue
        if line.startswith('END COMPONENTS'):
            in_components = False; continue
        if in_components and line.startswith('-'):
            # - inst_name macro_name + ( PLACED|FIXED ( x y ) orient )
            tokens = line.split()
            inst = tokens[1]; macro = tokens[2]
            fixed = '+ FIXED' in line
            placed = '+ PLACED' in line or fixed
            x, y, orient = 0, 0, 'N'
            if placed:
                # Find ( ... )
                lp = tokens.index('('); rp = tokens.index(')')
                x = float(tokens[lp + 1]); y = float(tokens[lp + 2])
                orient = tokens[rp + 1]
            components.append((inst, macro, x, y, orient, fixed))
        if line.startswith('NETS '):
            in_nets = True; continue
        if line.startswith('END NETS'):
            in_nets = False; continue
        if in_nets and line.startswith('-'):
            tokens = line.split()
            net_name = tokens[1]
            pins = []
            for i, tok in enumerate(tokens):
                if tok == '(' and i + 2 < len(tokens):
                    inst_pin = (tokens[i + 1], tokens[i + 2])
                    pins.append(inst_pin)
            nets.append((net_name, pins))
    return components, die, nets


# ============================================================
# NETLIST → PYG GRAPH
# ============================================================
def netlist_to_graph(macros_lef, components, nets, die, only_macros=True):
    """
    Build PyG Data object:
      - x: (N, F) macro features (size_w, size_h, num_pins, ...)
      - edge_index: (2, E) — clique expansion of hyperedges
      - edge_attr: (E, 1) — net weight (1 / hyperedge_size)
      - target_pos: (N, 2) ground-truth coords (normalized to [0,1])
      - target_orient: (N,) orient class index
      - sizes: (N, 2) macro sizes (normalized)
      - die: (2,) die size
      - inst_names: list of length N
    """
    # Filter to macros only (skip std cells)
    if only_macros:
        macro_set = set(macros_lef.keys())
        components = [c for c in components if c[1] in macro_set]
    inst_to_idx = {c[0]: i for i, c in enumerate(components)}
    N = len(components)
    if N == 0:
        return None

    # Features + targets
    sizes = []
    target_pos = []
    target_orient = []
    feats = []
    orient_map = {'N': 0, 'E': 1, 'S': 2, 'W': 3,
                  'FN': 4, 'FE': 5, 'FS': 6, 'FW': 7,
                  'R0': 0, 'R90': 1, 'R180': 2, 'R270': 3,
                  'MX': 4, 'MY': 5, 'MXR90': 6, 'MYR90': 7}
    die_w, die_h = die
    for inst, macro, x, y, orient, _fixed in components:
        w, h = macros_lef[macro]['size']
        sizes.append([w / die_w, h / die_h])
        target_pos.append([x / die_w, y / die_h])
        target_orient.append(orient_map.get(orient, 0))
        feats.append([w / die_w, h / die_h, len(macros_lef[macro].get('pins', []))])

    # Hyperedge clique expansion
    edge_src = []
    edge_dst = []
    edge_w = []
    for _net_name, pins in nets:
        macro_pins = [p for p in pins if p[0] in inst_to_idx]
        if len(macro_pins) < 2:
            continue
        weight = 1.0 / len(macro_pins)
        for i, (a, _) in enumerate(macro_pins):
            for b, _ in macro_pins[i + 1:]:
                ai = inst_to_idx[a]; bi = inst_to_idx[b]
                edge_src.extend([ai, bi])
                edge_dst.extend([bi, ai])
                edge_w.extend([weight, weight])

    if Data is None:
        # PyG not installed yet — return raw dict
        return {
            'x': torch.tensor(feats, dtype=torch.float32),
            'edge_index': torch.tensor([edge_src, edge_dst], dtype=torch.long),
            'edge_attr': torch.tensor(edge_w, dtype=torch.float32).unsqueeze(1),
            'target_pos': torch.tensor(target_pos, dtype=torch.float32),
            'target_orient': torch.tensor(target_orient, dtype=torch.long),
            'sizes': torch.tensor(sizes, dtype=torch.float32),
            'die': torch.tensor([1.0, 1.0]),  # normalized
            'inst_names': [c[0] for c in components],
        }

    data = Data(
        x=torch.tensor(feats, dtype=torch.float32),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_w, dtype=torch.float32).unsqueeze(1),
    )
    data.target_pos = torch.tensor(target_pos, dtype=torch.float32)
    data.target_orient = torch.tensor(target_orient, dtype=torch.long)
    data.sizes = torch.tensor(sizes, dtype=torch.float32)
    data.die = torch.tensor([1.0, 1.0])
    data.inst_names = [c[0] for c in components]
    return data


# ============================================================
# DATASET
# ============================================================
class MacroPlacementDataset(Dataset):
    """Iterates over designs in a benchmark suite, returning PyG Data objects."""

    def __init__(self, root: Path, suite: str = 'ispd15',
                 split: str = 'train', cache: bool = True):
        self.root = Path(root) / suite
        self.split = split
        self.cache = cache
        self.cache_path = self.root / f'_cache_{split}.pt'
        if cache and self.cache_path.exists():
            self.designs = torch.load(self.cache_path)
        else:
            self.designs = self._load_all()
            if cache:
                torch.save(self.designs, self.cache_path)

    def _load_all(self):
        # Hardcoded splits for now — easy to refactor
        ispd_train = ['bigblue1', 'bigblue2', 'bigblue3', 'newblue1', 'newblue2']
        ispd_test  = ['bigblue4', 'newblue3']
        asap_test  = ['ariane', 'nvdla']
        if 'ispd' in str(self.root) and self.split == 'train':
            names = ispd_train
        elif 'ispd' in str(self.root):
            names = ispd_test
        else:
            names = asap_test

        data_list = []
        for name in names:
            d = self.root / name
            if not d.exists():
                continue
            try:
                lef = parse_lef_macros(d / 'design.lef')
                comps, die, nets = parse_def_components(d / 'design.def')
                graph = netlist_to_graph(lef, comps, nets, die)
                if graph is not None:
                    graph.name = name
                    data_list.append(graph)
            except FileNotFoundError as e:
                print(f"[skip] {name}: {e}")
        return data_list

    def __len__(self):
        return len(self.designs)

    def __getitem__(self, idx):
        return self.designs[idx]
