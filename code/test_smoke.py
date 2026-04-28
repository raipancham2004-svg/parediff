"""
CPU-only smoke test — verifies the model can be instantiated and run a single
training step on synthetic dummy data, without needing a real dataset or GPU.

Run with:
    cd /home/panrai/Desktop/GraphDiff
    python -m code.test_smoke
"""
from __future__ import annotations

import torch

# Try imports
print("[1/5] Checking imports...")
try:
    from code.utils import PareDiffConfig
    from code.diffusion_model import PareDiff
    print("  ✓ Imports OK")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    raise SystemExit(1)


print("[2/5] Building model with default config...")
cfg = PareDiffConfig(
    gnn_hidden=64, gnn_layers=2,
    denoiser_hidden=128, denoiser_layers=2,
    T=100,  # smaller for smoke test
)
model = PareDiff(cfg, in_node_dim=3)
print(f"  ✓ Model built — {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")


print("[3/5] Building synthetic batch (10 macros, 20 nets)...")
N = 10
torch.manual_seed(42)
batch = type('B', (), {})()
batch.x = torch.randn(N, 3)
# Random edge_index — small graph
src = torch.randint(0, N, (40,))
dst = torch.randint(0, N, (40,))
batch.edge_index = torch.stack([src, dst])
batch.target_pos = torch.rand(N, 2)
batch.target_orient = torch.randint(0, 8, (N,))
batch.sizes = torch.rand(N, 2) * 0.1
print("  ✓ Synthetic batch built")


print("[4/5] Running one training step...")
losses = model.training_step(batch)
print(f"  ✓ Training step OK — loss={losses['loss'].item():.4f}, "
      f"eps={losses['eps_loss'].item():.4f}, "
      f"orient={losses['orient_loss'].item():.4f}")


print("[5/5] Running unguided sampling (10 DDIM steps)...")
coords, orient = model.sample(batch, num_steps=10)
print(f"  ✓ Sample shape: coords={tuple(coords.shape)}, orient={tuple(orient.shape)}")
print(f"  Coord range: [{coords.min().item():.3f}, {coords.max().item():.3f}]")

print("\n" + "=" * 50)
print("ALL SMOKE TESTS PASSED ✓")
print("=" * 50)
