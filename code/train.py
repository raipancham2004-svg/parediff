"""
Training entry point for PareDiff.

Usage (once GPU available):
    python -m code.train --data data/ispd15 --suite ispd15 --epochs 200 \
                         --lr 1e-4 --gnn_hidden 128 --denoiser_hidden 256

Two-stage training:
  Stage 1: Pre-train PareDiff on (netlist, expert-placement) pairs
           — supervised ε-prediction loss + orientation cross-entropy.
  Stage 2: Fine-tune RoutabilityNet on (placement, OpenROAD-routability) pairs
           — supervised regression loss.

For the 1-month sprint:
  - Stage 1: ~24-48 hr on RTX 4090
  - Stage 2: ~6-12 hr (smaller model, smaller dataset)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from .dataset import MacroPlacementDataset
from .diffusion_model import PareDiff
from .utils import PareDiffConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True, help='dataset root')
    p.add_argument('--suite', type=str, default='ispd15')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--gnn_hidden', type=int, default=128)
    p.add_argument('--gnn_layers', type=int, default=4)
    p.add_argument('--denoiser_hidden', type=int, default=256)
    p.add_argument('--denoiser_layers', type=int, default=6)
    p.add_argument('--T', type=int, default=1000)
    p.add_argument('--ckpt_dir', type=str, default='experiments/checkpoints')
    p.add_argument('--log_every', type=int, default=10)
    p.add_argument('--ckpt_every', type=int, default=20)
    p.add_argument('--use_wandb', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PareDiffConfig(
        gnn_hidden=args.gnn_hidden,
        gnn_layers=args.gnn_layers,
        denoiser_hidden=args.denoiser_hidden,
        denoiser_layers=args.denoiser_layers,
        T=args.T,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        use_wandb=args.use_wandb,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    dataset = MacroPlacementDataset(root=args.data, suite=args.suite, split='train')
    print(f"Loaded {len(dataset)} training designs")

    # Model
    model = PareDiff(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params")

    # Optimizer
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.05
    )

    # WandB (optional)
    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.project_name, config=vars(args))

    # Checkpoint dir
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    step = 0
    t_start = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses = []
        for design_idx in range(len(dataset)):
            data = dataset[design_idx]
            data = data.to(device)

            losses = model.training_step(data)
            loss = losses['loss']

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            epoch_losses.append({k: v.item() if torch.is_tensor(v) else v
                                  for k, v in losses.items()})
            step += 1
            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                print(f"step {step:5d} | loss {loss.item():.4f} | "
                      f"eps {losses['eps_loss']:.4f} | "
                      f"orient {losses['orient_loss']:.4f} | "
                      f"{elapsed:.0f}s")
                if cfg.use_wandb:
                    import wandb
                    wandb.log({**{k: v.item() if torch.is_tensor(v) else v
                                   for k, v in losses.items()},
                                'epoch': epoch, 'step': step, 'lr': opt.param_groups[0]['lr']})

        scheduler.step()

        if (epoch + 1) % args.ckpt_every == 0:
            ckpt_path = ckpt_dir / f'parediff_epoch{epoch+1:04d}.pt'
            torch.save({
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'cfg': cfg,
                'epoch': epoch,
            }, ckpt_path)
            print(f"  ✓ saved {ckpt_path}")

    # Final
    final_path = ckpt_dir / 'parediff_final.pt'
    torch.save({'model': model.state_dict(), 'cfg': cfg}, final_path)
    print(f"Training done. Final ckpt: {final_path}")


if __name__ == '__main__':
    main()
