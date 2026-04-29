"""
Training entry point for PareDiff.

Two modes:
  - --synthetic   : on-the-fly synthetic netlists (default, no dataset needed)
  - --data <path> : real LEF/DEF designs from disk

Usage (Kaggle-friendly default):
    python -m code.train --synthetic --epochs 200 --use_wandb

Usage (later, with real data):
    python -m code.train --data data/ispd15 --suite ispd15 --epochs 200
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from .diffusion_model import PareDiff
from .synthetic_dataset import SyntheticPlacementDataset
from .utils import PareDiffConfig

try:
    from .dataset import MacroPlacementDataset
    REAL_DATA_AVAILABLE = True
except Exception:
    REAL_DATA_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser()
    # Data source
    p.add_argument('--synthetic', action='store_true',
                   help='Use synthetic on-the-fly netlists (no disk dataset needed)')
    p.add_argument('--data', type=str, default=None, help='Real dataset root')
    p.add_argument('--suite', type=str, default='ispd15')
    p.add_argument('--n_designs_per_epoch', type=int, default=100,
                   help='Synthetic mode: how many designs per epoch')
    p.add_argument('--macros_min', type=int, default=10)
    p.add_argument('--macros_max', type=int, default=40)
    # Training
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--gnn_hidden', type=int, default=128)
    p.add_argument('--gnn_layers', type=int, default=4)
    p.add_argument('--denoiser_hidden', type=int, default=256)
    p.add_argument('--denoiser_layers', type=int, default=6)
    p.add_argument('--T', type=int, default=1000)
    # Logging + checkpointing
    p.add_argument('--ckpt_dir', type=str, default='experiments/checkpoints')
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--ckpt_every', type=int, default=10)
    p.add_argument('--use_wandb', action='store_true')
    p.add_argument('--resume', type=str, default=None,
                   help='Resume from checkpoint path')
    p.add_argument('--hf_repo', type=str, default=None,
                   help='If set, push checkpoints to this HF Hub repo every --hf_push_every epochs')
    p.add_argument('--hf_push_every', type=int, default=10,
                   help='Push to HF Hub every N epochs (only if --hf_repo set)')
    return p.parse_args()


def push_to_hf(ckpt_dir, hf_repo):
    """Push the entire ckpt dir to HF Hub. Idempotent."""
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi()
        create_repo(hf_repo, repo_type='model', exist_ok=True, private=True)
        api.upload_folder(
            folder_path=str(ckpt_dir),
            repo_id=hf_repo,
            repo_type='model',
        )
        print(f"  ✓ Pushed checkpoints to https://huggingface.co/{hf_repo}")
    except Exception as e:
        print(f"  ⚠ HF push failed: {e}")


def main():
    args = parse_args()
    cfg = PareDiffConfig(
        gnn_hidden=args.gnn_hidden,
        gnn_layers=args.gnn_layers,
        denoiser_hidden=args.denoiser_hidden,
        denoiser_layers=args.denoiser_layers,
        T=args.T,
        lr=args.lr,
        epochs=args.epochs,
        use_wandb=args.use_wandb,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)} | "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ----------------- Data -----------------
    if args.synthetic or args.data is None:
        print(f"Synthetic mode: {args.n_designs_per_epoch} designs/epoch, "
              f"macros {args.macros_min}-{args.macros_max}")
        dataset = SyntheticPlacementDataset(
            n_designs=args.n_designs_per_epoch,
            macros_range=(args.macros_min, args.macros_max),
            sa_steps=300,
        )
    else:
        if not REAL_DATA_AVAILABLE:
            raise RuntimeError("Real-data dataset module not importable")
        dataset = MacroPlacementDataset(root=args.data, suite=args.suite,
                                        split='train')
        print(f"Loaded {len(dataset)} real designs from {args.data}")

    # ----------------- Model -----------------
    model = PareDiff(cfg, in_node_dim=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params")

    # ----------------- Optimizer -----------------
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.05
    )

    # ----------------- Resume -----------------
    start_epoch = 0
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = args.resume
    if resume_path is None:
        # Try auto-resume from latest in ckpt_dir
        candidates = sorted(ckpt_dir.glob('parediff_epoch*.pt'))
        if candidates:
            resume_path = str(candidates[-1])
            print(f"Auto-resuming from {resume_path}")
    if resume_path:
        try:
            state = torch.load(resume_path, map_location=device)
            model.load_state_dict(state['model'])
            opt.load_state_dict(state['opt'])
            start_epoch = state.get('epoch', 0) + 1
            print(f"  ✓ Resumed at epoch {start_epoch}")
        except Exception as e:
            print(f"  Resume failed ({e}); starting fresh")

    # ----------------- WandB -----------------
    if cfg.use_wandb:
        import wandb
        run = wandb.init(project=cfg.project_name, config=vars(args),
                         resume='allow')
        print(f"  wandb run URL: {run.url}")

    # ============================================================
    # TRAIN
    # ============================================================
    step = 0
    t_start = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        if hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(epoch)
        model.train()
        epoch_losses = []
        for design_idx in range(len(dataset)):
            data = dataset[design_idx]
            # PyG Data has its own .to()
            try:
                data = data.to(device)
            except AttributeError:
                # dict fallback
                data = {k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in data.items()}

            losses = model.training_step(data)
            loss = losses['loss']

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            epoch_losses.append(loss.item())
            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t_start
                print(f"epoch {epoch:3d} step {step:6d} | "
                      f"loss {loss.item():.4f} | "
                      f"eps {losses['eps_loss']:.4f} | "
                      f"orient {losses['orient_loss']:.4f} | "
                      f"{elapsed:.0f}s")
                if cfg.use_wandb:
                    import wandb
                    wandb.log({
                        'loss': loss.item(),
                        'eps_loss': losses['eps_loss'].item(),
                        'orient_loss': losses['orient_loss'].item(),
                        'epoch': epoch, 'step': step,
                        'lr': opt.param_groups[0]['lr'],
                    })

        scheduler.step()
        avg_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        print(f"== epoch {epoch} done | avg loss {avg_epoch_loss:.4f} | "
              f"{(time.time()-t_start)/60:.1f}m total")

        if (epoch + 1) % args.ckpt_every == 0 or epoch == cfg.epochs - 1:
            ckpt_path = ckpt_dir / f'parediff_epoch{epoch+1:04d}.pt'
            torch.save({
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'cfg': cfg,
                'epoch': epoch,
                'avg_loss': avg_epoch_loss,
            }, ckpt_path)
            print(f"  ✓ saved {ckpt_path}")

        if args.hf_repo and (epoch + 1) % args.hf_push_every == 0:
            push_to_hf(ckpt_dir, args.hf_repo)

    final = ckpt_dir / 'parediff_final.pt'
    torch.save({'model': model.state_dict(), 'cfg': cfg}, final)
    print(f"\nTraining done. Final ckpt: {final}")

    # Final push regardless
    if args.hf_repo:
        push_to_hf(ckpt_dir, args.hf_repo)

    if cfg.use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
