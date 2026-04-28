# PareDiff

**Pareto-Aware Diffusion Sampling for Routability-Driven Macro Placement with Cross-Technology Transfer**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)

A conditional diffusion model that generates a **Pareto front of K macro placements** spanning the (HPWL ↔ routability) trade-off in a single sampling pass, with **cross-technology transfer** (NanGate 45nm → ASAP7).

> ⚠️ Research code under active development. Final paper / arXiv preprint coming May 2026.

## Key contributions

1. **Routability-aware classifier guidance** — first diffusion-based placer to optimize a learned routability score with a tunable inference-time scale
2. **Single-pass Pareto sampling** — generates K placements at K guidance scales in one inference, spanning the (HPWL, routability) trade-off
3. **Cross-PDK transfer** — train on N45, deploy on N7 without retraining

## Repository layout

```
parediff/
├── code/                    Python implementation
│   ├── utils.py             Diffusion schedule, HPWL, overlap penalty
│   ├── dataset.py           LEF/DEF parsers + PyG Data builder
│   ├── gnn_encoder.py       GraphGPS-style netlist encoder
│   ├── diffusion_model.py   PareDiff DDPM + classifier guidance
│   ├── routability_classifier.py  CNN routability predictor
│   ├── train.py             Training entry point
│   ├── sample.py            Pareto-front inference CLI
│   ├── evaluate.py          HPWL / diversity / hypervolume metrics
│   └── test_smoke.py        CPU smoke test (no GPU needed)
├── kaggle/                  Kaggle Free Tier training pipeline
├── paper/                   IEEE TCAD draft (LaTeX)
├── notes/                   Literature review
├── scripts/                 Data download helpers
├── baselines/               Scripts to reproduce TritonMacroPlace etc.
├── data/                    (gitignored) ISPD'15, ASAP7 PDK
└── experiments/             (gitignored) checkpoints, run logs
```

## Quick start

```bash
git clone https://github.com/raipancham2004-svg/parediff.git
cd parediff
conda create -n parediff python=3.10 -y
conda activate parediff
pip install torch torch-geometric diffusers wandb

# CPU smoke test (no GPU needed)
python -m code.test_smoke

# Full training (needs GPU; see kaggle/SETUP_KAGGLE.md for cloud option)
python -m code.train --data data/ispd15 --epochs 200 --use_wandb
```

## Citation

```bibtex
@misc{panrai2026parediff,
  title={PareDiff: Pareto-Aware Diffusion Sampling for Routability-Driven
         Macro Placement with Cross-Technology Transfer},
  author={Panrai and [Advisor]},
  year={2026},
  note={Manuscript in preparation}
}
```

## License

MIT — see [LICENSE](LICENSE)
