# PareDiff — Routability-Aware Diffusion for Cross-Technology Macro Placement

**Status:** Active research project. Targeting Q1 publication (IEEE TCAD / TVLSI).
**Owner of architecture + code + paper:** Claude (taking the wheel).
**User responsibilities:** GPU runs, tool installs, journal submission.

## One-line pitch

A conditional diffusion model that generates a **Pareto front of K macro placements** spanning the (HPWL ↔ routability) trade-off in a single sampling pass, with **cross-technology transfer** (train on NanGate 45nm → deploy on ASAP7).

## Two contributions (the novelty)

1. **Pareto-aware sampling:** Routability classifier guidance enables single-pass generation of K placements at different (HPWL, routability) trade-off points. Existing diffusion papers (Berkeley, GraphDiffusion, DiffPlace) give one answer; we give the front.

2. **Cross-technology generalization:** Train on N45 designs, evaluate on N7 (ASAP7). First demonstration that diffusion-based placement generalizes across technology nodes. Critical for industry adoption.

## Why these two together

PD engineers in industry need (a) routability-driven decisions, not just HPWL, and (b) the ability to deploy a learned model across tech nodes without retraining. Combined contribution = practical industry-relevant generative placement.

## Why this exists / Why now

- Google's RL approach (Mirhoseini 2021) is unstable, slow, controversial.
- Analytical placers (AutoDMP, TritonMacroPlace) are great but deterministic — one solution per netlist.
- GNN-based predictors decode greedily.
- **Diffusion is the next paradigm:** stable training (just MSE), diverse outputs, natural multi-objective via guidance.
- Diffusion-for-macro-placement is unclaimed in 2026.

## Project layout

```
GraphDiff/
  CLAUDE.md          ← this file (master plan, always up-to-date)
  notes/             ← lit review notes, design decisions
  baselines/         ← scripts to reproduce TritonMacroPlace / Hier-RTLMP / AutoDMP / Circuit Training
  data/              ← preprocessed ISPD'15, TILOS-AI MacroPlacement designs
  code/              ← model.py, dataset.py, train.py, sample.py, evaluate.py
  experiments/       ← trained checkpoints, run logs, hyperparam sweeps
  figures/           ← all paper plots (matplotlib)
  paper/             ← LaTeX draft + bib
  scripts/           ← utility shell/python scripts
```

## Phased plan

### Phase 1 — Literature & dataset prep (Week 1–2)
- 15-paper lit review → `notes/literature_review.md`
- Identify the precise gap → goes into Section 2 of paper
- Download ISPD'15 contest benchmarks (free, public)
- Download TILOS-AI MacroPlacement designs (Ariane, MemPool, BlackParrot, NVDLA)
- Preprocess into PyTorch Geometric format

### Phase 2 — Baselines (Week 2–3)
- Install OpenROAD (one-time on user's machine)
- Run TritonMacroPlace on all designs → log HPWL, runtime
- Run Hier-RTLMP → log same
- Run AutoDMP (NVIDIA, open-source) → log same
- Run Google Circuit Training (open-source RL) → log same
- All numbers go to `experiments/baseline_results.csv`

### Phase 3 — GraphDiff implementation (Week 3–6)
- `code/dataset.py`: Netlist → graph + macro coord targets
- `code/gnn_encoder.py`: GraphGPS / Graphormer for netlist
- `code/diffusion.py`: DDPM with continuous coord + categorical orient
- `code/guidance.py`: classifier-free guidance for overlap penalty
- `code/train.py`: training loop, mixed precision, wandb logging
- `code/sample.py`: reverse diffusion sampling with constraint projection
- `code/evaluate.py`: HPWL + routability (call OpenROAD detailed router) + diversity

### Phase 4 — Experiments (Week 6–9)
- Train on 5 ISPD designs, test on 2 unseen
- Ablations: no GNN cond, no overlap loss, sampling steps {10, 25, 50, 100}
- Compare against 4 baselines on HPWL, routability, runtime, diversity
- Generate all figures (placement viz, training curves, ablation tables)

### Phase 5 — Paper + submit (Week 9–12)
- LaTeX draft (IEEE TCAD format)
- arXiv preprint first (claim novelty)
- Submit to TCAD (or DAC'27 + parallel TVLSI)
- Iterate on reviewer feedback

## Critical decisions made (locked in)

- **Framework:** PyTorch + PyTorch Geometric + diffusers
- **Logging:** Weights & Biases (free academic)
- **Backbone:** GraphGPS for netlist, Transformer for denoising
- **Diffusion:** DDPM (cosine schedule, 1000 train steps, 50 sample steps)
- **Coord representation:** continuous (x, y) ∈ [0, 1]² normalized; orient as one-hot 8-class
- **Constraint enforcement:** soft (loss term) + projected sampling (clip to legal)
- **Code license:** MIT (release on GitHub for reproducibility — required for Q1)

## Locked-in constraints

- **GPU:** **Kaggle FREE** (P100 16GB, 30 hr/week) — primary
  - College lab RTX as backup if Kaggle has issues
  - User's laptop GTX 1650 = code editing only (4GB insufficient)
- **Deadline:** ~May 25, 2026 (1 month) for college submission
- **Co-author:** College advisor (standard student-work pattern)
- **Dev machine:** User's personal laptop (Lenovo IdeaPad Gaming 3, i5 10th, 16GB)
- **Strategy:** arXiv preprint by May 25 + parallel journal submission;
  real Q1 acceptance happens after college deadline (3-6 mo later)
- **Workflow:**
  1. Code/edit on laptop (or here)
  2. Push to GitHub
  3. Kaggle notebook clones GitHub repo + trains
  4. Checkpoints sync to HuggingFace Hub (so next 12hr session resumes)
  5. wandb logs → user shares URL → I monitor in chat

## REVISED 1-month sprint plan (Cross-PDK PareDiff)

| Week | Dates | Goal |
|------|-------|------|
| 1 | Apr 27 – May 3 | Lit review (DONE), download ISPD'15 + ASAP7 + NanGate45 PDKs, install OpenROAD, run TritonMacroPlace baseline |
| 2 | May 4 – May 10 | PareDiff core: GNN encoder + DDPM + classifier-guided sampling |
| 3 | May 11 – May 17 | Train on N45 designs + transfer eval on ASAP7 + Pareto generation + plots |
| 4 | May 18 – May 25 | Paper draft (LaTeX) + arXiv submit + journal submit |

## Scope cuts for sprint

- 2 baselines (TritonMacroPlace + Berkeley diffusion paper code) — not 4-5
- 3-4 ISPD designs train, 2 ASAP7 designs test
- One ablation: vary guidance scale (HPWL ↔ routability balance)
- Story angles: **(1) Pareto front sampling** + **(2) Cross-PDK transfer**

## Sessions log

| Date | What we did |
|------|-------------|
| 2026-04-27 | Project kicked off; scaffolding created; constraints locked in (1mo, RTX, advisor co-author, personal laptop) |
| 2026-04-27 (cont.) | Lit review (5 prior diffusion papers identified, gap confirmed); pivot to PareDiff (routability+Pareto+cross-PDK); full code skeleton written (utils/dataset/gnn/diffusion/routability/train/sample/evaluate); paper Sections 1+2 drafted; main.tex + refs.bib bootstrapped |
