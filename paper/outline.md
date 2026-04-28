# PareDiff — Paper Outline

**Working title:** PareDiff: Pareto-Aware Diffusion Sampling for Routability-Driven Macro Placement with Cross-Technology Transfer

**Target venue:** IEEE TCAD (primary), IEEE TVLSI (backup), DAC'27 / ICCAD'26 (conference parallel)

**Length target:** 12–14 pages double-column IEEE format

---

## Abstract (~150 words)

VLSI macro placement is a multi-objective optimization problem that trades wirelength against routability, congestion, and power. Existing learning-based methods — including recent diffusion-model approaches — produce a *single* deterministic placement per netlist, optimized primarily for half-perimeter wirelength (HPWL). We argue that this is a poor fit for industrial physical design, where engineers need (i) routability-aware decisions, not HPWL alone, and (ii) a *front* of options spanning the design trade-off space. We propose **PareDiff**, a conditional diffusion model for macro placement that, in a single sampling pass, generates K placements spanning the (HPWL, routability) Pareto front via routability-classifier guidance. We further demonstrate that PareDiff transfers across technology nodes: a model trained on NanGate 45nm produces high-quality placements on ASAP7 designs without retraining. Across ISPD'15 and TILOS-AI MacroPlacement benchmarks, PareDiff matches the Berkeley diffusion baseline on HPWL while reducing post-route DRC by X% and producing Y× more diverse solutions.

---

## 1. Introduction

- VLSI placement is the gateway to all downstream PD QoR
- Multi-objective by nature: HPWL, routability, IR/EM, timing
- Existing learning-based methods (RL: Mirhoseini 2021; analytical: AutoDMP; diffusion: Lee 2024) treat it as single-objective (HPWL)
- Real PD engineers care about routability, not HPWL alone (cite recent industry surveys)
- Diffusion models offer two underexploited advantages: (a) natural multi-modal sampling = diversity, (b) classifier-guidance = trade-off control at inference time
- Our contributions:
  1. PareDiff — first Pareto-aware diffusion sampler for macro placement
  2. Routability classifier guidance for trade-off control
  3. First demonstration of cross-technology (N45 → N7) transfer for diffusion-based placement
  4. Open-source code + reproducibility benchmark

---

## 2. Related Work

### 2.1 Macro Placement: Heuristic and Analytical
- Simulated annealing (TritonMacroPlace, Hier-RTLMP)
- Analytical (NTUplace, ePlace, AutoDMP)
- Bayesian optimization (WireMask-BBO)

### 2.2 Learning-Based Placement
- Reinforcement learning (Mirhoseini 2021; Cheng 2024 critique)
- GNN-based predictors (Princeton, MIT, Tsinghua works)
- Diffusion models — **the relevant prior art:**
  - Lee, Nguyen, Abbeel et al. 2024 — first diffusion-for-placement
  - Fang et al. 2025 — graph-conditioned diffusion (GraphDiffusion)
  - Trung & Hy 2025 — DiffPlace simultaneous placement
  - Yoon et al. 2025 — geometric/equivariant diffusion (LBR)
  - Ghazaryan 2025 — diffusion for macro optimization
- **What's missing:** All optimize HPWL, none address routability or Pareto sampling

### 2.3 Diffusion Models and Guidance
- DDPM (Ho 2020), DDIM (Song 2021)
- Classifier-free guidance (Ho & Salimans 2021)
- Graph diffusion: DiGress, GDSS
- Constraint-guided generation in adjacent domains (molecular, point cloud)

### 2.4 Cross-Technology Transfer in EDA
- Transfer learning for DRC prediction (CircuitNet)
- Domain adaptation for timing prediction
- **Gap:** No prior cross-PDK work for diffusion-based placement

---

## 3. Background

### 3.1 Problem Formulation
- Inputs: netlist G = (V, E), tile boundary T, halo/keepout constraints C
- Outputs: macro coordinates X = {(x_i, y_i, θ_i) : i ∈ macros}
- Objectives: HPWL(X, G), Routability(X, G), Legality(X, T, C)

### 3.2 Diffusion Models (compressed)
- Forward process: q(x_t | x_0) = N(x_t; α_t x_0, (1-α_t)I)
- Reverse: p_θ(x_{t-1} | x_t)
- Training: ε-prediction MSE
- Sampling: iterative denoising with optional classifier guidance

### 3.3 Classifier Guidance for Multi-Objective
- Standard classifier guidance: shift sampling toward class y with score ∇log p(y | x_t)
- Our extension: continuous guidance vector for (HPWL, routability) trade-off

---

## 4. Method: PareDiff

### 4.1 Architecture Overview
- Netlist G → GNN encoder (GraphGPS) → conditioning vector c(G)
- Coords X_t → Transformer denoising network ε_θ(X_t, t, c(G))
- Routability classifier R_φ(X) → guidance gradient

### 4.2 GNN Encoder
- Node features: cell type one-hot, area, pin count
- Edge features: net hyperedge → clique expansion
- Architecture: 4-layer GraphGPS with global attention

### 4.3 Diffusion Process
- Continuous (x, y) ∈ [0,1]² + discrete θ ∈ {R0, R90, R180, R270, MX, MY, MXR90, MYR90}
- Hybrid: Gaussian for coords, multinomial for orient (per DiGress)
- Cosine noise schedule, 1000 train steps, 50 sample steps (DDIM)

### 4.4 Routability Classifier
- Input: placed layout (rendered as multi-channel image)
- Output: predicted routability score (0 = uncongested, 1 = max overflow)
- Architecture: small ResNet (10M params)
- Trained separately on (placement, OpenROAD-router-output) pairs

### 4.5 Pareto-Front Sampling
- Sample K placements with K different guidance scales s ∈ [s_min, s_max]
- Each scale produces one point on the Pareto front
- Final user/auto-selector picks based on application priority

### 4.6 Cross-Technology Conditioning
- Add tech-embedding to GNN output: e_tech ∈ R^d
- Train with mixed N45 + (small amount of) ASAP7 data
- Evaluate zero-shot on held-out ASAP7 designs

---

## 5. Experiments

### 5.1 Setup
- Datasets: ISPD'15 (12 designs), TILOS-AI (Ariane, MemPool, BlackParrot, NVDLA)
- PDKs: NanGate 45nm (train), ASAP7 (transfer eval)
- Tools: OpenROAD for routability evaluation, OpenLane wrapper
- Hardware: 1× RTX 4090 (college lab)

### 5.2 Metrics
- HPWL (lower better)
- Post-route DRC count (lower better) — **our key metric**
- Routability overflow ratio (lower better)
- Diversity: pairwise placement distance across K samples
- Pareto coverage: hypervolume of sampled front
- Runtime: train + inference

### 5.3 Baselines
- TritonMacroPlace (analytical baseline)
- Hier-RTLMP (state-of-the-art SA)
- Lee et al. 2024 (Berkeley diffusion baseline) — re-run their open-source code
- Random + greedy (sanity check)

### 5.4 Main Results
- Table 1: HPWL + DRC + runtime across all baselines and PareDiff
- Figure 2: Pareto front visualization for 3 representative designs
- Figure 3: Cross-PDK transfer results (N45-trained → ASAP7-eval)

### 5.5 Ablations
- Guidance scale sweep
- Number of sampling steps {10, 25, 50, 100}
- With/without GNN conditioning
- With/without routability classifier (degrades to HPWL-only)

---

## 6. Discussion

### 6.1 Why Pareto > Single-Best
- Industry use case: PD engineer reviews 5 options per tile
- Comparison with multi-restart heuristics (much slower)

### 6.2 Why Cross-PDK Works
- Hypothesis: macro placement is more about topology than tech-specifics
- Evidence: tech-embedding ablation

### 6.3 Limitations
- Macro-only (no std cell joint placement)
- Routability classifier needs OpenROAD calls during training
- ASAP7 is academic PDK; industrial PDKs may differ

---

## 7. Conclusion

- First Pareto-aware diffusion sampler for macro placement
- First cross-technology demonstration
- Code released
- Future work: joint macro+std cell, IR/EM-aware guidance, industrial PDK validation

---

## Author contributions (suggested)

- **Panrai:** problem formulation, implementation, experiments, paper writing
- **[Advisor name]:** supervision, paper editing, technical guidance

---

## Estimated paper structure (page budget for IEEE TCAD)

| Section | Pages |
|---|---|
| Abstract + Intro | 1.5 |
| Related Work | 1.5 |
| Background | 1.0 |
| Method | 3.0 |
| Experiments | 3.5 |
| Discussion | 1.0 |
| Conclusion + Refs | 0.5 |
| **Total** | **12.0** |
