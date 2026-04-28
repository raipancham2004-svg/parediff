# Literature Review — Diffusion for Macro Placement

**Last updated:** 2026-04-27
**Status:** Initial scan; need to read full PDFs of Berkeley + GraphDiffusion papers

## ⚠ CRITICAL FINDING

Diffusion-for-macro-placement is **NOT** unclaimed. At least 5 papers exist from 2024-2025.
The "first diffusion approach" novelty claim is dead.
**Refined novelty needed** — see end of this doc.

---

## Prior Art — Diffusion-Based Placement

### 1. Lee et al. — "Chip Placement with Diffusion Models" (2024) [arXiv:2407.12282]
- **Affiliation:** UC Berkeley (Pieter Abbeel's group)
- **Code:** github.com/vint-1/chipdiffusion
- **Method:** "Backwards universal guidance" with learnable Lagrange multipliers
- **Guidance objective:** `φ_hpwl(x) + λ·φ_legality(x)` — HPWL + legality ONLY
- **Their own words:** *"The strong performance of our method on the congestion metric, despite NOT explicitly optimizing for it..."*
- **Single placement per inference** — no Pareto / multi-output
- **No cross-technology evaluation** — only ICCAD04 + ISPD2005, same tech
- **Limitations they list:** synthetic data lacks multimodal edge distributions; doesn't capture full mixed-size; recommends DDPO (diffusion+RL) as future work
- **GAPS WE EXPLOIT:** routability not in objective, no Pareto, no cross-PDK

### 2. Fang et al. — "GraphDiffusion: A Graph-conditioned Diffusion Model for Chip Placement" (2025)
- **Key claim:** Graph conditioning via GNN encoder
- **Status:** Need to read full paper
- **Note:** They've taken the name we wanted

### 3. Trung & Hy — "DiffPlace: A Conditional Diffusion Framework for Simultaneous VLSI Placement" (2025) [arXiv:2510.15897]
- **Key claim:** Reformulating placement as conditional denoising; "routability-first" perspective
- **Method:** Decoupled guidance mechanism — energy-based conditioning + manifold gradient injection
- **Single placement per inference** — no Pareto generation
- **No cross-PDK evaluation** — same-technology benchmarks only
- **DIFFERENTIATION FROM US:** They mention routability but use energy-based conditioning (compile-time);
  WE use classifier-guidance with tunable scale (inference-time, generates Pareto front)

### 4. Yoon, Jeon, Kang — "Late Breaking Results: A Geometric Diffusion Model for Macro Placement Generation" (2025)
- **Key claim:** Geometric (E(2)-equivariant?) diffusion; captures wirelength relationships
- **Status:** LBR = late breaking result, short paper, less rigorous baseline

### 5. Ghazaryan — "Macro Placement Optimization Using Diffusion Models" (2025)
- **Status:** Single-author; need to read

### 6. Pujari & Bali — "Hybrid AI–EDA Approaches for VLSI Floorplanning: SOTA and Future Directions" (2025) (SURVEY)
- Survey paper — useful for related-work section
- Calls out diffusion as emerging methodology
- We should cite this and POSITION ourselves within the future directions they list

---

## Prior Art — RL/Heuristic/Analytical Macro Placement (baselines)

### 7. Mirhoseini et al. — Nature 2021 (Google's RL paper)
- The famous one; also the controversial one
- Cheng et al. 2024 critique (Stanford/UCSD) — "Rethinking learned chip placement"

### 8. Hier-RTLMP — UCSD (Macro placement for very large designs)
- Open-source; one of our baselines

### 9. AutoDMP — NVIDIA (analytical placer with auto-tuning)
- Open-source; differentiable formulation

### 10. TritonMacroPlace — UCSD (open-source SA-based)
- Open-source; ships with OpenROAD

### 11. WireMask-BBO — Tsinghua (Bayesian Optimization based)
- Hou et al. 2024

---

## Prior Art — Graph Diffusion (Methodology references)

### 12. DiGress (Vignac et al., ICLR 2023)
- Discrete denoising diffusion for graph generation
- Foundational graph diffusion paper

### 13. GDSS (Jo et al., ICML 2022)
- Score-based graph generation
- Joint diffusion of node features + adjacency

### 14. DDPM (Ho et al., NeurIPS 2020)
- Foundational diffusion model paper — methodology citation

### 15. Classifier-Free Guidance (Ho & Salimans, NeurIPS 2021)
- Methodology reference for our guidance approach

---

### 7b. DALI-PD — Wu & Chhabria (2025) [arXiv:2507.10606]
- **NOT a competitor** — it generates SYNTHETIC heatmaps for ML training (orthogonal)
- Useful for us: cite as evidence that diffusion-for-EDA datasets are emerging, but doesn't compete
- Could even use their generated dataset to AUGMENT our training set

---

## CONFIRMED NOVELTY (after PDF reads)

After reading Berkeley + DiffPlace abstracts/HTML in detail (Apr 27):

| Capability | Berkeley 2024 | DiffPlace 2025 | GraphDiffusion 2025 | **PareDiff (us)** |
|---|---|---|---|---|
| Diffusion-based placement | ✓ | ✓ | ✓ | ✓ |
| Routability in objective | ✗ | △ (energy-based) | ✗ | **✓ (classifier guidance, tunable)** |
| Multi-output Pareto sampling | ✗ | ✗ | ✗ | **✓ (single inference pass)** |
| Cross-PDK / cross-tech transfer | ✗ | ✗ | ✗ | **✓ (N45 → ASAP7)** |
| Inference-time trade-off control | ✗ | ✗ | ✗ | **✓ (guidance scale knob)** |

**Bottom line:** 3 clean novelty contributions over the closest competitor. Strong Q1-grade paper.

---

## REFINED NOVELTY PITCH

The "first diffusion for macro placement" claim is taken. The remaining open real estate:

### Option A — **Routability-Aware Diffusion with Pareto-Front Sampling** (RECOMMENDED)
- **The gap:** Existing diffusion-for-placement papers optimize HPWL primarily. Routability (post-route DRC count, congestion overflow) is a downstream concern. PD engineers care more about routability than HPWL.
- **Our contribution:**
  1. Routability-aware classifier guidance during diffusion sampling
  2. Multi-objective Pareto sampling: in a single inference, generate K placements spanning the (HPWL, routability) trade-off curve
  3. Demonstrate that PD engineers can pick from K options instead of being stuck with one
- **Why publishable:** No prior diffusion-placement paper does multi-objective Pareto generation. This is a meaningful contribution beyond "we use diffusion."
- **Reviewer-proof story:** "Existing methods give you ONE answer. We give you the FRONT."

### Option B — Constraint-Projected Sampling with Theoretical Guarantees
- **The gap:** Existing work uses soft constraints (penalty in loss). We could be the first to formally project the diffusion sample onto the legal placement manifold and prove distribution preservation.
- **Risk:** Math-heavy; depends on whether you have time for formal proofs. Skip if no.

### Option C — Cross-PDK / Cross-Technology Transfer
- **The gap:** All existing work trains + tests on similar designs. Train on N45 NanGate → deploy on N28? Architectural insight: which features generalize, which don't.
- **Risk:** Need access to multiple PDKs (might be hard for an intern).

### Option D — Discrete Diffusion for Orientation + Continuous for Coordinates
- **The gap:** Most prior work treats orientation as continuous (rotated angle) or 2-class (R0/R180). We treat it as 8-class discrete + use mixed-modal diffusion.
- **Risk:** Methodologically novel but maybe too narrow.

### Recommendation: Option A

Combines the practical PD relevance (routability, what engineers actually care about) with diffusion's natural advantage (diversity → Pareto sampling).

Working title:
> **"PareDiff: Pareto-Aware Diffusion Sampling for Routability-Driven Macro Placement"**

---

## TODO before next session

- [ ] Download arXiv PDFs of Lee et al. (Berkeley), Fang et al. (GraphDiffusion), Trung & Hy (DiffPlace), Yoon et al.
- [ ] Read each paper's "Future Work" section — confirms gaps still open
- [ ] Confirm Berkeley paper does NOT do routability-aware sampling (they don't, based on summary)
- [ ] Confirm none do Pareto-front sampling (likely none — this is the diff's diversity advantage rarely exploited)
- [ ] Run `git clone` on Berkeley's repo so we can directly compare to their numbers
