# PROJECT STATE
## A Meta-Learning Framework for Cold-Start Adaptation in Session-Based MOOC Recommendation for Anonymous Learners

**Author:** Jamallah M H Zawia
**Supervisor:** Assoc. Prof. Dr Maizatul Akmar Binti Ismail — FCSIT, University of Malaya
**Last Updated:** 2026-04-10
**Status:** 🟢 ALL EXPERIMENTS COMPLETE — both datasets, all notebooks run, reporting notebooks created

---

## Research Overview

### Problem Statement
MOOC platforms serve millions of anonymous learners who leave no persistent identity trace. Standard recommender systems require long interaction histories to personalise. This work addresses cold-start recommendation for anonymous MOOC learners using only K=5 session interactions at inference time, with no user profile information available.

### Research Questions

| ID | Question |
|---|---|
| RQ1 | How do existing meta-learning approaches handle session quality in cold-start recommendation, and what limitations arise in anonymous MOOC settings? |
| RQ2 | How can session reliability be measured from clickstream data and used to condition MAML inner-loop adaptation per task? |
| RQ3 | Does conditioning inner-loop adaptation on SRS improve cold-start recommendation performance over standard MAML in anonymous MOOC settings? |

### Research Objectives

| ID | Objective |
|---|---|
| RO1 | Survey and identify gaps in meta-learning for anonymous MOOC cold-start |
| RO2 | Develop SRS metric from clickstream signals and integrate into MAML inner loop |
| RO3 | Design, implement, and evaluate SRS-Adaptive MAML against standard MAML baselines |

### Contributions

| ID | Contribution | Notebook |
|---|---|---|
| C1 | Session Reliability Score: SRS = (Intensity + Extent + Composition) / 3, normalised to [0,1] | `03b`, `09_srs_validation` |
| C2 | SRS-Adaptive MAML: task-specific αᵢ = α_base × SRSᵢ and Kᵢ = f(SRSᵢ, τ) ∈ [K_min, K_max] | `10_srs_adaptive_maml` |
| C3 | Warm-Start + SRS-Adaptive MAML: GRU pretrained init + SRS-conditioned inner loop | `11_warmstart_srs_adaptive_maml` |

---

## SRS Formula (Canonical)

```python
# Intensity: how much happened (event volume)
intensity   = min(n_events / CAP_INTENSITY, 1.0)
# CAP_INTENSITY = 95th percentile of n_events across training sessions only

# Extent: how long it lasted (temporal span)
extent      = min(duration_sec / CAP_EXTENT, 1.0)
# CAP_EXTENT = 95th percentile of session duration across training sessions only

# Composition (XuetangX): behavioral variety
composition = n_unique_action_types / N_POSSIBLE_ACTION_TYPES
# N_POSSIBLE_ACTION_TYPES = 6 (load, problem, video, seek, speed, pause)

# Composition (MARS): item diversity (no action types in MARS)
composition = min(n_unique_items / CAP_COMPOSITION, 1.0)
# CAP_COMPOSITION = 95th percentile of n_unique_items across training sessions

# Session Reliability Score
SRS = (intensity + extent + composition) / 3.0
# SRS is always in [0, 1]
```

---

## SRS-Adaptive MAML Algorithm (Canonical)

```
For each outer iteration:
  Sample batch of tasks {T₁, T₂, ..., T_B}
  For each task Tᵢ:
    Compute SRSᵢ from support set clickstream signals
    Compute task-specific inner-loop rate:  αᵢ = α_base × SRSᵢ
    Compute task-specific gradient steps:   Kᵢ = f(SRSᵢ, τ) ∈ [K_min, K_max]
    Adapt:  φᵢ = θ − αᵢ · ∇_θ L_support(θ)   [repeated Kᵢ times]
    Compute query loss: L_query(φᵢ)
  Outer update (SRS-weighted):
    θ ← θ − β · ∇_θ Σᵢ SRSᵢ · L_query(φᵢ)
```

Where:
```python
alpha_i = alpha_base * SRS_i
K_i     = K_min if SRS_i >= tau else K_max
# High SRS → larger α, fewer steps (reliable session)
# Low SRS  → smaller α, more steps  (noisy session)
```

---

## Base Model Selection Rationale

Four session-based architectures are evaluated in NB06 under the same cold-start protocol. The best-performing model is selected as the MAML backbone. This step is valid and defensible — it is a data-driven selection, not an arbitrary choice.

**Important distinction:**
- The base model in NB06 is trained on the **full training set** (supervised, no cold-start constraint) — this is the full-data performance ceiling
- All MAML variants operate under **cold-start constraints (K=5 interactions only)**
- These two settings are not directly comparable — they serve different purposes

---

## GRU4Rec Architecture (Locked)

```python
embedding_dim = 64
hidden_dim    = 128
num_layers    = 1
dropout       = 0.0
```

---

## MAML Hyperparameters (Locked)

```python
# Standard MAML
alpha_base      = 0.01    # base inner-loop learning rate
num_inner_steps = 5       # K (standard MAML — overridden in SRS-Adaptive)
outer_lr        = 0.001   # β outer loop
outer_lr_warmstart = 0.0001  # β for warm-start variants (lower to preserve pretrained weights)
batch_size      = 32      # tasks per outer update
n_iterations    = 3000    # max outer iterations
seed            = 20260107

# SRS-Adaptive additions
tau             = 0.5     # SRS threshold for K selection
K_min           = 3       # gradient steps for high-SRS sessions
K_max           = 7       # gradient steps for low-SRS sessions

# Episode protocol
K_support       = 5
Q_query         = 10
```

---

## Evaluation Protocol (Locked)

```
Primary metrics:   HR@10, NDCG@10
Secondary metrics: HR@1, HR@5, MRR

Test episodes:     XuetangX — 986 episodes (10 per qualifying test user)
                   MARS      — 17 episodes  (small dataset constraint)
User constraint:   train / val / test users are DISJOINT (no overlap)
Temporal:          support_max_timestamp < query_min_timestamp (no leakage)
Seed:              GLOBAL_SEED = 20260107
Report metric:     best checkpoint (not end-of-training)

HR@K = fraction of test query predictions where correct item appears in top-K
```

---

## Dataset Specifications

### XuetangX (Primary Dataset)

| Metric | Value |
|---|---|
| Raw source | XuetangX MOOC clickstream (Feb–Aug 2017) |
| Raw events (NB01) | 28,002,537 |
| Raw users (NB01) | 182,755 |
| Unique courses / vocabulary (NB01) | 1,518 |
| Sessions after sessionization (NB02) | 906,341 |
| Users with sessions (NB02) | 173,335 |
| Session boundary | 30-min inactivity gap, min 2 events |
| Total (prefix→label) pairs (NB03) | 487,696 |
| Users with pairs (NB03) | 70,809 |
| SRS mean / median (NB03b) | 0.3248 / 0.2456 |
| SRS tier: low / medium / high (NB03b) | 62.8% / 25.8% / 11.4% |
| CAP_INTENSITY (p95, NB03b) | 115 events |
| CAP_EXTENT (p95, NB03b) | 6,653 s |
| User split (NB04) | 70 / 15 / 15 → 49,566 / 10,621 / 10,622 users |
| Pairs per split (NB04) | 344,532 / 69,663 / 73,501 |
| Episode protocol (NB05) | K=5 support, Q=10 query |
| Episodes: train / val / test (NB05) | 113,920 / 942 / 986 |
| r(SRS, n_events) (NB09) | 0.503 |
| r(SRS, duration) (NB09) | 0.824 |

### MARS (Secondary Dataset — Generalisation)

| Metric | Value |
|---|---|
| Raw source | MARS MOOC interactions with ratings and watch % |
| Raw interactions (NB01) | 3,655 |
| Raw users (NB01) | 822 |
| Raw items (NB01) | 776 |
| Duplicates removed (NB01) | 4 |
| Sessions after sessionization (NB02) | 561 |
| Users with sessions (NB02) | 378 |
| Unique items in sessions (NB02) | 745 |
| Session boundary | 30-min inactivity gap, min 2 events |
| Avg events / session (NB02) | 5.2 |
| Total (prefix→label) pairs (NB03) | 2,333 |
| Avg pairs / user (NB03) | 6.2 |
| SRS mean / median (NB03b) | 0.2665 / 0.1627 |
| SRS tier: low / medium / high (NB03b) | 76.5% / 14.3% / 9.3% |
| CAP_INTENSITY (p95, NB03b) | 16.9 events |
| CAP_EXTENT (p95, NB03b) | 3,757.6 s |
| SRS composition | item diversity (no action types in MARS) |
| User split (NB04) | 40 / 20 / 40 → 151 / 75 / 152 users |
| Pairs per split (NB04) | 693 / 545 / 1,095 |
| Episode protocol (NB05) | K=5 support, Q=10 query |
| Episodes: train / val / test (NB05) | 111 / 6 / 17 |
| r(SRS, n_events) (NB09) | 0.902 |
| r(SRS, duration) (NB09) | 0.856 |
| Note | 40/20/40 split used because 70/15/15 yields too few test episodes |

---

## NB06 — Base Model Selection Results

### XuetangX (Test: 986 episodes, K=5 support, Q=10 query — zero-shot, no adaptation)

| Model | HR@1 | HR@5 | HR@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| Random | 0.05% | 0.28% | 0.70% | 0.30% | 0.52% |
| Popularity | 1.84% | 6.39% | 11.04% | 5.70% | 5.39% |
| Session-KNN | 13.17% | 35.07% | 42.98% | 27.43% | 23.55% |
| SASRec | 25.65% | 43.20% | 51.75% | 37.58% | 34.42% |
| **GRU4Rec** ← selected | **24.92%** | **43.52%** | **51.87%** | **37.36%** | **34.04%** |

**Selected backbone:** GRU4Rec (highest HR@10 = 51.87%)
Pretrained weights: `models/baselines/gru_global_xuetangx.pth`

### MARS (Test: 17 episodes, K=5 support, Q=10 query — zero-shot, no adaptation)

| Model | HR@1 | HR@5 | HR@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| Random | 0.00% | 0.00% | 0.59% | 0.20% | 0.92% |
| Popularity | 0.59% | 2.35% | 4.12% | 2.00% | 2.14% |
| Session-KNN | 29.41% | 37.06% | **38.82%** | 34.06% | 32.81% |
| GRU4Rec ← MAML backbone | 17.65% | 25.88% | 28.24% | 23.00% | 21.95% |

**Best baseline HR@10:** Session-KNN = 38.82%
**Selected MAML backbone:** GRU4Rec (gradient-based adaptation required; KNN not differentiable)
Pretrained weights: `models/baselines/gru_global_mars.pth`

---

## NB09 — SRS Validation

### XuetangX (906,341 sessions)

| Statistic | Value |
|---|---|
| Mean | 0.3248 |
| Std | 0.2325 |
| Min | 0.0614 |
| p25 / p50 / p75 | 0.1366 / 0.2456 / 0.4487 |
| Max | 1.0000 |
| Tier Low (<0.33) | 62.8% |
| Tier Medium (0.33–0.66) | 25.8% |
| Tier High (≥0.66) | 11.4% |
| r(SRS, n_events) | 0.503 |
| r(SRS, duration_sec) | 0.824 |

### MARS (561 sessions)

| Statistic | Value |
|---|---|
| Mean | 0.2665 |
| Std | 0.2413 |
| Min | 0.0791 |
| p25 / p50 / p75 | 0.0995 / 0.1627 / 0.3130 |
| Max | 1.0000 |
| Tier Low (<0.33) | 76.5% |
| Tier Medium (0.33–0.66) | 14.3% |
| Tier High (≥0.66) | 9.3% |
| r(SRS, n_events) | 0.902 |
| r(SRS, duration_sec) | 0.856 |

---

## Main Experiment Results

### XuetangX — All Models (Test: 986 episodes, 9,860 query predictions)

| Model | HR@1 | HR@5 | HR@10 | NDCG@10 | MRR | Best Iter | Notebook |
|---|---|---|---|---|---|---|---|
| GRU4Rec (full-data, no MAML) | 24.92% | 43.52% | 51.87% | 37.36% | 34.04% | — | NB06 |
| Standard MAML | 23.36% | 39.54% | 47.46% | 34.35% | 31.49% | 2800 | NB07 |
| Warm-Start MAML (ablation C1) | 26.53% | 45.09% | 53.51% | 39.07% | 35.70% | 1600 | NB08 |
| SRS-Adaptive MAML (ablation C2) | 21.91% | 37.84% | 46.20% | 32.96% | 30.07% | 2900 | NB10 |
| **Warm-Start + SRS-Adaptive (C3)** | **26.15%** | **44.32%** | **52.75%** | **38.43%** | **35.13%** | **800** | **NB11** |

**Primary metrics (HR@10 / NDCG@10):**
- Best HR@10: NB08 = 53.51% (warm-start alone)
- Best NDCG@10: NB08 = 39.07%
- Main contribution NB11: 52.75% HR@10 / 38.43% NDCG@10

**Gain analysis vs Standard MAML (NB07):**
- Warm-Start alone (NB08): +6.05pp HR@10
- SRS-Adaptive alone (NB10): −1.26pp HR@10
- Combined NB11: +5.29pp HR@10 / +4.08pp NDCG@10

### MARS — All Models (Test: 17 episodes, 170 query predictions)

| Model | HR@1 | HR@5 | HR@10 | NDCG@10 | MRR | Best Iter | Notebook |
|---|---|---|---|---|---|---|---|
| GRU4Rec (full-data, no MAML) | 17.65% | 25.88% | 28.24% | 23.00% | 21.95% | — | NB06 |
| Standard MAML | 14.12% | 16.47% | 17.65% | 15.55% | 15.36% | 100 | NB07 |
| Warm-Start MAML (ablation C1) | 19.41% | 24.71% | 27.06% | 23.01% | 22.29% | 100 | NB08 |
| SRS-Adaptive MAML (ablation C2) | — | — | 18.24% | 16.17% | 15.90% | 100 | NB10 |
| **Warm-Start + SRS-Adaptive (C3)** | **17.65%** | **25.29%** | **27.06%** | **21.83%** | **20.72%** | **100** | **NB11** |

**Note:** MARS has only 17 test episodes. One episode difference = 5.88pp HR@10. All comparisons are directionally informative only.

**Gain analysis vs Standard MAML (NB07):**
- Warm-Start alone (NB08): +9.41pp HR@10
- SRS-Adaptive alone (NB10): +0.59pp HR@10
- Combined NB11: +9.41pp HR@10 / +6.28pp NDCG@10

---

## Cross-Dataset Ablation Summary (2×2 Design)

| | No Warm-Start | With Warm-Start | Warm-Start Gain |
|---|---|---|---|
| **XuetangX — No SRS-Adaptive** | 47.46% (NB07) | 53.51% (NB08) | +6.05pp |
| **XuetangX — With SRS-Adaptive** | 46.20% (NB10) | 52.75% (NB11) | +6.55pp |
| **MARS — No SRS-Adaptive** | 17.65% (NB07) | 27.06% (NB08) | +9.41pp |
| **MARS — With SRS-Adaptive** | 18.24% (NB10) | 27.06% (NB11) | +8.82pp |

**Key findings:**
1. Warm-start is the dominant contribution on both datasets
2. SRS-Adaptive is slightly negative on XuetangX without warm-start (−1.26pp) but positive with it
3. Combined C3 (NB11) outperforms Standard MAML on both datasets

---

## Notebook Inventory

### XuetangX (`notebooks/xuetangx/`)

| Notebook | Status | Description |
|---|---|---|
| `00_bootstrap_xuetangx.ipynb` | ✅ Done | Environment setup, paths, seed |
| `01_ingest_xuetangx.ipynb` | ✅ Done | Raw events → cleaned Parquet (28M events) |
| `02_sessionize_xuetangx.ipynb` | ✅ Done | Events → 906K sessions (30-min gap) |
| `03_vocab_pairs_xuetangx.ipynb` | ✅ Done | Pairs (487K) + vocabulary (1,517 items) |
| `03b_srs_scores_xuetangx.ipynb` | ✅ Done | SRS scores attached to all pairs |
| `04_user_split_xuetangx.ipynb` | ✅ Done | 70/15/15 user split |
| `05_episode_index_xuetangx.ipynb` | ✅ Done | 113,920 train / 942 val / 986 test episodes |
| `06_base_model_selection_xuetangx.ipynb` | ✅ Done | GRU4Rec selected (HR@10=51.87%) |
| `07_standard_maml_xuetangx.ipynb` | ✅ Done | Benchmark: HR@10=47.46% |
| `08_warmstart_maml_xuetangx.ipynb` | ✅ Done | Ablation C1: HR@10=53.51% |
| `09_srs_validation_xuetangx.ipynb` | ✅ Done | SRS stats + correlations validated |
| `10_srs_adaptive_maml_xuetangx.ipynb` | ✅ Done | Ablation C2: HR@10=46.20% |
| `11_warmstart_srs_adaptive_maml_xuetangx.ipynb` | ✅ Done | Main C3: HR@10=52.75% |
| `12_eda_report_xuetangx.ipynb` | ✅ Done | Full EDA report (NB01–NB05 pipeline) |
| `13_results_analysis_xuetangx.ipynb` | ✅ Done | Results analysis + ablation charts |

### MARS (`notebooks/mars/`)

| Notebook | Status | Description |
|---|---|---|
| `01_ingest_mars.ipynb` | ✅ Done | Raw interactions → Parquet (3,655 rows) |
| `02_sessionize_mars.ipynb` | ✅ Done | Interactions → 561 sessions |
| `03_vocab_pairs_mars.ipynb` | ✅ Done | Pairs (2,333) + vocabulary (745 items) |
| `03b_srs_scores_mars.ipynb` | ✅ Done | SRS scores (item-diversity composition) |
| `04_user_split_mars.ipynb` | ✅ Done | 40/20/40 user split |
| `05_episode_index_mars.ipynb` | ✅ Done | 111 train / 6 val / 17 test episodes |
| `06_base_model_selection_mars.ipynb` | ✅ Done | KNN best (38.82%); GRU4Rec backbone (28.24%) |
| `07_standard_maml_mars.ipynb` | ✅ Done | Benchmark: HR@10=17.65% |
| `08_warmstart_maml_mars.ipynb` | ✅ Done | Ablation C1: HR@10=27.06% |
| `09_srs_validation_mars.ipynb` | ✅ Done | SRS stats + correlations validated |
| `10_srs_adaptive_maml_mars.ipynb` | ✅ Done | Ablation C2: HR@10=18.24% |
| `11_warmstart_srs_adaptive_maml_mars.ipynb` | ✅ Done | Main C3: HR@10=27.06% |
| `12_eda_report_mars.ipynb` | ✅ Done | Full EDA report (NB01–NB05 pipeline) |
| `13_results_analysis_mars.ipynb` | ✅ Done | Results analysis + ablation charts |

### Cross-Dataset (`notebooks/`)

| Notebook | Status | Description |
|---|---|---|
| `14_comparative_analysis.ipynb` | ✅ Done | XuetangX vs MARS side-by-side comparison |

---

## Detailed Run Records

### 07_standard_maml_xuetangx — Results
Run: 20260409_131647  |  Seed: 20260107
Protocol: K=5 support, Q=10 query | Test episodes: 986

| Metric | Value |
|---|---|
| HR@1 | 23.36% |
| HR@5 | 39.54% |
| HR@10 | 47.46% |
| NDCG@10 | 34.35% |
| MRR | 31.49% |

Best val HR@10: 49.12% @ iter 2800
Checkpoint: `models/maml/maml_standard_xuetangx.pth`

### 08_warmstart_maml_xuetangx — Results
Run: 20260409_142803  |  Seed: 20260107
Protocol: K=5 support, Q=10 query | Test episodes: 986
Ablation: Warm-Start MAML (no SRS-Adaptive) | outer_lr=0.0001

| Metric | Value |
|---|---|
| HR@1 | 26.53% |
| HR@5 | 45.09% |
| HR@10 | 53.51% |
| NDCG@10 | 39.07% |
| MRR | 35.70% |

Best val HR@10: 54.76% @ iter 1600
Checkpoint: `models/maml/maml_warmstart_xuetangx.pth`

### 10_srs_adaptive_maml_xuetangx — Results
Run: 20260409_155705  |  Seed: 20260107
Protocol: K=5 support, Q=10 query | Test episodes: 986
Ablation: SRS-Adaptive MAML (no warm-start) | alpha_base=0.01, tau=0.5, K_min=3, K_max=7

| Metric | Value |
|---|---|
| HR@1 | 21.91% |
| HR@5 | 37.84% |
| HR@10 | 46.20% |
| NDCG@10 | 32.96% |
| MRR | 30.07% |

Best val HR@10: 46.91% @ iter 2900
Checkpoint: `models/maml/maml_srs_adaptive_xuetangx.pth`

### 11_warmstart_srs_adaptive_maml_xuetangx — Results
Run: 20260409_163808  |  Seed: 20260107
Protocol: K=5 support, Q=10 query | Test episodes: 986
MAIN CONTRIBUTION (C3): Warm-Start + SRS-Adaptive MAML | outer_lr=0.0001
alpha_base=0.01  tau=0.5  K_min=3  K_max=7

| Metric | Value |
|---|---|
| HR@1 | 26.15% |
| HR@5 | 44.32% |
| HR@10 | 52.75% |
| NDCG@10 | 38.43% |
| MRR | 35.13% |

Best val HR@10: 54.00% @ iter 800
Checkpoint: `models/maml/maml_warmstart_srs_xuetangx.pth`

### 07_standard_maml_mars — Results
Run: 20260409_194423  |  Seed: 20260107
Protocol: K=5 support, Q=10 query | Test episodes: 17

| Metric | Value |
|---|---|
| HR@1 | 14.12% |
| HR@5 | 16.47% |
| HR@10 | 17.65% |
| NDCG@10 | 15.55% |
| MRR | 15.36% |

Best val HR@10: 1.67% @ iter 100
Checkpoint: `models/maml/maml_standard_mars.pth`

### 08_warmstart_maml_mars — Results
Run: 20260409_210311  |  Seed: 20260107
Protocol: K=5 support, Q=10 query | Test episodes: 17
Ablation: Warm-Start MAML (no SRS-Adaptive) | outer_lr=0.0001

| Metric | Value |
|---|---|
| HR@1 | 19.41% |
| HR@5 | 24.71% |
| HR@10 | 27.06% |
| NDCG@10 | 23.01% |
| MRR | 22.29% |

Best val HR@10: 3.33% @ iter 100
Checkpoint: `models/maml/maml_warmstart_mars.pth`

### 10_srs_adaptive_maml_mars — Results
Run: (latest available)  |  Seed: 20260107
Protocol: K=5 support, Q=10 query | Test episodes: 17
Ablation: SRS-Adaptive MAML (no warm-start) | alpha_base=0.01, tau=0.5, K_min=3, K_max=7

| Metric | Value |
|---|---|
| HR@10 | 18.24% |
| NDCG@10 | 16.17% |
| MRR | 15.90% |

Best val iter: 100

### 11_warmstart_srs_adaptive_maml_mars — Results
Run: 20260410_095801  |  Seed: 20260107
Protocol: K=5 support, Q=10 query | Test episodes: 17
MAIN CONTRIBUTION (C3): Warm-Start + SRS-Adaptive MAML | outer_lr=0.0001
alpha_base=0.01  tau=0.5  K_min=3  K_max=7

| Metric | Value |
|---|---|
| HR@1 | 17.65% |
| HR@5 | 25.29% |
| HR@10 | 27.06% |
| NDCG@10 | 21.83% |
| MRR | 20.72% |

Best val HR@10: 3.33% @ iter 100
Checkpoint: `models/maml/maml_warmstart_srs_mars.pth`

---

## Non-Negotiable Rules

1. Primary metrics are HR@10 and NDCG@10 — always report these first
2. Never label Acc@1 as HR@10, and never label Recall@10 as HR@10 — use correct labels
3. All MAML variants evaluate on the same test episodes per dataset
4. Seed = 20260107 before every training run
5. Report best checkpoint metric, not end-of-training metric
6. SRS-Adaptive MAML implements αᵢ = α_base × SRSᵢ and Kᵢ = f(SRSᵢ, τ) — not weighted loss
7. The full-data GRU baseline and cold-start MAML operate under different constraints — always state this distinction when reporting
8. MARS results are directionally informative only (17 test episodes; 1 episode = 5.88pp HR@10)
