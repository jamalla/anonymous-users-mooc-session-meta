# Documentation Guide

## Quick Start for Junior Engineers

Welcome! This project tackles **cold-start recommendation** in MOOCs (Massive Open Online Courses) using **meta-learning**. If you're new to this field, start here.

---

## What Problem Are We Solving?

**The Cold-Start Problem**: When a new student joins a MOOC platform like XuetangX, we have NO historical data about them. How can we recommend relevant courses?

**Traditional approaches** need hundreds of user interactions to make good recommendations. We want to do it with just **5 examples** from a new user.

---

## Key Concepts Explained Simply

### 1. What is an "Episode"?

> **One episode = One user's complete learning journey**

Think of an episode as a snapshot of a single user's activity on the platform:

```
Episode for User_ABC:
├── User ID: User_ABC
├── Support Set (K=5 pairs): The user's FIRST 5 course transitions
│   ├── [Python101] → DataScience101
│   ├── [Python101, DataScience101] → Statistics201
│   ├── [Python101, DataScience101, Statistics201] → ML301
│   ├── [WebDev101] → JavaScript201           ← New session started
│   └── [WebDev101, JavaScript201] → React301
│
└── Query Set (Q=10 pairs): The user's NEXT 10 course transitions
    ├── [WebDev101, JavaScript201, React301] → Node401
    ├── ... (9 more pairs from the user's FUTURE activity)
```

**Key insight**: Each episode contains multiple sessions from ONE user. Sessions are learning "bursts" (e.g., a weekend study session). A user might have 3-5 sessions total.

### 2. What is MAML?

**MAML = Model-Agnostic Meta-Learning** = "Learning to Learn"

Instead of training one model for all users, MAML finds a **starting point** (θ) that can quickly adapt to any new user:

```
Traditional ML:        Train once → Same predictions for everyone
MAML:                  Train θ → Adapt θ→φ per user → Personalized predictions
```

### 3. What is a "Pair"?

A pair is a training example: `(prefix sequence, next course to predict)`

```
User watches: Python101 → DataScience101 → Statistics201

This creates pairs:
  Pair 1: [Python101] → predict DataScience101
  Pair 2: [Python101, DataScience101] → predict Statistics201
```

---

## Project Results Summary

### Final Results (from results/*.json)

| Model | Test Acc@1 | vs Baseline |
|-------|------------|-------------|
| GRU4Rec (base model architecture) | 33.73% | N/A |
| Vanilla MAML Zero-shot | 23.50% | N/A |
| **Vanilla MAML Few-shot (K=5)** | **30.52%** | **BASELINE** |
| Residual Warm-Start MAML (Contribution 1) | **34.95%** | **+4.43 pp** |
| Recency-Weighted MAML (Contribution 2) | **34.35%** | **+3.83 pp** |

**Important**: The baseline for our contributions is **Vanilla MAML (30.52%)**, NOT GRU4Rec. GRU4Rec is just the neural network architecture we use inside MAML.

---

## Document Overview

| Document | Purpose | Audience |
|----------|---------|----------|
| [project_reference.md](project_reference.md) | Complete pipeline reference (Notebooks 00-09) | All levels |
| [gru4rec_vs_maml.md](gru4rec_vs_maml.md) | Explains difference between base model and meta-learning | Beginners |
| [architecture_table.md](architecture_table.md) | Detailed algorithm steps with examples | Intermediate |
| [RESEARCH_CONTRIBUTIONS.md](RESEARCH_CONTRIBUTIONS.md) | Academic contributions and publication strategy | Researchers |
| [meta_learning_architecture_justification.md](meta_learning_architecture_justification.md) | PhD-level architecture decision justification | Advanced |
| [paper_matrix_table.md](paper_matrix_table.md) | Literature review of 40+ meta-learning papers | Researchers |

---

## Pipeline Overview

```
00: Bootstrap → 01: Ingest → 02: Sessionize → 03: Vocab/Pairs → 04: User Split
                                                                     ↓
              09: Recency ← 08: Residual ← 07: MAML ← 06: Baselines ← 05: Episodes
```

### What Each Notebook Does:

| Notebook | What it does | Output |
|----------|--------------|--------|
| 00 | Sets up paths and conventions | Helper functions |
| 01 | Loads raw XuetangX data | `data/interim/xuetangx.duckdb` |
| 02 | Groups events into user sessions | Sessionized data |
| 03 | Creates course vocabulary and training pairs | `vocab/course2id.json`, `pairs/*.parquet` |
| 04 | Splits users into train/val/test (no overlap!) | User splits |
| 05 | Creates episodes for meta-learning | `episodes/*.parquet` |
| 06 | Evaluates baseline models (Random, KNN, GRU, etc.) | `results/baselines_K5_Q10.json` |
| 07 | Trains Vanilla MAML | `results/maml_K5_Q10.json` |
| 08 | **Contribution 1**: Residual Warm-Start MAML | `results/warmstart_residual_maml_K5_Q10.json` |
| 09 | **Contribution 2**: Recency-Weighted MAML | `results/warmstart_recency_maml_K5_Q10.json` |

---

## Understanding the Data Flow

```
Raw Logs → Sessions → Pairs → Episodes → Meta-Learning
         (per user)  (prefix→label)  (support+query)
```

### Example Data Flow for One User:

```
1. RAW LOGS (from XuetangX platform):
   user_id    | course_id | timestamp
   -----------|-----------|-------------------
   user_ABC   | Python101 | 2024-01-10 09:00
   user_ABC   | DataSci   | 2024-01-10 09:30
   user_ABC   | Stats201  | 2024-01-10 10:00
   user_ABC   | WebDev101 | 2024-01-15 14:00  ← 5+ day gap = new session
   user_ABC   | JS201     | 2024-01-15 14:30

2. SESSIONIZED:
   Session 1: [Python101, DataSci, Stats201]    (Jan 10)
   Session 2: [WebDev101, JS201]                (Jan 15)

3. PAIRS (training examples):
   Session 1 pairs:
     [Python101] → DataSci
     [Python101, DataSci] → Stats201
   Session 2 pairs:
     [WebDev101] → JS201

4. EPISODE (for meta-learning):
   Support Set (K=5): First 5 pairs (for adaptation)
   Query Set (Q=10):  Next 10 pairs (for evaluation)

   Important: support_max_timestamp < query_min_timestamp
              (No future leakage!)
```

---

## Our Research Contributions

### Contribution 1: Residual Warm-Start MAML (Notebook 08)

**Problem**: Vanilla MAML starts from random initialization, which is suboptimal.

**Solution**: Start from pre-trained GRU weights (FROZEN) and only learn a **residual delta**:
```
θ_effective = θ_pretrained (frozen) + Δθ (learnable)
```

**Result**: 34.95% Acc@1 (+4.43 pp over baseline)

**Why it works**: Similar to LoRA in LLMs - preserves pre-trained knowledge while allowing task-specific adaptation.

### Contribution 2: Recency-Weighted MAML (Notebook 09)

**Problem**: Standard MAML treats all K support pairs equally, but recent interactions are more predictive.

**Solution**: Weight support pairs by recency using exponential decay:
```python
weights = exp(-λ * (K - position))  # λ = 0.5
# weights ≈ [0.15, 0.17, 0.20, 0.22, 0.25]  (older → newer)
```

**Result**: 34.35% Acc@1 (+3.83 pp over baseline)

**Why it works**: In MOOCs, what you studied yesterday is more relevant to predicting today's activity than what you studied last month.

---

## Glossary

| Term | Definition |
|------|------------|
| **Cold-start** | New user with no historical data |
| **Episode** | One user's complete data (support + query sets) - represents a single user's learning journey across all their sessions |
| **Session** | A continuous learning "burst" within one user's history (e.g., one study day) |
| **Support set** | K=5 pairs used to adapt the model to a user |
| **Query set** | Q=10 pairs used to evaluate adaptation quality |
| **MAML** | Model-Agnostic Meta-Learning - learns good initialization θ |
| **Inner loop** | Adapting θ → φ using support set gradient descent |
| **Outer loop** | Updating θ based on query set performance |
| **FOMAML** | First-Order MAML - ignores second derivatives (faster) |
| **Base model** | The neural network architecture (GRU4Rec in our case) - NOT a baseline! |
| **Baseline** | The method we compare against (Vanilla MAML 30.52%) |

---

## FAQ

**Q: Why is GRU4Rec (33.73%) higher than Vanilla MAML (30.52%)?**

A: GRU4Rec is trained on ALL training data (225K pairs) and doesn't personalize. MAML learns to adapt from just K=5 examples. The zero-shot performance drops (23.50%), but few-shot adaptation recovers most of the gap (30.52%). Our contributions push this even higher (34.95%).

**Q: What's the difference between GRU4Rec and MAML?**

A: Same neural network, different training strategy:
- GRU4Rec: Train once, use same parameters for everyone
- MAML: Find parameters that can quickly adapt to each new user

See [gru4rec_vs_maml.md](gru4rec_vs_maml.md) for detailed explanation.

**Q: Why K=5 and Q=10?**

A: K=5 simulates a realistic cold-start scenario (new user has only ~5 interactions). Q=10 provides enough samples for reliable evaluation metrics.

**Q: Is GRU4Rec a baseline?**

A: **No!** GRU4Rec is the **base model architecture** - the neural network that both vanilla MAML and our contributions use internally. The **baseline for comparing our contributions** is Vanilla MAML Few-shot (30.52%).

---

## Architecture Decision Summary

**Decision**: MAML (Model-Agnostic Meta-Learning)

**Reasoning**:
- Reuses strong GRU architecture
- Conceptually aligned with cold-start personalization
- Rich ablation studies possible (K-shot, steps, layers)
- Interpretable adaptation mechanism
- Strong literature support (8K+ citations)
- PhD defensible

**Rejected**:
- Prototypical Networks: Loses temporal information
- MANNs: Too complex, limited precedent in recommendation

See [meta_learning_architecture_justification.md](meta_learning_architecture_justification.md) for full 30+ page justification.

---

*Last updated: 2026-02-03*
