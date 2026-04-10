# CLAUDE.md — Experiment Run Plan
## A Meta-Learning Framework for Cold-Start Adaptation in Session-Based MOOC Recommendation for Anonymous Learners

**Read this before touching any notebook or writing any code.**
**No previous results exist. Everything starts from scratch.**

---

## Repository Structure (Target State)

```
notebooks/
├── old/                               ← ALL current notebooks moved here untouched
│   ├── 00_bootstrap_and_conventions.ipynb
│   ├── 01_ingest_xuetangx.ipynb
│   ├── 02_sessionize_xuetangx.ipynb
│   ├── 03_build_vocab_pairs_xuetangx.ipynb
│   ├── 03b_build_pairs_with_events_xuetangx.ipynb
│   ├── 04_user_split_xuetangx.ipynb
│   ├── 05_episode_index_xuetangx.ipynb
│   ├── 06_base_model_selection_xuetangx.ipynb
│   ├── 07_maml_xuetangx.ipynb
│   ├── 08_warmstart_maml_xuetangx.ipynb
│   ├── 09_recency_weighted_maml_xuetangx.ipynb
│   ├── 10_reliability_validation.ipynb
│   ├── 11_reliability_weighted_maml_xuetangx.ipynb
│   ├── 12_warmstart_reliability_maml_xuetangx.ipynb
│   ├── xuetangx_deprecated_experiments/
│   └── MARS/
│
├── xuetangx/                          ← Dataset 1 — clean notebooks
│   ├── 00_bootstrap.ipynb
│   ├── 01_ingest.ipynb
│   ├── 02_sessionize.ipynb
│   ├── 03_vocab_pairs.ipynb
│   ├── 03b_srs_scores.ipynb
│   ├── 04_user_split.ipynb
│   ├── 05_episode_index.ipynb
│   ├── 06_base_model_selection.ipynb
│   ├── 07_standard_maml.ipynb
│   ├── 08_warmstart_maml.ipynb
│   ├── 09_srs_validation.ipynb
│   ├── 10_srs_adaptive_maml.ipynb
│   └── 11_warmstart_srs_adaptive_maml.ipynb
│
└── mars/                              ← Dataset 2 — same structure
    ├── 01_ingest.ipynb
    ├── 02_sessionize.ipynb
    ├── 03_vocab_pairs.ipynb
    ├── 03b_srs_scores.ipynb
    ├── 04_user_split.ipynb
    ├── 05_episode_index.ipynb
    ├── 06_base_model_selection.ipynb
    ├── 07_standard_maml.ipynb
    ├── 08_warmstart_maml.ipynb
    ├── 09_srs_validation.ipynb
    ├── 10_srs_adaptive_maml.ipynb
    └── 11_warmstart_srs_adaptive_maml.ipynb
```

---

## Step 0 — Restructure First (Do Before Anything Else)

```bash
# 1. Create the three folders
mkdir notebooks/old
mkdir notebooks/xuetangx
mkdir notebooks/mars

# 2. Move everything currently in notebooks/ into old/
mv notebooks/*.ipynb notebooks/old/
mv notebooks/xuetangx_deprecated_experiments notebooks/old/
mv notebooks/MARS notebooks/old/

# 3. notebooks/old/ now contains everything — do not touch it again
```

Then create fresh notebooks in `xuetangx/` and `mars/` as described below.

---

## What Each Notebook Does (Both Datasets Follow Identical Structure)

The `xuetangx/` and `mars/` folders follow the same notebook sequence.  
The code logic is identical — only the dataset paths and dataset-specific configs differ.

---

### NB00 — Bootstrap (`xuetangx/00_bootstrap.ipynb`) — XuetangX only

**Copy from:** `old/00_bootstrap_and_conventions.ipynb`  
**Clear all outputs.** No code changes needed.  
**Purpose:** Sets up repo root, paths, logger, and seed.

```python
GLOBAL_SEED = 20260107
```

---

### NB01 — Ingest (`01_ingest.ipynb`)

**Copy from:** `old/01_ingest_xuetangx.ipynb` → `xuetangx/01_ingest.ipynb`  
**Copy from:** `old/MARS/01_ingest_mars.ipynb` → `mars/01_ingest.ipynb`  
**Clear all outputs.**  
**Purpose:** Raw source data → cleaned Parquet events file.

---

### NB02 — Sessionize (`02_sessionize.ipynb`)

**Copy from:** `old/02_sessionize_xuetangx.ipynb` → `xuetangx/02_sessionize.ipynb`  
**Copy from:** `old/MARS/02_sessionize_mars.ipynb` → `mars/02_sessionize.ipynb`  
**Clear all outputs.**  
**Purpose:** Events → sessions using 30-min inactivity threshold, min 2 interactions per session.

---

### NB03 — Vocabulary and Pairs (`03_vocab_pairs.ipynb`)

**Copy from:** `old/03_build_vocab_pairs_xuetangx.ipynb` → `xuetangx/03_vocab_pairs.ipynb`  
**Copy from:** `old/MARS/03_build_vocab_and_pairs_mars.ipynb` → `mars/03_vocab_pairs.ipynb`  
**Clear all outputs.**  
**Purpose:** Build course vocabulary + (prefix → next item) pairs for next-item prediction.

---

### NB03b — SRS Scores (`03b_srs_scores.ipynb`)

**Copy from:** `old/03b_build_pairs_with_events_xuetangx.ipynb` → `xuetangx/03b_srs_scores.ipynb`  
**For MARS:** Create fresh `mars/03b_srs_scores.ipynb` using same logic.  
**Clear all outputs.**  
**Purpose:** Computes SRS score per session and attaches to pairs.

**SRS formula (must match exactly):**
```python
# Compute CAPs from training sessions only — no leakage
CAP_INTENSITY = np.percentile(train_sessions['n_events'], 95)
CAP_EXTENT    = np.percentile(train_sessions['duration_sec'], 95)
N_POSSIBLE    = 6  # load, problem, video, seek, speed, pause

def compute_srs(n_events, duration_sec, n_unique_action_types):
    intensity   = min(n_events / CAP_INTENSITY, 1.0)
    extent      = min(duration_sec / CAP_EXTENT, 1.0)
    composition = n_unique_action_types / N_POSSIBLE
    return round((intensity + extent + composition) / 3.0, 6)
```

---

### NB04 — User Split (`04_user_split.ipynb`)

**Copy from:** `old/04_user_split_xuetangx.ipynb` → `xuetangx/04_user_split.ipynb`  
**Copy from:** `old/MARS/04_user_split_mars.ipynb` → `mars/04_user_split.ipynb`  
**Clear all outputs.**  
**Purpose:** Split users into disjoint train / val / test sets (no user appears in two splits).

**XuetangX split ratio:** 70 / 15 / 15  
**MARS:** Use same ratio. Verify test set produces ≥ 50 episodes after NB05 — if not, adjust ratio.

---

### NB05 — Episode Index (`05_episode_index.ipynb`)

**Copy from:** `old/05_episode_index_xuetangx.ipynb` → `xuetangx/05_episode_index.ipynb`  
**Copy from:** `old/MARS/05_episode_index_mars.ipynb` → `mars/05_episode_index.ipynb`  
**Clear all outputs.**  
**Purpose:** Builds episode index (K=5 support pairs, Q=10 query pairs per user).

**Verify after running:**
```python
# XuetangX expected
assert len(episodes_train) == 47357
assert len(episodes_val)   == 341
assert len(episodes_test)  == 313

# MARS: must be ≥ 50 test episodes before proceeding
assert len(episodes_test) >= 50, "MARS test set too small — fix user split"
```

---

### NB06 — Base Model Selection (`06_base_model_selection.ipynb`)

**Copy from:** `old/06_base_model_selection_xuetangx.ipynb` → `xuetangx/06_base_model_selection.ipynb`  
**For MARS:** Copy and update paths → `mars/06_base_model_selection.ipynb`  
**Clear all outputs.**

**Required changes to the code:**
```python
# RENAME these metric labels everywhere in the notebook:
# "Accuracy@1"  →  "HR@1"
# "Recall@5"    →  "HR@5"
# "Recall@10"   →  "HR@10"

# ADD NDCG@10 computation (currently missing):
import math

def ndcg_at_k(ranked_items, true_item, k=10):
    for i, item in enumerate(ranked_items[:k]):
        if item == true_item:
            return 1.0 / math.log2(i + 2)
    return 0.0
```

**Final summary cell must use this format:**
```
========== BASE MODEL SELECTION — [DATASET] ==========
Protocol: K=5 support, Q=10 query | Test: [N] episodes

Model          HR@1    HR@5    HR@10   NDCG@10  MRR
------------------------------------------------------
Random         —       —       —       —        —
Popularity     —       —       —       —        —
Session-KNN    —       —       —       —        —
SASRec         —       —       —       —        —
GRU4Rec        —       —       —       —        —

Selected backbone: [name] (highest HR@10)
Note: Base model trained on full training data — not cold-start constrained.
======================================================
```

**Save pretrained model:**
```python
torch.save(model.state_dict(), f"models/baselines/gru_global_{DATASET}.pth")
```

---

### NB07 — Standard MAML (`07_standard_maml.ipynb`)

**Copy from:** `old/07_maml_xuetangx.ipynb` → `xuetangx/07_standard_maml.ipynb`  
**For MARS:** Copy and update paths → `mars/07_standard_maml.ipynb`  
**Clear all outputs.**

**Required changes:**
- Same metric label fixes as NB06 (HR@1, HR@5, HR@10, NDCG@10)
- Add NDCG@10 to evaluation function (copy from NB11 old version)

**Lock these hyperparameters:**
```python
alpha_base      = 0.01
num_inner_steps = 5
outer_lr        = 0.001
batch_size      = 32
n_iterations    = 3000
seed            = 20260107
```

**Keep ablation cells** (K ∈ {1,3,5,10} and steps ∈ {1,3,5,10}) — they are valid and needed.

---

### NB08 — Warm-Start MAML Ablation (`08_warmstart_maml.ipynb`)

**Copy from:** `old/08_warmstart_maml_xuetangx.ipynb` → `xuetangx/08_warmstart_maml.ipynb`  
**For MARS:** Same copy → `mars/08_warmstart_maml.ipynb`  
**Clear all outputs.**

**Purpose:** Ablation only. Warm-start init WITHOUT SRS-Adaptive inner loop. Shows warm-start alone is insufficient.

**Config changes from NB07:**
```python
outer_lr = 0.0001  # lower to preserve pretrained knowledge
model.load_state_dict(torch.load(f"models/baselines/gru_global_{DATASET}.pth"))
# Inner loop: standard uniform loss — no SRS modifications
```

---

### NB09 — SRS Validation (`09_srs_validation.ipynb`)

**Create fresh** for both `xuetangx/` and `mars/`.  
**Purpose:** Validates SRS formula before using it in NB10/NB11. Empirical justification for C1.

**Must produce:**
```python
# 1. SRS score distribution (histogram)
# 2. Summary stats: mean, std, min, max, 25th/50th/75th percentile
# 3. Tier breakdown
low    = (srs_scores < 0.33).mean()   # % of sessions
medium = ((srs_scores >= 0.33) & (srs_scores < 0.66)).mean()
high   = (srs_scores >= 0.66).mean()
# 4. Component breakdown: Intensity, Extent, Composition separately
# 5. Correlation: SRS vs session length (Pearson r)
```

---

### NB10 — SRS-Adaptive MAML (`10_srs_adaptive_maml.ipynb`) ← NEW CODE

**Create fresh** for both `xuetangx/` and `mars/`.  
**Purpose:** Ablation — tests SRS-Adaptive inner loop WITHOUT warm-start. Implements C2.

**This is the core new code. Implement exactly as follows:**

```python
# ── Config ────────────────────────────────────────────────────
ALPHA_BASE = 0.01
TAU        = 0.5    # SRS threshold for K selection
K_MIN      = 3      # gradient steps for high-SRS (reliable) sessions
K_MAX      = 7      # gradient steps for low-SRS (noisy) sessions
OUTER_LR   = 0.001  # same as standard MAML (no warm-start)
BATCH_SIZE = 32
N_ITER     = 3000
SEED       = 20260107

# ── Task-specific hyperparameter computation ──────────────────
def get_task_hyperparams(srs_i, alpha_base=ALPHA_BASE, tau=TAU,
                          k_min=K_MIN, k_max=K_MAX):
    """
    αᵢ = α_base × SRSᵢ
    Kᵢ = K_min if SRSᵢ >= τ else K_max
    High SRS → larger α, fewer steps (reliable session, fast adaptation)
    Low SRS  → smaller α, more steps  (noisy session, careful adaptation)
    """
    alpha_i = alpha_base * srs_i
    K_i     = k_min if srs_i >= tau else k_max
    return float(alpha_i), int(K_i)

# ── SRS-Adaptive inner loop ───────────────────────────────────
def srs_adaptive_inner_loop(model, params, support_seqs, support_lengths,
                             support_labels, alpha_i, K_i):
    """
    Standard MAML inner loop with task-specific α and K.
    alpha_i and K_i are computed from the session SRS score.
    Updates params and returns adapted phi_i.
    """
    for step in range(K_i):
        logits = functional_forward(model, params, support_seqs, support_lengths)
        loss   = F.cross_entropy(logits, support_labels)
        grads  = torch.autograd.grad(loss, params.values(), create_graph=False)
        params = {n: p - alpha_i * g
                  for (n, p), g in zip(params.items(), grads)}
    return params

# ── Outer loop with SRS-weighted gradient ─────────────────────
def outer_step(meta_model, batch_tasks, meta_optimizer):
    """
    θ ← θ − β · ∇_θ Σᵢ SRSᵢ · L_query(φᵢ)
    SRS-weighted sum: high-reliability tasks have stronger influence on θ.
    """
    meta_optimizer.zero_grad()
    total_loss = torch.tensor(0.0, requires_grad=True)

    for task in batch_tasks:
        srs_i             = task['srs']                    # float in [0,1]
        alpha_i, K_i      = get_task_hyperparams(srs_i)
        params            = {n: p.clone() for n, p in meta_model.named_parameters()}
        phi_i             = srs_adaptive_inner_loop(
                                meta_model, params,
                                task['support_seqs'],
                                task['support_lengths'],
                                task['support_labels'],
                                alpha_i, K_i)
        query_logits      = functional_forward(meta_model, phi_i,
                                task['query_seqs'],
                                task['query_lengths'])
        query_loss        = F.cross_entropy(query_logits, task['query_labels'])
        total_loss        = total_loss + srs_i * query_loss  # SRS-weighted

    total_loss.backward()
    meta_optimizer.step()
    return total_loss.item()
```

---

### NB11 — Warm-Start + SRS-Adaptive MAML (`11_warmstart_srs_adaptive_maml.ipynb`) ← MAIN CONTRIBUTION

**Create fresh** for both `xuetangx/` and `mars/`.  
**Purpose:** Full combined contribution C3. Warm-start init + SRS-Adaptive inner loop.

**Implementation — three differences from NB07:**
```python
# 1. Warm-start initialization
model.load_state_dict(
    torch.load(f"models/baselines/gru_global_{DATASET}.pth")
)

# 2. Lower outer LR to preserve pretrained knowledge
OUTER_LR = 0.0001   # vs 0.001 in standard MAML

# 3. SRS-Adaptive inner loop — exact same code as NB10
#    get_task_hyperparams() and srs_adaptive_inner_loop() unchanged
#    outer_step() unchanged
```

**Final cell must include full comparison:**
```
========== FINAL RESULTS — [DATASET] ==========
Protocol: K=5 support, Q=10 query | Test: [N] episodes

Model                            HR@10   NDCG@10  MRR    Best Iter
------------------------------------------------------------------
Standard MAML        (NB07)     —       —        —      —
Warm-Start MAML      (NB08)     —       —        —      —
SRS-Adaptive MAML    (NB10)     —       —        —      —
Warm-Start+SRS-Adapt (NB11)     —       —        —      —   ← MAIN
================================================
```

---

## Standard Result Block (Add to Final Cell of Every Experiment Notebook)

```python
print("=" * 55)
print(f"RESULTS — {NOTEBOOK_NAME} — {DATASET}")
print("=" * 55)
print(f"Protocol:      K={K_SUPPORT} support, Q={Q_QUERY} query")
print(f"Test episodes: {len(episodes_test)}")
print(f"Seed:          {GLOBAL_SEED}")
print()
print(f"HR@1:          {test_hr1*100:.2f}%")
print(f"HR@5:          {test_hr5*100:.2f}%")
print(f"HR@10:         {test_hr10*100:.2f}%   ← PRIMARY")
print(f"NDCG@10:       {test_ndcg10*100:.2f}%   ← PRIMARY")
print(f"MRR:           {test_mrr*100:.2f}%")
print()
print(f"Best val iter: {best_iter}")
print(f"Best val HR@10:{best_val_hr10*100:.2f}%")
print("=" * 55)
```

---

## Run Order

### XuetangX

```
Step 0: Create folders, move all old notebooks to old/
Step 1: 00_bootstrap              verify environment and paths
Step 2: Verify processed data exists (see NB05 assertions)
        → If missing: run 01 → 02 → 03 → 03b → 04 → 05
Step 3: 06_base_model_selection   produces gru_global_xuetangx.pth
Step 4: 07_standard_maml          THE BENCHMARK — must finish before NB10/NB11
Step 5: 08_warmstart_maml         ablation
Step 6: 09_srs_validation         validate SRS before NB10/NB11
Step 7: 10_srs_adaptive_maml      ablation (SRS-Adaptive alone)
Step 8: 11_warmstart_srs_adaptive_maml    MAIN CONTRIBUTION
Step 9: Fill XuetangX results in PROJECT_STATE.md
```

### MARS (run after XuetangX is complete)

```
Step 1: Verify test episodes ≥ 50 after NB05
        → If < 50: fix user split in 04_user_split.ipynb first
Step 2: 01 → 02 → 03 → 03b → 04 → 05   data pipeline
Step 3: 06_base_model_selection
Step 4: 07_standard_maml
Step 5: 08_warmstart_maml
Step 6: 09_srs_validation
Step 7: 10_srs_adaptive_maml
Step 8: 11_warmstart_srs_adaptive_maml
Step 9: Fill MARS results in PROJECT_STATE.md
```

---

## Non-Negotiable Rules

1. Never touch anything inside `old/` — it is a reference only
2. SRS-Adaptive code must implement `αᵢ = α_base × SRSᵢ` and `Kᵢ = f(SRSᵢ, τ)` — not weighted loss
3. Primary metrics are HR@10 and NDCG@10 — always report first
4. Metric labels: use HR@1, HR@5, HR@10 — never Acc@1 or Recall@K
5. All MAML variants evaluate on the same test episodes per dataset
6. Seed = 20260107 before every training run
7. Report best checkpoint metric — not end-of-training metric
8. No MARS results reported until test set has ≥ 50 episodes
9. Both datasets follow identical notebook structure and identical evaluation protocol
