# A Meta-Learning Framework for Cold-Start Adaptation in Session-Based MOOC Recommendation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://anonymous-users-mooc-session-meta-learning.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/jamalla/anonymous-users-mooc-session-meta)

> **Session-Based MOOC Recommendation for Anonymous Learners**  
> Author: Jamallah M H Zawia · Supervisor: Assoc. Prof. Dr Maizatul Akmar Binti Ismail  
> Faculty of Computer Science & Information Technology, University of Malaya

---

## 🌐 Live App

**[https://anonymous-users-mooc-session-meta-learning.streamlit.app/](https://anonymous-users-mooc-session-meta-learning.streamlit.app/)**

The interactive Streamlit app presents all experiment results and analysis across four pages:

| Page | Content |
|---|---|
| 🏠 Home | Project overview, problem statement, research questions, contributions |
| 📊 EDA | Exploratory data analysis of XuetangX and MARS datasets |
| 🔧 Data Processing | Full pipeline walkthrough: sessionisation, SRS scores, user splits, episode index |
| 📈 Results | Base model comparison, MAML experiment progression, 2×2 ablation, SRS validation |
| ⚖️ Comparative | Cross-dataset scale, SRS, and model performance comparison; conclusions |

---

## 🔍 Problem

MOOC platforms serve millions of **anonymous learners** with no persistent identity. Standard recommenders need long interaction histories. This work addresses **cold-start recommendation** where only **K=5 session interactions** are available at inference time — no user profile, no enrollment history.

**Key challenge:** Session quality varies enormously. Standard MAML treats all sessions equally, wasting adaptation capacity on noisy clickstream data.

---

## 💡 Contributions

| ID | Contribution |
|---|---|
| **C1** | **Session Reliability Score (SRS)** — composite metric from Intensity, Extent, and Composition of clickstream events. Normalised to [0, 1]. CAPs computed from training data only (no leakage). |
| **C2** | **SRS-Adaptive MAML** — task-specific inner-loop rate αᵢ = α_base × SRSᵢ and step count Kᵢ = f(SRSᵢ, τ). High-quality sessions adapt faster; noisy sessions adapt more cautiously. |
| **C3** | **Warm-Start + SRS-Adaptive MAML** — combines pretrained GRU4Rec initialisation with the SRS-conditioned inner loop. The full proposed framework. |

---

## 📐 SRS Formula

```
SRS = (Intensity + Extent + Composition) / 3

Intensity   = min(n_events / CAP_95, 1.0)
Extent      = min(duration_sec / CAP_95, 1.0)
Composition = action_type_diversity   # XuetangX (8 event types)
            = item_diversity           # MARS (no action types)

CAPs computed from training sessions only — no leakage.
```

---

## 🧪 Experiments

| Notebook | Model | Role |
|---|---|---|
| NB06 | GRU4Rec / KNN / Popularity / Random | Base model selection |
| NB07 | Standard MAML | Benchmark (random init) |
| NB08 | Warm-Start MAML | Ablation — warm-start only |
| NB09 | SRS Validation | Empirical justification of SRS |
| NB10 | SRS-Adaptive MAML | Ablation — SRS-adaptive only |
| NB11 ★ | Warm-Start + SRS-Adaptive MAML | **Main contribution** |

Datasets: **XuetangX** (28M events, 906K sessions) and **MARS** (compact MOOC interactions with ratings).  
Evaluation protocol: K=5 support interactions, Q=10 query items per episode (cold-start).  
Primary metrics: **HR@10** and **NDCG@10**.

---

## 🗂️ Repository Structure

```
app/                        ← Streamlit app
├── Home.py
├── pages/
│   ├── 1_EDA.py
│   ├── 2_Data_Processing.py
│   ├── 3_Results.py
│   └── 4_Comparative.py
└── utils/data_loader.py

notebooks/
├── xuetangx/               ← NB00–NB13 for XuetangX
└── mars/                   ← NB01–NB13 for MARS

reports/                    ← Pipeline output JSONs (loaded by the app)
```

---

## 🚀 Run Locally

```bash
pip install -r app/requirements.txt
streamlit run app/Home.py
```
