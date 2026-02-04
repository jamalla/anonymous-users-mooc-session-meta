"""
Base Model Selection Page - NB06 Results
Comparing base models for cold-start recommendation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import CURRENT_RESULTS

st.set_page_config(page_title="MAML Base Model", page_icon="🏋️", layout="wide")

st.title("🏋️ MAML Base Model Selection (NB06)")
st.markdown("Comparing base models to select the best architecture for meta-learning")

# Overview
st.header("Model Comparison Overview")

st.markdown("""
**Goal:** Select the best base model architecture for MAML meta-learning.

We evaluate 5 different models on cold-start users (zero-shot, no personalization):

| Model | Type | Description |
|-------|------|-------------|
| **Random** | Non-personalized | Uniform random prediction (sanity check) |
| **Popularity** | Non-personalized | Recommends most popular courses |
| **GRU (Global)** | Sequential | GRU trained on all training pairs |
| **SASRec** | Transformer | Self-attention based sequential model |
| **Session-KNN** | Neighborhood | k-NN based on session similarity |
""")

# Results from NB06
base_models = {
    "Random": {"acc1": 0.06, "recall5": 0.22, "recall10": 0.51, "mrr": 0.0047},
    "Popularity": {"acc1": 1.73, "recall5": 5.97, "recall10": 9.58, "mrr": 0.0498},
    "Session-KNN": {"acc1": 14.60, "recall5": 36.33, "recall10": 43.26, "mrr": 0.2503},
    "SASRec": {"acc1": 21.98, "recall5": 40.42, "recall10": 47.73, "mrr": 0.3124},
    "GRU (Global)": {"acc1": 33.55, "recall5": 48.56, "recall10": 55.05, "mrr": 0.4111},
}

# Performance metrics
st.header("Performance Results")

st.markdown("""
**What this shows:** Test set performance of each base model on cold-start users.
All models are evaluated zero-shot (no adaptation to the test user).
Higher is better for all metrics.
""")

# Key metrics
col1, col2, col3, col4 = st.columns(4)

best_model = "GRU (Global)"
best_acc1 = base_models[best_model]["acc1"]
best_recall10 = base_models[best_model]["recall10"]
best_mrr = base_models[best_model]["mrr"]

col1.metric("Best Model", best_model)
col2.metric("Best Acc@1", f"{best_acc1:.2f}%")
col3.metric("Best Recall@10", f"{best_recall10:.2f}%")
col4.metric("Best MRR", f"{best_mrr:.4f}")

# Results table
st.subheader("Full Results Table")

results_df = pd.DataFrame([
    {
        "Model": model,
        "Acc@1 (%)": f"{metrics['acc1']:.2f}",
        "Recall@5 (%)": f"{metrics['recall5']:.2f}",
        "Recall@10 (%)": f"{metrics['recall10']:.2f}",
        "MRR": f"{metrics['mrr']:.4f}",
    }
    for model, metrics in base_models.items()
])

# Sort by Acc@1 descending
results_df["_sort"] = [base_models[m]["acc1"] for m in results_df["Model"]]
results_df = results_df.sort_values("_sort", ascending=False).drop("_sort", axis=1)
st.table(results_df)

# Bar chart comparison
st.header("Visual Comparison")

st.markdown("""
**Bar Chart Interpretation:**
- **Acc@1**: Accuracy at position 1 (exact match in top recommendation)
- **Recall@10**: Whether the correct item appears in top-10 recommendations
- GRU significantly outperforms all other models
""")

# Create comparison chart
models = list(base_models.keys())
acc1_values = [base_models[m]["acc1"] for m in models]
recall10_values = [base_models[m]["recall10"] for m in models]

# Sort by acc1
sorted_indices = sorted(range(len(acc1_values)), key=lambda i: acc1_values[i], reverse=True)
models_sorted = [models[i] for i in sorted_indices]
acc1_sorted = [acc1_values[i] for i in sorted_indices]
recall10_sorted = [recall10_values[i] for i in sorted_indices]

fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy@1 (%)", "Recall@10 (%)"))

colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#95a5a6"]  # Green for best

fig.add_trace(
    go.Bar(
        x=models_sorted,
        y=acc1_sorted,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in acc1_sorted],
        textposition="outside",
        name="Acc@1"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Bar(
        x=models_sorted,
        y=recall10_sorted,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in recall10_sorted],
        textposition="outside",
        name="Recall@10"
    ),
    row=1, col=2
)

fig.update_layout(
    title="Base Model Comparison - GRU is Best",
    showlegend=False,
    height=450
)
fig.update_yaxes(range=[0, 60], row=1, col=1)
fig.update_yaxes(range=[0, 65], row=1, col=2)

st.plotly_chart(fig, use_container_width=True)

# Model architectures
st.header("Model Architectures")

col1, col2 = st.columns(2)

with col1:
    st.subheader("GRU (Global) - Best Model")
    st.markdown("""
    ```python
    class GRURecommender(nn.Module):
        def __init__(self, n_items, embed_dim=64,
                     hidden_dim=128):
            self.embedding = nn.Embedding(n_items, embed_dim)
            self.gru = nn.GRU(embed_dim, hidden_dim,
                              batch_first=True)
            self.fc = nn.Linear(hidden_dim, n_items)

        def forward(self, seq, lengths):
            emb = self.embedding(seq)
            packed = pack_padded_sequence(emb, lengths)
            _, hidden = self.gru(packed)
            logits = self.fc(hidden[-1])
            return logits
    ```

    **Parameters:** 367,470
    """)

with col2:
    st.subheader("SASRec (Self-Attention)")
    st.markdown("""
    ```python
    class SASRec(nn.Module):
        def __init__(self, n_items, hidden_dim=64,
                     num_heads=2, num_blocks=2):
            self.item_emb = nn.Embedding(n_items, hidden_dim)
            self.pos_emb = nn.Embedding(max_len, hidden_dim)
            self.blocks = nn.ModuleList([
                TransformerEncoderLayer(...)
            ])

        def forward(self, seq):
            x = self.item_emb(seq) + self.pos_emb(pos)
            for block in self.blocks:
                x = block(x, attn_mask)
            return self.fc(x[:, -1, :])
    ```

    **Parameters:** 299,182
    """)

# Key findings
st.header("Key Findings")

st.markdown(f"""
### Why GRU is Selected for MAML

1. **Best zero-shot performance**: 33.55% Acc@1 (vs 21.98% for SASRec)
2. **Efficient architecture**: Only 367K parameters
3. **Sequential modeling**: Captures course progression patterns
4. **Proven for meta-learning**: GRU-based models work well with MAML

### Performance Gap Analysis

| Comparison | Gap |
|------------|-----|
| GRU vs SASRec | +11.57% Acc@1 |
| GRU vs Session-KNN | +18.95% Acc@1 |
| GRU vs Popularity | +31.82% Acc@1 |

### Implications for Cold-Start

- **Zero-shot GRU achieves 33.55%** on cold-start users
- This is the baseline we aim to improve with MAML
- Meta-learning should enable quick adaptation with K=5 examples
""")

# Training configuration
st.header("Training Configuration")

config_data = {
    "Parameter": [
        "Vocabulary Size",
        "Embedding Dim",
        "Hidden Dim",
        "GRU Layers",
        "Dropout",
        "Batch Size",
        "Learning Rate",
        "Epochs",
        "Test Episodes",
    ],
    "Value": [
        "1,518 courses",
        "64",
        "128",
        "1",
        "0.2",
        "256",
        "0.001",
        "10",
        "313 cold-start users",
    ]
}

config_df = pd.DataFrame(config_data)
st.table(config_df)

# Outputs
st.header("Saved Models")

st.markdown("""
### Model Artifacts from NB06

| Model | Path | Format |
|-------|------|--------|
| Random | `models/baselines/random.pkl` | Pickle |
| Popularity | `models/baselines/popularity.pkl` | Pickle |
| **GRU (Global)** | **`models/baselines/gru_global.pth`** | PyTorch |
| SASRec | `models/baselines/sasrec.pth` | PyTorch |
| Session-KNN | `models/baselines/sessionknn.pkl` | Pickle |

### Usage in MAML

The GRU model (`gru_global.pth`) serves two purposes:
1. **Baseline comparison**: Zero-shot performance benchmark
2. **Warm-start initialization**: Pre-trained weights for NB08/NB12
""")
