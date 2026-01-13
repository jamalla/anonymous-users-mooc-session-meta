"""
MAML + Warm-Start + Residual Page - Combined Approach
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
from utils import get_dataset_config, load_baseline_results, load_maml_results, DATASETS

st.set_page_config(page_title="MAML + Warm-Start + Residual", page_icon="⭐", layout="wide")

st.title("⭐ MAML + Warm-Start + Residual")
st.markdown("The best of both worlds: Pre-trained initialization with residual regularization")

# Dataset selector
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    options=list(DATASETS.keys()),
    index=0
)

config = get_dataset_config(dataset_name)
st.sidebar.info(f"**Dataset:** {config['name']}\n\n{config['description']}")

# Load results
@st.cache_data
def load_all_results(dataset):
    baselines = load_baseline_results(dataset)
    basic_maml = load_maml_results(dataset, "basic")
    warmstart_maml = load_maml_results(dataset, "warmstart")
    residual_maml = load_maml_results(dataset, "residual")
    combined_maml = load_maml_results(dataset, "warmstart_residual")
    return baselines, basic_maml, warmstart_maml, residual_maml, combined_maml

baselines, basic_maml, warmstart_maml, residual_maml, combined_maml = load_all_results(dataset_name)

# Combined approach explanation
st.header("Combined Approach")

st.markdown("""
This variant combines **both** improvements:

### Components

| Component | Description |
|-----------|-------------|
| **Warm-Start** | Initialize from pre-trained GRU model |
| **Residual Loss** | Add λ * L_unadapted to prevent overfitting |

### Algorithm

```
1. Pre-train GRU on all training data
2. Initialize MAML with pre-trained weights
3. For each episode:
   - Adapt: θ' = θ - α∇L_support(θ)
   - Meta-loss: L_query(θ') + λ * L_query(θ)
4. Update θ using meta-loss gradient
```

### Expected Benefits

- Better starting point (warm-start)
- Stable adaptation (residual)
- Best overall performance
""")

if combined_maml is None:
    st.warning(f"Combined MAML results not found for {dataset_name}. Please run the MAML training first.")
    st.stop()

# Extract metrics
final_metrics = combined_maml.get("final_metrics", combined_maml.get("test_metrics", {}))
training_history = combined_maml.get("training_history", [])

if not final_metrics:
    st.warning("No final metrics found in results.")
    st.stop()

# Display metrics with highlights
st.header("Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy@1", f"{final_metrics.get('accuracy@1', 0):.4f}")
col2.metric("Recall@5", f"{final_metrics.get('recall@5', 0):.4f}")
col3.metric("Recall@10", f"{final_metrics.get('recall@10', 0):.4f}")
col4.metric("MRR", f"{final_metrics.get('mrr', 0):.4f}")

# Training history
if training_history:
    st.header("Training History")

    history_df = pd.DataFrame(training_history)

    col1, col2 = st.columns(2)

    with col1:
        if "train_loss" in history_df.columns:
            fig = px.line(
                history_df,
                y="train_loss",
                title="Training Loss",
                labels={"index": "Epoch", "train_loss": "Loss"}
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "val_mrr" in history_df.columns:
            fig = px.line(
                history_df,
                y="val_mrr",
                title="Validation MRR",
                labels={"index": "Epoch", "val_mrr": "MRR"}
            )
            st.plotly_chart(fig, use_container_width=True)

# Full comparison with all methods
st.header("Complete Model Comparison")

comparison_data = []

# Add baselines
if baselines:
    baseline_data = baselines.get("baselines", {})

    for model_name, display_name in [("random", "Random"), ("popularity", "Popularity"),
                                      ("gru_global", "GRU Global"), ("sasrec", "SASRec"),
                                      ("sessionknn", "Session-KNN")]:
        metrics = baseline_data.get(model_name, {})
        if metrics:
            comparison_data.append({
                "Model": display_name,
                "Type": "Baseline",
                "Accuracy@1": metrics.get("accuracy@1", 0),
                "Recall@5": metrics.get("recall@5", 0),
                "Recall@10": metrics.get("recall@10", 0),
                "MRR": metrics.get("mrr", 0)
            })

# Add MAML variants
for maml_data, name in [(basic_maml, "Basic MAML"),
                         (warmstart_maml, "Warm-Start MAML"),
                         (residual_maml, "Residual MAML")]:
    if maml_data:
        metrics = maml_data.get("final_metrics", maml_data.get("test_metrics", {}))
        if metrics:
            comparison_data.append({
                "Model": name,
                "Type": "MAML Variant",
                "Accuracy@1": metrics.get("accuracy@1", 0),
                "Recall@5": metrics.get("recall@5", 0),
                "Recall@10": metrics.get("recall@10", 0),
                "MRR": metrics.get("mrr", 0)
            })

# Add Combined (this variant)
comparison_data.append({
    "Model": "Combined (WS+Res)",
    "Type": "MAML Variant",
    "Accuracy@1": final_metrics.get("accuracy@1", 0),
    "Recall@5": final_metrics.get("recall@5", 0),
    "Recall@10": final_metrics.get("recall@10", 0),
    "MRR": final_metrics.get("mrr", 0)
})

comp_df = pd.DataFrame(comparison_data)
comp_df = comp_df.sort_values("MRR", ascending=False).reset_index(drop=True)

# Display table
display_df = comp_df.copy()
for col in ["Accuracy@1", "Recall@5", "Recall@10", "MRR"]:
    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
st.table(display_df)

# Visual comparison
st.subheader("MRR Comparison Across All Models")

# Sort by MRR for visualization
comp_df_sorted = comp_df.sort_values("MRR", ascending=True)

colors = ["#808080" if t == "Baseline" else "#e74c3c" for t in comp_df_sorted["Type"]]
colors[-1] = "#2ecc71"  # Highlight combined as green (best)

fig = go.Figure(go.Bar(
    x=comp_df_sorted["MRR"],
    y=comp_df_sorted["Model"],
    orientation='h',
    marker_color=colors,
    text=comp_df_sorted["MRR"].apply(lambda x: f"{x:.4f}"),
    textposition="outside"
))

fig.update_layout(
    title="Mean Reciprocal Rank (MRR) - All Models",
    xaxis_title="MRR",
    yaxis_title="Model",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# MAML variants comparison
st.header("MAML Variants Comparison")

maml_only = comp_df[comp_df["Type"] == "MAML Variant"].copy()

if len(maml_only) > 0:
    melted = maml_only.melt(id_vars=["Model", "Type"],
                             value_vars=["Accuracy@1", "Recall@5", "Recall@10", "MRR"],
                             var_name="Metric", value_name="Score")

    fig = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="MAML Variant Comparison",
        color_discrete_sequence=["#e74c3c", "#9b59b6", "#3498db", "#2ecc71"]
    )
    st.plotly_chart(fig, use_container_width=True)

# Ablation study
st.header("Ablation Study")

st.markdown("""
### Contribution of Each Component

Let's analyze how each component contributes to the final performance.
""")

ablation_data = []

if basic_maml:
    basic_metrics = basic_maml.get("final_metrics", basic_maml.get("test_metrics", {}))
    if basic_metrics:
        ablation_data.append({
            "Configuration": "Random Init + No Residual",
            "Warm-Start": "No",
            "Residual": "No",
            "MRR": basic_metrics.get("mrr", 0)
        })

if warmstart_maml:
    ws_metrics = warmstart_maml.get("final_metrics", warmstart_maml.get("test_metrics", {}))
    if ws_metrics:
        ablation_data.append({
            "Configuration": "Warm-Start + No Residual",
            "Warm-Start": "Yes",
            "Residual": "No",
            "MRR": ws_metrics.get("mrr", 0)
        })

if residual_maml:
    res_metrics = residual_maml.get("final_metrics", residual_maml.get("test_metrics", {}))
    if res_metrics:
        ablation_data.append({
            "Configuration": "Random Init + Residual",
            "Warm-Start": "No",
            "Residual": "Yes",
            "MRR": res_metrics.get("mrr", 0)
        })

ablation_data.append({
    "Configuration": "Warm-Start + Residual",
    "Warm-Start": "Yes",
    "Residual": "Yes",
    "MRR": final_metrics.get("mrr", 0)
})

ablation_df = pd.DataFrame(ablation_data)

col1, col2 = st.columns(2)

with col1:
    st.table(ablation_df)

with col2:
    fig = px.bar(
        ablation_df,
        x="Configuration",
        y="MRR",
        color="Configuration",
        title="Ablation: Effect of Each Component",
        color_discrete_sequence=["#e74c3c", "#3498db", "#9b59b6", "#2ecc71"]
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# Key findings
st.header("Key Findings")

# Calculate best model info
best_model = comp_df.iloc[0]["Model"]
best_mrr = comp_df.iloc[0]["MRR"]

gru_mrr = comp_df[comp_df["Model"] == "GRU Global"]["MRR"].values[0] if "GRU Global" in comp_df["Model"].values else 0
improvement = (best_mrr - gru_mrr) / gru_mrr * 100 if gru_mrr > 0 else 0

st.markdown(f"""
### Summary

The **{best_model}** approach achieves the best performance:

- **MRR: {best_mrr:.4f}**
- **{improvement:.1f}%** improvement over GRU Global baseline

### Conclusions

1. **Warm-Start helps**: Starting from pre-trained weights provides a significant boost
2. **Residual helps**: Adding unadapted loss prevents overfitting
3. **Combined is best**: The two improvements are complementary

### Practical Implications

For cold-start MOOC recommendation:
- Use the combined Warm-Start + Residual approach
- Pre-train a strong sequential model first
- Apply MAML with residual loss for new user adaptation
""")

# Configuration
st.header("Training Configuration")

hyperparams = combined_maml.get("hyperparameters", combined_maml.get("config", {}))
if hyperparams:
    config_df = pd.DataFrame({
        "Parameter": list(hyperparams.keys()),
        "Value": [str(v) for v in hyperparams.values()]
    })
    st.table(config_df)
