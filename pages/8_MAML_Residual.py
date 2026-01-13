"""
MAML + Residual Page
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_dataset_config, load_baseline_results, load_maml_results, DATASETS

st.set_page_config(page_title="MAML + Residual", page_icon="🔁", layout="wide")

st.title("🔁 MAML + Residual")
st.markdown("Meta-Learning with residual (unadapted) loss regularization")

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
    residual_maml = load_maml_results(dataset, "residual")
    return baselines, basic_maml, residual_maml

baselines, basic_maml, residual_maml = load_all_results(dataset_name)

# Residual explanation
st.header("What is Residual MAML?")

st.markdown("""
**Residual MAML** adds a regularization term using the unadapted (initial) model's loss.

### Motivation

- Standard MAML may overfit to specific users during adaptation
- The initial model (before adaptation) contains valuable global knowledge
- Adding an unadapted loss term helps prevent overfitting

### Loss Function

```
L_total = L_adapted + λ * L_unadapted
```

Where:
- `L_adapted`: Loss after K gradient steps on support set
- `L_unadapted`: Loss with initial (non-adapted) parameters
- `λ = 0.1`: Residual loss weight

### Benefits

- Prevents overfitting to small support sets
- Maintains global knowledge during adaptation
- More robust to noisy support examples
""")

if residual_maml is None:
    st.warning(f"Residual MAML results not found for {dataset_name}. Please run the MAML training first.")
    st.stop()

# Extract metrics
final_metrics = residual_maml.get("final_metrics", residual_maml.get("test_metrics", {}))
training_history = residual_maml.get("training_history", [])

if not final_metrics:
    st.warning("No final metrics found in results.")
    st.stop()

# Display metrics
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

# Comparison with other methods
st.header("Comparison with Other Methods")

comparison_data = []

# Add baselines
if baselines:
    baseline_data = baselines.get("baselines", {})
    gru_metrics = baseline_data.get("gru_global", {})
    if gru_metrics:
        comparison_data.append({
            "Model": "GRU Global",
            "Accuracy@1": gru_metrics.get("accuracy@1", 0),
            "Recall@5": gru_metrics.get("recall@5", 0),
            "Recall@10": gru_metrics.get("recall@10", 0),
            "MRR": gru_metrics.get("mrr", 0)
        })

# Add Basic MAML
if basic_maml:
    basic_metrics = basic_maml.get("final_metrics", basic_maml.get("test_metrics", {}))
    if basic_metrics:
        comparison_data.append({
            "Model": "Basic MAML",
            "Accuracy@1": basic_metrics.get("accuracy@1", 0),
            "Recall@5": basic_metrics.get("recall@5", 0),
            "Recall@10": basic_metrics.get("recall@10", 0),
            "MRR": basic_metrics.get("mrr", 0)
        })

# Add Residual MAML
comparison_data.append({
    "Model": "Residual MAML",
    "Accuracy@1": final_metrics.get("accuracy@1", 0),
    "Recall@5": final_metrics.get("recall@5", 0),
    "Recall@10": final_metrics.get("recall@10", 0),
    "MRR": final_metrics.get("mrr", 0)
})

comp_df = pd.DataFrame(comparison_data)

# Display table
display_df = comp_df.copy()
for col in ["Accuracy@1", "Recall@5", "Recall@10", "MRR"]:
    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
st.table(display_df)

# Bar chart comparison
melted = comp_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
fig = px.bar(
    melted,
    x="Metric",
    y="Score",
    color="Model",
    barmode="group",
    title="Model Comparison",
    color_discrete_sequence=["#3498db", "#e74c3c", "#9b59b6"]
)
st.plotly_chart(fig, use_container_width=True)

# Improvement over Basic MAML
if basic_maml:
    basic_metrics = basic_maml.get("final_metrics", basic_maml.get("test_metrics", {}))
    if basic_metrics:
        st.subheader("Improvement over Basic MAML")

        improvements = {}
        for metric in ["accuracy@1", "recall@5", "recall@10", "mrr"]:
            basic_val = basic_metrics.get(metric, 0)
            residual_val = final_metrics.get(metric, 0)
            if basic_val > 0:
                pct_change = (residual_val - basic_val) / basic_val * 100
                improvements[metric] = pct_change

        if improvements:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                delta = improvements.get("accuracy@1", 0)
                st.metric("Accuracy@1", f"{final_metrics.get('accuracy@1', 0):.4f}",
                         delta=f"{delta:+.1f}%")

            with col2:
                delta = improvements.get("recall@5", 0)
                st.metric("Recall@5", f"{final_metrics.get('recall@5', 0):.4f}",
                         delta=f"{delta:+.1f}%")

            with col3:
                delta = improvements.get("recall@10", 0)
                st.metric("Recall@10", f"{final_metrics.get('recall@10', 0):.4f}",
                         delta=f"{delta:+.1f}%")

            with col4:
                delta = improvements.get("mrr", 0)
                st.metric("MRR", f"{final_metrics.get('mrr', 0):.4f}",
                         delta=f"{delta:+.1f}%")

# Key insights
st.header("Key Insights")

st.markdown("""
### Residual Loss Benefits

1. **Regularization**: Prevents the model from drifting too far during adaptation
2. **Global Knowledge**: Maintains useful patterns from initial training
3. **Robustness**: More stable when support set is noisy or small

### Lambda Parameter (λ = 0.1)

The λ parameter controls the trade-off:
- **λ = 0**: Pure MAML (no residual)
- **λ = 1**: Equal weight to adapted and unadapted
- **λ = 0.1**: Light regularization (our choice)

### When Residual Helps

- Small support sets (K ≤ 5)
- Noisy or inconsistent user behavior
- When preventing overfitting is important
""")

# Configuration
st.header("Training Configuration")

hyperparams = residual_maml.get("hyperparameters", residual_maml.get("config", {}))
if hyperparams:
    config_df = pd.DataFrame({
        "Parameter": list(hyperparams.keys()),
        "Value": [str(v) for v in hyperparams.values()]
    })
    st.table(config_df)
