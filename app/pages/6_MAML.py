"""
MAML Page - Basic MAML Results
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
from utils import get_dataset_config, load_baseline_results, load_maml_results, DATASETS, render_dataset_selector

st.set_page_config(page_title="MAML", page_icon="🧠", layout="wide")

st.title("🧠 Basic MAML")
st.markdown("Model-Agnostic Meta-Learning for Cold-Start Recommendation")

# Global dataset selector at top of sidebar
dataset_name = render_dataset_selector()
config = get_dataset_config(dataset_name)

# Load results
@st.cache_data
def load_all_results(dataset):
    baselines = load_baseline_results(dataset)
    maml = load_maml_results(dataset, "basic")
    return baselines, maml

baselines, maml_results = load_all_results(dataset_name)

# MAML explanation
st.header("What is MAML?")

st.markdown("""
**Model-Agnostic Meta-Learning (MAML)** learns an initialization that can quickly adapt to new tasks (users).

### Key Idea

Instead of training a single model for all users, MAML:
1. Learns a good **initialization** of model parameters
2. **Adapts** to each new user with a few gradient steps
3. Evaluates on the user's held-out data

### Algorithm

```
For each training episode (user):
    1. Clone model parameters θ
    2. Compute loss on support set
    3. Update θ' = θ - α∇L_support
    4. Compute loss on query set with θ'
    5. Update θ using query loss gradient
```

### This Variant

- **Initialization**: Random
- **Residual Loss**: No
- **Base Model**: GRU
""")

if maml_results is None:
    st.warning(f"MAML results not found for {dataset_name}. Please run the MAML training first.")
    st.stop()

# Extract metrics
final_metrics = maml_results.get("final_metrics", maml_results.get("test_metrics", {}))
training_history = maml_results.get("training_history", [])

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

# Comparison with baselines
st.header("Comparison with Baselines")

if baselines:
    baseline_data = baselines.get("baselines", {})

    # Get GRU Global baseline for comparison
    gru_metrics = baseline_data.get("gru_global", {})

    comparison_data = []

    # Add GRU baseline
    if gru_metrics:
        comparison_data.append({
            "Model": "GRU Global",
            "Accuracy@1": gru_metrics.get("accuracy@1", 0),
            "Recall@5": gru_metrics.get("recall@5", 0),
            "Recall@10": gru_metrics.get("recall@10", 0),
            "MRR": gru_metrics.get("mrr", 0)
        })

    # Add MAML
    comparison_data.append({
        "Model": "Basic MAML",
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
        title="MAML vs GRU Global Baseline",
        color_discrete_sequence=["#3498db", "#e74c3c"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Improvement analysis
    if gru_metrics:
        st.subheader("Improvement Analysis")

        improvements = {}
        for metric in ["accuracy@1", "recall@5", "recall@10", "mrr"]:
            baseline_val = gru_metrics.get(metric, 0)
            maml_val = final_metrics.get(metric, 0)
            if baseline_val > 0:
                pct_change = (maml_val - baseline_val) / baseline_val * 100
                improvements[metric] = pct_change

        if improvements:
            imp_df = pd.DataFrame({
                "Metric": ["Accuracy@1", "Recall@5", "Recall@10", "MRR"],
                "Change (%)": [
                    improvements.get("accuracy@1", 0),
                    improvements.get("recall@5", 0),
                    improvements.get("recall@10", 0),
                    improvements.get("mrr", 0)
                ]
            })

            fig = px.bar(
                imp_df,
                x="Metric",
                y="Change (%)",
                title="Percentage Change vs GRU Baseline",
                color="Change (%)",
                color_continuous_scale=["red", "gray", "green"],
                color_continuous_midpoint=0
            )
            st.plotly_chart(fig, use_container_width=True)

# Configuration details
st.header("Training Configuration")

hyperparams = maml_results.get("hyperparameters", maml_results.get("config", {}))
if hyperparams:
    config_df = pd.DataFrame({
        "Parameter": list(hyperparams.keys()),
        "Value": [str(v) for v in hyperparams.values()]
    })
    st.table(config_df)
