"""
Baselines Page - Baseline Model Results
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
from utils import get_dataset_config, load_baseline_results, DATASETS

st.set_page_config(page_title="Baselines", page_icon="🏋️", layout="wide")

st.title("🏋️ Baseline Models")
st.markdown("Performance comparison of baseline recommendation models")

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
def load_results(dataset):
    return load_baseline_results(dataset)

results = load_results(dataset_name)

if results is None:
    st.error(f"Could not load baseline results for {dataset_name}. Please run the baseline evaluation first.")
    st.stop()

# K-shot info
k_shot = results.get("k_shot_config", {"K": 5, "Q": 10})
n_episodes = results.get("n_test_episodes", 0)

st.success(f"Loaded results for {dataset_name} (K={k_shot['K']}, Q={k_shot['Q']}, {n_episodes} test episodes)")

# Model descriptions
st.header("Baseline Models")

st.markdown("""
| Model | Description |
|-------|-------------|
| **Random** | Random item recommendation (lower bound) |
| **Popularity** | Recommend most popular items globally |
| **GRU Global** | GRU-based sequential model trained on all users |
| **SASRec** | Self-Attention for Sequential Recommendation |
| **Session-KNN** | K-Nearest Neighbors based on session similarity |
""")

# Extract baseline results
baselines = results.get("baselines", {})

if not baselines:
    st.error("No baseline results found in the results file.")
    st.stop()

# Create results dataframe
metrics = ["accuracy@1", "recall@5", "recall@10", "mrr"]
metric_labels = ["Accuracy@1", "Recall@5", "Recall@10", "MRR"]

results_data = []
for model, model_results in baselines.items():
    row = {"Model": model.replace("_", " ").title()}
    for metric in metrics:
        row[metric] = model_results.get(metric, 0)
    results_data.append(row)

results_df = pd.DataFrame(results_data)

# Sort by MRR (descending)
results_df = results_df.sort_values("mrr", ascending=False).reset_index(drop=True)

# Display results table
st.header("Performance Comparison")

# Format for display
display_df = results_df.copy()
for col in metrics:
    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

display_df.columns = ["Model"] + metric_labels
st.table(display_df)

# Bar charts
st.header("Visual Comparison")

# Define colors for models
model_colors = {
    "Random": "#808080",
    "Popularity": "#4ECDC4",
    "Gru Global": "#3498db",
    "Sasrec": "#9b59b6",
    "Sessionknn": "#e74c3c"
}

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        results_df,
        x="Model",
        y="accuracy@1",
        title="Accuracy@1 Comparison",
        color="Model",
        color_discrete_map=model_colors
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        results_df,
        x="Model",
        y="mrr",
        title="Mean Reciprocal Rank (MRR)",
        color="Model",
        color_discrete_map=model_colors
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Recall comparison
col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        results_df,
        x="Model",
        y="recall@5",
        title="Recall@5 Comparison",
        color="Model",
        color_discrete_map=model_colors
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        results_df,
        x="Model",
        y="recall@10",
        title="Recall@10 Comparison",
        color="Model",
        color_discrete_map=model_colors
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Grouped bar chart for all metrics
st.header("All Metrics Comparison")

# Reshape data for grouped bar chart
melted_df = results_df.melt(id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score")
melted_df["Metric"] = melted_df["Metric"].map(dict(zip(metrics, metric_labels)))

fig = px.bar(
    melted_df,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    title="All Metrics by Model",
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig, use_container_width=True)

# Key findings
st.header("Key Findings")

best_model = results_df.iloc[0]["Model"]
best_mrr = results_df.iloc[0]["mrr"]
random_mrr = results_df[results_df["Model"] == "Random"]["mrr"].values[0] if "Random" in results_df["Model"].values else 0

st.markdown(f"""
### Best Performing Model: **{best_model}**

- Achieves MRR of **{best_mrr:.4f}**
- Significantly outperforms Random baseline ({random_mrr:.4f})
- {best_mrr / random_mrr:.1f}x improvement over random

### Observations

1. **Neural models** (GRU, SASRec) significantly outperform simple baselines
2. **Popularity** baseline provides a reasonable starting point
3. **Session-KNN** offers competitive performance with interpretability

### Limitations

These baselines are trained on **all users** and evaluated on **new users**.
The challenge is: can we do better with meta-learning?
""")
