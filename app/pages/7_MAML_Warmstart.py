"""
MAML + Warm-Start Page (Notebook 07e)
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

st.set_page_config(page_title="MAML + Warm-Start", page_icon="🔥", layout="wide")

st.title("🔥 MAML + Warm-Start (07e)")
st.markdown("Meta-Learning initialized from pre-trained GRU baseline")

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
    baselines = load_baseline_results(dataset)
    warmstart_maml = load_maml_results(dataset, "warmstart")
    return baselines, warmstart_maml

baselines, warmstart_maml = load_results(dataset_name)

# Warm-start explanation
st.header("What is Warm-Start MAML?")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **Warm-Start** initializes MAML from a pre-trained model instead of random weights.

    ### Approach
    1. **Pre-train** a GRU model on all training data (global model)
    2. **Initialize** MAML with pre-trained GRU weights
    3. **Meta-train** using MAML algorithm to learn good initialization
    4. **Adapt** to new users with few gradient steps

    ### Benefits
    - Faster convergence during meta-training
    - Leverages knowledge from global training
    - Better starting point than random initialization
    """)

with col2:
    st.info("""
    **Key Insight**

    Pre-trained weights capture global sequential patterns.
    MAML learns to fine-tune these patterns for individual users.
    """)

if warmstart_maml is None:
    st.warning(f"Warm-Start MAML results not found for {dataset_name}. Please run notebook 07e first.")
    st.stop()

# Extract metrics
zero_shot = warmstart_maml.get("zero_shot_metrics", {})
few_shot = warmstart_maml.get("few_shot_metrics", {})
baseline_metrics = warmstart_maml.get("baseline_metrics", {})
sweep_results = warmstart_maml.get("sweep_results", {})

# Main Results Section
st.header("📊 Performance Results")

# Key metrics cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    baseline_acc = baseline_metrics.get("accuracy@1", 0)
    st.metric("GRU Baseline", f"{baseline_acc:.2%}")

with col2:
    zero_acc = zero_shot.get("accuracy@1", 0)
    delta_zero = ((zero_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
    st.metric("Zero-Shot", f"{zero_acc:.2%}", delta=f"{delta_zero:+.1f}%")

with col3:
    few_acc = few_shot.get("accuracy@1", 0)
    delta_few = ((few_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
    st.metric("Few-Shot (K=5)", f"{few_acc:.2%}", delta=f"{delta_few:+.1f}%")

with col4:
    improvement = warmstart_maml.get("improvement_over_baseline_pct", delta_few)
    st.metric("vs Baseline", f"{improvement:+.2f}%")

# Detailed comparison table
st.subheader("Detailed Metrics Comparison")

comparison_data = []

if baseline_metrics:
    comparison_data.append({
        "Model": "GRU Baseline",
        "Accuracy@1": baseline_metrics.get("accuracy@1", 0),
        "Recall@5": baseline_metrics.get("recall@5", 0),
        "Recall@10": baseline_metrics.get("recall@10", 0),
        "MRR": baseline_metrics.get("mrr", 0)
    })

if zero_shot:
    comparison_data.append({
        "Model": "MAML Zero-Shot",
        "Accuracy@1": zero_shot.get("accuracy@1", 0),
        "Recall@5": zero_shot.get("recall@5", 0),
        "Recall@10": zero_shot.get("recall@10", 0),
        "MRR": zero_shot.get("mrr", 0)
    })

if few_shot:
    comparison_data.append({
        "Model": "MAML Few-Shot (K=5)",
        "Accuracy@1": few_shot.get("accuracy@1", 0),
        "Recall@5": few_shot.get("recall@5", 0),
        "Recall@10": few_shot.get("recall@10", 0),
        "MRR": few_shot.get("mrr", 0)
    })

if comparison_data:
    comp_df = pd.DataFrame(comparison_data)

    # Format for display
    display_df = comp_df.copy()
    for col in ["Accuracy@1", "Recall@5", "Recall@10", "MRR"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Bar chart
    melted = comp_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
    fig = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Model Comparison",
        color_discrete_sequence=["#3498db", "#e74c3c", "#2ecc71"]
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

# Ablation Studies
st.header("🔬 Ablation Studies")

tab1, tab2 = st.tabs(["Support Set Size (K)", "Adaptation Steps"])

with tab1:
    ablation_k = warmstart_maml.get("ablation_support_size", {})
    if ablation_k:
        ablation_data = []
        for k, metrics in ablation_k.items():
            ablation_data.append({
                "K": int(k),
                "Accuracy@1": metrics.get("accuracy@1", 0),
                "Recall@5": metrics.get("recall@5", 0),
                "MRR": metrics.get("mrr", 0)
            })

        if ablation_data:
            abl_df = pd.DataFrame(ablation_data).sort_values("K")

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(abl_df.style.format({
                    "Accuracy@1": "{:.4f}",
                    "Recall@5": "{:.4f}",
                    "MRR": "{:.4f}"
                }), use_container_width=True, hide_index=True)

            with col2:
                fig = px.line(
                    abl_df, x="K", y="Accuracy@1",
                    markers=True,
                    title="Accuracy@1 vs Support Set Size"
                )
                fig.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No support set size ablation data available.")

with tab2:
    ablation_steps = warmstart_maml.get("ablation_adaptation_steps", {})
    if ablation_steps:
        steps_data = []
        for steps, metrics in ablation_steps.items():
            steps_data.append({
                "Steps": int(steps),
                "Accuracy@1": metrics.get("accuracy@1", 0),
                "Recall@5": metrics.get("recall@5", 0),
                "MRR": metrics.get("mrr", 0)
            })

        if steps_data:
            steps_df = pd.DataFrame(steps_data).sort_values("Steps")

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(steps_df.style.format({
                    "Accuracy@1": "{:.4f}",
                    "Recall@5": "{:.4f}",
                    "MRR": "{:.4f}"
                }), use_container_width=True, hide_index=True)

            with col2:
                fig = px.line(
                    steps_df, x="Steps", y="Accuracy@1",
                    markers=True,
                    title="Accuracy@1 vs Adaptation Steps"
                )
                fig.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No adaptation steps ablation data available.")

# Sweep Results (if available)
if sweep_results:
    st.header("🎯 Hyperparameter Tuning")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Optimal Configuration")
        if "checkpoint_used" in sweep_results:
            st.write(f"**Best Checkpoint:** {sweep_results['checkpoint_used']}")
        if "inner_lr" in sweep_results:
            st.write(f"**Optimal Inner LR:** {sweep_results['inner_lr']}")

    with col2:
        st.subheader("Tuned Performance")
        if "maml_few_shot_K5_acc1" in sweep_results:
            st.metric("Tuned Acc@1", f"{sweep_results['maml_few_shot_K5_acc1']:.2%}")

# Key Insights
st.header("💡 Key Insights")

st.markdown(f"""
### Results Summary

| Aspect | Finding |
|--------|---------|
| **Zero-shot** | {zero_shot.get('accuracy@1', 0):.2%} Acc@1 (without adaptation) |
| **Few-shot** | {few_shot.get('accuracy@1', 0):.2%} Acc@1 (with K=5 adaptation) |
| **vs Baseline** | {improvement:+.2f}% compared to GRU global |

### Observations

1. **Warm-start provides stable initialization** - Starting from pre-trained weights gives consistent results
2. **Adaptation improves performance** - Few-shot adaptation ({few_shot.get('accuracy@1', 0):.2%}) beats zero-shot ({zero_shot.get('accuracy@1', 0):.2%})
3. **Trade-off with baseline** - May not always beat the global GRU model
""")

# Configuration
with st.expander("📋 Experiment Configuration"):
    k_config = warmstart_maml.get("k_shot_config", {})
    st.write(f"**K (Support Size):** {k_config.get('K', 5)}")
    st.write(f"**Q (Query Size):** {k_config.get('Q', 10)}")
    st.write(f"**Test Episodes:** {warmstart_maml.get('n_test_episodes', 'N/A')}")
