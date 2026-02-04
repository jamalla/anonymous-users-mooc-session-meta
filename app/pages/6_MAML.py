"""
MAML Page - Vanilla MAML Results (NB07)
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
from utils import get_dataset_config, load_maml_results, get_all_maml_comparison, DATASETS, CURRENT_RESULTS

st.set_page_config(page_title="Vanilla MAML", page_icon="🧠", layout="wide")

st.title("🧠 Vanilla MAML (NB07)")
st.markdown("Model-Agnostic Meta-Learning for Cold-Start Recommendation")

# MAML explanation
st.header("What is MAML?")

st.markdown("""
**Model-Agnostic Meta-Learning (MAML)** learns an initialization that can quickly adapt to new tasks (users).

### Key Idea

Instead of training a single model for all users, MAML:
1. Learns a good **initialization** of model parameters
2. **Adapts** to each new user with a few gradient steps
3. Evaluates on the user's held-out data

### Algorithm (FOMAML)

```
For each training episode (user):
    1. Clone model parameters θ
    2. Compute loss on support set (K=5 examples)
    3. Update θ' = θ - α∇L_support (inner loop)
    4. Compute loss on query set (Q=10 examples) with θ'
    5. Update θ using query loss gradient (outer loop)
```

### Configuration (NB07)

- **Initialization**: Random
- **Base Model**: GRU4Rec (embedding_dim=64, hidden_dim=128)
- **Inner LR**: 0.01
- **Outer LR**: 0.001
- **Inner Steps**: 5
- **Meta Batch Size**: 32
- **Meta Iterations**: 3000
""")

# Load results
maml_results = load_maml_results("XuetangX", "vanilla")

# Display metrics
st.header("Performance Metrics")

# Use hardcoded values for reliability
hr10 = CURRENT_RESULTS["XuetangX"]["vanilla_maml"]["test_HR@10"]
ndcg10 = CURRENT_RESULTS["XuetangX"]["vanilla_maml"]["test_NDCG@10"]

col1, col2, col3 = st.columns(3)

col1.metric("Test HR@10", f"{hr10:.2f}%")
col2.metric("Test NDCG@10", f"{ndcg10:.2f}%")
col3.metric("Test Queries", "3,130")

# Training history
if maml_results and "training_log" in maml_results:
    st.header("Training History")

    history_df = pd.DataFrame(maml_results["training_log"])

    col1, col2 = st.columns(2)

    with col1:
        if "train_loss" in history_df.columns:
            fig = px.line(
                history_df,
                x="iteration",
                y="train_loss",
                title="Training Loss over Iterations",
                labels={"iteration": "Meta Iteration", "train_loss": "Loss"}
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "val_HR@10" in history_df.columns:
            fig = px.line(
                history_df,
                x="iteration",
                y="val_HR@10",
                title="Validation HR@10 over Iterations",
                labels={"iteration": "Meta Iteration", "val_HR@10": "HR@10 (%)"}
            )
            st.plotly_chart(fig, use_container_width=True)

# Comparison with other methods
st.header("Comparison with Other MAML Variants")

comparison_df = get_all_maml_comparison("XuetangX")

if not comparison_df.empty:
    # Display table
    display_df = comparison_df.copy()
    display_df["HR@10"] = display_df["HR@10"].apply(lambda x: f"{x:.2f}%")
    display_df["NDCG@10"] = display_df["NDCG@10"].apply(lambda x: f"{x:.2f}%")
    st.table(display_df)

    # Bar chart comparison
    fig = make_subplots(rows=1, cols=2, subplot_titles=("HR@10", "NDCG@10"))

    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    fig.add_trace(
        go.Bar(
            x=comparison_df["Method"],
            y=comparison_df["HR@10"],
            marker_color=colors,
            text=comparison_df["HR@10"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            name="HR@10"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=comparison_df["Method"],
            y=comparison_df["NDCG@10"],
            marker_color=colors,
            text=comparison_df["NDCG@10"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            name="NDCG@10"
        ),
        row=1, col=2
    )

    fig.update_layout(
        title="MAML Variants Comparison",
        showlegend=False,
        height=400
    )
    fig.update_yaxes(range=[0, 65], row=1, col=1)
    fig.update_yaxes(range=[0, 55], row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

# Key insights
st.header("Key Findings")

st.markdown(f"""
### Vanilla MAML Performance

- **Test HR@10**: {hr10:.2f}%
- **Test NDCG@10**: {ndcg10:.2f}%

### Observations

1. **Few-shot adaptation works**: With only K=5 support examples, the model can adapt to new users
2. **Room for improvement**: Vanilla MAML provides a baseline, but reliability weighting (NB11) and warm-start (NB12) can improve further
3. **Meta-iterations matter**: 3000 iterations were needed to converge

### Limitations

- Random initialization means slow training
- No reliability weighting - all samples treated equally
- Inner loop doesn't account for session quality
""")

# Configuration details
st.header("Training Configuration")

config_data = {
    "Parameter": [
        "K (support examples)",
        "Q (query examples)",
        "Inner Learning Rate",
        "Outer Learning Rate",
        "Inner Steps",
        "Meta Batch Size",
        "Meta Iterations",
        "Use Second Order",
        "Base Model",
        "Embedding Dim",
        "Hidden Dim",
    ],
    "Value": [
        "5",
        "10",
        "0.01",
        "0.001",
        "5",
        "32",
        "3000",
        "False (FOMAML)",
        "GRU4Rec",
        "64",
        "128",
    ]
}

config_df = pd.DataFrame(config_data)
st.table(config_df)
