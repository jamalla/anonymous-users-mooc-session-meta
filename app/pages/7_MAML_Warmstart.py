"""
MAML Warm-Start Page (NB08)
Initialize MAML from pre-trained GRU4Rec
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

st.set_page_config(page_title="MAML Warm-Start", page_icon="🔥", layout="wide")

st.title("🔥 Warm-Start MAML (NB08)")
st.markdown("Initialize MAML from pre-trained GRU4Rec instead of random weights")

# Approach explanation
st.header("Warm-Start Approach")

st.markdown("""
### Motivation

Standard MAML (NB07) starts from **random initialization**, which requires learning everything from scratch during meta-training.

**Warm-Start MAML** leverages the pre-trained GRU4Rec model from NB06:
- Load pre-trained weights as initialization
- Lower outer learning rate to preserve learned knowledge
- Fewer inner steps to prevent over-adaptation

### Key Idea

```python
# Standard MAML: Random init
model = GRURecommender(n_items)  # random weights

# Warm-Start MAML: Load pre-trained
model = GRURecommender(n_items)
model.load_state_dict(torch.load("gru_global.pth"))  # pre-trained weights
```
""")

# Results
st.header("Performance Results")

# NB08 results
warmstart_fewshot = 34.95
vanilla_fewshot = CURRENT_RESULTS["XuetangX"]["vanilla_maml"]["test_HR@10"]
gru_baseline = CURRENT_RESULTS["XuetangX"]["gru_baseline"]["test_HR@10"]

col1, col2, col3 = st.columns(3)

col1.metric("Few-Shot HR@10", f"{warmstart_fewshot:.2f}%")
col2.metric("vs Vanilla MAML", f"{warmstart_fewshot - vanilla_fewshot:+.2f}%")
col3.metric("vs GRU Baseline", f"{warmstart_fewshot - gru_baseline:+.2f}%")

# Comparison table
st.subheader("Comparison with Baselines")

comparison_data = [
    {"Model": "GRU4Rec (Global)", "HR@10": f"{gru_baseline:.2f}%", "Notes": "No adaptation"},
    {"Model": "Vanilla MAML (NB07)", "HR@10": f"{vanilla_fewshot:.2f}%", "Notes": "Random init"},
    {"Model": "Warm-Start MAML (NB08)", "HR@10": f"{warmstart_fewshot:.2f}%", "Notes": "Pre-trained init"},
]

st.table(pd.DataFrame(comparison_data))

# Visual comparison
st.header("Visual Comparison")

fig = go.Figure()

models = ["GRU Baseline", "Vanilla MAML", "Warm-Start MAML"]
accuracies = [gru_baseline, vanilla_fewshot, warmstart_fewshot]
colors = ["#95a5a6", "#3498db", "#e67e22"]

fig.add_trace(go.Bar(
    x=models,
    y=accuracies,
    marker_color=colors,
    text=[f"{a:.1f}%" for a in accuracies],
    textposition="outside",
))

fig.update_layout(
    title="HR@10 Comparison",
    yaxis_title="HR@10 (%)",
    yaxis=dict(range=[0, 60]),
    showlegend=False,
    height=400,
)

st.plotly_chart(fig, use_container_width=True)

# Training configuration
st.header("Training Configuration")

config_data = {
    "Parameter": [
        "Pre-trained Model",
        "Inner Learning Rate",
        "Outer Learning Rate",
        "Inner Steps",
        "Meta Batch Size",
    ],
    "Value": [
        "models/baselines/gru_global.pth",
        "0.01",
        "0.0001 (lowered to preserve pre-trained)",
        "3",
        "32",
    ]
}

st.table(pd.DataFrame(config_data))

# Key findings
st.header("Key Findings")

st.markdown(f"""
### Summary

| Metric | Value |
|--------|-------|
| **HR@10** | {warmstart_fewshot:.2f}% |
| **Improvement over Vanilla MAML** | +{warmstart_fewshot - vanilla_fewshot:.2f}% |
| **Improvement over GRU Baseline** | +{warmstart_fewshot - gru_baseline:.2f}% |

### Why Warm-Start Helps

1. **Pre-trained knowledge**: GRU4Rec already learned item co-occurrence patterns
2. **Better starting point**: Meta-training refines rather than learns from scratch
3. **Lower learning rate**: Preserves useful pre-trained representations

### Next: Reliability Weighting (NB11)

Warm-Start improves initialization. The next step is to improve the **inner loop** by weighting samples by session reliability.
""")
