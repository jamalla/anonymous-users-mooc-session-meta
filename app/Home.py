"""
Cold-Start MOOC Recommendation with Meta-Learning
Main Streamlit Application
"""

import streamlit as st
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Cold-Start MOOC Recommendation",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Find repo root
def find_repo_root():
    current = Path(__file__).resolve().parent
    for p in [current, *current.parents]:
        if (p / "PROJECT_STATE.md").exists():
            return p
    return current.parent

REPO_ROOT = find_repo_root()

# Main page content
st.title("🎓 Cold-Start MOOC Recommendation")
st.subheader("Meta-Learning for New User Recommendations")

st.markdown("""
---

## Research Overview

This application demonstrates a **meta-learning approach** for cold-start recommendation
in Massive Open Online Courses (MOOCs). The goal is to quickly adapt to new users with
only a few interactions.

### Key Concepts

- **Cold-Start Problem**: New users have no interaction history, making recommendations difficult
- **Meta-Learning (MAML)**: Learn an initialization that can quickly adapt to new users
- **Few-Shot Learning**: Adapt with only K=5 support examples per user
- **Session Reliability**: Weight training samples by session quality

### Dataset

| Dataset | Description | Size |
|---------|-------------|------|
| **XuetangX** | Large Chinese MOOC platform | 281K+ pairs, 3K+ users |

### Pipeline

1. **Data Ingestion** (NB01) - Raw interactions
2. **Sessionization** (NB02) - Group by 30-min gaps
3. **Pair Generation** (NB03) - prefix -> target sequences
4. **Reliability Scoring** (NB03b) - Session reliability computation
5. **User Splitting** (NB04) - Train/Val/Test (disjoint users)
6. **Episode Creation** (NB05) - K-shot support + query sets
7. **Baseline Training** (NB06) - GRU4Rec
8. **MAML Training** (NB07) - Vanilla MAML
9. **Reliability-Weighted MAML** (NB11) - Contribution 3
10. **Warm-Start + Reliability** (NB12) - Combined approach

---

## Research Contributions

### Contribution 3: Reliability-Weighted MAML

Weight the inner loop loss by session reliability scores:

```python
weighted_loss = (reliability * per_sample_loss).sum() / reliability.sum()
```

Where reliability = (intensity + extent + composition) / 3

### Results Summary

| Method | HR@10 | NDCG@10 | Improvement |
|--------|-------|---------|-------------|
| Vanilla MAML (NB07) | 47.35% | 37.41% | baseline |
| Reliability-Weighted MAML (NB11) | 48.34% | 37.71% | +0.99% |
| **Warm-Start + Reliability (NB12)** | **55.62%** | **44.80%** | **+8.27%** |

### Key Findings

1. **Reliability weighting improves adaptation** - Sessions with more engagement (higher intensity, extent, composition) provide more signal
2. **Warm-start provides strong initialization** - Pre-trained GRU4Rec weights preserve learned item relationships
3. **Contributions stack effectively** - Combining warm-start + reliability achieves 17.5% relative improvement

---

## Navigation

Use the **sidebar** to navigate between pages:

- 📊 **Dataset EDA** - Explore raw data statistics
- 📈 **Session Gap Analysis** - Validate sessionization
- 🔄 **Sessions** - View session statistics
- 🎯 **Prefix -> Target** - Training pair generation
- 🏋️ **Baselines** - GRU4Rec baseline results
- 🧠 **MAML** - Vanilla MAML results (NB07)
- 🎯 **Reliability MAML** - Reliability-Weighted MAML (NB11)
- ⭐ **Combined** - Warm-Start + Reliability (NB12)

---

*PhD Research: Meta-Learning for Cold-Start MOOC Recommendation*
""")

# Sidebar info
with st.sidebar:
    # Global dataset selector at top of sidebar
    st.markdown("### 📊 Dataset Selection")

    # Initialize session state for dataset if not exists
    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = "XuetangX"

    # Dataset selector dropdown
    dataset_options = ["XuetangX", "MARS"]
    selected_dataset = st.selectbox(
        "Select Dataset",
        options=dataset_options,
        index=dataset_options.index(st.session_state.selected_dataset),
        key="dataset_selector_home",
        help="Select the dataset to analyze across all pages"
    )

    # Update session state
    st.session_state.selected_dataset = selected_dataset

    # Show dataset info
    if selected_dataset == "XuetangX":
        st.caption("Large Chinese MOOC platform (212K+ pairs)")
    else:
        st.caption("Smaller English MOOC dataset (2.4K pairs)")

    st.divider()

    st.markdown("### 📁 Project Info")
    st.info(f"**Repo Root:**\n`{REPO_ROOT}`")

    st.markdown("### 📊 Latest Results")
    st.success("""
    **Best Model: Warm-Start + Reliability**
    - HR@10: 55.62%
    - NDCG@10: 44.80%
    - +8.27% over Vanilla MAML
    """)

    st.markdown("### 🔗 Quick Links")
    st.markdown("""
    - [MAML Paper](https://arxiv.org/abs/1703.03400)
    - [XuetangX Dataset](https://www.xuetangx.com/)
    """)
