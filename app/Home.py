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

## Project Overview

This application demonstrates a **meta-learning approach** for cold-start recommendation
in Massive Open Online Courses (MOOCs). The goal is to quickly adapt to new users with
only a few interactions.

### Key Concepts

- **Cold-Start Problem**: New users have no interaction history, making recommendations difficult
- **Meta-Learning (MAML)**: Learn an initialization that can quickly adapt to new users
- **Few-Shot Learning**: Adapt with only K=5 support examples per user

### Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| **XuetangX** | Large Chinese MOOC platform | 212K+ pairs, 3K+ users |
| **MARS** | Smaller English MOOC dataset | 2.4K pairs, 800+ users |

### Pipeline

1. **Data Ingestion** → Raw interactions
2. **Sessionization** → Group by 30-min gaps
3. **Pair Generation** → prefix → target sequences
4. **User Splitting** → Train/Val/Test (disjoint users)
5. **Episode Creation** → K-shot support + query sets
6. **Baseline Training** → GRU, SASRec, Session-KNN
7. **MAML Training** → Meta-learning variants

### MAML Variants

| Variant | Warm-Start | Residual | Description |
|---------|------------|----------|-------------|
| Basic MAML | ❌ | ❌ | Random init, standard MAML |
| Warm-Start | ✅ | ❌ | Initialize from GRU baseline |
| Residual | ❌ | ✅ | Add unadapted loss term |
| **Combined** | ✅ | ✅ | Best of both approaches |

---

## Navigation

Use the **sidebar** to navigate between pages:

- 📊 **Dataset EDA** - Explore raw data statistics
- 📈 **Session Gap Analysis** - Validate sessionization
- 🔄 **Sessions** - View session statistics
- 🎯 **Prefix → Target** - Training pair generation
- 🏋️ **Baselines** - Baseline model results
- 🧠 **MAML** - Basic MAML results
- 🚀 **MAML + Warm-Start** - Warm-start variant
- 🔁 **MAML + Residual** - Residual variant
- ⭐ **MAML + Warm-Start + Residual** - Combined approach

---

*Built for MSc Thesis: Meta-Learning for Cold-Start MOOC Recommendation*
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

    st.markdown("### 🔗 Quick Links")
    st.markdown("""
    - [MAML Paper](https://arxiv.org/abs/1703.03400)
    - [XuetangX Dataset](https://www.xuetangx.com/)
    - [MARS Dataset](https://github.com/marso/mars)
    """)
