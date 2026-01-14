"""
Prefix-Target Page - Training Pair Generation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_dataset_config, load_pairs, load_episodes, DATASETS, render_dataset_selector

st.set_page_config(page_title="Prefix → Target", page_icon="🎯", layout="wide")

st.title("🎯 Prefix → Target Pairs")
st.markdown("Training pair generation and episode creation for meta-learning")

# Global dataset selector at top of sidebar
dataset_name = render_dataset_selector()
config = get_dataset_config(dataset_name)

# K-shot configuration
st.sidebar.markdown("### K-Shot Configuration")
K = st.sidebar.selectbox("K (Support Size)", options=[3, 5, 10], index=1)
Q = st.sidebar.selectbox("Q (Query Size)", options=[5, 10, 15], index=1)

# Load data
@st.cache_data(show_spinner="Loading pairs data...")
def load_pair_data(dataset, split):
    return load_pairs(dataset, split)

@st.cache_data(show_spinner="Loading episodes data...")
def load_episode_data(dataset, split, k, q):
    return load_episodes(dataset, split, k, q)

# Load all splits
train_pairs = load_pair_data(dataset_name, "train")
val_pairs = load_pair_data(dataset_name, "val")
test_pairs = load_pair_data(dataset_name, "test")

# Pairs overview
st.header("Prefix-Target Pairs")

st.markdown("""
### Pair Generation Process

For each session, we generate multiple (prefix, target) pairs:
- **Prefix**: First k items of the session (k = 1 to n-1)
- **Target**: The (k+1)th item to predict

This creates training examples for next-item prediction.
""")

# Stats table with percentages
pairs_stats = []
total_pairs = 0
total_users = 0

# First pass to get totals
for split, df in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
    if df is not None:
        total_pairs += len(df)
        total_users += df["user_id"].nunique()

# Second pass to build stats with percentages
for split, df in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
    if df is not None:
        n_pairs = len(df)
        n_users = df["user_id"].nunique()
        avg_prefix_len = df["prefix"].apply(len).mean() if "prefix" in df.columns else 0
        pairs_pct = (n_pairs / total_pairs * 100) if total_pairs > 0 else 0
        users_pct = (n_users / total_users * 100) if total_users > 0 else 0
        pairs_stats.append({
            "Split": split.capitalize(),
            "Pairs": f"{n_pairs:,}",
            "Pairs %": f"{pairs_pct:.1f}%",
            "Users": f"{n_users:,}",
            "Users %": f"{users_pct:.1f}%",
            "Avg Prefix Length": f"{avg_prefix_len:.1f}"
        })

if pairs_stats:
    st.table(pd.DataFrame(pairs_stats))
else:
    st.warning(f"**Data files not available for {dataset_name}**")
    st.info("""
    This page requires pairs/episodes data files which are not included in the cloud deployment.

    **For cloud users:** Please visit the **Baselines** and **MAML** pages (5-10) to see experiment results.

    **For local users:** Run the pair generation pipeline first to generate the data files.
    """)
    st.stop()

# Prefix -> Target Examples
st.header("Prefix → Target Examples")

if train_pairs is not None and "prefix" in train_pairs.columns and "target" in train_pairs.columns:
    st.markdown("Here are some example pairs showing how prefixes map to targets:")

    # Get a few diverse examples (different prefix lengths)
    sample_pairs = train_pairs.sample(n=min(10, len(train_pairs)), random_state=42).sort_values(
        by="prefix", key=lambda x: x.apply(len)
    )

    examples = []
    for _, row in sample_pairs.iterrows():
        prefix = row["prefix"]
        target = row["target"]
        # Format prefix as list of items
        prefix_str = "[" + ", ".join([f"i{i}" for i in prefix[-5:]]) + "]"  # Show last 5 items
        if len(prefix) > 5:
            prefix_str = "[..., " + ", ".join([f"i{i}" for i in prefix[-5:]]) + "]"
        target_str = f"[i{target}]" if not isinstance(target, (list, tuple)) else "[" + ", ".join([f"i{t}" for t in target]) + "]"
        examples.append({
            "Prefix Length": len(prefix),
            "Prefix (last 5 items)": prefix_str,
            "→": "→",
            "Target": target_str
        })

    st.table(pd.DataFrame(examples))
else:
    st.info("No prefix-target pairs available to display examples.")

# Prefix length distribution
st.header("Prefix Length Distribution")

if train_pairs is not None and "prefix" in train_pairs.columns:
    prefix_lengths = train_pairs["prefix"].apply(len)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            prefix_lengths,
            nbins=int(min(30, prefix_lengths.max())),
            title="Distribution of Prefix Lengths (Train)",
            labels={"value": "Prefix Length", "count": "Number of Pairs"}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        length_stats = pd.DataFrame({
            "Statistic": ["Min", "Max", "Mean", "Median"],
            "Value": [
                f"{prefix_lengths.min():.0f}",
                f"{prefix_lengths.max():.0f}",
                f"{prefix_lengths.mean():.2f}",
                f"{prefix_lengths.median():.0f}"
            ]
        })
        st.table(length_stats)

# Episodes section
st.header(f"Meta-Learning Episodes (K={K}, Q={Q})")

st.markdown(f"""
### Episode Structure

Each episode contains data for one user:
- **Support Set**: {K} (prefix, target) pairs for adaptation
- **Query Set**: {Q} (prefix, target) pairs for evaluation

This simulates the cold-start scenario where we have few examples to learn from.
""")

# Load episodes
train_episodes = load_episode_data(dataset_name, "train", K, Q)
val_episodes = load_episode_data(dataset_name, "val", K, Q)
test_episodes = load_episode_data(dataset_name, "test", K, Q)

# Episode stats
episode_stats = []
for split, df in [("train", train_episodes), ("val", val_episodes), ("test", test_episodes)]:
    if df is not None:
        n_episodes = len(df)
        n_users = df["user_id"].nunique() if "user_id" in df.columns else n_episodes
        episode_stats.append({
            "Split": split.capitalize(),
            "Episodes": f"{n_episodes:,}",
            "Unique Users": f"{n_users:,}"
        })

if episode_stats:
    st.table(pd.DataFrame(episode_stats))
else:
    st.warning(f"No episode data found for K={K}, Q={Q}. Please run the episode generation pipeline first.")

# User split explanation
st.header("User-Disjoint Splitting")

st.markdown("""
### Why Disjoint Users?

To properly evaluate cold-start performance:
- **Train users**: Used only for meta-training
- **Validation users**: Used for hyperparameter tuning
- **Test users**: Used for final evaluation

No user appears in multiple splits, ensuring we truly test on "new" users.

### Split Ratios
- Train: 70% of users
- Validation: 15% of users
- Test: 15% of users
""")

# Visualize split
has_all_splits = (train_pairs is not None and val_pairs is not None and test_pairs is not None)
if has_all_splits:
    split_data = pd.DataFrame({
        "Split": ["Train", "Validation", "Test"],
        "Users": [
            train_pairs["user_id"].nunique(),
            val_pairs["user_id"].nunique(),
            test_pairs["user_id"].nunique()
        ],
        "Pairs": [
            len(train_pairs),
            len(val_pairs),
            len(test_pairs)
        ]
    })

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            split_data,
            values="Users",
            names="Split",
            title="User Distribution Across Splits",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            split_data,
            values="Pairs",
            names="Split",
            title="Pair Distribution Across Splits",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

# Sample data
st.header("Sample Data")

if train_pairs is not None:
    st.subheader("Sample Pairs")
    st.dataframe(train_pairs.head(10), use_container_width=True)

if train_episodes is not None:
    st.subheader("Sample Episode")
    st.dataframe(train_episodes.head(5), use_container_width=True)
