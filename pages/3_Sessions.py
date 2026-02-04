"""
Sessions Page - Session Statistics and Analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_dataset_config, load_sessions, load_interactions, DATASETS

st.set_page_config(page_title="Sessions", page_icon="🔄", layout="wide")

st.title("🔄 Sessions")
st.markdown("Analysis of sessionized interaction data")

# Dataset selector
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    options=list(DATASETS.keys()),
    index=0
)

config = get_dataset_config(dataset_name)
st.sidebar.info(f"**Dataset:** {config['name']}\n\n{config['description']}")

# Load data
@st.cache_data(show_spinner="Loading session data...")
def load_session_data(dataset):
    return load_sessions(dataset)

sessions_df = load_session_data(dataset_name)

if sessions_df is None:
    st.error(f"Could not load sessions for {dataset_name}. Please run the sessionization pipeline first.")
    st.stop()

st.success(f"Loaded {len(sessions_df):,} sessions from {dataset_name}")

# Basic statistics
st.header("Session Statistics")

col1, col2, col3, col4 = st.columns(4)

n_sessions = len(sessions_df)
n_users = sessions_df["user_id"].nunique()
avg_sessions_per_user = n_sessions / n_users

# Session length calculation - handle different structures
if "n_events" in sessions_df.columns:
    # XuetangX: one row per session with n_events
    session_lengths = sessions_df["n_events"]
elif "session_items" in sessions_df.columns:
    session_lengths = sessions_df["session_items"].apply(lambda x: len(x) if isinstance(x, list) else 1)
elif "items" in sessions_df.columns:
    session_lengths = sessions_df["items"].apply(lambda x: len(x) if isinstance(x, list) else 1)
elif "n_items" in sessions_df.columns:
    session_lengths = sessions_df["n_items"]
elif "session_id" in sessions_df.columns:
    # MARS: one row per item, count items per session
    session_lengths = sessions_df.groupby("session_id").size()
    n_sessions = len(session_lengths)  # Update n_sessions
else:
    session_lengths = pd.Series([1] * len(sessions_df))

avg_session_length = session_lengths.mean()

col1.metric("Total Sessions", f"{n_sessions:,}")
col2.metric("Unique Users", f"{n_users:,}")
col3.metric("Avg Sessions/User", f"{avg_sessions_per_user:.1f}")
col4.metric("Avg Session Length", f"{avg_session_length:.1f} items")

# Session length distribution
st.header("Session Length Distribution")

st.markdown("""
**What this shows:** The distribution of how many items (course interactions) are in each session.
Longer sessions indicate more engaged learning behavior. Sessions with only 1-2 items are filtered
out as they don't provide enough signal for recommendation.
""")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        session_lengths,
        nbins=int(min(50, session_lengths.max())),
        title="Distribution of Session Lengths",
        labels={"value": "Number of Items", "count": "Number of Sessions"}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("""
    **Box Plot Interpretation:**
    - The box shows the interquartile range (25th to 75th percentile)
    - The line inside the box is the median
    - Whiskers extend to 1.5x IQR
    - Points beyond whiskers are outliers (very long sessions)
    """)

    fig = px.box(
        session_lengths,
        title="Session Length (Box Plot)",
        labels={"value": "Number of Items"}
    )
    st.plotly_chart(fig, use_container_width=True)

# Session length stats
length_stats = pd.DataFrame({
    "Statistic": ["Min", "Max", "Mean", "Median", "Std Dev"],
    "Value": [
        f"{session_lengths.min():.0f}",
        f"{session_lengths.max():.0f}",
        f"{session_lengths.mean():.2f}",
        f"{session_lengths.median():.0f}",
        f"{session_lengths.std():.2f}"
    ]
})
st.table(length_stats)

# Sessions per user distribution
st.header("Sessions per User Distribution")

st.markdown("""
**What this shows:** How many learning sessions each user has. Users with more sessions
provide more training data. For cold-start evaluation, we use users with enough sessions
to create support and query sets.
""")

# For MARS (one row per item), count unique sessions per user
if "session_id" in sessions_df.columns and "n_events" not in sessions_df.columns:
    sessions_per_user = sessions_df.groupby("user_id")["session_id"].nunique()
else:
    sessions_per_user = sessions_df.groupby("user_id").size()

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        sessions_per_user,
        nbins=int(min(50, sessions_per_user.max())),
        title="Distribution of Sessions per User",
        labels={"value": "Number of Sessions", "count": "Number of Users"}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    user_stats = pd.DataFrame({
        "Statistic": ["Min", "Max", "Mean", "Median", "Std Dev"],
        "Value": [
            f"{sessions_per_user.min():.0f}",
            f"{sessions_per_user.max():.0f}",
            f"{sessions_per_user.mean():.2f}",
            f"{sessions_per_user.median():.0f}",
            f"{sessions_per_user.std():.2f}"
        ]
    })
    st.table(user_stats)

# Sessionization process explanation
st.header("Sessionization Process")

st.markdown(f"""
### How Sessions are Created

1. **Sort** interactions by user and timestamp
2. **Compute** inter-activity gaps for each user
3. **Split** into sessions when gap > 30 minutes
4. **Filter** sessions with < 2 items (not useful for recommendations)

### Results for {dataset_name}

| Metric | Value |
|--------|-------|
| Sessions Created | {n_sessions:,} |
| Unique Users | {n_users:,} |
| Avg Session Length | {avg_session_length:.1f} items |
""")

# Data sample
st.header("Session Data Sample")
st.dataframe(sessions_df.head(20), use_container_width=True)
