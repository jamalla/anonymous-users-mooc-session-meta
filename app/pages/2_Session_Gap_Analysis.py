"""
Session Gap Analysis Page - Validate Sessionization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_dataset_config, load_interactions, compute_gap_statistics, DATASETS, REPO_ROOT, render_dataset_selector

st.set_page_config(page_title="Session Gap Analysis", page_icon="📈", layout="wide")

st.title("📈 Session Gap Analysis")
st.markdown("Analyzing inter-activity gaps to validate the 30-minute session threshold")

# Global dataset selector at top of sidebar
dataset_name = render_dataset_selector()
config = get_dataset_config(dataset_name)

# Session threshold slider
threshold_minutes = st.sidebar.slider(
    "Session Threshold (minutes)",
    min_value=5,
    max_value=120,
    value=30,
    step=5
)
threshold_seconds = threshold_minutes * 60

# Load data with sampling for large datasets
@st.cache_data(show_spinner="Loading and computing gap statistics...")
def load_and_compute_gaps(dataset, sample_size=500000):
    df = load_interactions(dataset)
    if df is None:
        return None, None, 0
    total_count = len(df)
    # Sample for large datasets to avoid memory issues
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    stats = compute_gap_statistics(df)
    return df, stats, total_count

df, stats, total_count = load_and_compute_gaps(dataset_name)

if df is None or not stats:
    st.warning(f"**Data files not available for {dataset_name}**")
    st.info("""
    This page requires raw interaction data files which are not included in the cloud deployment.

    **For cloud users:** Please visit the **Baselines** and **MAML** pages (5-10) to see experiment results.

    **For local users:** Run the preprocessing pipeline first to generate the data files.
    """)
    st.stop()

gaps = stats["gaps"]

if len(df) < total_count:
    st.success(f"Analyzed {stats['n_gaps']:,} inter-activity gaps from {dataset_name} (sampled from {total_count:,} total)")
    st.info("Data sampled for visualization performance.")
else:
    st.success(f"Analyzed {stats['n_gaps']:,} inter-activity gaps from {dataset_name}")

# Key metrics
st.header("Gap Statistics Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Gaps", f"{stats['n_gaps']:,}")
col2.metric("Median Gap", f"{stats['median_seconds']/60:.1f} min")
col3.metric("Mean Gap", f"{stats['mean_seconds']/60:.1f} min")
col4.metric(f"Within {threshold_minutes} min", f"{(gaps <= threshold_seconds).mean()*100:.1f}%")

# Detailed statistics
st.header("Detailed Gap Statistics")

stats_df = pd.DataFrame({
    "Percentile": ["Min", "25th", "Median", "75th", "90th", "95th", "99th", "Max"],
    "Seconds": [
        stats["min_seconds"],
        stats["p25_seconds"],
        stats["median_seconds"],
        stats["p75_seconds"],
        stats["p90_seconds"],
        stats["p95_seconds"],
        stats["p99_seconds"],
        stats["max_seconds"]
    ]
})
stats_df["Minutes"] = stats_df["Seconds"] / 60
stats_df["Hours"] = stats_df["Minutes"] / 60

st.dataframe(stats_df.round(2), use_container_width=True)

# Gap distribution visualization
st.header("Gap Distribution")

col1, col2 = st.columns(2)

with col1:
    # Histogram of gaps (capped at 2 hours for visibility)
    gaps_capped = gaps[gaps <= 7200]  # 2 hours
    fig = px.histogram(
        gaps_capped / 60,
        nbins=100,
        title="Distribution of Gaps (up to 2 hours)",
        labels={"value": "Gap (minutes)", "count": "Frequency"}
    )
    fig.add_vline(x=threshold_minutes, line_dash="dash", line_color="red",
                  annotation_text=f"{threshold_minutes} min threshold")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Log-scale histogram
    fig = px.histogram(
        np.log10(gaps + 1),
        nbins=100,
        title="Distribution of Gaps (log scale)",
        labels={"value": "log10(Gap seconds + 1)", "count": "Frequency"}
    )
    fig.add_vline(x=np.log10(threshold_seconds + 1), line_dash="dash", line_color="red",
                  annotation_text=f"{threshold_minutes} min threshold")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# CDF plot
st.header("Cumulative Distribution Function")

sorted_gaps = np.sort(gaps)
cdf = np.arange(1, len(sorted_gaps) + 1) / len(sorted_gaps)

# Sample for performance
sample_size = min(10000, len(sorted_gaps))
indices = np.linspace(0, len(sorted_gaps) - 1, sample_size, dtype=int)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=sorted_gaps[indices] / 60,
    y=cdf[indices] * 100,
    mode='lines',
    name='CDF'
))
fig.add_vline(x=threshold_minutes, line_dash="dash", line_color="red",
              annotation_text=f"{threshold_minutes} min threshold")
fig.add_hline(y=(gaps <= threshold_seconds).mean() * 100, line_dash="dot", line_color="green")

fig.update_layout(
    title="CDF of Inter-Activity Gaps",
    xaxis_title="Gap (minutes)",
    yaxis_title="Cumulative Percentage (%)",
    xaxis=dict(range=[0, 120])  # Show up to 2 hours
)
st.plotly_chart(fig, use_container_width=True)

# Threshold sensitivity analysis
st.header("Threshold Sensitivity Analysis")

thresholds = [5, 10, 15, 20, 30, 45, 60, 90, 120]
pct_within = [(gaps <= t * 60).mean() * 100 for t in thresholds]

sensitivity_df = pd.DataFrame({
    "Threshold (min)": thresholds,
    "% Gaps Within": pct_within
})

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        sensitivity_df,
        x="Threshold (min)",
        y="% Gaps Within",
        title="Percentage of Gaps Within Threshold",
        text="% Gaps Within"
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.dataframe(sensitivity_df.round(2), use_container_width=True)

    st.markdown(f"""
    ### Threshold Recommendation

    With a **{threshold_minutes}-minute threshold**:
    - **{(gaps <= threshold_seconds).mean()*100:.1f}%** of gaps are within-session
    - **{(gaps > threshold_seconds).mean()*100:.1f}%** of gaps are session boundaries

    The 30-minute threshold is a common choice in literature that balances:
    - Capturing natural study breaks
    - Not over-segmenting continuous activity
    """)

# Show existing gap analysis image if available
st.header("Pre-computed Analysis")

gap_img_path = REPO_ROOT / "reports" / f"02_sessionize_{dataset_name.lower()}" / "gap_analysis" / f"{dataset_name.lower()}_gap_analysis.png"
if gap_img_path.exists():
    st.image(str(gap_img_path), caption=f"Gap Analysis for {dataset_name}")
else:
    st.info("No pre-computed gap analysis image found. The analysis above is computed from raw data.")
