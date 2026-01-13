"""
Dataset EDA Page - Exploratory Data Analysis
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
from utils import get_dataset_config, load_interactions, load_raw_data, DATASETS

st.set_page_config(page_title="Dataset EDA", page_icon="📊", layout="wide")

st.title("📊 Dataset EDA")
st.markdown("Exploratory Data Analysis of MOOC Interaction Data")

# Dataset selector
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    options=list(DATASETS.keys()),
    index=0
)

config = get_dataset_config(dataset_name)
st.sidebar.info(f"**Dataset:** {config['name']}\n\n{config['description']}")

# Load data with sampling for large datasets
@st.cache_data(show_spinner="Loading interactions data...")
def load_data(dataset, sample_size=500000):
    df = load_interactions(dataset)
    if df is not None and len(df) > sample_size:
        # Sample for visualization to avoid memory issues
        return df.sample(n=sample_size, random_state=42), len(df)
    return df, len(df) if df is not None else 0

df, total_count = load_data(dataset_name)

if df is None:
    st.error(f"Could not load interactions for {dataset_name}. Please run the preprocessing pipeline first.")
    st.stop()

if len(df) < total_count:
    st.success(f"Loaded {len(df):,} sampled interactions from {dataset_name} (total: {total_count:,})")
    st.info("Data sampled for visualization performance. Statistics computed on sample.")
else:
    st.success(f"Loaded {len(df):,} interactions from {dataset_name}")

# Basic statistics
st.header("Basic Statistics")

col1, col2, col3, col4 = st.columns(4)

n_users = df["user_id"].nunique()
n_items = df["item_id"].nunique() if "item_id" in df.columns else df["course_id"].nunique()
n_interactions = len(df)
avg_per_user = n_interactions / n_users

col1.metric("Total Interactions", f"{n_interactions:,}")
col2.metric("Unique Users", f"{n_users:,}")
col3.metric("Unique Items", f"{n_items:,}")
col4.metric("Avg per User", f"{avg_per_user:.1f}")

# Interactions per user distribution
st.header("User Activity Distribution")

user_counts = df["user_id"].value_counts()

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        user_counts,
        nbins=50,
        title="Distribution of Interactions per User",
        labels={"value": "Number of Interactions", "count": "Number of Users"}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(
        user_counts,
        title="Interactions per User (Box Plot)",
        labels={"value": "Number of Interactions"}
    )
    st.plotly_chart(fig, use_container_width=True)

# User activity stats
st.subheader("User Activity Statistics")
user_stats = pd.DataFrame({
    "Statistic": ["Min", "Max", "Mean", "Median", "Std Dev"],
    "Value": [
        f"{user_counts.min():.0f}",
        f"{user_counts.max():.0f}",
        f"{user_counts.mean():.2f}",
        f"{user_counts.median():.0f}",
        f"{user_counts.std():.2f}"
    ]
})
st.table(user_stats)

# Item popularity distribution
st.header("Item Popularity Distribution")

item_col = "item_id" if "item_id" in df.columns else "course_id"
item_counts = df[item_col].value_counts()

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        item_counts,
        nbins=50,
        title="Distribution of Interactions per Item",
        labels={"value": "Number of Interactions", "count": "Number of Items"}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Top 20 items
    top_items = item_counts.head(20)
    fig = px.bar(
        x=top_items.values,
        y=[str(i) for i in top_items.index],
        orientation='h',
        title="Top 20 Most Popular Items",
        labels={"x": "Number of Interactions", "y": "Item ID"}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# Temporal analysis
st.header("Temporal Analysis")

if "ts_epoch" in df.columns:
    df_time = df.copy()
    df_time["datetime"] = pd.to_datetime(df_time["ts_epoch"], unit="s")
    df_time["date"] = df_time["datetime"].dt.date
    df_time["hour"] = df_time["datetime"].dt.hour
    df_time["day_of_week"] = df_time["datetime"].dt.day_name()

    col1, col2 = st.columns(2)

    with col1:
        # Interactions by hour
        hour_counts = df_time.groupby("hour").size().reset_index(name="count")
        fig = px.bar(
            hour_counts,
            x="hour",
            y="count",
            title="Interactions by Hour of Day",
            labels={"hour": "Hour", "count": "Number of Interactions"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Interactions by day of week
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_counts = df_time.groupby("day_of_week").size().reindex(day_order).reset_index()
        dow_counts.columns = ["day", "count"]
        fig = px.bar(
            dow_counts,
            x="day",
            y="count",
            title="Interactions by Day of Week",
            labels={"day": "Day", "count": "Number of Interactions"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Daily interactions over time
    daily_counts = df_time.groupby("date").size().reset_index(name="count")
    fig = px.line(
        daily_counts,
        x="date",
        y="count",
        title="Daily Interactions Over Time",
        labels={"date": "Date", "count": "Number of Interactions"}
    )
    st.plotly_chart(fig, use_container_width=True)

# Data sample
st.header("Data Sample")
st.dataframe(df.head(20), use_container_width=True)
