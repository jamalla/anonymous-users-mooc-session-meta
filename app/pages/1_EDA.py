"""
1_EDA.py — Exploratory Data Analysis of the raw dataset (NB01).
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_dataset_reports, metrics, key_findings, run_tag, DATASET_PREFIX

st.set_page_config(page_title="EDA | MOOC Recommendation", page_icon="📊", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 MOOC Recommendation")
    st.markdown("---")
    dataset = st.selectbox(
        "Select Dataset",
        options=list(DATASET_PREFIX.keys()),
        key="dataset",
    )
    st.markdown("---")
    st.markdown("**Page:** 📊 EDA")

reports = load_dataset_reports(dataset)
m01 = metrics(reports, "01_ingest")
r01 = reports.get("01_ingest", {})

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"📊 Exploratory Data Analysis — {dataset}")
st.caption(f"Source: NB01 ingest report · Run: {run_tag(reports, '01_ingest')}")
st.markdown("---")

# ── Key metrics ───────────────────────────────────────────────────────────────
st.subheader("Dataset Scale")

if dataset == "XuetangX":
    raw_key, raw_label = "n_events", "Raw Events"
    col_defs = [
        (raw_label,         f"{m01.get('n_events', 0):,}",  "All learning interaction events Feb–Aug 2017"),
        ("Unique Users",    f"{m01.get('n_users', 0):,}",   "Distinct learner IDs"),
        ("Unique Courses",  f"{m01.get('n_courses', 0):,}", "Courses in the vocabulary"),
        ("Event Types",     str(m01.get("n_event_types", 0)), "Distinct interaction types"),
    ]
else:
    col_defs = [
        ("Interactions",    f"{m01.get('n_interactions', 0):,}", "Raw (user, item) interaction records"),
        ("Unique Users",    f"{m01.get('n_users', 0):,}",        "Distinct learner IDs"),
        ("Unique Items",    f"{m01.get('n_items', 0):,}",        "Distinct learning resources"),
        ("Duplicates Removed", str(m01.get("n_dedup_removed", 0)), "Exact duplicate triples removed"),
    ]

cols = st.columns(len(col_defs))
for col, (label, value, help_text) in zip(cols, col_defs):
    col.metric(label, value, help=help_text)

st.markdown("---")

# ── Dataset-specific content ──────────────────────────────────────────────────
if dataset == "XuetangX":
    st.subheader("Event Type Distribution")
    st.markdown(
        "XuetangX records 8 distinct learning event types. "
        "**pause_video** is the most frequent, reflecting widespread video-watching behaviour. "
        "Problem interactions (problem_get, problem_check) indicate active engagement."
    )

    top_events = r01.get("sanity_samples", {}).get("top_event_types", {})
    if top_events:
        sorted_events = sorted(top_events.items(), key=lambda x: x[1], reverse=True)
        labels = [e[0].replace("_", " ") for e in sorted_events]
        counts = [e[1] for e in sorted_events]
        pct    = [c / sum(counts) * 100 for c in counts]

        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            fig = px.bar(
                x=counts, y=labels,
                orientation="h",
                labels={"x": "Event count", "y": "Event type"},
                color=counts,
                color_continuous_scale="Blues",
                text=[f"{p:.1f}%" for p in pct],
                title="Event Type Frequency",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
                coloraxis_showscale=False,
                height=380,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.markdown("**Event Type Breakdown**")
            df_ev = pd.DataFrame({
                "Event Type": labels,
                "Count": [f"{c:,}" for c in counts],
                "Share": [f"{p:.1f}%" for p in pct],
            })
            st.dataframe(df_ev, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("**Event categories:**")
            st.markdown(
                "- 🎬 **Video**: load, pause, seek, speed  \n"
                "- 📝 **Problem**: problem_get, problem_check, problem_check_correct, problem_check_incorrect, problem_save  \n"
                "- 🔢 **Unique types recorded**: 8"
            )

        # ── Pie chart ──────────────────────────────────────────────────────────
        st.subheader("Event Composition")
        video_types    = ["load video", "pause video", "seek video", "speed video"]
        problem_types  = ["problem get", "problem check", "problem check correct",
                          "problem check incorrect", "problem save"]
        video_count   = sum(top_events.get(l.replace(" ", "_"), 0) for l in video_types)
        problem_count = sum(top_events.get(l.replace(" ", "_"), 0) for l in problem_types)
        other_count   = sum(counts) - video_count - problem_count

        fig_pie = px.pie(
            names=["Video events", "Problem events", "Other"],
            values=[video_count, problem_count, other_count],
            color_discrete_sequence=["#2166AC", "#D73027", "#FDAE61"],
            title="Event Category Share",
            hole=0.4,
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

else:
    # MARS
    st.subheader("Interaction Sample")
    st.markdown(
        "MARS is a compact, curated dataset of MOOC learner interactions with ratings (1–10) "
        "and watch percentages (0–100%). Unlike XuetangX, MARS does not have multiple event types — "
        "each row represents a single item view event."
    )

    sample = r01.get("sanity_samples", {}).get("head5", [])
    if sample:
        col_sample, col_info = st.columns([3, 2])
        with col_sample:
            st.markdown("**Sample interactions (head-5):**")
            df_sample = pd.DataFrame(sample)
            st.dataframe(df_sample, use_container_width=True, hide_index=True)

        with col_info:
            st.markdown("**Schema:**")
            schema = {
                "user_id": "Learner identifier",
                "item_id": "Learning resource ID",
                "ts_epoch": "Interaction timestamp (Unix)",
                "watch_percentage": "% of resource watched (0–100)",
                "rating": "Explicit rating (1–10)",
            }
            for field, desc in schema.items():
                st.markdown(f"- **{field}**: {desc}")

    # Watch % vs Rating scatter
    if sample:
        st.subheader("Watch Percentage vs Rating")
        df_s = pd.DataFrame(sample)
        fig_sc = px.scatter(
            df_s, x="watch_percentage", y="rating",
            labels={"watch_percentage": "Watch Percentage (%)", "rating": "Rating (1–10)"},
            color="rating",
            color_continuous_scale="RdYlGn",
            size_max=14,
            title="Watch % vs Rating (head-5 sample)",
        )
        fig_sc.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_sc, use_container_width=True)

    # Scale comparison
    st.subheader("MARS vs XuetangX — Scale Comparison")
    xue_reports = load_dataset_reports("XuetangX")
    m01_x = metrics(xue_reports, "01_ingest")

    comp_data = {
        "Metric": ["Raw events/interactions", "Unique users", "Unique items/courses"],
        "XuetangX": [
            m01_x.get("n_events", 0),
            m01_x.get("n_users", 0),
            m01_x.get("n_courses", 0),
        ],
        "MARS": [
            m01.get("n_interactions", 0),
            m01.get("n_users", 0),
            m01.get("n_items", 0),
        ],
    }
    df_comp = pd.DataFrame(comp_data)
    df_comp["Ratio (X/M)"] = [
        f"{a/b:,.0f}×" if b > 0 else "—"
        for a, b in zip(df_comp["XuetangX"], df_comp["MARS"])
    ]
    df_comp["XuetangX"] = df_comp["XuetangX"].apply(lambda x: f"{x:,}")
    df_comp["MARS"]      = df_comp["MARS"].apply(lambda x: f"{x:,}")
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.info(
        "MARS is **~7,700× smaller** in raw events and **~220× smaller** in users than XuetangX. "
        "This fundamental scale difference constrains the number of meta-learning episodes available "
        "and limits statistical power in evaluation."
    )

# ── Notes ─────────────────────────────────────────────────────────────────────
st.markdown("---")
notes = r01.get("notes", [])
if notes:
    st.subheader("Processing Notes")
    for note in notes:
        st.markdown(f"- {note}")
