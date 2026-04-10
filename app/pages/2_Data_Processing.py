"""
2_Data_Processing.py — Full data pipeline (NB02–NB05): sessionization, pairs,
SRS scores, user split, and episode index.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_dataset_reports, metrics, run_tag, DATASET_PREFIX

st.set_page_config(page_title="Data Processing | MOOC Recommendation", page_icon="🔬", layout="wide")

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
    st.markdown("**Page:** 🔬 Data Processing")

reports = load_dataset_reports(dataset)
m01 = metrics(reports, "01_ingest")
m02 = metrics(reports, "02_sessionize")
m03 = metrics(reports, "03_vocab_pairs")
m03b = metrics(reports, "03b_srs_scores")
m04 = metrics(reports, "04_user_split")
m05 = metrics(reports, "05_episode_index")

raw_events = m01.get("n_events_raw", m01.get("n_events", m01.get("n_interactions", 0)))

st.title(f"🔬 Data Processing Pipeline — {dataset}")
st.markdown(
    "This page traces every transformation from raw events through to the "
    "meta-learning episode index. All numbers are loaded from pipeline run reports."
)
st.markdown("---")

# ── Pipeline progress bar ─────────────────────────────────────────────────────
steps = ["NB01\nIngest", "NB02\nSessionize", "NB03\nPairs", "NB03b\nSRS", "NB04\nSplit", "NB05\nEpisodes"]
st.subheader("Pipeline Funnel")

funnel_users = [
    m01.get("n_users", 0),
    m02.get("n_users", 0),
    m03.get("n_users", 0),
    m03.get("n_users", 0),
    m04.get("n_users_total", 0),
    m04.get("n_users_total", 0),
]
funnel_events = [
    raw_events,
    m02.get("n_events", 0),
    m03.get("n_pairs", 0),
    m03b.get("n_pairs", 0),
    m04.get("n_pairs_train", 0) + m04.get("n_pairs_val", 0) + m04.get("n_pairs_test", 0),
    (m05.get("n_episodes_train", 0) + m05.get("n_episodes_val", 0) + m05.get("n_episodes_test", 0)),
]
funnel_labels_u = [f"{v:,}" for v in funnel_users]
funnel_labels_e = [f"{v:,}" for v in funnel_events]

col_f1, col_f2 = st.columns(2)

with col_f1:
    fig_u = go.Figure(go.Funnel(
        y=["NB01 — Raw Users", "NB02 — With Sessions", "NB03 — With Pairs",
           "NB03b — SRS Scored", "NB04 — Split", "NB05 — In Episodes"],
        x=funnel_users,
        textposition="inside",
        textinfo="value+percent initial",
        marker={"color": ["#08519C", "#2171B5", "#4292C6", "#6BAED6", "#9ECAE1", "#C6DBEF"]},
        connector={"line": {"color": "#2166AC", "width": 2}},
    ))
    fig_u.update_layout(title="User Retention Funnel", height=360,
                        margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_u, use_container_width=True)

with col_f2:
    stage_labels = ["NB01\nRaw", "NB02\nEvents", "NB03\nPairs",
                    "NB03b\nPairs+SRS", "NB04\nAll Pairs", "NB05\nEpisodes"]
    fig_bar = px.bar(
        x=stage_labels, y=funnel_events,
        labels={"x": "Pipeline Stage", "y": "Count"},
        color=funnel_events,
        color_continuous_scale="Blues",
        text=[f"{v:,}" for v in funnel_events],
        log_y=True,
        title="Events / Pairs / Episodes per Stage (log scale)",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False, coloraxis_showscale=False,
                          height=360, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# NB02 — Sessionization
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("**NB02 — Sessionization** · 30-min inactivity threshold", expanded=True):
    st.caption(f"Run: {run_tag(reports, '02_sessionize')}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sessions Created",    f"{m02.get('n_sessions', 0):,}")
    col2.metric("Users with Sessions", f"{m02.get('n_users', 0):,}")
    col3.metric("Gap Threshold",       "1,800 s  (30 min)")
    col4.metric("Min Events / Session", str(m02.get("min_events_per_session", 2)))

    st.markdown("**Session boundary rule:**")
    st.markdown(
        "- If the time gap between two consecutive events by the same user exceeds **30 minutes**, "
        "a new session begins.  \n"
        "- Sessions with fewer than **2 events** are discarded as too sparse to carry learning signal."
    )

    gs = m02.get("gap_stats", {})
    if gs:
        pct_names  = ["p50 (median)", "p90", "p95", "p99", "Max"]
        pct_values = [
            gs.get("p50_sec", 0), gs.get("p90_sec", 0),
            gs.get("p95_sec", 0), gs.get("p99_sec", 0),
            gs.get("max_sec", 0),
        ]
        col_chart2, col_info2 = st.columns([3, 2])
        with col_chart2:
            fig_gap = px.bar(
                x=pct_names, y=pct_values,
                labels={"x": "Percentile", "y": "Gap (seconds)"},
                color=pct_values,
                color_continuous_scale="Blues",
                text=[f"{v:,.0f}s" for v in pct_values],
                title="Inter-event Gap Distribution",
            )
            fig_gap.add_hline(y=1800, line_dash="dash", line_color="red",
                              annotation_text="Session threshold (1800 s)")
            fig_gap.update_traces(textposition="outside")
            fig_gap.update_layout(showlegend=False, coloraxis_showscale=False,
                                  height=320, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_gap, use_container_width=True)

        with col_info2:
            st.markdown("**Gap percentiles:**")
            for name, val in zip(pct_names, pct_values):
                st.markdown(f"- **{name}**: {val:,.0f} s")
            st.markdown(f"- **Threshold**: 1,800 s")
            st.markdown(f"- **Total gaps measured**: {gs.get('n_gaps', 0):,}")

    avg_events = m02.get("n_events", 0) / m02.get("n_sessions", 1)
    avg_sessions = m02.get("n_sessions", 0) / m02.get("n_users", 1)
    c_a, c_b = st.columns(2)
    c_a.metric("Avg events / session", f"{avg_events:.1f}")
    c_b.metric("Avg sessions / user",  f"{avg_sessions:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# NB03 — Vocabulary & Pairs
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("**NB03 — Vocabulary & Sequence Pairs**", expanded=True):
    st.caption(f"Run: {run_tag(reports, '03_vocab_pairs')}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vocabulary Size",  f"{m03.get('n_items', 0):,}")
    col2.metric("Total Pairs",      f"{m03.get('n_pairs', 0):,}")
    col3.metric("Users with Pairs", f"{m03.get('n_users', 0):,}")
    if "n_sessions_with_pairs" in m03:
        col4.metric("Sessions with Pairs", f"{m03.get('n_sessions_with_pairs', 0):,}")
    else:
        col4.metric("Avg pairs / user",
                    f"{m03.get('n_pairs', 0) / m03.get('n_users', 1):.1f}")

    st.markdown(
        "Each session is converted into a sequence of item IDs visited in chronological order. "
        "For every position in the sequence, a **(prefix → next item)** pair is created. "
        "These pairs are the prediction targets for next-item recommendation."
    )
    st.markdown(
        "- **Prefix**: all items seen so far in the session  \n"
        "- **Label**: the next item the learner accessed  \n"
        "- Consecutive duplicate items within a session are removed before pair generation"
    )

    # Sample pairs from sanity_samples
    r03 = reports.get("03_vocab_pairs", {})
    sample_pairs = r03.get("sanity_samples", {}).get("pairs_head3", [])
    if sample_pairs:
        st.markdown("**Sample pairs:**")
        df_pairs = pd.DataFrame(sample_pairs)[["pair_id", "user_id", "session_id", "prefix", "label", "prefix_len"]]
        st.dataframe(df_pairs, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# NB03b — SRS Scores
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("**NB03b — Session Reliability Scores (SRS)**", expanded=True):
    st.caption(f"Run: {run_tag(reports, '03b_srs_scores')}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sessions Scored",  f"{m03b.get('n_sessions', 0):,}")
    col2.metric("SRS Mean",         f"{m03b.get('srs_mean', 0):.4f}")
    col3.metric("SRS Median (p50)", f"{m03b.get('srs_p50', 0):.4f}")
    col4.metric("CAP Intensity",    f"{m03b.get('cap_intensity', 0):.1f} events")

    st.markdown(
        "Each session receives an **SRS score ∈ [0, 1]** — a composite reliability measure "
        "computed from three components. CAPs are derived from **training sessions only** (no leakage)."
    )

    if dataset == "XuetangX":
        st.markdown(
            "| Component | Formula | Captures |\n"
            "|---|---|---|\n"
            "| **Intensity** | `min(n_events / CAP_95, 1.0)` | Interaction volume |\n"
            "| **Extent** | `min(duration_sec / CAP_95, 1.0)` | Temporal span |\n"
            "| **Composition** | `n_unique_action_types / 6` | Behavioural diversity |\n"
            "| **SRS** | `(Intensity + Extent + Composition) / 3` | Overall reliability |"
        )
    else:
        st.markdown(
            "| Component | Formula | Captures |\n"
            "|---|---|---|\n"
            "| **Intensity** | `min(n_events / CAP_95, 1.0)` | Interaction volume |\n"
            "| **Extent** | `min(duration_sec / CAP_95, 1.0)` | Temporal span |\n"
            "| **Composition** | `min(n_unique_items / CAP_95, 1.0)` | Item diversity (no action types in MARS) |\n"
            "| **SRS** | `(Intensity + Extent + Composition) / 3` | Overall reliability |"
        )

    tier_low  = m03b.get("tier_low",    0)
    tier_med  = m03b.get("tier_medium", 0)
    tier_high = m03b.get("tier_high",   0)

    col_pie, col_stats = st.columns([2, 3])

    with col_pie:
        fig_pie = px.pie(
            names=["Low (<0.33)", "Medium (0.33–0.66)", "High (≥0.66)"],
            values=[tier_low, tier_med, tier_high],
            color_discrete_sequence=["#D73027", "#FEE090", "#4DAC26"],
            title="Session Tier Distribution",
            hole=0.45,
        )
        fig_pie.update_traces(
            textinfo="percent+label",
            textfont_size=12,
        )
        fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10),
                              showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_stats:
        n_sess = m03b.get("n_sessions", 0)
        st.markdown("**Tier breakdown:**")
        tier_df = pd.DataFrame({
            "Tier":       ["Low (<0.33)", "Medium (0.33–0.66)", "High (≥0.66)"],
            "Sessions":   [f"{int(tier_low*n_sess):,}", f"{int(tier_med*n_sess):,}", f"{int(tier_high*n_sess):,}"],
            "Percentage": [f"{tier_low*100:.1f}%", f"{tier_med*100:.1f}%", f"{tier_high*100:.1f}%"],
        })
        st.dataframe(tier_df, use_container_width=True, hide_index=True)
        st.markdown(
            f"- **Mean SRS**: {m03b.get('srs_mean', 0):.4f}  \n"
            f"- **CAP Intensity (p95)**: {m03b.get('cap_intensity', 0):.1f} events  \n"
            f"- **CAP Extent (p95)**: {m03b.get('cap_extent', 0):.1f} s"
        )

    # NB09 SRS validation data
    m09 = metrics(reports, "09_srs_validation")
    if m09:
        st.markdown("**SRS Validation (NB09) — Percentiles & Correlations:**")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            pct_data = {
                "Percentile": ["min", "p25", "p50", "p75", "max"],
                "SRS Value":  [
                    f"{m09.get('min', 0):.4f}",
                    f"{m09.get('p25', 0):.4f}",
                    f"{m09.get('p50', 0):.4f}",
                    f"{m09.get('p75', 0):.4f}",
                    f"{m09.get('max', 0):.4f}",
                ],
            }
            st.dataframe(pd.DataFrame(pct_data), use_container_width=True, hide_index=True)
        with col_v2:
            corr_data = {
                "Attribute":   ["n_events (intensity)", "duration_sec (extent)"],
                "Pearson r":   [
                    f"{m09.get('corr_srs_n_events', 0):.4f}",
                    f"{m09.get('corr_srs_duration', 0):.4f}",
                ],
                "Interpretation": [
                    "Moderate — SRS ≠ just event count",
                    "Strong — duration is a key SRS driver",
                ] if dataset == "XuetangX" else [
                    "Very strong — MARS SRS dominated by event count",
                    "Strong — duration also highly correlated",
                ],
            }
            st.dataframe(pd.DataFrame(corr_data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# NB04 — User Split
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("**NB04 — User Split (Cold-Start Guarantee)**", expanded=True):
    st.caption(f"Run: {run_tag(reports, '04_user_split')}")

    split_cfg = m04.get("split", {})
    train_pct = split_cfg.get("train", 0)
    val_pct   = split_cfg.get("val", 0)
    test_pct  = split_cfg.get("test", 0)
    split_label = f"{int(train_pct*100)} / {int(val_pct*100)} / {int(test_pct*100)}"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", f"{m04.get('n_users_total', 0):,}")
    col2.metric(f"Train ({int(train_pct*100)}%)", f"{m04.get('n_users_train', 0):,} users")
    col3.metric(f"Val ({int(val_pct*100)}%)",     f"{m04.get('n_users_val', 0):,} users")
    col4.metric(f"Test ({int(test_pct*100)}%)",   f"{m04.get('n_users_test', 0):,} users")

    st.markdown(
        f"Split: **{split_label}** · "
        "No user appears in more than one split (**cold-start guarantee**). "
        "The model never observes test users during training."
    )
    if dataset == "MARS":
        st.warning(
            "MARS uses a **40/20/40** split instead of the standard 70/15/15. "
            "The standard split produces too few test users to form meaningful episodes "
            "given MARS's low average pair count (~6.2 pairs/user)."
        )

    # Users + pairs bar chart
    split_labels_chart = [f"Train\n({int(train_pct*100)}%)", f"Val\n({int(val_pct*100)}%)", f"Test\n({int(test_pct*100)}%)"]
    user_counts  = [m04.get("n_users_train", 0),  m04.get("n_users_val", 0),  m04.get("n_users_test", 0)]
    pair_counts  = [m04.get("n_pairs_train", 0),  m04.get("n_pairs_val", 0),  m04.get("n_pairs_test", 0)]
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        fig_u = px.bar(
            x=split_labels_chart, y=user_counts,
            labels={"x": "Split", "y": "Users"},
            color=split_labels_chart,
            color_discrete_sequence=["#2166AC", "#74ADD1", "#ABD9E9"],
            text=[f"{v:,}" for v in user_counts],
            title="Users per Split",
        )
        fig_u.update_traces(textposition="outside", showlegend=False)
        fig_u.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_u, use_container_width=True)

    with col_c2:
        fig_p = px.bar(
            x=split_labels_chart, y=pair_counts,
            labels={"x": "Split", "y": "Pairs"},
            color=split_labels_chart,
            color_discrete_sequence=["#2166AC", "#74ADD1", "#ABD9E9"],
            text=[f"{v:,}" for v in pair_counts],
            title="Pairs per Split",
        )
        fig_p.update_traces(textposition="outside", showlegend=False)
        fig_p.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_p, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# NB05 — Episode Index
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("**NB05 — Meta-Learning Episode Index**", expanded=True):
    st.caption(f"Run: {run_tag(reports, '05_episode_index')}")

    K = m05.get("K", 5)
    Q = m05.get("Q", 10)
    eps_train = m05.get("n_episodes_train", 0)
    eps_val   = m05.get("n_episodes_val", 0)
    eps_test  = m05.get("n_episodes_test", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Protocol", f"K={K} support + Q={Q} query")
    col2.metric("Train Episodes", f"{eps_train:,}")
    col3.metric("Val Episodes",   f"{eps_val:,}")
    col4.metric("Test Episodes",  f"{eps_test:,}")

    st.markdown(
        f"An **episode** is built per qualifying user:  \n"
        f"- **Support set** (K={K}): the chronologically earliest {K} pairs — used for inner-loop adaptation  \n"
        f"- **Query set** (Q={Q}): the next {Q} pairs — used for evaluation / outer-loop loss  \n"
        f"- Minimum {K + Q} pairs required per user to form one episode  \n"
        f"- Support timestamps are strictly earlier than query timestamps (**no temporal leakage**)"
    )

    if dataset == "MARS":
        st.warning(
            f"MARS has only **{eps_test} test episodes** — most users have fewer than "
            f"{K+Q} pairs and do not qualify. One episode difference equals "
            f"{1/eps_test*100:.1f}pp HR@10. Results are directionally informative only."
        )

    eps_per_user_test  = eps_test / m04.get("n_users_test", 1)
    eps_per_user_train = eps_train / m04.get("n_users_train", 1)

    col_chart, col_info = st.columns([3, 2])
    with col_chart:
        fig_eps = px.bar(
            x=["Train", "Val", "Test"],
            y=[eps_train, eps_val, eps_test],
            labels={"x": "Split", "y": "Episodes"},
            color=["Train", "Val", "Test"],
            color_discrete_sequence=["#1B7837", "#41AB5D", "#74C476"],
            text=[f"{v:,}" for v in [eps_train, eps_val, eps_test]],
            title=f"Episodes per Split (K={K}, Q={Q})",
        )
        fig_eps.update_traces(textposition="outside", showlegend=False)
        fig_eps.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_eps, use_container_width=True)

    with col_info:
        st.markdown("**Derived stats:**")
        st.markdown(
            f"- Train episodes / train user: **{eps_per_user_train:.2f}**  \n"
            f"- Test episodes / test user: **{eps_per_user_test:.2f}**  \n"
            f"- Total test query predictions: **{eps_test * Q:,}**  \n"
        )
