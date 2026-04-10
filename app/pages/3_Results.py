"""
3_Results.py — Model results: baselines, MAML progression, SRS validation (NB06–NB11).
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_dataset_reports, metrics, key_findings, run_tag, DATASET_PREFIX

st.set_page_config(page_title="Results | MOOC Recommendation", page_icon="📈", layout="wide")

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
    st.markdown("**Page:** 📈 Results")

reports = load_dataset_reports(dataset)

st.title(f"📈 Model Results — {dataset}")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🏁 Base Models", "🔬 MAML Experiments", "📊 SRS Validation"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Baseline Models (NB06)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Base Model Comparison")
    st.caption(f"Source: NB06 base model selection · Run: {run_tag(reports, '06_base_model_selection')}")

    m06 = metrics(reports, "06_base_model_selection")

    if not m06:
        st.warning("NB06 report not available for this dataset.")
    else:
        # ── Build model comparison table — only include models present in report ─
        all_model_names = ["random", "popularity", "session_knn", "sasrec", "gru4rec"]
        display_names = {
            "random": "Random",
            "popularity": "Popularity",
            "session_knn": "Session-KNN",
            "sasrec": "SASRec",
            "gru4rec": "GRU4Rec",
        }
        metric_keys = ["hr1", "hr5", "hr10", "ndcg10", "mrr"]
        metric_labels = ["HR@1", "HR@5", "HR@10", "NDCG@10", "MRR"]

        rows = []
        for mn in all_model_names:
            m_data = m06.get(mn, {})
            if m_data:
                row = {"Model": display_names[mn]}
                for mk, ml in zip(metric_keys, metric_labels):
                    row[ml] = m_data.get(mk, 0)
                rows.append(row)

        # Determine actual selected backbone (highest HR@10 among present models)
        best_model_key = max(
            [mn for mn in all_model_names if m06.get(mn)],
            key=lambda mn: m06.get(mn, {}).get("hr10", 0),
            default="gru4rec"
        )
        best_model_data = m06.get(best_model_key, {})
        best_model_name = display_names.get(best_model_key, best_model_key)

        if rows:
            df_base = pd.DataFrame(rows)

            # ── Grouped bar chart ─────────────────────────────────────────────
            fig_bar = go.Figure()
            colors = ["#2166AC", "#4393C3", "#92C5DE", "#D73027", "#F4A582"]
            for ml, color in zip(metric_labels, colors):
                fig_bar.add_trace(go.Bar(
                    name=ml,
                    x=df_base["Model"],
                    y=df_base[ml] * 100,
                    marker_color=color,
                    text=[f"{v*100:.1f}%" for v in df_base[ml]],
                    textposition="outside",
                ))
            fig_bar.update_layout(
                barmode="group",
                title="Base Model Performance Comparison",
                yaxis_title="Score (%)",
                xaxis_title="Model",
                height=420,
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Results table ─────────────────────────────────────────────────
            st.subheader("Full Results Table")
            df_display = df_base.copy()
            for ml in metric_labels:
                df_display[ml] = df_display[ml].apply(lambda v: f"{v*100:.2f}%")
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # ── Selected backbone callout — dynamic ───────────────────────────
            st.success(
                f"**Selected backbone: {best_model_name}** "
                f"(HR@10 = {best_model_data.get('hr10', 0)*100:.2f}%, "
                f"NDCG@10 = {best_model_data.get('ndcg10', 0)*100:.2f}%) — "
                "highest HR@10 among evaluated models. "
                "Pre-trained GRU4Rec weights saved for warm-start initialisation in NB08 and NB11."
            )

            # ── HR@10 ranking bar ─────────────────────────────────────────────
            df_ranked = df_base.sort_values("HR@10", ascending=True)
            fig_rank = px.bar(
                df_ranked, x="HR@10", y="Model",
                orientation="h",
                text=df_ranked["HR@10"].apply(lambda v: f"{v*100:.2f}%"),
                color="HR@10",
                color_continuous_scale="Blues",
                title="HR@10 Ranking",
            )
            fig_rank.update_traces(textposition="outside")
            fig_rank.update_layout(
                coloraxis_showscale=False,
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_tickformat=".0%",
            )
            st.plotly_chart(fig_rank, use_container_width=True)

        # ── Key findings ──────────────────────────────────────────────────────
        findings = key_findings(reports, "06_base_model_selection")
        if findings:
            st.subheader("Key Findings")
            for f in findings:
                st.markdown(f"- {f}")

        st.info(
            "**Note:** Base models (NB06) are trained on **full training data** — not cold-start constrained. "
            "These represent an upper bound on performance available to a system with complete user history. "
            "The MAML baseline and ablations (NB07–NB11) operate under the cold-start constraint: only K=5 support interactions."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MAML Experiments (NB07–NB11)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("MAML Model Progression")
    st.caption("Evaluation protocol: K=5 support, Q=10 query per episode (cold-start)")

    m06 = metrics(reports, "06_base_model_selection")
    m07 = metrics(reports, "07_standard_maml")
    m08 = metrics(reports, "08_warmstart_maml")
    m10 = metrics(reports, "10_srs_adaptive_maml")
    m11 = metrics(reports, "11_warmstart_srs_adaptive_maml")

    # Determine actual selected backbone dynamically
    all_model_keys = ["random", "popularity", "session_knn", "sasrec", "gru4rec"]
    disp = {"random": "Random", "popularity": "Popularity", "session_knn": "Session-KNN",
            "sasrec": "SASRec", "gru4rec": "GRU4Rec"}
    best_key = max(
        [k for k in all_model_keys if m06 and m06.get(k)],
        key=lambda k: m06.get(k, {}).get("hr10", 0),
        default="gru4rec"
    ) if m06 else "gru4rec"
    backbone_data = m06.get(best_key, {}) if m06 else {}
    backbone_label = disp.get(best_key, best_key)

    model_data = [backbone_data, m07, m08, m10 if m10 else {}, m11 if m11 else {}]
    colors_prog = ["#999999", "#4393C3", "#2166AC", "#D73027", "#B30000"]

    hr10_vals  = [d.get("hr10", 0)   if d else 0 for d in model_data]
    ndcg_vals  = [d.get("ndcg10", 0) if d else 0 for d in model_data]
    mrr_vals   = [d.get("mrr", 0)    if d else 0 for d in model_data]

    # ── Progression chart ─────────────────────────────────────────────────────
    short_labels = [f"{backbone_label}\n(NB06)", "NB07", "NB08", "NB10", "NB11 ★"]

    fig_prog = go.Figure()
    fig_prog.add_trace(go.Bar(
        name="HR@10",
        x=short_labels,
        y=[v * 100 for v in hr10_vals],
        marker_color=colors_prog,
        text=[f"{v*100:.2f}%" if v > 0 else "—" for v in hr10_vals],
        textposition="outside",
    ))
    fig_prog.add_trace(go.Scatter(
        name="NDCG@10",
        x=short_labels,
        y=[v * 100 for v in ndcg_vals],
        mode="lines+markers",
        line=dict(color="#F4A582", width=2, dash="dot"),
        marker=dict(size=8),
        yaxis="y",
    ))
    fig_prog.update_layout(
        title="Model Progression: HR@10 and NDCG@10",
        yaxis_title="Score (%)",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="group",
    )
    st.plotly_chart(fig_prog, use_container_width=True)

    # ── Full results table ─────────────────────────────────────────────────────
    st.subheader("Detailed Results Table")

    table_rows = []
    for label, tag, d in zip(
        [f"{backbone_label} (full data)", "Standard MAML", "Warm-Start MAML",
         "SRS-Adaptive MAML", "Warm-Start + SRS-Adapt"],
        ["NB06", "NB07", "NB08", "NB10", "NB11 ★"],
        model_data,
    ):
        if d:
            table_rows.append({
                "Model": label,
                "NB": tag,
                "HR@1":    f"{d.get('hr1', 0)*100:.2f}%",
                "HR@5":    f"{d.get('hr5', 0)*100:.2f}%",
                "HR@10":   f"{d.get('hr10', 0)*100:.2f}%",
                "NDCG@10": f"{d.get('ndcg10', 0)*100:.2f}%",
                "MRR":     f"{d.get('mrr', 0)*100:.2f}%",
            })
        else:
            table_rows.append({
                "Model": label, "NB": tag,
                "HR@1": "—", "HR@5": "—", "HR@10": "—", "NDCG@10": "—", "MRR": "—",
            })

    df_results = pd.DataFrame(table_rows)
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    # ── 2×2 Ablation Heatmap ──────────────────────────────────────────────────
    st.subheader("2×2 Ablation: Warm-Start × SRS-Adaptive (HR@10)")

    ws_off_srs_off = m07.get("hr10", 0) if m07 else 0   # NB07
    ws_on_srs_off  = m08.get("hr10", 0) if m08 else 0   # NB08
    ws_off_srs_on  = (m10.get("hr10", 0) if m10 else 0) # NB10
    ws_on_srs_on   = m11.get("hr10", 0) if m11 else 0   # NB11

    heatmap_z = [
        [ws_off_srs_off * 100, ws_on_srs_off * 100],
        [ws_off_srs_on  * 100, ws_on_srs_on  * 100],
    ]
    heatmap_text = [
        [f"NB07\n{ws_off_srs_off*100:.2f}%", f"NB08\n{ws_on_srs_off*100:.2f}%"],
        [f"NB10\n{ws_off_srs_on*100:.2f}%",  f"NB11 ★\n{ws_on_srs_on*100:.2f}%"],
    ]

    fig_hm = go.Figure(go.Heatmap(
        z=heatmap_z,
        x=["No Warm-Start", "Warm-Start"],
        y=["No SRS-Adaptive", "SRS-Adaptive"],
        text=heatmap_text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="HR@10 (%)"),
    ))
    fig_hm.update_layout(
        title="2×2 Ablation Heatmap — HR@10 (%)",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Warm-Start Initialisation",
        yaxis_title="SRS-Adaptive Inner Loop",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Gain analysis ─────────────────────────────────────────────────────────
    if m07 and m11:
        total_gain = (m11.get("hr10", 0) - m07.get("hr10", 0)) * 100
        ws_gain    = (m08.get("hr10", 0) - m07.get("hr10", 0)) * 100 if m08 else 0
        srs_alone  = (ws_off_srs_on - ws_off_srs_off) * 100
        combined   = (ws_on_srs_on  - ws_off_srs_off) * 100

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Total Gain (NB11 vs NB07)", f"{total_gain:+.2f}pp")
        col_b.metric("Warm-Start Only (NB08–NB07)", f"{ws_gain:+.2f}pp")
        col_c.metric("SRS-Alone (NB10–NB07)", f"{srs_alone:+.2f}pp")
        col_d.metric("Combined (NB11–NB07)", f"{combined:+.2f}pp")

    # ── Key findings ──────────────────────────────────────────────────────────
    for nb in ["07_standard_maml", "08_warmstart_maml", "10_srs_adaptive_maml",
               "11_warmstart_srs_adaptive_maml"]:
        findings = key_findings(reports, nb)
        if findings:
            nb_label = nb.replace("_", " ").replace("xuetangx", "").replace("mars", "").strip()
            with st.expander(f"Key findings — {nb_label}"):
                for f in findings:
                    st.markdown(f"- {f}")

    # ── MARS caveat ────────────────────────────────────────────────────────────
    if dataset == "MARS":
        m05 = metrics(reports, "05_episode_index")
        n_test = m05.get("n_episodes_test", 0) if m05 else 0
        if n_test < 50:
            ep_pp = round(100 / n_test, 2) if n_test > 0 else 0
            st.warning(
                f"**MARS test set is very small ({n_test} episodes).** "
                f"Each episode contributes ~{ep_pp:.2f} pp to HR@10. "
                "Results have high variance and should be interpreted with caution. "
                "Cross-dataset comparison requires acknowledging this limitation."
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SRS Validation (NB09)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Session Reliability Score (SRS) Validation")
    st.caption(f"Source: NB09 SRS validation · Run: {run_tag(reports, '09_srs_validation')}")

    m09 = metrics(reports, "09_srs_validation")

    if not m09:
        st.warning("NB09 SRS validation report not available for this dataset.")
    else:
        # ── Summary stats ─────────────────────────────────────────────────────
        st.subheader("SRS Score Statistics")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean SRS",   f"{m09.get('mean', 0):.4f}")
        col2.metric("Std Dev",    f"{m09.get('std', 0):.4f}")
        col3.metric("Min",        f"{m09.get('min', 0):.4f}")
        col4.metric("Max",        f"{m09.get('max', 0):.4f}")

        col5, col6, col7 = st.columns(3)
        col5.metric("25th pct",   f"{m09.get('p25', 0):.4f}")
        col6.metric("Median",     f"{m09.get('p50', 0):.4f}")
        col7.metric("75th pct",   f"{m09.get('p75', 0):.4f}")

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            # ── Tier breakdown ─────────────────────────────────────────────────
            st.subheader("SRS Tier Breakdown")
            tier_labels = ["Low (< 0.33)", "Medium (0.33–0.66)", "High (≥ 0.66)"]
            tier_vals = [
                m09.get("tier_low", 0) * 100,
                m09.get("tier_medium", 0) * 100,
                m09.get("tier_high", 0) * 100,
            ]
            fig_tier = px.bar(
                x=tier_labels, y=tier_vals,
                labels={"x": "SRS Tier", "y": "% of sessions"},
                color=tier_vals,
                color_continuous_scale=["#D73027", "#FDAE61", "#2166AC"],
                text=[f"{v:.1f}%" for v in tier_vals],
                title="Session Distribution by SRS Tier",
            )
            fig_tier.update_traces(textposition="outside")
            fig_tier.update_layout(
                coloraxis_showscale=False,
                height=320,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_tier, use_container_width=True)

        with col_right:
            # ── Correlation analysis ───────────────────────────────────────────
            st.subheader("Correlation with SRS")
            corr_labels = ["N events per session", "Session duration (sec)"]
            corr_vals   = [
                m09.get("corr_srs_n_events", 0),
                m09.get("corr_srs_duration", 0),
            ]

            fig_corr = px.bar(
                x=corr_labels, y=corr_vals,
                labels={"x": "Variable", "y": "Pearson r with SRS"},
                color=corr_vals,
                color_continuous_scale="RdBu",
                range_color=[-1, 1],
                text=[f"r={v:.3f}" for v in corr_vals],
                title="Correlation of SRS with Session Features",
            )
            fig_corr.update_traces(textposition="outside")
            fig_corr.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_corr.update_layout(
                coloraxis_showscale=False,
                height=320,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # ── Interpretation ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            """
            **SRS Interpretation:**
            - **Intensity** = min(n_events / CAP₉₅, 1.0) — how active the session was
            - **Extent** = min(duration_sec / CAP₉₅, 1.0) — how long the session lasted
            - **Composition** = action type diversity (XuetangX) or item diversity (MARS)
            - **SRS** = (Intensity + Extent + Composition) / 3 ∈ [0, 1]

            CAPs computed from **training sessions only** — no leakage into val/test.
            """
        )

        findings = key_findings(reports, "09_srs_validation")
        if findings:
            st.subheader("Key Findings")
            for f in findings:
                st.markdown(f"- {f}")
