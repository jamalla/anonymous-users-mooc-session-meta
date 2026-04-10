"""
Home.py — Project overview page for the MOOC Meta-Learning Recommendation app.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_loader import load_dataset_reports, metrics, DATASET_PREFIX

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MOOC Cold-Start Recommendation",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Uniiversity_of_Malaya_Logo.png/240px-Uniiversity_of_Malaya_Logo.png", width=80, use_container_width=False)
    st.markdown("### 🎓 MOOC Recommendation")
    st.markdown("---")
    dataset = st.selectbox(
        "**Select Dataset**",
        options=list(DATASET_PREFIX.keys()),
        key="dataset",
        help="Choose the dataset to explore across all pages.",
    )
    st.markdown("---")
    st.caption("Navigate using the pages in the sidebar above.")

# ── Load top-level stats ──────────────────────────────────────────────────────
reports = load_dataset_reports(dataset)
m01 = metrics(reports, "01_ingest")
m05 = metrics(reports, "05_episode_index")
m11 = metrics(reports, "11_warmstart_srs_adaptive_maml")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center; color:#2166AC; margin-bottom:4px;'>
        A Meta-Learning Framework for Cold-Start Adaptation
    </h1>
    <h3 style='text-align:center; color:#555; font-weight:400; margin-top:0;'>
        Session-Based MOOC Recommendation for Anonymous Learners
    </h3>
    """,
    unsafe_allow_html=True,
)
st.markdown("")

col_a, col_b, col_c = st.columns(3)
col_a.info("**Author:** Jamallah M H Zawia")
col_b.info("**Supervisor:** Assoc. Prof. Dr Maizatul Akmar Binti Ismail")
col_c.info("**Institution:** FCSIT, University of Malaya")

st.markdown("---")

# ── Quick stats for selected dataset ─────────────────────────────────────────
st.subheader(f"📌 Dataset at a Glance — {dataset}")

if dataset == "XuetangX":
    raw_key, raw_label = "n_events", "Raw Events"
else:
    raw_key, raw_label = "n_interactions", "Raw Interactions"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(raw_label, f"{m01.get(raw_key, m01.get('n_events', 0)):,}")
c2.metric("Raw Users", f"{m01.get('n_users', 0):,}")
c3.metric("Vocabulary", f"{m01.get('n_items', m01.get('n_courses', 0)):,} items")
c4.metric("Test Episodes", f"{m05.get('n_episodes_test', 0):,}")
if m11:
    c5.metric("Best HR@10 (C3)", f"{m11.get('hr10', 0)*100:.2f}%")
else:
    c5.metric("Best HR@10 (C3)", "—")

st.markdown("---")

# ── Problem statement ─────────────────────────────────────────────────────────
col_l, col_r = st.columns([3, 2])

with col_l:
    st.subheader("🔍 Problem Statement")
    st.markdown(
        """
        MOOC platforms serve **millions of anonymous learners** who leave no persistent identity trace.
        Standard recommender systems require long interaction histories to personalise recommendations.

        This work addresses **cold-start recommendation** for anonymous MOOC learners:
        - Only **K = 5 session interactions** are available at inference time
        - No user profile, no enrollment history, no demographic information
        - The system must adapt in real time based solely on within-session behaviour

        **Key challenge:** Session quality varies enormously — a 5-minute distracted browse
        carries far less learning signal than a focused 90-minute problem-solving session.
        Standard MAML treats all sessions equally, wasting adaptation capacity on noisy data.
        """
    )

    st.subheader("🎯 Research Questions")
    rqs = [
        ("RQ1", "How do existing meta-learning approaches handle session quality in cold-start recommendation?"),
        ("RQ2", "How can session reliability be measured from clickstream data and used to condition MAML inner-loop adaptation?"),
        ("RQ3", "Does conditioning inner-loop adaptation on SRS improve cold-start performance over standard MAML?"),
    ]
    for rid, rq in rqs:
        st.markdown(f"**{rid}** — {rq}")

with col_r:
    st.subheader("💡 Contributions")
    st.markdown(
        """
        | ID | Contribution |
        |---|---|
        | **C1** | **Session Reliability Score (SRS)** — composite metric from intensity, extent, and composition of clickstream events. Normalised to [0, 1]. |
        | **C2** | **SRS-Adaptive MAML** — task-specific inner-loop rate αᵢ = α_base × SRSᵢ and step count Kᵢ = f(SRSᵢ, τ). High-quality sessions adapt faster; noisy sessions adapt more cautiously. |
        | **C3** | **Warm-Start + SRS-Adaptive MAML** — combines pretrained GRU4Rec initialisation (C1) with the SRS-conditioned inner loop (C2). The full proposed framework. |
        """
    )

    st.subheader("📐 SRS Formula")
    st.code(
        """SRS = (Intensity + Extent + Composition) / 3

Intensity   = min(n_events / CAP_95, 1.0)
Extent      = min(duration_sec / CAP_95, 1.0)
Composition = action_type_diversity  # XuetangX
            = item_diversity          # MARS (no action types)

CAPs computed from training sessions only (no leakage).""",
        language="python",
    )

st.markdown("---")

# ── Experiment overview ───────────────────────────────────────────────────────
st.subheader("🧪 Experiment Overview")

exp_data = {
    "Notebook": ["NB06", "NB07", "NB08", "NB10", "NB11 ★"],
    "Model": [
        "Baselines (Random / Popularity / KNN / GRU4Rec)",
        "Standard MAML",
        "Warm-Start MAML",
        "SRS-Adaptive MAML",
        "Warm-Start + SRS-Adaptive MAML",
    ],
    "Role": [
        "Performance ceiling (full training data)",
        "MAML benchmark (random init)",
        "Ablation — warm-start only (C1)",
        "Ablation — SRS-adaptive only (C2)",
        "Main contribution (C1 + C2)",
    ],
}

if m11:
    m07 = metrics(reports, "07_standard_maml")
    m08 = metrics(reports, "08_warmstart_maml")
    m10 = metrics(reports, "10_srs_adaptive_maml")
    m06 = metrics(reports, "06_base_model_selection")
    gru = m06.get("gru4rec", {})
    hr10_vals = [
        f"{gru.get('hr10', 0)*100:.2f}%",
        f"{m07.get('hr10', 0)*100:.2f}%",
        f"{m08.get('hr10', 0)*100:.2f}%",
        f"{m10.get('hr10', 0)*100:.2f}%" if m10 else "—",
        f"{m11.get('hr10', 0)*100:.2f}%",
    ]
    exp_data["HR@10"] = hr10_vals

import pandas as pd
df_exp = pd.DataFrame(exp_data)
st.dataframe(df_exp, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "Navigate using the sidebar pages: **EDA** → **Data Processing** → **Model Results** → **Comparative Analysis**. "
    "Use the **Select Dataset** dropdown to switch between XuetangX and MARS."
)
