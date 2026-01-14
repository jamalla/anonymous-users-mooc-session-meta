"""
Comparative Analysis Page - All MAML Variants Comparison
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
from utils import get_dataset_config, load_baseline_results, load_maml_results, load_all_maml_results, DATASETS

st.set_page_config(page_title="Comparative Analysis", page_icon="📊", layout="wide")

st.title("📊 Comparative Analysis")
st.markdown("Complete comparison of all MAML variants for cold-start recommendation")

# Dataset selector
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    options=list(DATASETS.keys()),
    index=0
)

config = get_dataset_config(dataset_name)
st.sidebar.info(f"**Dataset:** {config['name']}\n\n{config['description']}")

# Load all results
@st.cache_data
def load_all_data(dataset):
    baselines = load_baseline_results(dataset)
    all_maml = load_all_maml_results(dataset)
    return baselines, all_maml

baselines, all_maml = load_all_data(dataset_name)

# Executive Summary
st.header("📋 Executive Summary")

# Get baseline and best MAML results
baseline_acc = 0
if baselines:
    baseline_data = baselines.get("baselines", {})
    gru_metrics = baseline_data.get("gru_global", {})
    baseline_acc = gru_metrics.get("accuracy@1", 0)

# Build comparison data
comparison_data = []

# Add baselines
if baselines:
    baseline_data = baselines.get("baselines", {})
    for model_key, model_name in [("gru_global", "GRU Baseline")]:
        metrics = baseline_data.get(model_key, {})
        if metrics:
            comparison_data.append({
                "Model": model_name,
                "Type": "Baseline",
                "Accuracy@1": metrics.get("accuracy@1", 0),
                "Recall@5": metrics.get("recall@5", 0),
                "Recall@10": metrics.get("recall@10", 0),
                "MRR": metrics.get("mrr", 0),
                "vs_Baseline": 0.0
            })

# Add MAML variants
variant_names = {
    "warmstart": "MAML + Warm-Start (07e)",
    "residual": "MAML + Residual (07f)",
    "warmstart_residual": "MAML + WS + Res (07g)"
}

for variant_key, variant_name in variant_names.items():
    maml_data = all_maml.get(variant_key)
    if maml_data:
        few_shot = maml_data.get("few_shot_metrics", {})
        if few_shot:
            acc = few_shot.get("accuracy@1", 0)
            improvement = ((acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
            comparison_data.append({
                "Model": variant_name,
                "Type": "MAML",
                "Accuracy@1": acc,
                "Recall@5": few_shot.get("recall@5", 0),
                "Recall@10": few_shot.get("recall@10", 0),
                "MRR": few_shot.get("mrr", 0),
                "vs_Baseline": improvement
            })

if comparison_data:
    comp_df = pd.DataFrame(comparison_data)

    # Find best model
    maml_df = comp_df[comp_df["Type"] == "MAML"]
    if len(maml_df) > 0:
        best_idx = maml_df["Accuracy@1"].idxmax()
        best_model = comp_df.loc[best_idx, "Model"]
        best_acc = comp_df.loc[best_idx, "Accuracy@1"]
        best_improvement = comp_df.loc[best_idx, "vs_Baseline"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("GRU Baseline", f"{baseline_acc:.2%}")

        with col2:
            st.metric("Best MAML Variant", best_model.split("(")[0].strip())

        with col3:
            delta_color = "normal" if best_improvement > 0 else "inverse"
            st.metric("Improvement", f"{best_improvement:+.2f}%", delta=f"{best_acc:.2%} Acc@1")

# Main Comparison Table
st.header("📊 Full Results Comparison")

if comparison_data:
    # Format for display
    display_df = comp_df.copy()
    display_df["Accuracy@1"] = display_df["Accuracy@1"].apply(lambda x: f"{x:.2%}")
    display_df["Recall@5"] = display_df["Recall@5"].apply(lambda x: f"{x:.2%}")
    display_df["Recall@10"] = display_df["Recall@10"].apply(lambda x: f"{x:.2%}")
    display_df["MRR"] = display_df["MRR"].apply(lambda x: f"{x:.4f}")
    display_df["vs_Baseline"] = display_df["vs_Baseline"].apply(lambda x: f"{x:+.2f}%")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Grouped bar chart
    st.subheader("Accuracy@1 Comparison")

    colors = ["#3498db" if t == "Baseline" else "#2ecc71" if "WS + Res" in m else "#e74c3c"
              for t, m in zip(comp_df["Type"], comp_df["Model"])]

    fig = go.Figure(go.Bar(
        x=comp_df["Model"],
        y=comp_df["Accuracy@1"],
        marker_color=colors,
        text=comp_df["Accuracy@1"].apply(lambda x: f"{x:.2%}"),
        textposition="outside"
    ))

    fig.update_layout(
        title="Accuracy@1 Across All Models",
        yaxis_title="Accuracy@1",
        yaxis_tickformat=".0%",
        height=400
    )

    # Add baseline reference line
    fig.add_hline(y=baseline_acc, line_dash="dash", line_color="gray",
                  annotation_text=f"GRU Baseline: {baseline_acc:.2%}")

    st.plotly_chart(fig, use_container_width=True)

# Multi-metric comparison
st.header("📈 Multi-Metric Comparison")

if comparison_data:
    metrics_to_plot = ["Accuracy@1", "Recall@5", "Recall@10", "MRR"]

    melted = comp_df.melt(
        id_vars=["Model", "Type"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="All Metrics Comparison",
        color_discrete_sequence=["#3498db", "#e74c3c", "#9b59b6", "#2ecc71"]
    )
    fig.update_layout(yaxis_tickformat=".0%", height=500)
    st.plotly_chart(fig, use_container_width=True)

# Radar Chart
st.header("🎯 Radar Chart Comparison")

if comparison_data and len(comp_df) > 1:
    fig = go.Figure()

    categories = ["Accuracy@1", "Recall@5", "Recall@10", "MRR"]
    colors = ["#3498db", "#e74c3c", "#9b59b6", "#2ecc71"]

    for i, (_, row) in enumerate(comp_df.iterrows()):
        values = [row[cat] for cat in categories]
        values.append(values[0])  # Close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=row["Model"],
            line_color=colors[i % len(colors)],
            opacity=0.6
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(comp_df[categories].max())])
        ),
        showlegend=True,
        title="Model Performance Radar",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# MAML Variant Analysis
st.header("🔬 MAML Variant Analysis")

st.markdown("""
### Component Contribution Analysis

| Variant | Warm-Start | Residual | Result |
|---------|------------|----------|--------|
| 07e: Warm-Start | ✅ | ❌ | Good initialization, may overfit |
| 07f: Residual | ❌ | ✅ | Stable adaptation, weak start |
| 07g: Combined | ✅ | ✅ | **Best of both worlds** |
""")

# Component contribution bar chart
if len(maml_df) >= 2:
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="vs GRU Baseline",
        x=maml_df["Model"],
        y=maml_df["vs_Baseline"],
        text=maml_df["vs_Baseline"].apply(lambda x: f"{x:+.2f}%"),
        textposition="outside",
        marker_color=["#e74c3c" if x < 0 else "#2ecc71" for x in maml_df["vs_Baseline"]]
    ))

    fig.update_layout(
        title="Improvement vs GRU Baseline",
        yaxis_title="% Change",
        height=400
    )
    fig.add_hline(y=0, line_dash="solid", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

# Zero-shot vs Few-shot Comparison
st.header("🔄 Zero-Shot vs Few-Shot Adaptation")

zeroshot_data = []
fewshot_data = []

for variant_key, variant_name in variant_names.items():
    maml_data = all_maml.get(variant_key)
    if maml_data:
        zero_shot = maml_data.get("zero_shot_metrics", {})
        few_shot = maml_data.get("few_shot_metrics", {})

        if zero_shot and few_shot:
            zero_acc = zero_shot.get("accuracy@1", 0)
            few_acc = few_shot.get("accuracy@1", 0)
            gain = ((few_acc - zero_acc) / zero_acc * 100) if zero_acc > 0 else 0

            zeroshot_data.append({
                "Model": variant_name,
                "Zero-Shot": zero_acc,
                "Few-Shot": few_acc,
                "Adaptation Gain": gain
            })

if zeroshot_data:
    zs_df = pd.DataFrame(zeroshot_data)

    col1, col2 = st.columns(2)

    with col1:
        # Display table
        display_zs = zs_df.copy()
        display_zs["Zero-Shot"] = display_zs["Zero-Shot"].apply(lambda x: f"{x:.2%}")
        display_zs["Few-Shot"] = display_zs["Few-Shot"].apply(lambda x: f"{x:.2%}")
        display_zs["Adaptation Gain"] = display_zs["Adaptation Gain"].apply(lambda x: f"{x:+.1f}%")
        st.dataframe(display_zs, use_container_width=True, hide_index=True)

    with col2:
        # Grouped bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Zero-Shot",
            x=zs_df["Model"],
            y=zs_df["Zero-Shot"],
            marker_color="#e74c3c"
        ))

        fig.add_trace(go.Bar(
            name="Few-Shot (K=5)",
            x=zs_df["Model"],
            y=zs_df["Few-Shot"],
            marker_color="#2ecc71"
        ))

        fig.update_layout(
            barmode="group",
            title="Zero-Shot vs Few-Shot Performance",
            yaxis_title="Accuracy@1",
            yaxis_tickformat=".0%"
        )
        st.plotly_chart(fig, use_container_width=True)

# Key Takeaways
st.header("💡 Key Takeaways")

if comparison_data and len(maml_df) > 0:
    best_improvement = maml_df["vs_Baseline"].max()
    best_variant = maml_df.loc[maml_df["vs_Baseline"].idxmax(), "Model"]

    st.markdown(f"""
    ### Research Findings

    1. **Best Approach: {best_variant}**
       - Achieves **{best_improvement:+.2f}%** improvement over GRU baseline
       - Combines warm-start initialization with residual regularization

    2. **Warm-Start is Essential**
       - Pre-trained weights provide a strong starting point
       - Pure MAML from random init underperforms baseline

    3. **Residual Loss Helps Stability**
       - Prevents overfitting during adaptation
       - Most effective when combined with warm-start

    4. **Few-Shot Adaptation Works**
       - Significant improvement from zero-shot to few-shot
       - K=5 examples sufficient for adaptation

    ### Practical Recommendations

    - **Use Warm-Start + Residual (07g)** for production
    - **Early stopping** (checkpoint at iter 1000) prevents overfitting
    - **Tune inner_lr** (0.02 optimal for this dataset)
    - **K=5 support examples** is a good balance
    """)

# LaTeX Table Export
st.header("📄 Export for Publication")

if comparison_data:
    st.subheader("LaTeX Table")

    latex_df = comp_df[["Model", "Accuracy@1", "Recall@5", "MRR", "vs_Baseline"]].copy()

    # Find best values for bolding
    best_acc = latex_df["Accuracy@1"].max()
    best_recall = latex_df["Recall@5"].max()
    best_mrr = latex_df["MRR"].max()

    latex_table = r"""
\begin{table}[h]
\centering
\caption{Cold-Start Recommendation Performance on XuetangX Dataset}
\label{tab:results}
\begin{tabular}{lcccr}
\toprule
\textbf{Model} & \textbf{Acc@1} & \textbf{Recall@5} & \textbf{MRR} & \textbf{vs Baseline} \\
\midrule
"""

    for _, row in latex_df.iterrows():
        acc_str = f"{row['Accuracy@1']:.4f}"
        rec_str = f"{row['Recall@5']:.4f}"
        mrr_str = f"{row['MRR']:.4f}"
        vs_str = f"{row['vs_Baseline']:+.2f}\\%"

        # Bold best values
        if row["Accuracy@1"] == best_acc and row["Model"] != "GRU Baseline":
            acc_str = f"\\textbf{{{acc_str}}}"
        if row["Recall@5"] == best_recall and row["Model"] != "GRU Baseline":
            rec_str = f"\\textbf{{{rec_str}}}"
        if row["MRR"] == best_mrr and row["Model"] != "GRU Baseline":
            mrr_str = f"\\textbf{{{mrr_str}}}"

        model_name = row["Model"].replace("_", "\\_").replace("+", "+")
        latex_table += f"{model_name} & {acc_str} & {rec_str} & {mrr_str} & {vs_str} \\\\\n"

    latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""

    st.code(latex_table, language="latex")

    # CSV download
    st.subheader("CSV Download")
    csv = comp_df.to_csv(index=False)
    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name="maml_comparison_results.csv",
        mime="text/csv"
    )
