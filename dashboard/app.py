from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.predict import predict, explain
from src.risk import load_thresholds, ZONE_COLORS, ZONE_LABELS

st.set_page_config(page_title="Industrial IoT Predictive Maintenance", layout="wide", page_icon="🏭")

if not (ROOT / "models/failure_model.joblib").exists() or not (ROOT / "models/rul_model.joblib").exists():
    st.error("Trained models are missing. Run: python src/train_models.py")
    st.stop()

THRESHOLDS = load_thresholds()


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data/raw/factory_sensor_simulator_2040.csv")


@st.cache_data(show_spinner="Scoring fleet...")
def score_fleet(df: pd.DataFrame) -> pd.DataFrame:
    scored = predict(df)
    return pd.concat([df[["Machine_ID", "Machine_Type"]].reset_index(drop=True), scored], axis=1)


def zone_badge(zone: str) -> str:
    color = ZONE_COLORS[zone]
    label = ZONE_LABELS[zone]
    return (
        f'<span style="background-color:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-weight:600;font-size:0.85em;">{zone} - {label}</span>'
    )


def gauge(value: float, title: str, max_value: float, zones: list, suffix: str = "") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix},
        title={"text": title},
        gauge={
            "axis": {"range": [0, max_value]},
            "bar": {"color": "#1f2937"},
            "steps": zones,
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


st.title("🏭 Industrial IoT Predictive Maintenance")
st.caption(
    "Failure risk and Remaining Useful Life (RUL) predictions from factory sensor data, "
    "with fleet-wide Green / Yellow / Red condition monitoring."
)

df = load_data()

tab_fleet, tab_machine, tab_model = st.tabs(["🚦 Fleet Overview", "🔍 Machine Inspector", "📊 Model Insights"])

# ---------------------------------------------------------------------------
# Fleet Overview
# ---------------------------------------------------------------------------
with tab_fleet:
    st.subheader("Fleet health at a glance")

    with st.spinner("Loading fleet scores..."):
        scored = score_fleet(df)

    zone_counts = scored["Risk_Zone"].value_counts().reindex(["Green", "Yellow", "Red"]).fillna(0).astype(int)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total machines", f"{len(scored):,}")
    c2.markdown(
        f"<div style='background:#e8f5e9;padding:14px;border-radius:10px;text-align:center;'>"
        f"<div style='font-size:1.6em;font-weight:700;color:#2e7d32;'>{zone_counts['Green']:,}</div>"
        f"<div style='color:#2e7d32;'>🟢 Healthy</div></div>", unsafe_allow_html=True,
    )
    c3.markdown(
        f"<div style='background:#fff8e1;padding:14px;border-radius:10px;text-align:center;'>"
        f"<div style='font-size:1.6em;font-weight:700;color:#f9a825;'>{zone_counts['Yellow']:,}</div>"
        f"<div style='color:#f9a825;'>🟡 Warning</div></div>", unsafe_allow_html=True,
    )
    c4.markdown(
        f"<div style='background:#ffebee;padding:14px;border-radius:10px;text-align:center;'>"
        f"<div style='font-size:1.6em;font-weight:700;color:#c62828;'>{zone_counts['Red']:,}</div>"
        f"<div style='color:#c62828;'>🔴 Critical</div></div>", unsafe_allow_html=True,
    )

    st.divider()

    filt1, filt2, filt3 = st.columns([2, 2, 3])
    with filt1:
        zone_filter = st.multiselect("Risk zone", ["Red", "Yellow", "Green"], default=["Red", "Yellow"])
    with filt2:
        type_filter = st.multiselect("Machine type", sorted(df["Machine_Type"].unique()))
    with filt3:
        sort_by = st.selectbox("Sort by", ["Failure_Probability", "Predicted_RUL_days"], index=0)

    view = scored.copy()
    if zone_filter:
        view = view[view["Risk_Zone"].isin(zone_filter)]
    if type_filter:
        view = view[view["Machine_Type"].isin(type_filter)]
    ascending = sort_by == "Predicted_RUL_days"
    view = view.sort_values(sort_by, ascending=ascending).head(300)

    display = view.copy()
    display["Failure_Probability"] = (display["Failure_Probability"] * 100).round(2).astype(str) + "%"
    display["Predicted_RUL_days"] = display["Predicted_RUL_days"].round(1)
    display["Zone"] = display["Risk_Zone"].map(lambda z: f"{z}")

    st.caption(f"Showing {len(view):,} of {len(scored):,} machines (top 300 by sort order)")

    def style_zone(row):
        color = ZONE_COLORS.get(row["Risk_Zone"], "#ffffff")
        return [f"background-color: {color}22"] * len(row)

    st.dataframe(
        display[["Machine_ID", "Machine_Type", "Risk_Zone", "Failure_Probability", "Predicted_RUL_days"]]
        .style.apply(style_zone, axis=1),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

# ---------------------------------------------------------------------------
# Machine Inspector
# ---------------------------------------------------------------------------
with tab_machine:
    st.subheader("Inspect a single machine")

    c1, c2 = st.columns(2)
    with c1:
        machine_type = st.selectbox("Machine type", sorted(df["Machine_Type"].unique()))
    subset = df[df["Machine_Type"] == machine_type].reset_index(drop=True)
    with c2:
        row_number = st.number_input("Record number", 0, max(len(subset) - 1, 0), 0, 1)

    row = subset.iloc[[int(row_number)]].copy()

    if st.button("Run prediction", type="primary"):
        result = predict(row).iloc[0]
        zone = result["Risk_Zone"]

        st.markdown(zone_badge(zone), unsafe_allow_html=True)
        st.write("")

        gc1, gc2 = st.columns(2)
        with gc1:
            fig = gauge(
                result["Failure_Probability"] * 100,
                "Failure probability (7 days)",
                100,
                zones=[
                    {"range": [0, THRESHOLDS.yellow_probability * 100], "color": ZONE_COLORS["Green"] + "55"},
                    {"range": [THRESHOLDS.yellow_probability * 100, THRESHOLDS.red_probability * 100], "color": ZONE_COLORS["Yellow"] + "55"},
                    {"range": [THRESHOLDS.red_probability * 100, 100], "color": ZONE_COLORS["Red"] + "55"},
                ],
                suffix="%",
            )
            st.plotly_chart(fig, use_container_width=True)
        with gc2:
            max_rul = max(200.0, float(result["Predicted_RUL_days"]) * 1.2)
            fig = gauge(
                result["Predicted_RUL_days"],
                "Predicted Remaining Useful Life (days)",
                max_rul,
                zones=[
                    {"range": [0, THRESHOLDS.red_rul_days], "color": ZONE_COLORS["Red"] + "55"},
                    {"range": [THRESHOLDS.red_rul_days, THRESHOLDS.yellow_rul_days], "color": ZONE_COLORS["Yellow"] + "55"},
                    {"range": [THRESHOLDS.yellow_rul_days, max_rul], "color": ZONE_COLORS["Green"] + "55"},
                ],
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Why this prediction? (top SHAP risk factors)")
        factors = explain(row, top_n=6)[0]
        factor_df = pd.DataFrame(factors)
        factor_df["direction"] = np.where(factor_df["shap"] >= 0, "Increases risk", "Decreases risk")
        factor_df = factor_df.sort_values("shap", key=abs, ascending=True)

        fig2 = go.Figure(go.Bar(
            x=factor_df["shap"],
            y=factor_df["feature"],
            orientation="h",
            marker_color=np.where(factor_df["shap"] >= 0, "#c62828", "#2e7d32"),
            text=[f"{v:.3g}" if isinstance(v, float) else str(v) for v in factor_df["value"]],
            textposition="outside",
        ))
        fig2.update_layout(
            height=280, margin=dict(l=10, r=60, t=20, b=10),
            xaxis_title="SHAP value (impact on failure probability)",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Bars to the right (red) push the failure risk up; bars to the left (green) push it down. Labels show the machine's actual sensor value for that feature.")

        st.subheader("Selected machine - raw readings")
        display_cols = [
            "Machine_ID", "Machine_Type", "Operational_Hours", "Temperature_C",
            "Vibration_mms", "Sound_dB", "Oil_Level_pct", "Coolant_Level_pct",
            "Power_Consumption_kW", "Last_Maintenance_Days_Ago",
            "Failure_History_Count", "Error_Codes_Last_30_Days",
        ]
        st.dataframe(row[display_cols], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Model Insights
# ---------------------------------------------------------------------------
with tab_model:
    st.subheader("Model performance & explainability")
    import json
    metrics_path = ROOT / "reports" / "model_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

        fp = metrics["failure_prediction"]
        rp = metrics["rul_prediction"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Failure ROC-AUC", f"{fp['test_metrics_tuned_threshold']['roc_auc']:.3f}")
        m2.metric("Failure F1 (tuned threshold)", f"{fp['test_metrics_tuned_threshold']['f1']:.3f}")
        m3.metric("RUL MAE (days)", f"{rp['test_metrics']['mae_days']:.1f}")
        m4.metric("RUL R²", f"{rp['test_metrics']['r2']:.3f}")

        st.caption(
            f"Operating threshold tuned on the validation split: "
            f"{fp['validation_threshold_tuning']['best_f1_threshold']:.3f} "
            f"(precision {fp['test_metrics_tuned_threshold']['precision']:.2f}, "
            f"recall {fp['test_metrics_tuned_threshold']['recall']:.2f} on the held-out test set)."
        )

        fig_dir = ROOT / "reports" / "figures"
        c1, c2 = st.columns(2)
        if (fig_dir / "failure_roc_curve.png").exists():
            c1.image(str(fig_dir / "failure_roc_curve.png"), caption="Failure model ROC curve")
        if (fig_dir / "failure_pr_curve.png").exists():
            c2.image(str(fig_dir / "failure_pr_curve.png"), caption="Failure model Precision-Recall curve")

        c3, c4 = st.columns(2)
        if (fig_dir / "failure_feature_importance.png").exists():
            c3.image(str(fig_dir / "failure_feature_importance.png"), caption="Failure model - global SHAP importance")
        if (fig_dir / "rul_feature_importance.png").exists():
            c4.image(str(fig_dir / "rul_feature_importance.png"), caption="RUL model - global SHAP importance")

        with st.expander("Full metrics report (JSON)"):
            st.json(metrics)
    else:
        st.info("Run `python src/train_models.py` to generate the metrics report.")

st.divider()
st.subheader("Dataset preview")
st.dataframe(df.head(20), use_container_width=True, hide_index=True)
