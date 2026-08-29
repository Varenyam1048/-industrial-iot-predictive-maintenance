from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap

try:
    from .data_pipeline import clean_and_engineer, build_datasets
    from .risk import classify, load_thresholds
except ImportError:
    from data_pipeline import clean_and_engineer, build_datasets
    from risk import classify, load_thresholds

ROOT = Path(__file__).resolve().parents[1]

_failure_model = None
_rul_model = None
_explainer = None


def load_models():
    """Load (and cache) both trained pipelines."""
    global _failure_model, _rul_model
    if _failure_model is None or _rul_model is None:
        _failure_model = joblib.load(ROOT / "models" / "failure_model.joblib")
        _rul_model = joblib.load(ROOT / "models" / "rul_model.joblib")
    return _failure_model, _rul_model


def _get_explainer(failure_model):
    global _explainer
    if _explainer is None:
        _explainer = shap.TreeExplainer(failure_model.named_steps["model"])
    return _explainer


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Score raw machine rows: failure probability, predicted RUL, and risk zone."""
    prepared = clean_and_engineer(df)
    X, _, _ = build_datasets(prepared)
    failure_model, rul_model = load_models()

    failure_probability = failure_model.predict_proba(X)[:, 1]
    rul_days = np.clip(rul_model.predict(X), a_min=0, a_max=None)

    thresholds = load_thresholds()
    zones = [
        classify(p, r, thresholds) for p, r in zip(failure_probability, rul_days)
    ]

    return pd.DataFrame({
        "Failure_Probability": failure_probability,
        "Predicted_RUL_days": rul_days,
        "Risk_Zone": zones,
    })


def explain(df: pd.DataFrame, top_n: int = 5) -> list:
    """Return the top SHAP-driven risk factors for each row in df (raw columns).

    Output: one list of ``{"feature": ..., "value": ..., "shap": ...}`` dicts
    per input row, sorted by |SHAP value| descending. Positive SHAP values
    push the failure probability up (worse); negative push it down (better).
    """
    prepared = clean_and_engineer(df)
    X, _, _ = build_datasets(prepared)
    failure_model, _ = load_models()

    preprocessor = failure_model.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())

    explainer = _get_explainer(failure_model)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    results = []
    for row_idx in range(len(df)):
        order = np.argsort(-np.abs(shap_values[row_idx]))[:top_n]
        row_factors = []
        for j in order:
            name = feature_names[j]
            # Report the original raw value when it's a plain (non one-hot) column.
            raw_value = X.iloc[row_idx][name] if name in X.columns else None
            row_factors.append({
                "feature": name,
                "value": None if raw_value is None else float(raw_value) if isinstance(raw_value, (int, float, np.number)) else str(raw_value),
                "shap": float(shap_values[row_idx, j]),
            })
        results.append(row_factors)
    return results
