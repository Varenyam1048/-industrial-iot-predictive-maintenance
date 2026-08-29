"""Train the failure-classification and RUL-regression models.

Compared to a "fit once, evaluate once" baseline, this script:

1. Splits data into train / validation / test (60/20/20), stratified on
   the failure label, so the test set is only ever touched once, at the
   very end.
2. Tunes hyperparameters with ``RandomizedSearchCV`` (3-fold CV, scored on
   average precision / R^2, since failure is a rare/imbalanced class).
3. Uses the validation split - never the test split - to pick an
   operating threshold (best-F1 and a high-precision alternative) instead
   of hard-coding 0.5.
4. Refits the tuned model on train+validation before final scoring, so the
   shipped model sees as much data as possible while the test set stays
   untouched for the headline numbers.
5. Explains both models with SHAP (TreeExplainer) and saves global
   importance plots + ROC/PR curves to reports/.
6. Derives Green/Yellow/Red risk-zone thresholds from the tuned
   probability threshold and writes them to reports/model_metrics.json so
   the dashboard and training stay in sync.
"""
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, confusion_matrix,
    precision_recall_curve, roc_curve,
)

try:
    from .data_pipeline import load_raw_data, clean_and_engineer, build_datasets
except ImportError:
    from data_pipeline import load_raw_data, clean_and_engineer, build_datasets

import shap

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
RANDOM_STATE = 42
SHAP_SAMPLE_SIZE = 3000


def make_preprocessor(X):
    categorical = ["Machine_Type"]
    numeric = [c for c in X.columns if c not in categorical]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def best_thresholds(y_true, y_prob, min_precision=0.75):
    """Pick a best-F1 operating threshold and a higher-precision alternative."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision/recall have one extra point (for threshold=inf); align them.
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0)

    best_idx = int(np.argmax(f1))
    best_f1_threshold = float(thresholds[best_idx])

    high_precision_candidates = np.where(precision >= min_precision)[0]
    if len(high_precision_candidates) > 0:
        # Among thresholds hitting the precision bar, take the one with the
        # best recall (i.e. the least conservative one that still qualifies).
        hp_idx = high_precision_candidates[np.argmax(recall[high_precision_candidates])]
        high_precision_threshold = float(thresholds[hp_idx])
    else:
        high_precision_threshold = float(np.clip(best_f1_threshold * 1.6, 0.05, 0.95))

    return {
        "best_f1_threshold": best_f1_threshold,
        "precision_at_best_f1": float(precision[best_idx]),
        "recall_at_best_f1": float(recall[best_idx]),
        "f1_at_best_f1": float(f1[best_idx]),
        "high_precision_threshold": high_precision_threshold,
    }


def classification_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def shap_global_importance(fitted_pipeline, X_sample, top_n=15):
    """Mean |SHAP value| per raw feature, aggregating one-hot columns back
    to their original column (e.g. all Machine_Type_* -> Machine_Type)."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X_sample)
    feature_names = list(preprocessor.get_feature_names_out())

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):  # binary classifiers sometimes return [class0, class1]
        shap_values = shap_values[-1]
    if shap_values.ndim == 3:  # (n_samples, n_features, n_classes)
        shap_values = shap_values[:, :, -1]

    mean_abs = np.abs(shap_values).mean(axis=0)

    # Collapse one-hot encoded Machine_Type_* columns back into "Machine_Type".
    collapsed = {}
    for name, val in zip(feature_names, mean_abs):
        base = "Machine_Type" if name.startswith("Machine_Type") else name
        collapsed[base] = collapsed.get(base, 0.0) + float(val)

    ranked = sorted(collapsed.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return ranked


def save_bar_chart(ranked_importance, title, path):
    labels = [k for k, _ in ranked_importance][::-1]
    values = [v for _, v in ranked_importance][::-1]
    plt.figure(figsize=(8, max(4, len(labels) * 0.35)))
    plt.barh(labels, values, color="#3b6ea5")
    plt.xlabel("Mean |SHAP value| (impact on model output)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def save_roc_pr_curves(y_true, y_prob, roc_path, pr_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color="#3b6ea5", label=f"ROC-AUC = {roc_auc_score(y_true, y_prob):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Failure Model - ROC Curve (test set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=140)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, color="#c62828",
              label=f"AP = {average_precision_score(y_true, y_prob):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Failure Model - Precision-Recall Curve (test set)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=140)
    plt.close()


def tune_classifier(X_train, y_train, sample_weight):
    preprocessor = make_preprocessor(X_train)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])
    param_distributions = {
        "model__max_iter": [100, 150, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__l2_regularization": [0.0, 0.5, 1.0, 2.0],
        "model__min_samples_leaf": [20, 50, 100],
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="average_precision",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train, model__sample_weight=sample_weight)
    return search


def tune_regressor(X_train, y_train):
    preprocessor = make_preprocessor(X_train)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])
    param_distributions = {
        "model__max_iter": [150, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__l2_regularization": [0.0, 0.5, 1.0, 2.0],
        "model__min_samples_leaf": [20, 50, 100],
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="r2",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def main():
    t_start = time.time()
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)

    print("Loading and engineering features...")
    df = clean_and_engineer(load_raw_data())
    X, y_failure, y_rul = build_datasets(df)

    # 60 / 20 / 20 train / validation / test split.
    X_temp, X_test, yf_temp, yf_test, yr_temp, yr_test = train_test_split(
        X, y_failure, y_rul, test_size=0.20, random_state=RANDOM_STATE, stratify=y_failure,
    )
    X_train, X_val, yf_train, yf_val, yr_train, yr_val = train_test_split(
        X_temp, yf_temp, yr_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=yf_temp,
    )  # 0.25 * 0.8 = 0.20 of the full set

    X_trainval = pd.concat([X_train, X_val])
    yf_trainval = pd.concat([yf_train, yf_val])
    yr_trainval = pd.concat([yr_train, yr_val])

    split_sizes = {
        "train": len(X_train), "validation": len(X_val), "test": len(X_test),
        "train_pct": round(len(X_train) / len(X) * 100, 1),
        "validation_pct": round(len(X_val) / len(X) * 100, 1),
        "test_pct": round(len(X_test) / len(X) * 100, 1),
    }
    print(f"Split sizes: {split_sizes}")

    # ------------------------------------------------------------------
    # Failure classification
    # ------------------------------------------------------------------
    print("\nTuning failure classifier (RandomizedSearchCV, 3-fold)...")
    train_weights = compute_sample_weight(class_weight="balanced", y=yf_train)
    search = tune_classifier(X_train, yf_train, train_weights)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV average precision: {search.best_score_:.4f}")

    print("Selecting operating threshold on the validation set...")
    val_prob = search.best_estimator_.predict_proba(X_val)[:, 1]
    threshold_info = best_thresholds(yf_val, val_prob)
    print(f"Threshold info: {threshold_info}")

    print("Refitting best configuration on train+validation...")
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    final_preprocessor = make_preprocessor(X_trainval)
    final_classifier = Pipeline([
        ("preprocessor", final_preprocessor),
        ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE, **best_params)),
    ])
    trainval_weights = compute_sample_weight(class_weight="balanced", y=yf_trainval)
    final_classifier.fit(X_trainval, yf_trainval, model__sample_weight=trainval_weights)

    test_prob = final_classifier.predict_proba(X_test)[:, 1]
    test_metrics_default = classification_metrics(yf_test, test_prob, 0.5)
    test_metrics_tuned = classification_metrics(yf_test, test_prob, threshold_info["best_f1_threshold"])
    print(f"Test metrics @0.5: {test_metrics_default}")
    print(f"Test metrics @tuned threshold: {test_metrics_tuned}")

    joblib.dump(final_classifier, MODEL_DIR / "failure_model.joblib")

    print("Computing SHAP global importance for the failure model...")
    shap_sample = X_test.sample(n=min(SHAP_SAMPLE_SIZE, len(X_test)), random_state=RANDOM_STATE)
    failure_importance = shap_global_importance(final_classifier, shap_sample)
    save_bar_chart(
        failure_importance, "Failure Model - Top Feature Importance (SHAP)",
        FIGURE_DIR / "failure_feature_importance.png",
    )
    save_roc_pr_curves(
        yf_test, test_prob,
        FIGURE_DIR / "failure_roc_curve.png", FIGURE_DIR / "failure_pr_curve.png",
    )

    # ------------------------------------------------------------------
    # RUL regression
    # ------------------------------------------------------------------
    print("\nTuning RUL regressor (RandomizedSearchCV, 3-fold)...")
    reg_search = tune_regressor(X_train, yr_train)
    print(f"Best params: {reg_search.best_params_}")
    print(f"Best CV R2: {reg_search.best_score_:.4f}")

    print("Refitting best configuration on train+validation...")
    reg_best_params = {k.replace("model__", ""): v for k, v in reg_search.best_params_.items()}
    final_reg_preprocessor = make_preprocessor(X_trainval)
    final_regressor = Pipeline([
        ("preprocessor", final_reg_preprocessor),
        ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE, **reg_best_params)),
    ])
    final_regressor.fit(X_trainval, yr_trainval)

    rul_pred = final_regressor.predict(X_test)
    rul_metrics = {
        "mae_days": float(mean_absolute_error(yr_test, rul_pred)),
        "rmse_days": float(mean_squared_error(yr_test, rul_pred) ** 0.5),
        "r2": float(r2_score(yr_test, rul_pred)),
    }
    print(f"Test metrics: {rul_metrics}")

    joblib.dump(final_regressor, MODEL_DIR / "rul_model.joblib")

    print("Computing SHAP global importance for the RUL model...")
    rul_importance = shap_global_importance(final_regressor, shap_sample)
    save_bar_chart(
        rul_importance, "RUL Model - Top Feature Importance (SHAP)",
        FIGURE_DIR / "rul_feature_importance.png",
    )

    # ------------------------------------------------------------------
    # Risk zone thresholds (feed the dashboard)
    # ------------------------------------------------------------------
    risk_zone_thresholds = {
        "red_probability": threshold_info["high_precision_threshold"],
        "yellow_probability": threshold_info["best_f1_threshold"],
        "red_rul_days": 30.0,
        "yellow_rul_days": 90.0,
    }

    metrics = {
        "generated_with": "train_models.py (tuned + validated pipeline)",
        "random_state": RANDOM_STATE,
        "data_split": split_sizes,
        "failure_prediction": {
            "model": "HistGradientBoostingClassifier",
            "best_hyperparameters": best_params,
            "cv_average_precision_mean": float(search.best_score_),
            "cv_average_precision_std": float(
                search.cv_results_["std_test_score"][search.best_index_]
            ),
            "validation_threshold_tuning": threshold_info,
            "test_metrics_default_threshold": test_metrics_default,
            "test_metrics_tuned_threshold": test_metrics_tuned,
            "top_shap_features": [{"feature": k, "mean_abs_shap": v} for k, v in failure_importance],
        },
        "rul_prediction": {
            "model": "HistGradientBoostingRegressor",
            "best_hyperparameters": reg_best_params,
            "cv_r2_mean": float(reg_search.best_score_),
            "cv_r2_std": float(reg_search.cv_results_["std_test_score"][reg_search.best_index_]),
            "test_metrics": rul_metrics,
            "top_shap_features": [{"feature": k, "mean_abs_shap": v} for k, v in rul_importance],
        },
        "risk_zone_thresholds": risk_zone_thresholds,
    }
    with open(REPORT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone in {time.time() - t_start:.1f}s. Models saved in models/, report in reports/.")


if __name__ == "__main__":
    main()
