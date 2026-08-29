"""Shared Green / Yellow / Red risk-zone logic.

The thresholds are derived at training time from the validation set
(``reports/model_metrics.json``) so the same operating point that was
tuned during training is what the dashboard displays. If the metrics
report is unavailable, sane fixed defaults are used instead.
"""
from dataclasses import dataclass
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports" / "model_metrics.json"

ZONE_ORDER = ["Green", "Yellow", "Red"]

ZONE_COLORS = {
    "Green": "#2e7d32",
    "Yellow": "#f9a825",
    "Red": "#c62828",
}

ZONE_LABELS = {
    "Green": "Healthy",
    "Yellow": "Warning",
    "Red": "Critical",
}


@dataclass(frozen=True)
class RiskThresholds:
    """Boundaries used to bucket a machine into Green / Yellow / Red.

    A machine is Red if its failure probability is at or above
    ``red_probability`` OR its predicted RUL is at or below
    ``red_rul_days`` (whichever signal is worse wins). Yellow works the
    same way one tier down. Everything else is Green.
    """

    red_probability: float = 0.50
    yellow_probability: float = 0.20
    red_rul_days: float = 30.0
    yellow_rul_days: float = 90.0


DEFAULT_THRESHOLDS = RiskThresholds()


def load_thresholds(metrics_path: Path = METRICS_PATH) -> RiskThresholds:
    """Load tuned thresholds from the training report, falling back to defaults."""
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        zone = metrics.get("risk_zone_thresholds", {})
        return RiskThresholds(
            red_probability=zone.get("red_probability", DEFAULT_THRESHOLDS.red_probability),
            yellow_probability=zone.get("yellow_probability", DEFAULT_THRESHOLDS.yellow_probability),
            red_rul_days=zone.get("red_rul_days", DEFAULT_THRESHOLDS.red_rul_days),
            yellow_rul_days=zone.get("yellow_rul_days", DEFAULT_THRESHOLDS.yellow_rul_days),
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return DEFAULT_THRESHOLDS


def classify(
    failure_probability: float,
    predicted_rul_days: float,
    thresholds: RiskThresholds = DEFAULT_THRESHOLDS,
) -> str:
    """Return 'Red', 'Yellow', or 'Green' for a single machine's prediction."""
    if failure_probability >= thresholds.red_probability or predicted_rul_days <= thresholds.red_rul_days:
        return "Red"
    if failure_probability >= thresholds.yellow_probability or predicted_rul_days <= thresholds.yellow_rul_days:
        return "Yellow"
    return "Green"
