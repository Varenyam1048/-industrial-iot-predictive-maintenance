import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from risk import classify, RiskThresholds, DEFAULT_THRESHOLDS  # noqa: E402


def test_healthy_machine_is_green():
    zone = classify(failure_probability=0.05, predicted_rul_days=300, thresholds=DEFAULT_THRESHOLDS)
    assert zone == "Green"


def test_high_probability_forces_red_even_with_high_rul():
    zone = classify(failure_probability=0.9, predicted_rul_days=300, thresholds=DEFAULT_THRESHOLDS)
    assert zone == "Red"


def test_low_rul_forces_red_even_with_low_probability():
    zone = classify(failure_probability=0.01, predicted_rul_days=5, thresholds=DEFAULT_THRESHOLDS)
    assert zone == "Red"


def test_moderate_probability_is_yellow():
    zone = classify(failure_probability=0.25, predicted_rul_days=300, thresholds=DEFAULT_THRESHOLDS)
    assert zone == "Yellow"


def test_moderate_rul_is_yellow():
    zone = classify(failure_probability=0.01, predicted_rul_days=60, thresholds=DEFAULT_THRESHOLDS)
    assert zone == "Yellow"


def test_custom_thresholds_are_respected():
    custom = RiskThresholds(red_probability=0.3, yellow_probability=0.1, red_rul_days=10, yellow_rul_days=40)
    assert classify(0.35, 500, custom) == "Red"
    assert classify(0.15, 500, custom) == "Yellow"
    assert classify(0.05, 500, custom) == "Green"
