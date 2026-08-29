import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import clean_and_engineer, build_datasets, SENSOR_COLUMNS  # noqa: E402


def make_sample_df(n=20):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Machine_ID": [f"MC_{i:06d}" for i in range(n)],
        "Machine_Type": rng.choice(["Mixer", "Industrial_Chiller"], n),
        "Installation_Year": rng.integers(2015, 2040, n),
        "Operational_Hours": rng.integers(0, 100000, n),
        "Temperature_C": rng.uniform(20, 100, n),
        "Vibration_mms": rng.uniform(-1, 20, n),  # includes invalid negatives
        "Sound_dB": rng.uniform(50, 100, n),
        "Oil_Level_pct": rng.uniform(0, 100, n),
        "Coolant_Level_pct": rng.uniform(0, 100, n),
        "Power_Consumption_kW": rng.uniform(-1, 200, n),  # includes invalid negatives
        "Last_Maintenance_Days_Ago": rng.integers(0, 400, n),
        "Maintenance_History_Count": rng.integers(0, 20, n),
        "Failure_History_Count": rng.integers(0, 10, n),
        "AI_Supervision": rng.choice([True, False], n),
        "Error_Codes_Last_30_Days": rng.integers(0, 10, n),
        "Remaining_Useful_Life_days": rng.uniform(0, 400, n),
        "Failure_Within_7_Days": rng.choice([True, False], n),
        "Laser_Intensity": [np.nan] * n,
        "Hydraulic_Pressure_bar": rng.uniform(0, 300, n),
        "Coolant_Flow_L_min": [np.nan] * n,
        "Heat_Index": rng.uniform(0, 100, n),
        "AI_Override_Events": rng.integers(0, 5, n),
    })


def test_negative_readings_are_treated_as_missing_then_imputed():
    df = make_sample_df()
    out = clean_and_engineer(df)
    assert (out["Vibration_mms"] >= 0).all()
    assert (out["Power_Consumption_kW"] >= 0).all()


def test_sensor_availability_flags_created():
    df = make_sample_df()
    out = clean_and_engineer(df)
    for col in SENSOR_COLUMNS:
        assert f"{col}_Available" in out.columns
    # Laser_Intensity and Coolant_Flow_L_min were entirely missing in the sample.
    assert out["Laser_Intensity_Available"].sum() == 0
    assert out["Coolant_Flow_L_min_Available"].sum() == 0


def test_no_missing_values_after_cleaning():
    df = make_sample_df()
    out = clean_and_engineer(df)
    assert out.isna().sum().sum() == 0


def test_engineered_features_present():
    df = make_sample_df()
    out = clean_and_engineer(df)
    expected = [
        "Machine_Age", "Maintenance_Overdue_Ratio", "Failure_History_Rate",
        "Maintenance_Frequency", "Sensor_Stress_Index", "Vibration_Power_Ratio",
        "Fluid_Level_Deficit", "Error_Rate_30d", "AI_Override_Rate",
        "Age_Overdue_Interaction", "Hours_Per_Maintenance",
    ]
    for col in expected:
        assert col in out.columns


def test_build_datasets_excludes_targets_and_id():
    df = make_sample_df()
    cleaned = clean_and_engineer(df)
    X, y_failure, y_rul = build_datasets(cleaned)
    for leak_col in ["Machine_ID", "Failure_Within_7_Days", "Remaining_Useful_Life_days"]:
        assert leak_col not in X.columns
    assert y_failure.isin([0, 1]).all()
    assert (y_rul >= 0).all()


def test_build_datasets_converts_boolean_to_numeric():
    df = make_sample_df()
    cleaned = clean_and_engineer(df)
    X, _, _ = build_datasets(cleaned)
    assert pd.api.types.is_numeric_dtype(X["AI_Supervision"])
