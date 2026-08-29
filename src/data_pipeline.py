from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "factory_sensor_simulator_2040.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

# Optional sensors that are only fitted to some machine types and are
# frequently missing (NaN) rather than invalid.
SENSOR_COLUMNS = [
    "Laser_Intensity",
    "Hydraulic_Pressure_bar",
    "Coolant_Flow_L_min",
    "Heat_Index",
]

CURRENT_YEAR = 2040


def load_raw_data(path=RAW_DATA):
    return pd.read_csv(path)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Clean invalid sensor readings and engineer model-ready features.

    Kept deliberately as one deterministic, side-effect-free function so it
    can be reused identically by training, batch scoring, and the live
    dashboard (no train/serve skew).
    """
    df = df.copy()

    # --- Data quality -----------------------------------------------------
    # Physically impossible negative readings are treated as missing rather
    # than clipped, so they get imputed the same way as true sensor dropouts.
    df.loc[df["Vibration_mms"] < 0, "Vibration_mms"] = np.nan
    df.loc[df["Power_Consumption_kW"] < 0, "Power_Consumption_kW"] = np.nan

    # Structural missingness (a sensor a machine type doesn't have) is
    # informative on its own, so capture it as a feature before imputing.
    for col in SENSOR_COLUMNS:
        df[f"{col}_Available"] = df[col].notna().astype(int)

    # --- Core domain features (from the original pipeline) ----------------
    df["Machine_Age"] = CURRENT_YEAR - df["Installation_Year"]
    df["Maintenance_Overdue_Ratio"] = df["Last_Maintenance_Days_Ago"] / 365
    df["Failure_History_Rate"] = (
        df["Failure_History_Count"] / (df["Maintenance_History_Count"] + 1)
    )
    df["Maintenance_Frequency"] = (
        df["Maintenance_History_Count"] / (df["Operational_Hours"] + 1) * 10000
    )
    df["Sensor_Stress_Index"] = (
        df["Temperature_C"].abs()
        + df["Vibration_mms"].abs()
        + df["Sound_dB"]
        + df["Power_Consumption_kW"].abs()
    ) / 4

    # --- Additional engineered features ------------------------------------
    # Mechanical inefficiency: vibration that's high relative to the power
    # actually being drawn tends to indicate bearing/alignment wear.
    df["Vibration_Power_Ratio"] = df["Vibration_mms"] / (df["Power_Consumption_kW"].abs() + 1)

    # Fluid health: low oil/coolant levels together are a stronger signal
    # than either alone.
    df["Fluid_Level_Deficit"] = 100 - ((df["Oil_Level_pct"] + df["Coolant_Level_pct"]) / 2)

    # Recent electrical/control instability, normalized to a daily rate.
    df["Error_Rate_30d"] = df["Error_Codes_Last_30_Days"] / 30
    df["AI_Override_Rate"] = df["AI_Override_Events"] / (df["Operational_Hours"] + 1) * 10000

    # Older machines that are also overdue on maintenance compound risk
    # faster than either factor alone would suggest.
    df["Age_Overdue_Interaction"] = df["Machine_Age"] * df["Maintenance_Overdue_Ratio"]

    # Wear accumulated per unit of maintenance effort - low values mean the
    # machine is being run hard relative to how often it's serviced.
    df["Hours_Per_Maintenance"] = df["Operational_Hours"] / (df["Maintenance_History_Count"] + 1)

    # --- Impute (features only - targets are never imputed) ---------------
    # This is a best-effort fallback for batch/offline use. A column that is
    # entirely missing within a given batch (e.g. one machine scored alone)
    # has an undefined median, so it falls back to 0. The model pipeline's
    # own ColumnTransformer imputer (fit on the full training set at train
    # time) is the real safety net and runs downstream of this.
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median if pd.notna(median) else 0)

    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


def build_datasets(df: pd.DataFrame):
    X = df.drop(
        columns=["Machine_ID", "Failure_Within_7_Days", "Remaining_Useful_Life_days"]
    ).copy()

    # Convert boolean model input to numeric.
    X["AI_Supervision"] = X["AI_Supervision"].astype(int)

    y_failure = df["Failure_Within_7_Days"].astype(int)
    y_rul = df["Remaining_Useful_Life_days"].astype(float)
    return X, y_failure, y_rul