# src/model_training/time_series.py
"""
Time series training utilities for arrivals and exits. Trains per-gate Prophet models and saves forecasts.
Also provides an XGBoost lag regressor fallback.

- TODO: Tune Prophet according to seasonality, holidays, events

"""
from pathlib import Path
import logging
import joblib
import pandas as pd
import numpy as np
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("time_series")

try:
    from prophet import Prophet
except Exception:
    Prophet = None

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
except Exception:
    xgb = None

def train_prophet_per_gate(
    df: pd.DataFrame,
    gate_col: str,
    time_col: str,
    count_col: str,
    save_dir: str = "models/forecasting"
) -> dict:
    """
    Trains and saves a Prophet model per gate using all available historical data.
    Does NOT generate a forecast. Use `generate_forecast` for that.

    Args:
        df: DataFrame containing historical data [gate_col, time_col, count_col]
        gate_col: Name of the column containing gate identifiers.
        time_col: Name of the column with datetime data.
        count_col: Name of the column with count data to forecast.
        save_dir: Directory to save models.

    Returns:
        dict: A dictionary mapping gate name -> path to saved model.
    """
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    model_paths = {}
    gates = df[gate_col].unique()

    for gate in gates:
        # Prepare the gate-specific data
        df_g = (
            df[df[gate_col] == gate]
            [[time_col, count_col]]
            .rename(columns={time_col: "ds", count_col: "y"})
            .sort_values("ds")
        )

        if df_g.empty:
            logger.warning(f"No data found for gate {gate}. Skipping.")
            model_paths[gate] = None
            continue

        if Prophet is None:
            logger.warning("Prophet not installed; skipping Prophet training.")
            model_paths[gate] = None
            continue

        # Train the model
        m = Prophet()
        m.fit(df_g)
        
        # Save the model
        model_filename = f"prophet_{gate}.pkl"
        model_path = p / model_filename
        joblib.dump(m, model_path)
        
        model_paths[gate] = str(model_path)
        logger.info(f"Trained and saved Prophet model for gate {gate} at {model_path}")

    return model_paths

def generate_forecast(
    model_path: str | Path,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    freq: str = "5min"
) -> list[dict]:
    """
    Generates a forecast for a specific time window using a pre-trained Prophet model.

    Args:
        model_path: Path to the saved joblib model file.
        forecast_start: Start timestamp for the forecast period.
        forecast_end: End timestamp for the forecast period.
        freq: Frequency of the forecast (e.g., '5min', '1H').

    Returns:
        list[dict]: Forecast results as a list of records with keys
                   'ds', 'yhat', 'yhat_lower', 'yhat_upper'.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load the model
    model = joblib.load(model_path)
    
    # Generate the date range for the requested window
    future_dates = pd.date_range(start=forecast_start, end=forecast_end, freq=freq)
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Generate the forecast
    forecast_df = model.predict(future_df)
    
    # Format the output
    result_df = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    
    # Convert timestamps to strings for JSON serialization
    result_df['ds'] = result_df['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return result_df.to_dict(orient="records")

def estimate_exits_from_entries(
    arrivals_fc: dict,
    method: str = "mirror_delay",
    event_end_time: pd.Timestamp | None = None,
    decay_rate: float = 0.02,
    transit_schedule: list[pd.Timestamp] | None = None
) -> dict:
    """
    Estimate exit forecasts when no exit data is available.

    Args:
        arrivals_fc: Dict of {gate -> list of forecasted arrivals}.
        method: "mirror_delay", "decay", or "transit_aligned".
        event_end_time: Required if using "mirror_delay" or "transit_aligned".
        decay_rate: Exponential decay rate for "decay" method.
        transit_schedule: List of public transport departure times.

    Returns:
        Dict of {gate -> list of forecasted exits}.
    """
    exits_fc = {}

    for gate, fc in arrivals_fc.items():
        df = pd.DataFrame(fc)
        df["ds"] = pd.to_datetime(df["ds"])

        if method == "mirror_delay":
            if event_end_time is None:
                raise ValueError("event_end_time required for mirror_delay")
            # Shift arrivals forward so that they align around event_end_time
            shift = event_end_time - df["ds"].min()
            df["ds"] = df["ds"] + shift
            exits_fc[gate] = df.to_dict(orient="records")

        elif method == "decay":
            peak_time = df.loc[df["yhat"].idxmax(), "ds"]
            df["minutes_after_peak"] = (df["ds"] - peak_time).dt.total_seconds() / 60
            df["yhat"] = df["yhat"].max() * np.exp(-decay_rate * df["minutes_after_peak"].clip(lower=0))
            df["yhat_lower"] = df["yhat"] * 0.8
            df["yhat_upper"] = df["yhat"] * 1.2
            exits_fc[gate] = df[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")

        elif method == "transit_aligned":
            if event_end_time is None or transit_schedule is None:
                raise ValueError("event_end_time and transit_schedule required for transit_aligned")
            df["yhat"] = 0.0
            for ts in transit_schedule:
                window_mask = (df["ds"] >= ts - pd.Timedelta(minutes=10)) & (df["ds"] <= ts)
                df.loc[window_mask, "yhat"] += df["yhat"].max() / len(transit_schedule)
            df["yhat_lower"] = df["yhat"] * 0.8
            df["yhat_upper"] = df["yhat"] * 1.2
            exits_fc[gate] = df[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")

        else:
            raise ValueError(f"Unknown method {method}")

        # Convert timestamp to string for JSON
        df["ds"] = df["ds"].dt.strftime('%Y-%m-%d %H:%M:%S')

        exits_fc[gate] = df[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")

    return exits_fc


# def train_xgb_lag(arrivals_df: pd.DataFrame, gate: str, time_col: str="ds", count_col: str="y", lags=(1,2,3,4,5), save_dir: str="models/forecasting"):
#     if xgb is None:
#         raise ImportError("xgboost not installed")
#     df = arrivals_df[arrivals_df["gate"]==gate].sort_values(time_col).reset_index(drop=True)
#     df[count_col] = df[count_col].astype(float)
#     for lag in lags:
#         df[f"lag_{lag}"] = df[count_col].shift(lag)
#     df = df.dropna().reset_index(drop=True)
#     features = [f"lag_{lag}" for lag in lags]
#     X = df[features].values
#     y = df[count_col].values
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#     model = xgb.XGBRegressor(n_estimators=200)
#     model.fit(X_train, y_train)
#     joblib.dump(model, Path(save_dir)/f"xgb_arrival_{gate}.pkl")
#     logger.info(f"Saved XGB arrival model for gate {gate}")
#     return model
