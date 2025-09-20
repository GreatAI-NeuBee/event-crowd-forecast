from pathlib import Path
import logging
import pandas as pd
import numpy as np
import json

from src.model_training.time_series import estimate_exits_from_entries, generate_forecast, train_prophet_per_gate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("time_series")

def train_inout_models(
    arrivals_df: pd.DataFrame,
    gate_col: str,
    time_col: str,
    count_col: str,
    save_dir: str = "models/forecasting"
) -> dict:
    """
    Offline step: trains Prophet models for all gates and saves them.
    Returns a dict of gate -> model path.
    """
    return train_prophet_per_gate(
        arrivals_df,
        gate_col=gate_col,
        time_col=time_col,
        count_col=count_col,
        save_dir=save_dir
    )

def update_inout_models(
    new_data: pd.DataFrame,
    gate_col: str,
    time_col: str,
    count_col: str,
    history_data_path: str = "data/simulations/event_entries_history.csv",
    save_dir: str = "models/forecasting"
) -> dict:
    """
    Update models with new observed data (not forecasts).
    Appends new data to historical file (with metadata) and retrains Prophet models
    using only the essential columns.
    """
    hist_path = Path(history_data_path)
    
    # 1. Load old history (if exists), else create empty
    if hist_path.exists():
        df_hist = pd.read_csv(hist_path, parse_dates=[time_col])
    else:
        df_hist = pd.DataFrame(columns=new_data.columns)

    # 2. Append new data
    df_all = pd.concat([df_hist, new_data], ignore_index=True)

    # 3. Drop exact duplicates (all columns match)
    df_all = df_all.drop_duplicates()

    # 4. Save updated history (preserve metadata)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(hist_path, index=False)

    # 5. Train Prophet only on the essential columns
    df_train = df_all[[gate_col, time_col, count_col]].copy()

    model_paths = train_prophet_per_gate(
        df_train,
        gate_col=gate_col,
        time_col=time_col,
        count_col=count_col,
        save_dir=save_dir
    )

    return model_paths


def forecast_inout(
    model_dir: str,
    gates: list[str] | None = None,
    schedule_start_time: pd.Timestamp = None,
    event_end_time: pd.Timestamp | None = None,
    method_exits: str = "mirror_delay",
    save_path: str = "models/forecasting/forecast.json",
    freq: str = "5min"
) -> dict:
    """
    Online step: loads trained models and produces forecasts for given gates.

    Args:
        model_dir: Directory containing trained Prophet models.
        gates: List of gate identifiers. If None, auto-detect from available models.
        schedule_start_time: Start time of the event (forecast will expand around it).
        event_end_time: End time of the event (needed for exit estimation).
        method_exits: Method for estimating exits.
        save_path: Where to save JSON results.
        freq: Forecast frequency.
    """
    model_dir = Path(model_dir)

    # Auto-detect available gates if not provided
    if gates is None:
        gates = [p.stem.replace("prophet_", "") for p in model_dir.glob("prophet_*.pkl")]
        logger.info(f"Auto-detected gates from models: {gates}")

    forecast_start = schedule_start_time - pd.Timedelta(hours=2.5)
    forecast_end = schedule_start_time + pd.Timedelta(hours=1.0)

    arrivals_fc = {}
    for gate in gates:
        model_path = model_dir / f"prophet_{gate}.pkl"
        if model_path.exists():
            arrivals_fc[gate] = generate_forecast(
                model_path=model_path,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
                freq=freq
            )
        else:
            logger.warning(f"No model found for gate {gate}. Skipping.")

    exits_fc = estimate_exits_from_entries(
        arrivals_fc,
        method=method_exits,
        event_end_time=event_end_time
    )

    result = {"arrivals": arrivals_fc, "exits": exits_fc}

    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"✅ Forecasts saved to {save_path}")
    return result