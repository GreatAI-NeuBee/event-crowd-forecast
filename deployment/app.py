# deployment/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib, json
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from prediction.forecast_pipeline import forecast_inout, update_inout_models


app = FastAPI(title="Event Crowd Model API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ------------------------- Request Schemas ----------------------------
#------------------- Time Series -----------------------
class UpdateModelsRequest(BaseModel):
    data: List[dict]   # [{gate: "A", timestamp: "...", count: 123}, ...]

class ForecastInOutRequest(BaseModel):
    gates: Optional[List[str]] = None
    schedule_start_time: str
    event_end_time: Optional[str] = None
    method_exits: str = "mirror_delay"
    freq: str = "5min"


# ------------------------ Endpoints ---------------------------
## -----------------------Time Series---------------------------
@app.post("/update_models")
def update_models(req: UpdateModelsRequest):
    try:
        df_new = pd.DataFrame(req.data)
        model_paths = update_inout_models(
            new_data=df_new,
            gate_col="gate",
            time_col="timestamp",
            count_col="count"
        )
        return {"message": "✅ Models updated", "models": model_paths}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating models: {str(e)}")


@app.post("/forecast_inout")
def forecast_inout_endpoint(req: ForecastInOutRequest):
    try:
        result = forecast_inout(
            model_dir="models/forecasting",
            gates=req.gates,
            schedule_start_time=pd.Timestamp(req.schedule_start_time),
            event_end_time=pd.Timestamp(req.event_end_time) if req.event_end_time else None,
            method_exits=req.method_exits,
            freq=req.freq,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating in/out forecast: {str(e)}")