import os
import pandas as pd
import mlflow
import mlflow.artifacts
import shutil
import gc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

# 1. DagsHub & MLflow Configuration
os.environ["MLFLOW_TRACKING_USERNAME"] = "skarthiksubramanian0704"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "263e741bed9dbcc09a222274fda5c798ed1984b9" 
mlflow.set_tracking_uri("https://dagshub.com/skarthiksubramanian0704/Income-Classification.mlflow")

app = FastAPI(title="On-Demand Quad-Engine API", version="4.0")

# 2. Your specific Run IDs
RUN_IDS = {
    "XGBoost_Balanced": "6f32a87ed3584be1900800b9a7231e71",
    "XGBoost_Unbalanced": "88126842597646fb85314d70ef18197a",
    "Random Forest_Balanced": "fe509cdc68c645e7885e208fd86421d3",
    "Random Forest_Unbalanced": "244f8b07576740e5a4c4d0cb9fa44993"
}

# Global state to manage the active model
current_model_key = None
loaded_model = None
LOCAL_STORAGE = "./model_cache"

def load_specific_model(model_key: str):
    global loaded_model, current_model_key
    
    if current_model_key == model_key:
        return # Already loaded, skip to inference
    
    try:
        # Clear RAM before loading a new heavy model
        print(f"[MEMORY] Releasing {current_model_key}...")
        loaded_model = None
        gc.collect() 

        run_id = RUN_IDS[model_key]
        model_path = os.path.join(LOCAL_STORAGE, model_key)

        # Download if not already on disk
        if not os.path.exists(model_path):
            print(f"[DISK] Downloading {model_key} from DagsHub...")
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="model",
                dst_path=model_path
            )

        # Load from local disk
        print(f"[RAM] Loading {model_key} from local storage...")
        loaded_model = mlflow.pyfunc.load_model(model_path)
        current_model_key = model_key
        
    except Exception as e:
        print(f"[ERROR] Failed to switch to {model_key}: {e}")
        raise HTTPException(status_code=500, detail=f"Model Swap Failed: {str(e)}")

class IncomePredictionRequest(BaseModel):
    mode: str
    data: List[Dict[str, Any]]

@app.post("/predict")
async def predict_income(request: IncomePredictionRequest):
    if request.mode not in RUN_IDS:
        raise HTTPException(status_code=400, detail="Invalid model mode.")

    load_specific_model(request.mode)

    try:
        input_df = pd.DataFrame(request.data)
        predictions = loaded_model.predict(input_df)
        
        results = [{"row_index": i, "prediction": ">$50k" if pred == 1 else "<$50k"} for i, pred in enumerate(predictions)]
        return {"status": "success", "engine": request.mode, "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)