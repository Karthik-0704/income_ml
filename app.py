import os
import pandas as pd
import mlflow
import mlflow.artifacts
import gc
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

# 1. DagsHub & MLflow Configuration (Pulling from GitHub Secrets/Env)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
USERNAME = "skarthiksubramanian0704"

if TRACKING_URI and DAGSHUB_TOKEN:
    os.environ["MLFLOW_TRACKING_USERNAME"] = USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    mlflow.set_tracking_uri(TRACKING_URI)
else:
    print("Warning: MLflow credentials not found in environment. Local mode active.")

app = FastAPI(title="On-Demand Quad-Engine API", version="4.0")

# 2. Model Mapping (Update these Run IDs as you re-train)
RUN_IDS = {
    "XGBoost_Balanced": "eb8507ccd6c9472c94db2883904e4235",
    "XGBoost_Unbalanced": "8f0ce3abf9424c789b6dd1c040f07195",
    "Random Forest_Balanced": "e3e900f610af45bf9ed333da33eed8f7",
    "Random Forest_Unbalanced": "282e80fae8e964327a73e725e070499ce"
}

# Global state
current_model_key = None
loaded_model = None
LOCAL_STORAGE = "./model_cache"

# 3. Model Loading Logic
def load_specific_model(model_key: str):
    global loaded_model, current_model_key
    
    if current_model_key == model_key:
        return 
    
    try:
        # Clear RAM to prevent OOM (Out of Memory) errors on t2.small
        print(f"[MEMORY] Releasing {current_model_key}...")
        loaded_model = None
        gc.collect() 

        run_id = RUN_IDS[model_key]
        dest_folder = os.path.join(LOCAL_STORAGE, model_key)
        # MLflow downloads the artifact folder, so model.pkl is inside 'model/'
        pkl_file_path = os.path.join(dest_folder, "model", "model.pkl")

        # Download from DagsHub if not cached locally
        if not os.path.exists(pkl_file_path):
            print(f"[DISK] Downloading {model_key} pkl from DagsHub...")
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="model",
                dst_path=dest_folder
            )

        # Load using manual pickle
        print(f"[RAM] Loading {model_key} via Pickle...")
        with open(pkl_file_path, "rb") as f:
            loaded_model = pickle.load(f)
            
        current_model_key = model_key
        print(f"✅ {model_key} is now active.")
        
    except Exception as e:
        print(f"[ERROR] Failed to switch to {model_key}: {e}")
        raise HTTPException(status_code=500, detail=f"Model Swap Failed: {str(e)}")

# 4. Data Models
class IncomePredictionRequest(BaseModel):
    mode: str
    data: List[Dict[str, Any]]

# 5. Prediction Endpoint
@app.post("/predict")
async def predict_income(request: IncomePredictionRequest):
    if request.mode not in RUN_IDS:
        raise HTTPException(status_code=400, detail="Invalid model mode.")

    # Trigger the model swap/load
    load_specific_model(request.mode)

    try:
        # Convert incoming JSON to DataFrame for the sklearn pipeline
        input_df = pd.DataFrame(request.data)
        predictions = loaded_model.predict(input_df)
        
        # Map binary 0/1 back to human-readable labels
        results = [
            {"row_index": i, "prediction": ">$50k" if pred == 1 else "<$50k"} 
            for i, pred in enumerate(predictions)
        ]
        return {"status": "success", "engine": request.mode, "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)