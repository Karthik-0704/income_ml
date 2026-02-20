import os
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

# 1. Configure MLflow credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = "skarthiksubramanian0704"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "263e741bed9dbcc09a222274fda5c798ed1984b9" 
mlflow.set_tracking_uri("https://dagshub.com/skarthiksubramanian0704/Income-Classification.mlflow")

app = FastAPI(title="Dual-Engine Census API", version="2.0")

# 2. Define BOTH Run IDs from DagsHub
# You must paste the specific 32-character Run IDs from your DagsHub experiments here
RUN_ID_UNBALANCED = "88126842597646fb85314d70ef18197a"
RUN_ID_BALANCED = "6f32a87ed3584be1900800b9a7231e71"

# Global dictionary to store the models
models = {
    "Unbalanced": None,
    "Balanced": None
}

@app.on_event("startup")
def load_models():
    """Loads both models into memory when the server starts."""
    try:
        print("[INFO] Loading Unbalanced Model...")
        models["Unbalanced"] = mlflow.pyfunc.load_model(f"runs:/{RUN_ID_UNBALANCED}/model")
        
        print("[INFO] Loading Balanced Model...")
        models["Balanced"] = mlflow.pyfunc.load_model(f"runs:/{RUN_ID_BALANCED}/model")
        
        print("[INFO] Both models successfully loaded into memory!")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")

# 3. Update the expected payload to include the 'mode' toggle
class IncomePredictionRequest(BaseModel):
    mode: str  # Expects "Balanced" or "Unbalanced" from the frontend
    data: List[Dict[str, Any]]

@app.post("/predict")
async def predict_income(request: IncomePredictionRequest):
    """Routes the data to the correct model based on the requested mode."""
    
    # Validate the requested mode
    if request.mode not in models or models[request.mode] is None:
        raise HTTPException(status_code=400, detail=f"Invalid or unloaded model mode: {request.mode}")

    try:
        # Convert incoming JSON into a DataFrame
        input_df = pd.DataFrame(request.data)

        # Route the data to the specifically requested model pipeline
        selected_model = models[request.mode]
        predictions = selected_model.predict(input_df)
        
        results = [{"row_index": i, "prediction": ">$50k" if pred == 1 else "<$50k"} for i, pred in enumerate(predictions)]

        return {
            "status": "success", 
            "engine_used": request.mode,
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    # This tells the script to start the server on port 8000 when executed
    uvicorn.run(app, host="0.0.0.0", port=8000)