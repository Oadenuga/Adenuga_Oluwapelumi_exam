from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import uvicorn

model= joblib.load("Adenuga_exam3\Model.pki")
scaler= joblib.load("Adenuga_exam3\scaler.pki")

app= FastAPI()

class wine_chemical_properties(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def Home():
     return {"Welcome to our Final Exams wine predictor"}

@app.post("/predict")
def predict_quality(payload: wine_chemical_properties):
    features = np.array([[payload.fixed_acidity, payload.volatile_acidity, payload.citric_acid,
                         payload.residual_sugar, payload.chlorides, payload.free_sulfur_dioxide,
                         payload.total_sulfur_dioxide, payload.density, payload.pH,
                         payload.sulphates, payload.alcohol]])



    scaled_features = scaler.transform(features)
    preds = model.predict(scaled_features)
    return {"predicted quality":str(preds[0])}

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8080
    uvicorn.run(app, host=host, port=port)
    
 






