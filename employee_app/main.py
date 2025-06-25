from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Load model + preprocessors
with open("model.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
scaler = artifacts["scaler"]
selector = artifacts["selector"]
selected_columns = artifacts["selected_columns"]
le_dept = artifacts["le_dept"]
le_salary = artifacts["le_salary"]

# FastAPI app
app = FastAPI(title="Employee Retention Predictor")

# Define input schema
class EmployeeData(BaseModel):
    satisfaction_level: float
    average_montly_hours: int
    time_spend_company: int
    Work_accident: int
    promotion_last_5years: int

@app.post("/predict")
def predict(data: EmployeeData):
    # Build input data dict
    input_dict = {
        'satisfaction_level': data.satisfaction_level,
        'average_montly_hours': data.average_montly_hours,
        'time_spend_company': data.time_spend_company,
        'Work_accident': data.Work_accident,
        'promotion_last_5years': data.promotion_last_5years,
    }

    df_input = {col: [input_dict[col]] for col in input_dict}
    df_input = pd.DataFrame(df_input)

    # Select features
    df_selected = df_input[selected_columns]

    # Scale
    scaled_input = scaler.transform(df_selected)

    # Predict
    pred = model.predict(scaled_input)[0]
    return {"prediction": "Will Leave" if pred == 1 else "Will Stay"}
