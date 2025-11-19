from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_no_show

app = FastAPI(title="No-Show Risk Predictor API")

class AppointmentInput(BaseModel):
    Age: int = Field(..., ge=0, le=120)
    WaitingDays: int = Field(..., ge=-30, le=365)
    Scholarship: int  # 0 or 1
    Hipertension: int
    Diabetes: int
    Alcoholism: int
    Handcap: int
    SMS_received: int
    Gender: str      # "F" or "M"
    ApptWeekday: str # e.g. "Monday"

class PredictionOutput(BaseModel):
    no_show_probability: float

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOutput)
def predict_appointment(input_data: AppointmentInput):
    proba = predict_no_show(input_data.dict())
    return PredictionOutput(no_show_probability=proba)
