from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("server_model.pkl")

@app.get("/")
def home():
    return {"message": "AI Server Monitor v2.0 - CI/CD Active"}

@app.get("/predict")
def predict(temp: float, cpu: float):
    # The AI makes a prediction based on inputs
    prediction = model.predict([[temp, cpu]])
    result = "CRASH" if prediction[0] == 1 else "HEALTHY"
    return {
        "input_temp": temp,
        "input_cpu": cpu,
        "prediction": result
    }