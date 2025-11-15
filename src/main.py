from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware 
from prediction_logic import (
    load_models_info, fetch_real_time_price,
    crypto_ids, generate_predictions, generate_future_dates
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes (para desarrollo)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los headers
)

class PredictionRequest(BaseModel):
    symbol: str     # "0", "1", "2"
    model: str      # "arima", "lstm", "gru"
    days: int = 10

@app.post("/predict")
async def predict(req: PredictionRequest):
    models_info = load_models_info()

    # Validar símbolo
    if req.symbol not in models_info["symbols"]:
        return {"error": "Símbolo no válido"}

    symbol_info = models_info["models"][req.symbol]

    # Validar modelo
    model_data = symbol_info["models"].get(req.model)
    if not model_data:
        return {"error": "Modelo no encontrado"}

    # Precio en tiempo real
    real_time = await fetch_real_time_price(crypto_ids[req.symbol])
    base_price = real_time if real_time else symbol_info["last_price"]

    # Predicciones
    predictions = generate_predictions(base_price, model_data, req.days)
    future_dates = generate_future_dates(len(predictions))

    avg_price = sum(predictions) / len(predictions)

    return {
        "symbol": req.symbol,
        "model_used": req.model,
        "real_time_price": real_time,
        "base_price": base_price,
        "predictions": predictions,
        "dates": future_dates,
        "stats": {
            "avg_price": avg_price,
            "min_price": min(predictions),
            "max_price": max(predictions),
        }
    }
