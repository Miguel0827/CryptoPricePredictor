import json
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import httpx
import os

# -------------------------------------------------------------------
BASE_DIR = "C:/Users/juanm/Documents/GitHub/CryptoPricePredictor"
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODELS_INFO_PATH = os.path.join(BASE_DIR, "models", "models_info.json")

# -------------------------------------------------------------------
crypto_ids = {
    "0": "bitcoin",
    "1": "ethereum",
    "2": "solana"
}

# -------------------------------------------------------------------
def load_models_info():
    with open(MODELS_INFO_PATH, "r") as f:
        return json.load(f)

# -------------------------------------------------------------------
def generate_future_dates(days):
    from datetime import datetime, timedelta
    today = datetime.today()
    return [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, days+1)]

# -------------------------------------------------------------------
async def fetch_real_time_price(crypto_name: str):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_name}&vs_currencies=usd"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    if response.status_code != 200:
        return None
    return response.json()[crypto_name]["usd"]

# -------------------------------------------------------------------
def generate_predictions(base_price, model_data, days: int):
    """
    Genera predicciones según tipo de modelo
    """
    model_type = model_data.get("architecture", "").lower()
    
    if "arima" in model_data.get("file", "") or model_type == "arima":
        # Usar forecast ARIMA
        forecast = model_data["forecast_30d"]
        return forecast[:days]

    # Para LSTM o GRU: usamos last_30_predictions como ejemplo
    predictions = model_data.get("last_30_predictions", [])
    if len(predictions) >= days:
        return predictions[:days]
    
    # Si pedimos más días de los disponibles, repetimos el último
    while len(predictions) < days:
        predictions.append(predictions[-1])
    return predictions
