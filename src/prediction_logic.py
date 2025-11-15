import json
import math
from datetime import datetime, timedelta
import httpx

crypto_names = {
    "0": "Bitcoin",
    "1": "Ethereum",
    "2": "Dogecoin"
}

crypto_ids = {
    "0": "bitcoin",
    "1": "ethereum",
    "2": "dogecoin"
}

def load_models_info():
    with open("models_info.json", "r") as f:
        return json.load(f)


async def fetch_real_time_price(crypto_id: str):
    """Consulta el precio en Coingecko."""
    url = f"https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": crypto_id,
        "vs_currencies": "usd",
        "include_24hr_change": "true"
    }
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(url, params=params)
            data = r.json()
            return {
                "price": data[crypto_id]["usd"],
                "change24h": data[crypto_id]["usd_24h_change"]
            }
        except:
            return None


def generate_predictions(base_price, model_data, days):
    """Replica tu lógica matemática del frontend."""
    model_predictions = model_data.get("forecast_30d") if model_data.get("forecast_30d") else model_data.get("last_30_predictions")

    if not model_predictions:
        raise ValueError("No hay predicciones en el modelo.")

    first_pred = model_predictions[0]
    last_pred = model_predictions[-1]
    total_change_percent = (last_pred - first_pred) / first_pred
    avg_change_percent = total_change_percent / len(model_predictions)

    conservative_factor = 0.3

    volatility_sum = 0
    for i in range(1, len(model_predictions)):
        change = abs((model_predictions[i] - model_predictions[i-1]) / model_predictions[i-1])
        volatility_sum += change

    avg_volatility = volatility_sum / (len(model_predictions)-1)

    predictions = [base_price]
    last_price = base_price

    for i in range(1, min(days, 30)):
        trend_change = avg_change_percent * conservative_factor
        volatility_component = avg_volatility * conservative_factor * (math.sin(i * 0.5) * 0.5)
        total_change = trend_change + volatility_component
        predicted_price = last_price * (1 + total_change)

        predictions.append(predicted_price)
        last_price = predicted_price

    return predictions


def generate_future_dates(n):
    today = datetime.now()
    return [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]
