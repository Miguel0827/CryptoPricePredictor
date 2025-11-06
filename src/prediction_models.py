# src/prediction_models.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
from keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import joblib
import json
from datetime import datetime

# ==========================================================
# 1️⃣ Configuración inicial
# ==========================================================
print("🔥 Cargando dataset...")
df = pd.read_csv("data/crypto_dataset_final.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")

# Crear carpetas de salida si no existen
os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("models/scalers", exist_ok=True)
os.makedirs("models/metadata", exist_ok=True)

# Identificar las monedas únicas
cryptos = df["symbol_id"].unique()
print(f"🔍 Monedas encontradas: {cryptos}")

# Diccionario para guardar información de todos los modelos
models_info = {}

# ==========================================================
# 2️⃣ Función para crear secuencias para LSTM / GRU
# ==========================================================
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# ==========================================================
# 3️⃣ Entrenar y predecir por cada moneda
# ==========================================================
for symbol in cryptos:
    print(f"\n🚀 Procesando symbol_id = {symbol}...")
    crypto_df = df[df["symbol_id"] == symbol].sort_values("date")

    # Usamos la columna 'close'
    prices = crypto_df["close"].values.reshape(-1, 1)

    # Normalizar para redes neuronales
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    # Guardar el scaler para poder desnormalizar después
    scaler_path = f"models/scalers/{symbol}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler guardado en {scaler_path}")

    time_steps = 10
    X, y = create_sequences(scaled_prices, time_steps)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Información del modelo para metadata
    model_metadata = {
        "symbol": str(symbol),
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_points": int(len(prices)),
        "time_steps": int(time_steps),
        "last_price": float(prices[-1][0]),
        "last_date": str(crypto_df["date"].iloc[-1].strftime("%Y-%m-%d")),
        "min_price": float(prices.min()),
        "max_price": float(prices.max()),
        "models": {}
    }

    # ======================================================
    # 🔹 Modelo ARIMA
    # ======================================================
    print("🧩 Entrenando modelo ARIMA...")
    try:
        arima_model = ARIMA(prices, order=(5, 1, 2))
        arima_result = arima_model.fit()
        forecast_arima = arima_result.forecast(steps=30)

        # Guardar modelo ARIMA
        arima_path = f"models/{symbol}_arima.pkl"
        with open(arima_path, 'wb') as f:
            pickle.dump(arima_result, f)
        print(f"💾 Modelo ARIMA guardado en {arima_path}")

        # Metadata ARIMA
        model_metadata["models"]["arima"] = {
            "file": f"{symbol}_arima.pkl",
            "order": [5, 1, 2],
            "aic": float(arima_result.aic),
            "bic": float(arima_result.bic),
            "forecast_30d": [float(x) for x in forecast_arima.tolist()]
        }

        plt.figure(figsize=(10, 5))
        plt.plot(crypto_df["date"], prices, label="Real")
        plt.plot(pd.date_range(crypto_df["date"].iloc[-1], periods=31, freq="D")[1:], forecast_arima, label="ARIMA Forecast", color="orange")
        plt.title(f"Predicción de precios {symbol} - ARIMA")
        plt.legend()
        plt.savefig(f"output/{symbol}_forecast_arima.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Error en ARIMA para {symbol}: {e}")
        model_metadata["models"]["arima"] = {"error": str(e)}

    # ======================================================
    # 🔹 Modelo LSTM
    # ======================================================
    print("🧠 Entrenando modelo LSTM...")
    lstm_model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(time_steps, 1)),
        Dense(1)
    ])
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    history_lstm = lstm_model.fit(X, y, epochs=10, batch_size=16, verbose=1)

    # Guardar modelo LSTM (formato .keras es más moderno)
    lstm_path = f"models/{symbol}_lstm.keras"
    lstm_model.save(lstm_path)
    print(f"💾 Modelo LSTM guardado en {lstm_path}")

    # También guardar en formato TensorFlow.js para usar directamente en el navegador
    tfjs_lstm_path = f"models/tfjs/{symbol}_lstm"
    os.makedirs(tfjs_lstm_path, exist_ok=True)
    
    predicted_lstm = lstm_model.predict(X[-30:])
    predicted_lstm_rescaled = scaler.inverse_transform(predicted_lstm)

    # Metadata LSTM
    model_metadata["models"]["lstm"] = {
        "file": f"{symbol}_lstm.keras",
        "tfjs_path": f"tfjs/{symbol}_lstm",
        "units": 50,
        "time_steps": int(time_steps),
        "final_loss": float(history_lstm.history['loss'][-1]),
        "architecture": "LSTM(50) -> Dense(1)",
        "last_30_predictions": [float(x) for x in predicted_lstm_rescaled.flatten().tolist()]
    }

    plt.figure(figsize=(10, 5))
    plt.plot(crypto_df["date"], prices, label="Real")
    plt.plot(crypto_df["date"].iloc[-30:], predicted_lstm_rescaled, label="LSTM Forecast", color="red")
    plt.title(f"Predicción de precios {symbol} - LSTM")
    plt.legend()
    plt.savefig(f"output/{symbol}_forecast_lstm.png")
    plt.close()

    # ======================================================
    # 🔹 Modelo GRU
    # ======================================================
    print("🧩 Entrenando modelo GRU...")
    gru_model = Sequential([
        GRU(50, return_sequences=False, input_shape=(time_steps, 1)),
        Dense(1)
    ])
    gru_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    history_gru = gru_model.fit(X, y, epochs=10, batch_size=16, verbose=1)

    # Guardar modelo GRU
    gru_path = f"models/{symbol}_gru.keras"
    gru_model.save(gru_path)
    print(f"💾 Modelo GRU guardado en {gru_path}")

    # También guardar en formato TensorFlow.js
    tfjs_gru_path = f"models/tfjs/{symbol}_gru"
    os.makedirs(tfjs_gru_path, exist_ok=True)

    predicted_gru = gru_model.predict(X[-30:])
    predicted_gru_rescaled = scaler.inverse_transform(predicted_gru)

    # Metadata GRU
    model_metadata["models"]["gru"] = {
        "file": f"{symbol}_gru.keras",
        "tfjs_path": f"tfjs/{symbol}_gru",
        "units": 50,
        "time_steps": int(time_steps),
        "final_loss": float(history_gru.history['loss'][-1]),
        "architecture": "GRU(50) -> Dense(1)",
        "last_30_predictions": [float(x) for x in predicted_gru_rescaled.flatten().tolist()]
    }

    plt.figure(figsize=(10, 5))
    plt.plot(crypto_df["date"], prices, label="Real")
    plt.plot(crypto_df["date"].iloc[-30:], predicted_gru_rescaled, label="GRU Forecast", color="green")
    plt.title(f"Predicción de precios {symbol} - GRU")
    plt.legend()
    plt.savefig(f"output/{symbol}_forecast_gru.png")
    plt.close()

    # Guardar metadata del símbolo
    metadata_path = f"models/metadata/{symbol}_info.json"
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    print(f"📄 Metadata guardada en {metadata_path}")

    # Agregar al diccionario general (convertir symbol a string para JSON)
    models_info[str(symbol)] = model_metadata

    print(f"✅ Modelos de {symbol} entrenados y guardados en /models/")
    print(f"✅ Gráficos guardados en /output/")

# Guardar información general de todos los modelos
general_info_path = "models/models_info.json"
with open(general_info_path, 'w') as f:
    json.dump({
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_symbols": int(len(cryptos)),
        "symbols": [str(s) for s in cryptos],
        "models": models_info
    }, f, indent=2)

print("\n🎯 Proceso completo para todas las monedas finalizado.")
print("\n📂 Estructura de archivos guardados:")
print("   models/")
print("   ├── models_info.json (información general)")
print("   ├── {symbol}_arima.pkl")
print("   ├── {symbol}_lstm.keras")
print("   ├── {symbol}_gru.keras")
print("   ├── metadata/")
print("   │   └── {symbol}_info.json")
print("   ├── scalers/")
print("   │   └── {symbol}_scaler.pkl")
print("   └── tfjs/ (para usar en navegador)")
print("       ├── {symbol}_lstm/")
print("       └── {symbol}_gru/")
print("\n💡 Tip: Usa models_info.json para listar todos los modelos disponibles en tu web")