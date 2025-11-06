# src/time_series_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# ==========================
# CONFIGURACIÓN DE RUTAS
# ==========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "crypto_dataset_final.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================
# CARGA Y PREPROCESAMIENTO
# ==========================
print("📥 Cargando dataset...")
df = pd.read_csv(DATA_PATH)

# Convertir la columna de fecha
df["date"] = pd.to_datetime(df["date"], format='mixed', errors='coerce')

# Eliminar filas con fechas inválidas
df = df.dropna(subset=["date"])
df = df.sort_values("date")

# Mapear los IDs a nombres
id_map = {0: "Bitcoin", 1: "Ethereum", 2: "Dogecoin"}
df["symbol"] = df["symbol_id"].map(id_map)

# ==========================
# FUNCIÓN DE ANÁLISIS
# ==========================
def analyze_crypto(symbol: str):
    print(f"\n🔍 Analizando {symbol}...")
    data = df[df["symbol"] == symbol].copy()

    # Usar solo la columna 'close'
    ts = data.set_index("date")["close"]

    # ==========================
    # DESCOMPOSICIÓN ESTACIONAL
    # ==========================
    result = seasonal_decompose(ts, model="additive", period=30)
    result.plot()
    plt.suptitle(f"Descomposición de la serie temporal - {symbol}", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"decomposition_{symbol.lower()}.png")
    plt.close()

    # ==========================
    # MODELO ARIMA (manual)
    # ==========================
    print(f"🧠 Ajustando modelo ARIMA(1,1,1) para {symbol}...")
    model = ARIMA(ts, order=(1, 1, 1))
    model_fit = model.fit()

    # ==========================
    # PRONÓSTICO
    # ==========================
    n_periods = 30
    forecast = model_fit.forecast(steps=n_periods)
    forecast_index = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=n_periods)

    # Crear DataFrame de resultados
    forecast_df = pd.DataFrame({
        "date": forecast_index,
        "forecast": forecast.values,
        "symbol": symbol
    })

    # ==========================
    # GRAFICAR RESULTADOS
    # ==========================
    plt.figure(figsize=(10, 5))
    plt.plot(ts.index, ts, label="Histórico")
    plt.plot(forecast_index, forecast, label="Pronóstico", color="orange")
    plt.title(f"Predicción de precios - {symbol}")
    plt.xlabel("Fecha")
    plt.ylabel("Precio de cierre (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"forecast_{symbol.lower()}.png")
    plt.close()

    return forecast_df


# ==========================
# EJECUCIÓN PRINCIPAL
# ==========================
if __name__ == "__main__":
    all_forecasts = []

    for name in ["Bitcoin", "Ethereum", "Dogecoin"]:
        fc = analyze_crypto(name)
        all_forecasts.append(fc)

    # Concatenar y guardar predicciones
    final_df = pd.concat(all_forecasts)
    final_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    print("\n✅ Análisis completado. Resultados guardados en /output/")
