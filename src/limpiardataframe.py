
#importar pandas para dataframes y yfinance para descargar datos de criptomonedas

import pandas as pd

# Descargar datos de Bitcoin y Ethereum desde la API de yfinance
import yfinance as yf

#importar os para manejo de archivos
import os

#Importar para normalizar los datos
from sklearn.preprocessing import MinMaxScaler

# Definir las criptomonedas que se van a descargar de la API yfinance (Bitcoin y Ethereum)
symbols = ['BTC-USD', 'ETH-USD']

# Se crea una variable llamada api_data para almacenar los datos descargados
api_data = pd.DataFrame()

# Descargar datos de cada criptomoneda con un periodo más largo
for symbol in symbols:
    print(f"Descargando {symbol}...")
    data = yf.download(symbol, period="5y", interval="1d")  # Aumentar el periodo para tener más filas
    print(data.head())

    # Aplanar el MultiIndex de columnas (por seguridad)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    data['symbol'] = symbol
    api_data = pd.concat([api_data, data])

# Al descargar los datos de la api, estos traen columnas de Date (fecha sin hora), close, high, low, open y volume

# Estandarizar nombres
api_data.reset_index(inplace=True)
api_data.rename(columns={
    'Date': 'date',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
}, inplace=True)

print("Datos desde API listos:", api_data.shape)


######################################################################################################################

# Recuperar los datos de Dogecoin desde un archivo .CSV local
doge_csv_path = "data/coin_Dogecoin.csv"

doge_csv = pd.read_csv(doge_csv_path)

# En el archivo .csv hay columnas que no son necesarias por lo que se deben eliminar
# como lo son Unix, tradecount, etc.


# Estandarizar columnas al igual que Bitcoin y Ethereum
doge_csv.rename(columns={
    'Date': 'date',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume USDT': 'volume'
}, inplace=True)

# Convertir fecha ya que es Datetime, y en la apyi trae solo Date. se agrega la columna de "Symbol"
#para diferenciar entre cada moneda

doge_csv['symbol'] = 'DOGE-USD'

# Mantener solo columnas necesarias
doge_csv = doge_csv[['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']]

# Tomar la misma cantidad de datos para cada símbolo
target_rows = 2000 // 3  # aproximadamente 666 por criptomoneda

balanced_data = pd.DataFrame()
for symbol in ['BTC-USD', 'ETH-USD', 'DOGE-USD']:
    if symbol == 'DOGE-USD':
        df = doge_csv.copy()
    else:
        df = api_data[api_data['symbol'] == symbol].copy()

    # Ordenar por fecha y tomar muestras uniformemente distribuidas
    df = df.sort_values(by='date').reset_index(drop=True)
    if len(df) > target_rows:
        df = df.iloc[-target_rows:]  # tomar los últimos n registros


    balanced_data = pd.concat([balanced_data, df], ignore_index=True)

# Limpiar duplicados y ordenar
balanced_data.drop_duplicates(subset=['date', 'symbol'], keep='last', inplace=True)
balanced_data.sort_values(by=['symbol', 'date'], inplace=True)


# Asignar variable numérica a cada símbolo
symbol_map = {
    'BTC-USD': 0,
    'ETH-USD': 1,
    'DOGE-USD': 2
}

balanced_data['symbol_id'] = balanced_data['symbol'].map(symbol_map)

# Eliminar la columna 'symbol', ya no es necesaria
balanced_data.drop(columns=['symbol'], inplace=True)


## Revisar si hay valores nulos
print("Valores nulos:", balanced_data.isnull().sum())

## No se encontraron valores nulos en el df.






# Guardar resultado final
output_path = "data/crypto_dataset_final.csv"
os.makedirs("data", exist_ok=True)
balanced_data.to_csv(output_path, index=False)
print(f"Archivo guardado como {output_path}")




# Mostrar resumen
print("\nPrimeras filas del dataset combinado:")
print(balanced_data.head())


# Cargar el dataset limpio
input_path = "data/crypto_dataset_final.csv"
balanced_data = pd.read_csv(input_path)

# Seleccionar columnas numéricas
numeric_cols = ['open', 'high', 'low', 'close', 'volume']

# Crear el objeto scaler
scaler = MinMaxScaler()

# Ajustar el scaler y transformar los datos
balanced_data[numeric_cols] = scaler.fit_transform(balanced_data[numeric_cols])

# Guardar el dataset normalizado
output_path = "data/crypto_dataset_normalized.csv"
os.makedirs("data", exist_ok=True)
balanced_data.to_csv(output_path, index=False)

print(f"Datos normalizados guardados en: {output_path}")

# Mostrar ejemplo
print("\nPrimeras filas de los datos normalizados:")
print(balanced_data.head())



# Ver resumen estadístico general
stats = df.describe()
print("📊 Estadísticas Descriptivas:")
print(stats)

# Mostrar valores nulos
print("\n🔍 Valores nulos por columna:")
print(df.isnull().sum())

