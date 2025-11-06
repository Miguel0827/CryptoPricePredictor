# ==========================================================
# 📊 Análisis Exploratorio - Estadísticas, Visualización y Correlación
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ruta al archivo limpio
DATA_PATH = Path("data/crypto_dataset_final.csv")

# ==========================================================
# 1️⃣ Cargar los datos
# ==========================================================
print("📥 Cargando datos...")
df = pd.read_csv(DATA_PATH)

# Asegurar que la columna 'Date' esté en formato datetime
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

# ==========================================================
# 2️⃣ Estadísticas descriptivas
# ==========================================================
print("\n📊 Estadísticas Descriptivas:")

# Guardar en CSV
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

symbol_map = {
    0: "BTC-USD",
    1: "ETH-USD",
    2: "DOGE-USD"
}

# Imprimir estadísticas descriptivas para cada moneda
for symbol_id, symbol_name in symbol_map.items():
    print("\n" + "="*70)
    print(f"📊 ESTADÍSTICAS DESCRIPTIVAS PARA {symbol_name}")
    print("="*70)
    
    # Filtrar los datos de la moneda
    df_symbol = df[df['symbol_id'] == symbol_id]
    
    # Calcular estadísticas
    stats = df_symbol.describe()
    
    # Mostrar estadísticas
    print(stats)
    
    # Mostrar valores nulos por columna
    print("\n🔍 Valores nulos por columna:")
    print(df_symbol.isnull().sum())


# Mostrar valores nulos
print("\n🔍 Valores nulos por columna:")

# ==========================================================
# 3️⃣ Visualización de tendencias
# ==========================================================
# Graficar cada símbolo con su propia línea
# Obtener lista de símbolos únicos
symbols = df['symbol_id'].unique()

# Crear una gráfica independiente para cada símbolo
for symbol in symbols:
    data = df[df['symbol_id'] == symbol]

    plt.figure(figsize=(10, 5))
    plt.plot(data['date'], data['close'], label=f'{symbol}', color='blue')
    plt.title(f"Tendencia del Precio de Cierre de {symbol} (2023–2025)")
    plt.xlabel("Fecha")
    plt.ylabel("Precio (USD)")
    plt.legend()
    plt.tight_layout()
    # Guardar la gráfica en la carpeta output
    output_path = f"output/tendencia_{symbol}.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Gráfica guardada en: {output_path}")
# ==========================================================
# 4️⃣ Correlaciones
# ==========================================================
print("\n🔗 Matriz de Correlación:")
corr = df.corr(numeric_only=True)
print(corr)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlación entre Variables')
plt.tight_layout()
plt.savefig(output_dir / "matriz_correlacion.png")
plt.show()

print("\n✅ Análisis exploratorio completado.")
print(f"📂 Resultados guardados en: {output_dir.resolve()}")
