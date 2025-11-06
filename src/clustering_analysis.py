# src/clustering_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# ==========================================================
# Cargar dataset
# ==========================================================
df = pd.read_csv("data/crypto_dataset_final.csv")

# Seleccionamos variables numéricas relevantes
features = ["open", "high", "low", "close", "volume"]
df_features = df[features].copy()

# Normalizamos los datos para evitar sesgos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# ==========================================================
# 2️⃣ K-MEANS (Particional)
# ==========================================================
print("🔹 Aplicando K-Means...")
kmeans = KMeans(n_clusters=3, random_state=42)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(7, 5))
sns.scatterplot(x=df["close"], y=df["volume"], hue=df["kmeans_cluster"], palette="viridis")
plt.title("K-Means Clustering - Criptomonedas")
plt.xlabel("Precio de Cierre")
plt.ylabel("Volumen")
plt.savefig("output/kmeans_clusters.png")
plt.close()

# ==========================================================
# 3️⃣ DBSCAN (Basado en densidad)
# ==========================================================
print("🔹 Aplicando DBSCAN...")
dbscan = DBSCAN(eps=0.8, min_samples=10)
df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)

plt.figure(figsize=(7, 5))
sns.scatterplot(x=df["close"], y=df["volume"], hue=df["dbscan_cluster"], palette="tab10")
plt.title("DBSCAN Clustering - Criptomonedas")
plt.xlabel("Precio de Cierre")
plt.ylabel("Volumen")
plt.savefig("output/dbscan_clusters.png")
plt.close()

# ==========================================================
# 4️⃣ JERÁRQUICO
# ==========================================================
print("🔹 Aplicando clustering jerárquico...")
linkage_matrix = linkage(X_scaled, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, truncate_mode="level", p=5)
plt.title("Clustering Jerárquico (Dendrograma)")
plt.xlabel("Muestras")
plt.ylabel("Distancia")
plt.savefig("output/hierarchical_dendrogram.png")
plt.close()

# ==========================================================
# 5️⃣ Guardar resultados
# ==========================================================
output_path = "output/clustering_results.csv"
df.to_csv(output_path, index=False)
print(f"Resultados guardados en {output_path}")
