# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:31:11 2022

@author: Alex
"""

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Cargar datos desde el archivo CSV
df = pd.read_csv('datos.csv')

# Convertir la columna 'Fecha' a formato datetime
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Seleccionar las columnas relevantes para el clustering
data = df[['CoordX', 'CoordY', 'Dato']].values

# Definir factores de escala para cada característica
escala_coordx = 0.25
escala_coordy = 0.25
escala_dato = 0.5

# Escalar las características manualmente
data_scaled = np.column_stack((
    data[:, 0] * escala_coordx,
    data[:, 1] * escala_coordy,
    data[:, 2] * escala_dato
))

# Entrenar un modelo de clustering K-Means
n_clusters = 35  # Define el número de clusters deseado
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(data_scaled)

# Agregar las etiquetas de cluster al DataFrame original
df['Cluster'] = kmeans.labels_

# Eliminar duplicados en la columna 'Nombre'
df = df.drop_duplicates(subset='Nombre')

# Exportar el DataFrame df con un solo nombre por grupo
df_export = df[['Nombre', 'CoordX', 'CoordY', 'Cluster']]

# Visualizar los resultados de clustering
plt.figure(figsize=(12, 6))

for cluster_id in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster_id]
    plt.scatter(cluster_data['CoordX'], cluster_data['CoordY'], label=f'Cluster {cluster_id}')

plt.xlabel('CoordX')
plt.ylabel('CoordY')
plt.title(f'Clustering de Puntos Espaciales (Escalado Prioritario)')
plt.legend()
plt.grid(True)
plt.show()

# Imprimir el DataFrame exportado
df_export.to_csv('datos_agrupados.csv')

