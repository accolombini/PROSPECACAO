"""
Trabalhando com clusters
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Carregar o arquivo CSV
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Criar o codificador de rótulos
label_encoder = LabelEncoder()

# Codificar a coluna "role" para valores numéricos
df['role_encoded'] = label_encoder.fit_transform(df['role'])

# Selecionar as colunas para clusterização
columns = ['offset_seconds', 'role_encoded']
df_selected = df[columns]

# Criar o modelo de clusterização
kmeans = KMeans(n_clusters=4, n_init=10)

# Ajustar o modelo aos dados
kmeans.fit(df_selected)

# Obter os rótulos dos clusters
labels = kmeans.labels_

# Obter as coordenadas dos centroides dos clusters
centroids = kmeans.cluster_centers_

# Criar um dicionário para mapear os rótulos dos clusters às cores
color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple'}

# Atribuir cores aos pontos de acordo com os rótulos dos clusters
colors = [color_map[label] for label in labels]

# Configurar a figura em 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotar os pontos com as cores correspondentes aos clusters
scatter = ax.scatter(df_selected['offset_seconds'], df_selected['role_encoded'], np.zeros_like(
    df_selected['role_encoded']), c=colors, alpha=0.5)

# Plotar os centroides dos clusters com destaque
centroid_scatter = ax.scatter(centroids[:, 0], centroids[:, 1], np.zeros_like(
    centroids[:, 0]), marker='x', s=200, linewidths=3, color='black')

# Configurar os rótulos dos eixos
ax.set_xlabel('Offset Seconds')
ax.set_ylabel('Role Encoded')
ax.set_zlabel('Z')

# Contar o número de amostras em cada cluster
unique_labels, counts = np.unique(labels, return_counts=True)

# Exibir o número de amostras em cada cluster
for label, count in zip(unique_labels, counts):
    print(f'Cluster {label+1}: {count} amostra(s)')

# Criar legendas para os clusters
legend_labels = [f'Cluster {label+1}' for label in unique_labels]
legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=color_map[label], markersize=10) for label in unique_labels]

# Adicionar a legenda
ax.legend(legend_handles, legend_labels)

# Exibir o gráfico em 3D
plt.show()
