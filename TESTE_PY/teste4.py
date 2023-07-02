"""
Refinando o algogitmo --> Trabalhando com clusters e adicionando um pouco de estatística
"""

from sklearn.cluster import KMeans, DBSCAN
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

# Criar novas características
df['mean_offset'] = df.groupby('role_encoded')[
    'offset_seconds'].transform('mean')
df['std_offset'] = df.groupby('role_encoded')[
    'offset_seconds'].transform('std')
df['var_offset'] = df.groupby('role_encoded')[
    'offset_seconds'].transform('var')

# Selecionar as colunas para clusterização
columns = ['offset_seconds', 'role_encoded',
           'mean_offset', 'std_offset', 'var_offset']
df_selected = df[columns]

# Criar o modelo de clusterização K-means
kmeans = KMeans(n_clusters=4, n_init=10)
kmeans.fit(df_selected)
labels_kmeans = kmeans.labels_
unique_labels_kmeans, counts_kmeans = np.unique(
    labels_kmeans, return_counts=True)

# Exibir o número de amostras em cada cluster pelo K-means
for label, count in zip(unique_labels_kmeans, counts_kmeans):
    print("K-means Cluster {}: {} amostra(s)".format(label, count))

# Plotar os resultados do K-means em um gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors_kmeans = ['green', 'red', 'blue', 'orange']

for label, color in zip(unique_labels_kmeans, colors_kmeans):
    cluster_samples = df_selected[labels_kmeans == label]
    ax.scatter(cluster_samples['offset_seconds'], cluster_samples['role_encoded'], cluster_samples['mean_offset'],
               c=color, label='K-means Cluster {}'.format(label))

ax.set_xlabel('offset_seconds')
ax.set_ylabel('role_encoded')
ax.set_zlabel('mean_offset')

# Adicionar uma legenda
ax.legend()

# Exibir o gráfico do K-means
plt.show()

# Criar o modelo de clusterização DBSCAN
dbscan = DBSCAN(eps=0.6, min_samples=9)
dbscan.fit(df_selected)
labels_dbscan = dbscan.labels_
unique_labels_dbscan, counts_dbscan = np.unique(
    labels_dbscan, return_counts=True)

# Exibir o número de amostras em cada cluster pelo DBSCAN
for label, count in zip(unique_labels_dbscan, counts_dbscan):
    print("DBSCAN Cluster {}: {} amostra(s)".format(label, count))

# Plotar os resultados do DBSCAN em um gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors_dbscan = ['green', 'red', 'blue', 'orange', 'yellow']

for label, color in zip(unique_labels_dbscan, colors_dbscan):
    if label == -1:
        cluster_samples = df_selected[labels_dbscan == label]
        ax.scatter(cluster_samples['offset_seconds'], cluster_samples['role_encoded'], cluster_samples['mean_offset'],
                   c=color, label='DBSCAN Cluster {}'.format(label))
    else:
        cluster_samples = df_selected[labels_dbscan == label]
        ax.scatter(cluster_samples['offset_seconds'], cluster_samples['role_encoded'], cluster_samples['mean_offset'],
                   c=color, label='DBSCAN Cluster {}'.format(label))

ax.set_xlabel('offset_seconds')
ax.set_ylabel('role_encoded')
ax.set_zlabel('mean_offset')

# Adicionar uma legenda
ax.legend()

# Exibir o gráfico do DBSCAN
plt.show()
