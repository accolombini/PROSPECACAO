"""
Refinando o algogitmo --> Explorando os resultados do teste6.py

"""

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Carregar o arquivo CSV
df = pd.read_csv('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Criar o codificador de rótulos
label_encoder = LabelEncoder()

# Codificar a coluna "role" para valores numéricos
df['role_encoded'] = label_encoder.fit_transform(df['role'])

# Selecionar as colunas para clusterização
columns = ['offset_seconds', 'role_encoded']
df_selected = df[columns]

# Criar o modelo de clusterização K-means
kmeans = KMeans(n_clusters=4, n_init=10)

# Ajustar o modelo aos dados
kmeans.fit(df_selected)

# Obter os rótulos dos clusters
labels_kmeans = kmeans.labels_

# Adicionar os rótulos dos clusters ao DataFrame
df['cluster'] = labels_kmeans

# Contar o número de amostras em cada cluster do K-means
unique_labels_kmeans, counts_kmeans = np.unique(
    labels_kmeans, return_counts=True)

# Exibir o número de amostras em cada cluster do K-means
for label, count in zip(unique_labels_kmeans, counts_kmeans):
    print("K-means Cluster {}: {} amostra(s)".format(label, count))

# Criar o modelo de clusterização GMM
gmm = GaussianMixture(n_components=6)

# Ajustar o modelo aos dados
gmm.fit(df_selected)

# Obter os rótulos dos clusters do GMM
labels_gmm = gmm.predict(df_selected)

# Contar o número de amostras em cada cluster do GMM
unique_labels_gmm, counts_gmm = np.unique(labels_gmm, return_counts=True)

# Exibir o número de amostras em cada cluster do GMM
for label, count in zip(unique_labels_gmm, counts_gmm):
    print("GMM Cluster {}: {} amostra(s)".format(label, count))

# Análise dos clusters do GMM
for cluster in range(len(unique_labels_gmm)):
    cluster_samples = df_selected[labels_gmm == cluster]
    print("\nCluster {}: Características do cluster".format(cluster))
    print(cluster_samples.describe())

# Selecionar apenas os dados do Cluster 5
cluster_5_data = df[df['cluster'] == 4]

# Selecionar os dados dos outros clusters
other_clusters_data = df[df['cluster'] != 4]

# Calcular as estatísticas descritivas para cada variável em relação ao Cluster 5
cluster_5_stats = cluster_5_data.describe().stack()

# Calcular as estatísticas descritivas para cada variável nos outros clusters
other_clusters_stats = other_clusters_data.groupby(
    'cluster').describe().stack()

# Comparar as estatísticas do Cluster 5 com as dos outros clusters
comparison = pd.concat([cluster_5_stats, other_clusters_stats], axis=1, keys=[
                       'Cluster 5', 'Other Clusters'])

# Exibir a comparação
print(comparison)
