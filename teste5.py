"""
Refinando o algogitmo --> Trabalhando com clusters e adicionando um pouco de estatística e usando GMM
"""

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


# Carregar o arquivo CSV
df = pd.read_csv('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

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

# Ajustar o modelo K-means aos dados
kmeans.fit(df_selected)

# Obter os rótulos dos clusters do K-means
labels_kmeans = kmeans.labels_

# Contar o número de amostras em cada cluster do K-means
unique_labels_kmeans, counts_kmeans = np.unique(
    labels_kmeans, return_counts=True)

# Exibir o número de amostras em cada cluster do K-means
for label, count in zip(unique_labels_kmeans, counts_kmeans):
    print("K-means Cluster {}: {} amostra(s)".format(label, count))


# Normalizar os dados
scaler = StandardScaler()
df_selected_scaled = scaler.fit_transform(df_selected)

# Parâmetros do GMM
min_components = 2
max_components = 6
best_gmm = None
best_bic = np.inf

# Iterar sobre diferentes números de componentes
for n_components in range(min_components, max_components + 1):
    # Criar o modelo GMM
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type='full', random_state=42)

    # Ajustar o modelo aos dados
    gmm.fit(df_selected_scaled)

    # Obter o BIC
    bic = gmm.bic(df_selected_scaled)

    # Verificar se o BIC é o menor encontrado até o momento
    if bic < best_bic:
        best_bic = bic
        best_gmm = gmm

# Obter os rótulos dos clusters com o melhor modelo GMM
labels_gmm = best_gmm.predict(df_selected_scaled)

# Contar o número de amostras em cada cluster do GMM
unique_labels_gmm, counts_gmm = np.unique(labels_gmm, return_counts=True)

# Exibir o número de amostras em cada cluster do GMM
for label, count in zip(unique_labels_gmm, counts_gmm):
    print("GMM Cluster {}: {} amostra(s)".format(label, count))

# Plote os clusters do GMM em um gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors_gmm = ['green', 'red', 'blue', 'orange', 'yellow', 'purple']

for label in unique_labels_gmm:
    cluster_samples = df_selected[labels_gmm == label]
    ax.scatter(cluster_samples['offset_seconds'], cluster_samples['role_encoded'],
               cluster_samples['mean_offset'], c=colors_gmm[label], label='GMM Cluster {}'.format(label))

ax.set_xlabel('offset_seconds')
ax.set_ylabel('role_encoded')
ax.set_zlabel('mean_offset')

# Adicione uma legenda
ax.legend()

# Exiba o gráfico do GMM
plt.show()
