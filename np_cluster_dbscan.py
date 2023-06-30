"""
A proposta deste algoritmo é tipo de aprendizado não supervisionado conhecido como agrupamento ou clustering.
Neste caso, nosso foco será usar o DBSCAN e aaliar seu desemepenho, sendo nosso objetivo seeparar o conjunto de
dados em três clusteress, um para operação Normal, outrto para operação sob Anomalias e por fim um para Faltas.

Nota:
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Carregando a base de dados
df = pd.read_excel('DADOS/OUTLIERS_FREE.xlsx')

# Removendo as colunas 'role' e 'offset_seconds'
df = df.select_dtypes(include=[np.number])

# Padronização dos dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Aplicando DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(df_scaled)

# Adicionando os labels dos clusters ao dataframe
df['Cluster'] = dbscan.labels_

# Renomeando os clusters
df['Cluster'] = df['Cluster'].map({0: 'Dados_1', 1: 'Dados_2', 2: 'Dados_3', -1: 'Noise'})

# Contando o número de elementos em cada cluster
print(df['Cluster'].value_counts())

# Aplicando PCA para visualização
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Plotando os clusters
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=dbscan.labels_, cmap='viridis')
plt.title('Clusters (visualização PCA 2D)')

# Adicionando a legenda
plt.legend(handles=scatter.legend_elements()[0], labels=['Dados_1', 'Dados_2', 'Dados_3', 'Noise'])
plt.show()

# Salvando os resultados em um arquivo Excel
df.to_excel('DADOS/DBSCAN.xlsx', index=False)
