"""
A proposta deste algoritmo é tipo de aprendizado não supervisionado conhecido como agrupamento ou clustering.
Neste caso, nosso foco será usar o k-means e aaliar seu desemepenho, sendo nosso objetivo seeparar o conjunto de
dados em três clusteress, um para operação Normal, outrto para operação sob Anomalias e por fim um para Faltas.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer

# Carregando a base de dados
df = pd.read_excel('DADOS/OUTLIERS_FREE.xlsx')

# Imprime o número de linhas no dataframe original
print(f'Número de linhas no dataframe original: {len(df)}')

# Removendo as colunas 'role' e 'offset_seconds'
df = df.select_dtypes(include=[np.number])

# Imprime o número de linhas após a remoção das colunas não numéricas
print(f'Número de linhas após a remoção das colunas não numéricas: {len(df)}')

# Padronização dos dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Método do cotovelo para encontrar o número ideal de clusters
model = KMeans(n_init=10)
visualizer = KElbowVisualizer(model, k=(2,12))
visualizer.fit(df_scaled)
n_clusters = visualizer.elbow_value_

# Aplicando K-means com o número ótimo de clusters
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
kmeans.fit(df_scaled)

# Adicionando os labels dos clusters ao dataframe
df['Cluster'] = kmeans.labels_

# Imprime o número de linhas após a atribuição dos clusters
print(f'Número de linhas após a atribuição dos clusters: {len(df)}')

# Verificando se há algum registro sem cluster atribuído
missing_clusters = df[df['Cluster'].isnull()]
print('Registros sem clusters atribuídos:')
print(missing_clusters)

# Verificando os labels únicos antes de renomear os clusters
print('Labels únicos antes de renomear os clusters:')
print(df['Cluster'].unique())

# Renomeando os clusters
df['Cluster'] = df['Cluster'].map({0: 'Dados_1', 1: 'Dados_2', 2: 'Dados_3', 3: 'Dados_4'})

# Verificando os labels únicos após renomear os clusters
print('Labels únicos após renomear os clusters:')
print(df['Cluster'].unique())

# Contando o número de elementos em cada cluster
print(df['Cluster'].value_counts())

# Aplicando PCA para visualização
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df_scaled), columns=['PC1', 'PC2'])
df_pca['Cluster'] = df['Cluster']

# Plotando os clusters
sns.set(rc={'figure.figsize':(10,7)})
scatterplot = sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis')
scatterplot.set_title('Clusters (visualização PCA 2D)')
plt.show()

# Salvando os resultados em um arquivo Excel
df.to_excel('DADOS/KMEANS.xlsx', index=False)
