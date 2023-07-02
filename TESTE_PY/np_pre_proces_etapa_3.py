"""
A proposta deste algoritmo é traballhar com a base OUTLIERS_FREE.xlsx e seguir os seguintes passos.

Análise de correlação: Isso irá nos dar uma ideia inicial de quais variáveis podem ser importantes para o nosso modelo. Nós removeremos as variáveis com correlação <= 0.5.

PCA: Após a análise de correlação, pode ser aplicada a PCA para reduzir a dimensionalidade dos dados. As variáveis selecionadas após a análise de correlação serão usadas para a PCA. Após a PCA, teremos um conjunto de componentes principais, que são combinações lineares das variáveis originais.

considerando que todas as variáveis são numéricas e que você está procurando um método com uma precisão muito alta, eu recomendaria o método Incorporado Lasso (Least Absolute Shrinkage and Selection Operator).

Análise de Componentes Principais (PCA) que já utilizamos. Outra alternativa seria a Análise de Fatores (Factor Analysis), que é uma técnica semelhante ao PCA, mas que faz suposições ligeiramente diferentes sobre a estrutura dos dados.
Análise de Fatores tenta identificar variáveis latentes ou não observadas (chamadas fatores) que explicam a correlação entre as variáveis observadas. A Análise de Fatores pode ser útil se você acredita que existem tais variáveis latentes em seus dados e deseja identificá-las.
"""

# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt

# Carregando a base de dados limpa
df = pd.read_excel('DADOS/OUTLIERS_FREE.xlsx')

# Removendo as colunas 'role' e 'offset_seconds'
df = df.select_dtypes(include=[np.number])

# Calculando a matriz de correlação
corr_matrix = df.corr().abs()

# Plotando o mapa de calor da correlação
plt.figure(figsize=(24,14))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f")
plt.show()

# Selecionando as variáveis com a menor correlação média
mean_correlation = corr_matrix.mean()
low_corr_vars = mean_correlation.nsmallest(10).index.tolist()  # substitua 10 pelo número de variáveis que você quer selecionar

print("\nVariáveis selecionadas com a menor correlação média:")
print(low_corr_vars)

# Atualizando o DataFrame com as variáveis selecionadas
df = df[low_corr_vars]

# Aplicando PCA
pca = PCA()
df_pca = pca.fit_transform(df)

# Convertendo os resultados do PCA para um DataFrame e renomeando as colunas
df_pca = pd.DataFrame(data = df_pca, columns = ['PC' + str(i) for i in range(1, df_pca.shape[1]+1)])

print("\nPrimeiros 10 registros dos resultados da PCA:")
print(df_pca.head(10))

# Matriz de coeficientes para a PCA
pca_loadings = pd.DataFrame(pca.components_, columns=df.columns)
print("\nMatriz de coeficientes para a PCA:")
print(pca_loadings)

# Aplicando Análise de Fatores
fa = FactorAnalyzer()
df_fa = fa.fit_transform(df)

# Convertendo os resultados da Análise de Fatores para um DataFrame e renomeando as colunas
df_fa = pd.DataFrame(data = df_fa, columns = ['FA' + str(i) for i in range(1, df_fa.shape[1]+1)])

print("\nPrimeiros 10 registros dos resultados da Análise de Fatores:")
print(df_fa.head(10))

# Matriz de coeficientes para a Análise de Fatores
fa_loadings = pd.DataFrame(fa.loadings_, columns=['FA' + str(i) for i in range(1, fa.loadings_.shape[1]+1)])
print("\nMatriz de coeficientes para a Análise de Fatores:")
print(fa_loadings)

# Salvando os resultados em um arquivo Excel
with pd.ExcelWriter('../DADOS/VAR_SELEC.xlsx') as writer:
    df.to_excel(writer, sheet_name='Low Correlation Variables', index=False)
    df_pca.to_excel(writer, sheet_name='PCA Results', index=False)
    pca_loadings.to_excel(writer, sheet_name='PCA Loadings', index=False)
    df_fa.to_excel(writer, sheet_name='FA Results', index=False)
    fa_loadings.to_excel(writer, sheet_name='FA Loadings', index=False)
