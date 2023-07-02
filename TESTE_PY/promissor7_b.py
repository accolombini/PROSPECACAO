"""
Neste teste vamos adicionar um estágio de data cleanning antes de
trabalhar combinando os melhores algoritmos até aqui utilizados.
Em busca de outliers.
"""
import pandas as pd
import numpy as np
from scipy import stats

# Carregando os dados
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Separa os dados pelas categorias 'normal' e 'test-0'
df_normal = df[df['role'] == 'normal']
df_test = df[df['role'] == 'test-0']

# Verifica a quantidade de dados faltantes em cada categoria
missing_data_normal = df_normal.isnull().sum()
missing_data_test = df_test.isnull().sum()

# Adiciona zeros para colunas ausentes
missing_data_normal = missing_data_normal.reindex(df.columns, fill_value=0)
missing_data_test = missing_data_test.reindex(df.columns, fill_value=0)

# Imprimindo informações do DataFrame
print("\nInformações do DataFrame 'normal':")
print(df_normal.info())

print("\nColunas e Dados faltantes na categoria 'normal':")
print(missing_data_normal)

print("\nInformações do DataFrame 'test-0':")
print(df_test.info())

print("\nColunas e Dados faltantes na categoria 'test-0':")
print(missing_data_test)

# Cálculo dos outliers
outliers_normal = {}
outliers_test = {}

# Para a categoria 'normal'
for col in df_normal.columns:
    if col != 'role':  # Não aplicamos para a coluna 'role'
        z_scores = np.abs(stats.zscore(df_normal[col]))
        outliers = np.where(z_scores > 3)[0]
        outliers_normal[col] = outliers

# Para a categoria 'test-0'
for col in df_test.columns:
    if col != 'role':  # Não aplicamos para a coluna 'role'
        z_scores = np.abs(stats.zscore(df_test[col]))
        outliers = np.where(z_scores > 3)[0]
        outliers_test[col] = outliers

# Resumo com totais
print("\nResumo:")
print("Total de variáveis analisadas: ", len(df.columns))
print("Total de variáveis com dados faltantes em 'normal': ",
      missing_data_normal.sum())
print("Total de variáveis com dados faltantes em 'test-0': ",
      missing_data_test.sum())
print("Total de registros com outliers na categoria 'normal': ",
      sum([len(val) for val in outliers_normal.values()]))
print("Total de registros com outliers na categoria 'test-0': ",
      sum([len(val) for val in outliers_test.values()]))

# Detalhes dos outliers
print("\nDetalhes dos outliers:")
print("Na categoria 'normal':")
for var, indices in outliers_normal.items():
    if len(indices) > 0:
        print(f"A variável '{var}' tem {len(indices)} outliers.")
        print(df_normal.iloc[indices][var])
        print()

print("Na categoria 'test-0':")
for var, indices in outliers_test.items():
    if len(indices) > 0:
        print(f"A variável '{var}' tem {len(indices)} outliers.")
        print(df_test.iloc[indices][var])
        print()
