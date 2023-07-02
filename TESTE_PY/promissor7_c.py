"""
Neste teste vamos adicionar um estágio de data cleanning antes de
trabalhar combinando os melhores algoritmos até aqui utilizados.
Em busca de outliers --> salvando outliers no arquivo.
"""

import pandas as pd
from scipy import stats
import numpy as np

# Carregando os dados
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Separa os dados pelas categorias 'normal' e 'test-0'
df_normal = df[df['role'] == 'normal']
df_test = df[df['role'] == 'test-0']

# Verifica a quantidade de dados faltantes em cada categoria
missing_data_normal = df_normal.isnull().sum()
missing_data_test = df_test.isnull().sum()

# Calcula os outliers
z_scores_normal = stats.zscore(
    df_normal.drop(columns=['role']), nan_policy='omit')
z_scores_test = stats.zscore(df_test.drop(columns=['role']), nan_policy='omit')

outliers_normal = df_normal[(np.abs(z_scores_normal) > 3).any(axis=1)]
outliers_test = df_test[(np.abs(z_scores_test) > 3).any(axis=1)]

# Cria um writer do pandas para Excel
writer = pd.ExcelWriter('../DADOS/data_cleanning.xlsx', engine='xlsxwriter')

# Escreve cada DataFrame em uma folha diferente
outliers_normal.to_excel(writer, sheet_name='Outliers_Normal', index=False)
outliers_test.to_excel(writer, sheet_name='Outliers_Test0', index=False)

# Fecha o escritor e salva o arquivo Excel
writer.close()

# Imprime as informações
print("\nResumo:")
print("Total de variáveis analisadas: ", len(df.columns))
print("Total de variáveis com dados faltantes em 'normal': ",
      missing_data_normal.sum())
print("Total de variáveis com dados faltantes em 'test-0': ",
      missing_data_test.sum())
print("Total de registros com outliers na categoria 'normal': ", len(outliers_normal))
print("Total de registros com outliers na categoria 'test-0': ", len(outliers_test))
