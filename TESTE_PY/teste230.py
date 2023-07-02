"""
Particionando e calculando as Estatísticas Descritivas da Base
"""
import pandas as pd

# Carregando os dados em csv e salvando em Exxcel
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')
df.to_excel('DADOS/Adendo A.2_Conjunto de Dados_DataSet.xlsx', index=False)

# Calculando as estatísticas descritivas e salvando em Exxcel

desc_total = df.describe()
desc_total.to_excel('DADOS/EST_DESC_TOTAL.xlsx')

# Dividindo a base em normal e test-0 e salvando em Exxcel

df_normal = df[df['role'] == 1]
df_test0 = df[df['role'] == -1]

df_normal.to_excel('DADOS/DADOS_NORMAL.xlsx', index=False)
df_test0.to_excel('DADOS/DADOS_TEST-0.xlsx', index=False)

# CAlculando as estatísticas descritivas de cada base e salvando em Excel

desc_normal = df_normal.describe()
desc_test0 = df_test0.describe()

desc_normal.to_excel('DADOS/EST_DESC_NORMAL.xlsx')
desc_test0.to_excel('DADOS/EST_DESC_TEST-0.xlsx')
