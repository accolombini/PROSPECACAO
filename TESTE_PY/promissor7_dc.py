""" 
Neste teste vamos adicionar um estágio de data cleanning antes de
trabalhar combinando os melhores algoritmos até aqui utilizados.
"""

import pandas as pd

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

