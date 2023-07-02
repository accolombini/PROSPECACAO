"""
Faremos neste algoritmo a análise exploratória de dados, que deverá abranger as seguintes etapas:

Primeiro, precisamos entender completamente os dados disponíveis. Faremos uma análise exploratória, que inclui a compreensão das variáveis existentes, sua distribuição, como elas estão relacionadas umas às outras e como são correlacionadas com a falha do equipamento.

"""

# Importando as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carregar os dados
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Visualizar as primeiras linhas do dataframe
print(f"\nVisualização das 5 primeiras linhas do dataframe\n")
print(df.head())

# Informações gerais sobre os dados
print(f"\nInformações gerais do dataframe\n")
print(df.info())

# Estatísticas descritivas dos dados (excluindo 'role' e 'offset_seconds')
print(f"Estatísticas descritivas descartando se as colunas role e offset_seconds\n")
df_to_describe = df.drop(columns=['role', 'offset_seconds'])
print(df_to_describe.describe())

# Contagem de valores para a coluna 'role'
print(f"\nContagem de dados da coluna role\n")
print(df['role'].value_counts())

# Verificar se há valores faltantes
print(f"\nVerificando se há dados faltantes na base\n")
print(df.isna().sum())

# Visualizar a distribuição das variáveis (excluindo 'role' e 'offset_seconds')
print(f"\nVisualizar a distribuição das variáveis excluindo role e offset_seconds\n")
for column in df_to_describe.columns:
    plt.figure(figsize=(8,6))
    sns.kdeplot(x=df_to_describe[column])  # Mudança aqui
    plt.axvline(df_to_describe[column].quantile(0.25), color='g', linestyle='dashed', linewidth=2)  # Primeiro quartil
    plt.axvline(df_to_describe[column].quantile(0.75), color='r', linestyle='dashed', linewidth=2)  # Terceiro quartil
    plt.title(column)
    plt.xlabel(column)  # Nome do eixo x
    plt.ylabel('Densidade')  # Nome do eixo y
    plt.show()

# Mapa de calor das correlações (excluindo 'role' e 'offset_seconds')
print(f"\nVisualizar correlação das variáveis excluindo role e offset_seconds\n")
plt.figure(figsize=(24,12))  # Ajustar o tamanho da figura
sns.heatmap(df_to_describe.corr(), annot=True, cmap='coolwarm')
plt.show()
