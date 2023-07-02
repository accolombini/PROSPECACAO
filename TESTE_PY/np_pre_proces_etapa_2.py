"""
VNesta ETAPA faremos o pré processamento do dados, vamos levar em conta alguns passos importantes:
    Tratar os valores ausentes: Isso pode ser feito de várias maneiras, como excluir as linhas com valores ausentes, preenchê-las com a média ou mediana dos dados (para dados numéricos) ou a moda (para dados categóricos). O método escolhido dependerá do seu conhecimento específico sobre os dados e os requisitos do seu projeto.

Remover outliers: Outliers podem ser identificados usando o método do escore Z, o intervalo interquartil (IQR) ou outros métodos. Eles podem ser tratados excluindo as linhas com outliers, substituindo-os por um valor (como a média ou mediana) ou deixando-os intactos, dependendo do contexto.

Converter a coluna "offset_seconds" para o formato datetime: Para isso, primeiro precisamos entender o que exatamente essa coluna representa. Se for uma representação do tempo em segundos desde um determinado ponto de partida, podemos converter para o formato datetime usando a função pd.to_datetime.

"""

# Importando as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import numpy as np

# Configurações para exibir todas as colunas e linhas
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)  # Nenhum limite para o número de linhas

# Carregar os dados
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Convertendo offset_seconds para o formato datetime
# Considerando que o offset_seconds é o tempo em segundos desde 1970-01-01
df['offset_seconds'] = pd.to_datetime(df['offset_seconds'], unit='s')

# Tratando os valores ausentes - preenchendo com a mediana
for column in df.columns:
    if np.issubdtype(df[column].dtype, np.number):  # Verificar se o dtype é subclasse de np.number
        median = df[column].median()
        df[column].fillna(median, inplace=True)

# Seleciona colunas numéricas, exceto 'offset_seconds'
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col not in ['offset_seconds']]

# Calcula os quantiles e o IQR para as colunas numéricas
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Identifica os outliers
outliers = (df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))

# Cria um dataframe separado que contém apenas os outliers
df_outliers = df[outliers.any(axis=1)].copy()

print("Outliers identificados:")
print(df_outliers)

# Substitui os outliers pelas medianas correspondentes
for col in numeric_cols:
    median = df[col].median()
    df.loc[outliers[col], col] = median

# Agora, a base de dados está pronta para a próxima fase de análise.

# Salva o DataFrame sem outliers em um arquivo .xlsx
df.to_excel('DADOS/OUTLIERS_FREE.xlsx', index=False)
