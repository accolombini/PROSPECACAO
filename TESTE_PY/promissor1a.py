"""
    Refinando o algogitmo --> hoje 12/06 => vamos explorar o cluster 5 na busca de possível
    solução para o problema
    Este algoritmo tem como referência para rollback o algoritmo teste10.py
    Problema encontrado o Cluster 5 esta vazio - vamos tentar uma nova abordagem. Uso do GMM levou a muitos problemas.Vamos seguir continuando com novos testes.
    Neste teste trabalhamos com uma variação de 3 sigmas

"""

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o arquivo CSV
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Dimensões do DataFrame original
print(f"\nDimensões do DataFrame original => {df.shape}")

# Separar o DataFrame em treinamento e teste
treinamento = df[df['role'] == 'normal'].copy()
teste = df[df['role'] == 'test-0'].copy()

# Dimensões do DataFrame de treinamento
print(
    f"\nDimensões do DataFrame de treinamento. Antes de dropar as colunas => {treinamento.shape}")

# Dimensões do DataFrame de teste
print(
    f"\nDimensões do DataFrame de teste. Antes de dropar as colunas => {teste.shape}\n")
print(teste.shape)

# Remover as colunas 'role' e 'offset_seconds' dos dataframes
treinamento.drop(['role', 'offset_seconds'], axis=1, inplace=True)
teste.drop(['role', 'offset_seconds'], axis=1, inplace=True)

# Dimensões do DataFrame de treinamento
print(
    f"\nDimensões do DataFrame de treinamento. Após dropar as colunas => {treinamento.shape}")

# Dimensões do DataFrame de teste
print(
    f"\nDimensões do DataFrame de teste. Após dropar as colunas => {teste.shape}\n")
print(teste.shape)

# Calcular as estatísticas descritivas para treinamento
treinamento_stats = treinamento.describe(include='all')

# Calcular as estatísticas descritivas para teste
teste_stats = teste.describe(include='all')

# Exibir as estatísticas descritivas completas
print("\nEstatísticas descritivas para treinamento:\n")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(treinamento_stats)
print("\nEstatísticas descritivas para teste:\n")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(teste_stats)

# Identificar as variáveis com variação superior a três sigmas
variables_with_outliers = []
outlier_records = {}
multiple_outliers_records = []

for column in teste.columns:
    std = teste[column].std()
    mean = teste[column].mean()
    upper_threshold = mean + 3 * std
    lower_threshold = mean - 3 * std

    outliers = teste[(teste[column] > upper_threshold) |
                     (teste[column] < lower_threshold)]
    if not outliers.empty:
        variables_with_outliers.append(column)
        outlier_records[column] = outliers.index.tolist()
        multiple_outliers_records.extend(outliers.index.tolist())

# Contar a ocorrência dos registros com mais de uma variável com valores superiores a três sigmas
record_counts = {}
for record in multiple_outliers_records:
    if record in record_counts:
        record_counts[record] += 1
    else:
        record_counts[record] = 1

# Mostrar as variáveis com variação superior a três sigmas, os registros correspondentes, a contagem de registros e as variáveis múltiplas
for variable in variables_with_outliers:
    print(f"\nVariável: {variable}")
    print("\nRegistros com variação superior a três sigmas:")
    print(outlier_records[variable])
    print("\nTotal de registros com variação superior a três sigmas:",
          len(outlier_records[variable]))
    print()

print("\nRegistros com mais de uma variável com valores superiores a três sigmas:")
for record, count in record_counts.items():
    if count > 1:
        print(f"Registro: {record}")
        print(
            f"Número de variáveis com valores superiores a três sigmas: {count}")
        print("Variáveis:", end=" ")
        for variable in variables_with_outliers:
            if record in outlier_records[variable]:
                print(variable, end=" ")
        print()

# Identificar eventos com valores superiores a três sigmas seguidos por registros subsequentes zerados
print("\nEventos com valores superiores a três sigmas seguidos por registros subsequentes zerados:")
for record, count in record_counts.items():
    if count > 1:
        variables = []
        for variable in variables_with_outliers:
            if record in outlier_records[variable]:
                variables.append(variable)
        for variable in variables:
            index = outlier_records[variable].index(record)
            subsequent_records = teste.iloc[index+1:]
            zero_subsequent_records = subsequent_records[subsequent_records[variable] == 0]
            if zero_subsequent_records.shape[0] > 0:
                print(f"\nRegistro: {record}")
                print("\nVariáveis afetadas:", end=" ")
                print(*variables, sep=", ")
                print("Registros subsequentes zerados:")
                print(zero_subsequent_records)
                print()
                break
