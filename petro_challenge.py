""" 
Este projeto tem por objetivo atender ao desafio da Petrobras visando atender a seguinte demanda:

"A  busca  é  por  uma  ferramenta  capaz  de  realizar  predição  de  falhasem equipamentosutilizando  modelos computacionais  baseados  emtécnicas modernas e disruptivas, como Inteligência Artificial, de forma integrada aos processos   e   softwares   de   gestão   da   companhia,   e   que   forneça   as informações  com  antecedência  adequadapara  ação, com  boa  acurácia(poucos falsospositivos/negativos), indicando o diagnóstico e ações claras e efetivas para que as equipes de operação e manutenção possam evitar ou  dirimir  falhas  potenciais, além  de otimizar  os  planos de  manutenção preventivados equipamentos e sistemas"

"""

# Importe as bibliotecas necessárias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Carregando os dados

data = pd.read_csv('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Manipulando os dados conforme necessidade

# Visualização inicial dos dados
# Exibir as primeiras 5 linhas do DataFrame

print(f"As 5 primeiras linhas do arquivo \n{data.head()} \n")

# Verificar o número de linhas e colunas do DataFrame
print(f"O formato dos dados -> {data.shape} \n")

# Verificar os nomes das colunas
print(f"O nome das colunas \n{data.columns}\n")

# Separando os dados entre Normal e Teste_0 -> coluna role será a referência

# Verificar as variáveis únicas na coluna "role"

unique_roles = data['role'].unique()

print(f"Variáveis únicas na coluna 'role':\n")
for role in unique_roles:
    print(role)

print("\n")

# Estatísticas descritivas
# Obter estatísticas descritivas básicas

print(f"Estatísticas descritivas \n{data.describe()}\n")

# Remover a primeira coluna ("role")

data_numeric = data.drop('role', axis=1)

# Calcular a média de cada coluna

mean_values = data_numeric.mean()
print(f"A Média por coluna do DataFrame é:\n{mean_values}\n")

# Calcular o desvio padrão de cada coluna

std_values = data_numeric.std()
print(f" O Desvio padrão por coluna do DataFrame é:\n{std_values}\n")

# Calcular o valor máximo e mínimo de cada coluna
max_values = data_numeric.max()
min_values = data_numeric.min()

# Obter o tamanho máximo de valor entre os máximos e mínimos
# Mínimo de 4 caracteres para "Máx" e "Mín"
max_value_width = max(max_values.astype(str).map(len).max(), 4)
column_width = max(max(data_numeric.columns.astype(str).map(len).max(), 10), len(
    "Coluna"))  # Tamanho mínimo de 10 para nome da coluna

# Exibir valor máximo e mínimo lado a lado com alinhamento
print("Valores máximo e mínimo por coluna:")
for column, max_value, min_value in zip(data_numeric.columns, max_values, min_values):
    formatted_column = f"{column:{column_width}}"
    formatted_max = f"Máx = {max_value:.2f}".replace(".", ",")
    formatted_min = f"Mín = {min_value:.2f}".replace(".", ",")
    print(f"{formatted_column}: {formatted_max:{max_value_width + 6}} , {formatted_min:{max_value_width + 6}}")


# Análise gráfica => vamos deixar para depois o seu aprimoaramento
# Definir estilo do seaborn

sns.set_style('whitegrid')

# Loop para gerar histogramas e gráficos de caixa
for i, column in enumerate(data.columns):
    # Plotar histograma
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], kde=True, color='skyblue')
    plt.xlabel(column)
    plt.ylabel('Frequência')
    plt.title('Histograma de ' + column)

# Loop para gerar gráficos de boxplot
for i, column in enumerate(data.columns):
    # Plotar gráfico de boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data[column], color='lightcoral')
    plt.xlabel('Variável')
    plt.ylabel(column)
    plt.title('Gráfico de Boxplot: ' + column)
    plt.show()

    # Loop para gerar gráficos de dispersão com as colunas seguintes
    for j in range(i + 1, len(data.columns)):
        # Plotar gráfico de dispersão
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=data, x=column,
                        y=data.columns[j], color='lightcoral')
        plt.xlabel(column)
        plt.ylabel(data.columns[j])
        plt.title('Gráfico de Dispersão: ' +
                  column + ' vs ' + data.columns[j])
        plt.show()

# Analisando a correlação entre as variáveis
# Calculando a matriz de correlação

correlation_matrix = data.corr()

# Criando um mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Mapa de Calor - Correlação entre Variáveis')
plt.show()

# Identificando as principais correlações positivas e negativas
threshold = 0.5  # Definindo um valor de limite para destacar as correlações significativas
correlation_pairs = correlation_matrix.unstack().sort_values(ascending=False)
significant_correlations = correlation_pairs[(correlation_pairs > threshold) & (correlation_pairs < 1)]
positive_correlations = significant_correlations[significant_correlations > 0]
negative_correlations = significant_correlations[significant_correlations < 0]

# Imprimindo as principais correlações positivas
print('Principais correlações positivas:')
for pair, correlation in positive_correlations.items():
    variable_1, variable_2 = pair
    print(f'{variable_1} e {variable_2}: {correlation:.2f}')

# Imprimindo as principais correlações negativas
print('\nPrincipais correlações negativas:')
for pair, correlation in negative_correlations.items():
    variable_1, variable_2 = pair
    print(f'{variable_1} e {variable_2}: {correlation:.2f}')

# ================================== Iniciando com os testes mesmo ==========================
# Vamos agora começar a olhar para o arquivo de forma particular, separando operação Normal e
# dados de Teste

# Separar o DataFrame para operação normal
df_normal = data.loc[data['role'] == 'normal']

# Separar o DataFrame para teste
df_test = data.loc[data['role'] == 'test-0']

# Preparando os dados - identificando as variáveis

def preprocess_data(df):
    # Identificar colunas numéricas e categóricas
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    # Tratar colunas categóricas usando codificação one-hot
    df_encoded = pd.get_dummies(df, columns=categorical_columns)

    # Separar a variável de destino do restante dos dados
    target_column = 'target'  # Substitua pelo nome da sua coluna de destino
    target = df_encoded.pop(target_column)

    return df_encoded, target

# Exemplo de uso
df_normal_encoded, target = preprocess_data(df_normal)
df_test_encoded, target_test = preprocess_data(df_test)
