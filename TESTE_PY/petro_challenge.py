""" 
Este projeto tem por objetivo atender ao desafio da Petrobras visando atender a seguinte demanda:

"A  busca  é  por  uma  ferramenta  capaz  de  realizar  predição  de  falhasem equipamentosutilizando  modelos computacionais  baseados  emtécnicas modernas e disruptivas, como Inteligência Artificial, de forma integrada aos processos   e   softwares   de   gestão   da   companhia,   e   que   forneça   as informações  com  antecedência  adequadapara  ação, com  boa  acurácia(poucos falsospositivos/negativos), indicando o diagnóstico e ações claras e efetivas para que as equipes de operação e manutenção possam evitar ou  dirimir  falhas  potenciais, além  de otimizar  os  planos de  manutenção preventivados equipamentos e sistemas"

"""

# Importe as bibliotecas necessárias

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Carregando os dados

data = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

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

# =========================================================================================

# Análise gráfica => vamos deixar para depois o seu aprimoaramento
# Definir estilo do seaborn

'''
# Remover a coluna 'role' do DataFrame
data_without_role = data.drop('role', axis=1)

# Loop para gerar histogramas e gráficos de caixa
for column in data_without_role.columns:
    # Plotar histograma
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data_without_role[column], kde=True, color='skyblue')
    plt.xlabel(column)
    plt.ylabel('Frequência')
    plt.title('Histograma de ' + column)
    plt.show()

    # Plotar gráfico de boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data_without_role[column], color='lightcoral')
    plt.xlabel('Variável')
    plt.ylabel(column)
    plt.title('Gráfico de Boxplot: ' + column)
    plt.show()

    # Loop para gerar gráficos de dispersão com as colunas seguintes
    for column2 in data_without_role.columns:
        if column != column2:
            # Plotar gráfico de dispersão
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=data_without_role, x=column,
                            y=column2, color='lightcoral')
            plt.xlabel(column)
            plt.ylabel(column2)
            plt.title('Gráfico de Dispersão: ' + column + ' vs ' + column2)
            plt.show()

# Analisando a correlação entre as variáveis
# Calculando a matriz de correlação
correlation_matrix = data_without_role.corr()

# Criando um mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Mapa de Calor - Correlação entre Variáveis')
plt.show()

# Identificando as principais correlações positivas e negativas
threshold = 0.5  # Definindo um valor de limite para destacar as correlações significativas
correlation_pairs = correlation_matrix.unstack().sort_values(ascending=False)
significant_correlations = correlation_pairs[(
    correlation_pairs > threshold) & (correlation_pairs < 1)]
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

'''
# ================================== Iniciando com os testes mesmo ==========================
# Vamos agora começar a olhar para o arquivo de forma particular, separando operação Normal e
# dados de Teste

# Separar o DataFrame para operação normal
df_normal = data.loc[data['role'] == 'normal']

# Separar o DataFrame para teste
df_test = data.loc[data['role'] == 'test-0']

# Preparando os dados - identificando as variáveis

# Preparação dos dados
# Remover a coluna "role" dos dataframes
df_normal = df_normal.drop('role', axis=1)
df_test = df_test.drop('role', axis=1)

# Realizar codificação one-hot para colunas categóricas
encoder = OneHotEncoder()
df_normal_encoded = pd.get_dummies(df_normal, columns=['role'])
df_test_encoded = pd.get_dummies(df_test, columns=['role'])

# Separar a variável de destino (target) do restante dos dados
X_train = df_normal_encoded.drop('target_column', axis=1)
y_train = df_normal_encoded['target_column']
X_test = df_test_encoded.drop('target_column', axis=1)
y_test = df_test_encoded['target_column']

# Treinamento do modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Avaliação do modelo
y_pred = model.predict(X_test)

# Métricas de avaliação
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Cálculo de falsos positivos e falsos negativos em termos percentuais
total_samples = len(y_test)
false_positives = cm[0, 1]
false_negatives = cm[1, 0]
false_positives_percent = false_positives / total_samples * 100
false_negatives_percent = false_negatives / total_samples * 100

# Antecedência de detecção de falha
failure_detected = y_test[y_pred == 1].index.min()

# Visualização da matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusão')
plt.show()

# Visualização dos resultados
print('Métricas de Avaliação:')
print('Acurácia:', accuracy)
print('Precisão:', precision)
print('Recall:', recall)
print('F1-score:', f1)
print('\nFalsos Positivos (%):', false_positives_percent)
print('Falsos Negativos (%):', false_negatives_percent)
print('Antecedência de Detecção de Falha:', failure_detected)
