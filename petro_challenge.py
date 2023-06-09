""" 
Este projeto tem por objetivo atender ao desafio da Petrobras visando atender a seguinte demanda:

"A  busca  é  por  uma  ferramenta  capaz  de  realizar  predição  de  falhasem equipamentosutilizando  modelos computacionais  baseados  emtécnicas modernas e disruptivas, como Inteligência Artificial, de forma integrada aos processos   e   softwares   de   gestão   da   companhia,   e   que   forneça   as informações  com  antecedência  adequadapara  ação, com  boa  acurácia(poucos falsospositivos/negativos), indicando o diagnóstico e ações claras e efetivas para que as equipes de operação e manutenção possam evitar ou  dirimir  falhas  potenciais, além  de otimizar  os  planos de  manutenção preventivados equipamentos e sistemas"

"""

# Importe as bibliotecas necessárias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


# Análise gráfica => vamos deixar para depois

'''
import matplotlib.pyplot as plt

# Exemplo de histograma
data['nome_da_coluna'].hist()
plt.show()

# Exemplo de gráfico de dispersão
plt.scatter(data['coluna_x'], data['coluna_y'])
plt.show()

'''
