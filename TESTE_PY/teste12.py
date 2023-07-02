""" 
Vamos agora tentar uma abordagem que não envolva clusterização

"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Verificar as dimensões da base de dados
print("Dimensões da base de dados:")
print("Número de linhas:", df.shape[0])
print("Número de colunas:", df.shape[1])

# Verificar valores ausentes
print("Valores ausentes por coluna:")
print(df.isnull().sum())

# Verificar linhas com valores ausentes
print("\nLinhas com valores ausentes:")
print(df[df.isnull().any(axis=1)])

# Verificar o tipo de cada variável
print("\nTipos de dados das variáveis:")
print(df.dtypes)

# Verificar estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe())

# Transformar a coluna 'role' em uma representação numérica
df['role_numeric'] = df['role'].map({'normal': 0, 'test-0': 1})

# Exibir o DataFrame atualizado
print("\nDataFrame com coluna 'role' transformada:")
print(df.head())


# Realizar a codificação one-hot da coluna "role"
encoded_data = pd.get_dummies(df, columns=['role'])

# Obter as colunas numéricas
numeric_columns = encoded_data.select_dtypes(include=['float64']).columns

# Realizar a normalização dos dados numéricos
scaler = StandardScaler()
encoded_data[numeric_columns] = scaler.fit_transform(
    encoded_data[numeric_columns])

# Visualizar as primeiras linhas do dataframe resultante
print(encoded_data.head())

# Verificar as estatísticas descritivas do dataframe resultante
print(encoded_data.describe())


# Separar os dados com base na coluna "role"
train_data = encoded_data[encoded_data['role_test-0'] == 0]
test_data = encoded_data[encoded_data['role_test-0'] == 1]


print("\nImprimindo encoded_data\n")
print(encoded_data.columns)


# Definir as features e o target
features = encoded_data.columns.drop(['role_normal', 'role_numeric'])
target = 'role_test-0'


print("Recursos (features):")
print(features)
print("Variável alvo (target):")
print(target)


# Divisão dos dados de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    encoded_data.drop(columns=['role_normal', 'role_test-0']), encoded_data['role_test-0'], test_size=0.2, random_state=42)


# Treinamento do modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predição no conjunto de teste
y_pred = model.predict(X_test)

# Cálculo das métricas
confusion = confusion_matrix(y_pred, y_test)

false_positives = confusion[0, 1] / len(y_test) * 100
false_negatives = confusion[1, 0] / len(y_test) * 100

# Exibição das métricas
print(f"Porcentagem de falsos positivos: {false_positives:.2f}%")
print(f"Porcentagem de falsos negativos: {false_negatives:.2f}%")
print("Matriz de Confusão:")
print(confusion)


# Exibir a matriz de confusão usando Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion/len(y_test), annot=True, fmt=".2%", cmap="Blues")

plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Verdadeiro")
plt.show()
