""" 
Vamos agora tentar uma abordagem que não envolva clusterização

Para uso futuro se resolver voltar a esse código -> df_train = df_train.drop('offset_datetime', axis=1)
esse trecho de código deve ser adicionando antes de definir as features <>

"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Verificar as dimensões da base de dados
print("\nDimensões da base de dados:")
print("\nNúmero de linhas:", df.shape[0])
print("\nNúmero de colunas:", df.shape[1])
print(f"\nO formtato da base de DADOS sob análise é => {df.shape}")

# Verificar valores ausentes
print("\nValores ausentes por coluna:")
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

# Converter a coluna 'offset_seconds' para datetime
df['offset_datetime'] = pd.to_datetime(df['offset_seconds'], unit='s')

# Transformar a coluna 'role' em uma representação numérica
df['role_numeric'] = df['role'].map({'normal': 0, 'test-0': 1})

# Exibir o DataFrame atualizado
print("\nDataFrame com coluna 'role' transformada:")
print(df.head())

# Realizar a codificação one-hot da coluna "role"
encoded_data = pd.get_dummies(df, columns=['role'])

# Obter as colunas numéricas
numeric_columns = encoded_data.select_dtypes(include=['float64']).columns

# Exibir estatísticas descritivas antes da normalização
print("\nEstatísticas descritivas antes da normalização. Testando antes de normalizar =====")
print(encoded_data[numeric_columns].describe())

# Realizar a normalização dos dados numéricos
scaler = StandardScaler()
encoded_data[numeric_columns] = scaler.fit_transform(
    encoded_data[numeric_columns])

# Exibir estatísticas descritivas após a normalização
print("\nEstatísticas descritivas após a normalização testnado após normalizar ====")
print(encoded_data[numeric_columns].describe())

# Redefinir o índice do DataFrame
encoded_data.reset_index(drop=True, inplace=True)

# Definir as features e o target
features = encoded_data.columns.drop(['role_normal', 'role_test-0'])
target = ['role_normal', 'role_test-0']


# Divisão dos dados de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    encoded_data[features], encoded_data[target], test_size=0.2, random_state=42)

# Imprimir o tamanho dos conjuntos
print("\nVerificando a divisão de dados procurando erros ===========\n")
print("\nTamanho do conjunto de treinamento (X_train):", X_train.shape)
print("\nTamanho do conjunto de teste (X_test):", X_test.shape)
print("\nTamanho do conjunto de treinamento (y_train):", y_train.shape)
print("\nTamanho do conjunto de teste (y_test):", y_test.shape)

# Imprimir as primeiras linhas dos conjuntos de treinamento e teste
print("\nConjunto de treinamento:")
print(y_train.head())

print("\nConjunto de teste:")
print(y_test.head())

print()

# Remover a coluna 'offset_datetime' do conjunto de recursos X_train
X_train = X_train.drop('offset_datetime', axis=1)

# Remover a coluna 'offset_datetime' do conjunto de teste X_test
X_test = X_test.drop('offset_datetime', axis=1)

# Treinamento do modelo
# model = RandomForestClassifier()
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

accuracy = model.score(X_train, y_train)
print("\nPrecisão do modelo nos dados de treinamento -> ", accuracy)

# Predição no conjunto de teste
y_pred = model.predict(X_test)


# Cálculo das métricas
confusion = confusion_matrix(y_pred.argmax(
    axis=1), y_test.values.argmax(axis=1))

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
