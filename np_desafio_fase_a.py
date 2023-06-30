""" "
Vamos trabalhar nesta primeira tentativa de solução do problema utilizando o algoritmo RandomForestClassifier

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Carregar os dados
df = pd.read_csv('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Converter a coluna "offset_seconds" para formato datetime
df['offset_seconds'] = pd.to_datetime(
    df['offset_seconds'], unit='s') + pd.DateOffset(years=2023, months=5, days=10)

# Separar a base em treinamento e teste
df_train = df[df['role'] == 'normal'].copy()
df_test = df[df['role'] == 'test-0'].copy()

# Criar coluna "classificacao" como alvo
df_train['classificacao'] = 1
df_test['classificacao'] = 0

# Selecionar apenas as colunas numéricas
numeric_columns = df_train.select_dtypes(include=['float64', 'int64']).columns

# Separar as features e o alvo do conjunto de treinamento
X_train = df_train[numeric_columns]
y_train = df_train['classificacao']

# Separar as features e o alvo do conjunto de teste
X_test = df_test[numeric_columns]
y_test = df_test['classificacao']

# Criar o modelo de classificação
model = RandomForestClassifier()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer a predição no conjunto de teste
y_pred = model.predict(X_test)

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Calcular as porcentagens de falsos positivos e falsos negativos
false_positive_rate = fp / (fp + tn) * 100
false_negative_rate = fn / (fn + tp) * 100

# Imprimir a matriz de confusão e as porcentagens de falsos positivos e falsos negativos
print("Matriz de Confusão:")
print(cm)
print("\nPorcentagem de Falsos Positivos: {:.2f}%".format(false_positive_rate))
print("Porcentagem de Falsos Negativos: {:.2f}%".format(false_negative_rate))
