""" 
Novo algoritmo ->  algoritmo IsolationForest
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Filtrar os dados da classe 'normal' e 'test-0' para treinamento
df_train = df[(df['role'] == 'normal') | (df['role'] == 'test-0')].copy()

# Remover a coluna 'offset_datetime'
df_train = df_train.drop('offset_datetime', axis=1, errors='ignore')

# Transformar a coluna 'role' em uma representação numérica
label_encoder = LabelEncoder()
df_train['role_numeric'] = label_encoder.fit_transform(df_train['role'])

# Definir as features
features = df_train.columns.drop(['role', 'role_numeric'])

# Padronização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train[features])

# Ajustar o hiperparâmetro contamination
contamination = 0.5
print(f"\nO parâmetro contamination foi ajustado para {contamination}\n")

# Treinamento do modelo Isolation Forest
model = IsolationForest(contamination=contamination)
model.fit(X_train)

# Predição no conjunto de treinamento
y_train_pred = model.predict(X_train)

# Transformar as predições em rótulos (1 para anomalia, -1 para normal)
y_train_pred_labels = [1 if pred == -1 else 0 for pred in y_train_pred]

# Cálculo das métricas
confusion = confusion_matrix(df_train['role_numeric'], y_train_pred_labels)
false_positives = confusion[0, 1] / len(df_train) * 100
false_negatives = confusion[1, 0] / len(df_train) * 100

# Exibição das métricas
print(f"Porcentagem de falsos positivos: {false_positives:.2f}%")
print(f"Porcentagem de falsos negativos: {false_negatives:.2f}%")
print("Matriz de Confusão:")
print(confusion)

# Exibir a matriz de confusão usando Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion / len(df_train), annot=True, fmt=".2%", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Verdadeiro")
plt.show()

# Carregar o arquivo CSV de teste
# Carregar o arquivo de teste
df_test = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Filtrar os dados da classe 'test-0' para teste
df_test = df_test[df_test['role'] == 'test-0'].copy()

# Remover a coluna 'offset_datetime'
df_test = df_test.drop('offset_datetime', axis=1, errors='ignore')

# Transformar a coluna 'role' em uma representação numérica nos dados de teste
df_test['role_numeric'] = label_encoder.transform(df_test['role'])

# Selecionar as features nos dados de teste
X_test = scaler.transform(df_test[features])

# Predição nos dados de teste
y_test_pred = model.predict(X_test)

# Transformar as predições em rótulos (1 para anomalia, -1 para normal) nos dados de teste
y_test_pred_labels = [1 if pred == -1 else 0 for pred in y_test_pred]

# Cálculo das métricas para os dados de teste
confusion_test = confusion_matrix(df_test['role_numeric'], y_test_pred_labels)
false_positives_test = confusion_test[0, 1] / len(df_test) * 100
false_negatives_test = confusion_test[1, 0] / len(df_test) * 100

# Exibição das métricas para os dados de teste
print(
    f"Porcentagem de falsos positivos (dados de teste): {false_positives_test:.2f}%")
print(
    f"Porcentagem de falsos negativos (dados de teste): {false_negatives_test:.2f}%")
print("Matriz de Confusão (dados de teste):")
print(confusion_test)

# Exibir a matriz de confusão para os dados de teste usando Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_test / len(df_test), annot=True, fmt=".2%", cmap="Blues")
plt.title("Matriz de Confusão (dados de teste)")
plt.xlabel("Previsto")
plt.ylabel("Verdadeiro")
plt.show()
