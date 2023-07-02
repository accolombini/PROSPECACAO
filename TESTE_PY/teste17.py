"""
Alteração do algoritmo para melhorar a busca por defeitos -> OneClassSVM
dado que nosso primeiro sucesso aconteceu no teste16, vamos seguir a partir dele, pois os resultados estão longe da precisão desejada
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.svm import OneClassSVM
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

# Treinamento do modelo One-Class SVM
# Nu é um parâmetro de controle da taxa de anomalias esperada
model = OneClassSVM(nu=0.795)
model.fit(X_train)
#print(
 #   f"Para acompanhar os ajustes de nu segue o valor adotado nesta análise -> {.795}")

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
