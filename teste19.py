""" 
Vamos trabalhar agora com o algoritmo Random Florest e tentar uma convergência mais rápida
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Filtrar os dados da classe 'normal' e 'test-0' para treinamento
df_train = df[(df['role'] == 'normal') | (df['role'] == 'test-0')].copy()

# Remover a coluna 'offset_datetime'
df_train = df_train.drop('offset_datetime', axis=1, errors='ignore')

# Transformar a coluna 'role' em uma representação numérica
label_encoder = LabelEncoder()
df_train['role_numeric'] = label_encoder.fit_transform(df_train['role'])

# Definir as features
features = df_train.columns.drop(['role', 'role_numeric'])

# Separar os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    df_train[features], df_train['role_numeric'], test_size=0.2, stratify=df_train['role_numeric'], random_state=42)

# Padronização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinamento do modelo Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predição no conjunto de teste
y_test_pred = model.predict(X_test_scaled)

# Cálculo das métricas
confusion = confusion_matrix(y_test, y_test_pred)
false_positives = confusion[0, 1] / (confusion[0, 0] + confusion[0, 1]) * 100
false_negatives = confusion[1, 0] / (confusion[1, 0] + confusion[1, 1]) * 100

# Exibição das métricas
print(f"Porcentagem de falsos positivos: {false_positives:.2f}%")
print(f"Porcentagem de falsos negativos: {false_negatives:.2f}%")
print("Matriz de Confusão:")
print(confusion)

# Exibir a matriz de confusão usando Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion / confusion.sum(), annot=True, fmt=".2%", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Verdadeiro")
plt.show()
