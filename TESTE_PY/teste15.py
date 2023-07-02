""" 
Nesta aboradagem vamos rever alguns pontos que possam ter impactado nos resultados negativos do teste anterior

"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Filtrar os dados da classe 'normal' para treinamento
df_train = df[df['role'] == 'normal'].copy()

# Filtrar os dados da classe 'test-0' para teste
df_test = df[df['role'] == 'test-0'].copy()

# Remover a coluna 'offset_datetime'
df_train = df_train.drop('offset_datetime', axis=1, errors='ignore')
df_test = df_test.drop('offset_datetime', axis=1, errors='ignore')

# Transformar a coluna 'role' em uma representação numérica
df_train['role_numeric'] = 0
df_test['role_numeric'] = 1

# Concatenar os dataframes de treinamento e teste
df_combined = pd.concat([df_train, df_test], ignore_index=True)

# Definir as features e o target
features = df_combined.columns.drop(['role', 'role_numeric'])
target = 'role_numeric'

# Divisão dos dados em treinamento e teste
X = df_combined[features]
y = df_combined[target]

# Padronização dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separação novamente dos dados em treinamento e teste
X_train = X[:len(df_train)]
y_train = y[:len(df_train)]
X_test = X[len(df_train):]
y_test = y[len(df_train):]

# Treinamento do modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Precisão do modelo nos dados de treinamento
accuracy_train = model.score(X_train, y_train)
print("Precisão do modelo nos dados de treinamento:", accuracy_train)

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
sns.heatmap(confusion / len(y_test), annot=True, fmt=".2%", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Verdadeiro")
plt.show()
