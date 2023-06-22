"""
Novo teste => IsolationForest
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Carregando os dados
df = pd.read_csv('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Transformando a coluna categórica "role" em numérica
df['role'] = df['role'].map({'normal': 1, 'test-0': -1})

# Separando o target do resto dos dados
X = df.drop('role', axis=1)
y = df['role']

# Aplicando o modelo IsolationForest para a detecção de anomalias
# O valor da contaminação é ajustado aqui
clf = IsolationForest(contamination=float(.50),
                      max_samples='auto', random_state=42)
clf.fit(X)

# Fazendo as previsões
y_pred = clf.predict(X)

# Avaliando o modelo
print("Classification Report:")
print(classification_report(y, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
