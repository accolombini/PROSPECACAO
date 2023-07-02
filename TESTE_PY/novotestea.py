"""
Novo teste => OneClassSVM
"""

from sklearn.svm import OneClassSVM
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Carregando os dados
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Transformando a coluna categórica "role" em numérica
df['role'] = df['role'].map({'normal': 1, 'test-0': -1})

# Separando o target do resto dos dados
X = df.drop('role', axis=1)
y = df['role']

# Normalizando os dados, que é um passo importante quando se utiliza SVM
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Aplicando o modelo OneClassSVM
clf = OneClassSVM(kernel='rbf', nu=0.95, gamma='auto').fit(X)

# Fazendo as previsões
y_pred = clf.predict(X)

# Avaliando o modelo
print("Classification Report:")
print(classification_report(y, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
