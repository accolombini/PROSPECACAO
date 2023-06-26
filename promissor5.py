"""
Vamos agora trabalhar com o método LOF -> sendo este mais um algoritmo a ser explorado na busca de uma solução para nosso problema
"""

from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Carregando os dados
df = pd.read_csv('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Transformando a coluna categórica "role" em numérica
df['role'] = df['role'].map({'normal': 1, 'test-0': -1})

# Separando o target do resto dos dados
X = df.drop('role', axis=1)
y = df['role']

# Normalizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Instanciando o modelo LOF
lof = LocalOutlierFactor(n_neighbors=15, contamination=0.05)

# Ajustar o modelo aos dados e fazer previsões
y_pred = lof.fit_predict(X_scaled)

# Como o LOF retorna -1 para anomalias e 1 para valores normais, podemos reverter isso para ficar igual ao seu caso
y_pred = [1 if x == -1 else -1 for x in y_pred]

# Imprimindo a matriz de confusão e o relatório de classificação
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

print("Classification Report:")
print(classification_report(y, y_pred))

# Calculando a taxa de falsos positivos e falsos negativos
conf_matrix = confusion_matrix(y, y_pred)
fp = conf_matrix[0][1]
fn = conf_matrix[1][0]
total = conf_matrix.sum()

fp_rate = fp / total * 100
fn_rate = fn / total * 100

print(f'False Positive Rate: {fp_rate}%')
print(f'False Negative Rate: {fn_rate}%')
