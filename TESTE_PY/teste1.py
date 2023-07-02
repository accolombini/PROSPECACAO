import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o arquivo CSV
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Criar o modelo de Isolation Forest
model = IsolationForest(contamination=0.1, random_state=0)
model.fit(df.select_dtypes(include=['float64', 'int64']))

# Classificar as amostras
df['anomaly'] = model.predict(df.select_dtypes(include=['float64', 'int64']))
df['anomaly'] = np.where(df['anomaly'] == 1, 'normal', 'anomaly')

# Criar DataFrame de teste apenas com as amostras de teste
df_test = df[df['role'] == 'test-0']

# Calcular a matriz de confusão
# Calcule a matriz de confusão
cm = confusion_matrix(df_test['role'], df_test['anomaly'], labels=[
                      'normal', 'test-0'])


# Calcular falsos positivos e falsos negativos
false_positives = cm[0, 1]
false_negatives = cm[1, 0]
total_normal = np.sum(cm[0, :])
total_anomaly = np.sum(cm[1, :])
false_positives_percent = false_positives / total_normal * 100
false_negatives_percent = false_negatives / total_anomaly * 100

# Calcular o tempo de antecedência de detecção
df_test['offset_seconds'] = pd.to_datetime(df_test['offset_seconds'])
df_test = df_test.sort_values('offset_seconds')
failure_detected = df_test[df_test['anomaly']
                           == 'anomaly']['offset_seconds'].min()

# Exibir resultados
print('Resultados:')
print('Falsos Positivos (%):', false_positives_percent)
print('Falsos Negativos (%):', false_negatives_percent)
print('Antecedência de Detecção:', failure_detected)

# Plotar a matriz de confusão
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.xticks([0.5, 1.5], ['Normal', 'Anomalia'])
plt.yticks([0.5, 1.5], ['Normal', 'Anomalia'])
plt.show()
