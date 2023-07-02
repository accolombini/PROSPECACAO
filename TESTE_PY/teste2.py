import pandas as pd
from sklearn.ensemble import IsolationForest

# Carregar o arquivo CSV
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Criar um DataFrame de treinamento apenas com as instâncias normais
df_train = df[df['role'] == 'normal']

# Treinar o modelo Isolation Forest
model = IsolationForest(contamination='auto')
model.fit(df_train.select_dtypes(include=['float64', 'int64']))

# Aplicar o modelo aos dados completos
df['anomaly'] = model.predict(df.select_dtypes(include=['float64', 'int64']))
df['anomaly'] = df['anomaly'].map({1: 'normal', -1: 'anomaly'})

# Verificar os resultados
print(df['anomaly'].value_counts())

# Calcular a porcentagem de anomalias
anomaly_percentage = (df['anomaly'].value_counts()['anomaly'] / len(df)) * 100
print(f'Porcentagem de anomalias: {anomaly_percentage:.2f}%')
