""" 
Neste teste vamos traalhar combinando os melhores algoritmos
"""

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

# Carregando os dados
df = pd.read_csv('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Separando em dados normais e de teste
df_normal = df[df['role'] == 'normal'].drop(columns=['role', 'offset_seconds'])
df_test = df[df['role'] == 'test-0'].drop(columns=['role', 'offset_seconds'])

# Normalizando os dados
scaler = StandardScaler()
df_normal_scaled = scaler.fit_transform(df_normal)
df_test_scaled = scaler.transform(df_test)

# Instanciando o modelo
lof = LocalOutlierFactor(n_neighbors=15, contamination=0.029, novelty=True)

# Treinando o modelo apenas com os dados normais
lof.fit(df_normal_scaled)

# Usando o modelo para prever se um registro em 'test-0' é uma anomalia
df_test['anomaly'] = lof.predict(df_test_scaled)

# Mapeando as previsões para 1 para anomalias e 0 para não-anomalias
df_test['anomaly'] = df_test['anomaly'].map({1: 0, -1: 1})

# Calculando o total de registros anômalos
total_anomalies = df_test['anomaly'].sum()

print("Total de registros: ", len(df_test))
print("Total de registros anômalos previstos: ", total_anomalies)
print("Registros anômalos:")
print(df_test[df_test['anomaly'] == 1])
