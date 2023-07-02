"""
Alteração do algoritmo para melhorar a busca por defeitos -> OneClassSVM
Este algoritmo parte do algoritmo teste17.py (ter isso como referência para rollback)
dado que nosso primeiro sucesso aconteceu no teste16, vamos seguir a partir dele, pois os resultados estão longe da precisão desejada
"""

import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# Carregar o arquivo CSV
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Tratamento da coluna "role"
label_encoder = LabelEncoder()
df['role_numeric'] = label_encoder.fit_transform(df['role'])

# Conversão da coluna "offset_seconds" para formato de data e hora (opcional)
# df['offset_seconds'] = pd.to_datetime(df['offset_seconds'], unit='s')

# Filtragem dos dados das classes "normal" e "test-0"
df_filtered = df[df['role'].isin(['normal', 'test-0'])].copy()

# Balanceamento dos dados
df_normal = df_filtered[df_filtered['role'] == 'normal']
df_test = df_filtered[df_filtered['role'] == 'test-0']
df_normal_resampled = resample(
    df_normal, replace=True, n_samples=len(df_test), random_state=42)
df_balanced = pd.concat([df_normal_resampled, df_test])

# Separação dos dados de treinamento e teste
X_train = df_balanced[df_balanced['role'] == 'normal'].drop(
    ['role', 'role_numeric'], axis=1)
X_test = df_balanced[df_balanced['role'] ==
                     'test-0'].drop(['role', 'role_numeric'], axis=1)

# Treinamento do modelo OneClassSVM
model = OneClassSVM(nu=0.1)  # Ajuste o parâmetro "nu" conforme necessário
model.fit(X_train)

# Predição no conjunto de teste
y_pred = model.predict(X_test)

# Identificação dos registros classificados como anomalias
anomaly_indices = X_test[y_pred == -1].index
anomaly_records = df.loc[anomaly_indices]

# Análise dos resultados
print("Registros classificados como anomalias:")
print(anomaly_records)

# Análise das variáveis que contribuíram para a detecção das anomalias
# Obtendo os scores de decisão para cada amostra
decision_scores = model.decision_function(X_train)

# Calculando a importância das variáveis
decision_scores = model.score_samples(X_train)

# Exibindo a importância das variáveis
print("Importância das variáveis:")
for feature, importance in zip(X_train.columns[:-1], decision_scores):
    print(f"{feature}: {importance}")
