""" 
Vamos agora explorar redes neurais iniciemos com --> redes neurais autoencoder
Como referência para rollback esse algoritm tem como base o algoritmo teste22.py
Agora que funcionou em teste21, vamos incluir os cálculos de falsos positivos e negativos e calcular a precisão do modelo

Métricas avaliadas

Aqui está a interpretação dos resultados que você obteve:

- Mean Squared Error (MSE): O MSE é uma métrica que mede a média dos erros quadrados entre as previsões do modelo e os valores reais. No seu caso, você obteve uma lista de MSEs para cada amostra de teste. Valores menores indicam que as previsões do modelo estão mais próximas dos valores reais. Por exemplo, o primeiro valor de MSE é 0.00011193.

- Threshold: O threshold é um valor definido para determinar a partir de qual MSE uma amostra é considerada uma anomalia. No seu caso, o threshold calculado foi de 0.0016539611080088014. Portanto, amostras com MSE acima desse valor serão classificadas como anomalias.

- Predictions: Essa lista representa as previsões feitas pelo modelo para as amostras de teste. Cada valor na lista representa se a amostra foi classificada como normal (0) ou anomalia (1). Por exemplo, o primeiro valor na lista de previsões é 0, o que significa que a primeira amostra de teste foi classificada como normal.

- True Labels: Essa lista representa os rótulos verdadeiros das amostras de teste. Cada valor na lista indica se a amostra é normal (0) ou anomalia (1). Por exemplo, o primeiro valor na lista de rótulos verdadeiros é 1, indicando que a primeira amostra de teste é uma anomalia.

- Accuracy: A precisão do modelo é calculada comparando as previsões do modelo com os rótulos verdadeiros das amostras de teste. A precisão indica a porcentagem de amostras que foram corretamente classificadas pelo modelo. No seu caso, a precisão obtida foi de 0.2533814247069432, o que significa que o modelo acertou cerca de 25,3% das classificações das amostras de teste.

Com base nos resultados, parece que o modelo não está performando muito bem. A precisão de 25,3% indica que o modelo está tendo dificuldade em distinguir entre anomalias e operação normal. Talvez seja necessário ajustar o threshold ou explorar outras técnicas para melhorar o desempenho do modelo.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Carregar o arquivo CSV
df = pd.read_csv('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Pré-processamento dos dados
# Remover colunas desnecessárias, como 'offset_seconds'
df = df.drop('offset_seconds', axis=1)

# Separar as features (X) e os rótulos (y)
X = df.drop('role', axis=1)
y = df['role']

# Codificar as variáveis categóricas (se houver)
# Exemplo: se 'role' for 'normal', codificar como 0, se for 'test-0', codificar como 1
y = y.map({'normal': 0, 'test-0': 1})

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Normalizar os dados numéricos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar o modelo do autoencoder
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(X_train.shape[1], activation='linear'))

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model.fit(X_train, X_train, epochs=50, batch_size=32, verbose=1)

# Avaliar o modelo nos dados de teste
predictions = model.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)

# Definir um limiar para detecção de anomalias
threshold = np.percentile(mse, 95)

# Classificar as amostras como normal (0) ou anomalia (1)
y_pred = np.where(mse > threshold, 1, 0)

# Calcular a precisão do modelo
accuracy = np.mean(y_pred == y_test)

# Exibir as métricas e precisão do modelo
print(f"Mean Squared Error (MSE): {mse}")
print(f"Threshold: {threshold}")
print(f"Predictions: {y_pred}")
print(f"True Labels: {y_test}")
print(f"Accuracy: {accuracy}")

