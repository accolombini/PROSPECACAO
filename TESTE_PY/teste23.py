""" 
Vamos agora explorar redes neurais iniciemos com --> redes neurais autoencoder
Agora que funcionou em teste21, vamos incluir os cálculos de falsos positivos e negativos e calcular a precisão do modelo

Objetivo algora é trabalhar os hiperparâmetros

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Carregar o arquivo CSV
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

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
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(X_train.shape[1], activation='linear'))

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Tunar os hiperparâmetros do autoencoder
epochs = 100
batch_size = 64

# Treinar o modelo
model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=1)

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
