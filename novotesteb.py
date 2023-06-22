"""
Novo teste => Redes Neurais autoencoder
"""

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.layers import Dropout

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

# Definindo a arquitetura do autoencoder
input_dim = X.shape[1]
encoding_dim = int(input_dim / 2) - 1
hidden_dim = int(encoding_dim / 2)

input_layer = Input(shape=(input_dim, ))

# Adicionando mais camadas ao encoder
encoder = Dense(encoding_dim, activation="tanh")(input_layer)
encoder = Dropout(0.5)(encoder)
encoder = Dense(hidden_dim, activation="relu")(encoder)

# Adicionando mais camadas ao decoder
decoder = Dense(hidden_dim, activation='tanh')(encoder)
decoder = Dropout(0.5)(decoder)
decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Treinando o novo autoencoder
autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=32, shuffle=True)

# Obtendo a reconstrução dos dados
predictions = autoencoder.predict(X_scaled)

# Calculando o erro de reconstrução
mse = np.mean(np.power(X_scaled - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse, 'true_class': y})

# Definindo um threshold e calculando a matriz de confusão
threshold = np.quantile(error_df.reconstruction_error, 0.99)
y_pred = [1 if e > threshold else -
          1 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(y, y_pred)


print("Confusion Matrix:")
print(conf_matrix)

print("Classification Report:")
print(classification_report(y, y_pred))

# Calculando a porcentagem de falsos positivos e falsos negativos
fp = conf_matrix[0][1]
fn = conf_matrix[1][0]
total = conf_matrix.sum()

fp_rate = fp / total * 100
fn_rate = fn / total * 100

print(f'False Positive Rate: {fp_rate}%')
print(f'False Negative Rate: {fn_rate}%')
