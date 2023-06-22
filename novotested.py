"""
Novo teste => Redes Neurais autoencoder
dividndo a base em normal para treinamento e test-0 para teste
Realizando tambe´m o balanceamento dos dados
Tentando técnicas mais sofisticadas para balanceamento e aumentando o número de épocas.
"""

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from imblearn.over_sampling import ADASYN  # Alterado para ADASYN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN

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

# Aplicando o ADASYN para balancear os dados
adasyn = ADASYN(random_state=42)
X_res, y_res = adasyn.fit_resample(X_scaled, y)

# Separando os dados de treinamento e teste
X_normal = X_res[y_res == 1]
X_test0 = X_res[y_res == -1]

# Continua a construção e treinamento do modelo como antes ...
# Definindo a arquitetura do autoencoder
input_dim = X_normal.shape[1]
encoding_dim = int(input_dim / 2) - 1
hidden_dim = int(encoding_dim / 2)

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh")(input_layer)
encoder = Dense(hidden_dim, activation="relu")(encoder)
decoder = Dense(hidden_dim, activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Treinando o autoencoder (com 500 épocas)  # Aumentado para 500 épocas
autoencoder.fit(X_normal, X_normal, epochs=500, batch_size=32, shuffle=True)

# Obtendo a reconstrução dos dados
predictions_test0 = autoencoder.predict(X_test0)

# Calculando o erro de reconstrução
mse = np.mean(np.power(X_test0 - predictions_test0, 2), axis=1)
error_df = pd.DataFrame(
    {'reconstruction_error': mse, 'true_class': y_res[y_res == -1]})  # Alterado para y_res

# Definindo um threshold e calculando a matriz de confusão
threshold = np.quantile(error_df.reconstruction_error, 0.9)


# Definindo um threshold e calculando a matriz de confusão
threshold = np.quantile(error_df.reconstruction_error, 0.9)
y_pred = [1 if e > threshold else -
          1 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)

# Imprimindo a matriz de confusão e o relatório de classificação
print("Confusion Matrix:")
print(conf_matrix)

print("Classification Report:")
print(classification_report(error_df.true_class, y_pred, zero_division=1))

# O código permanece o mesmo até o relatório de classificação

print("Classification Report:")
print(classification_report(error_df.true_class, y_pred, zero_division=1))

# Plotando a curva Precision-Recall
precision, recall, _ = precision_recall_curve(error_df.true_class, y_pred)
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# Calculando a taxa de falsos positivos e falsos negativos
fp = conf_matrix[0][1]
fn = conf_matrix[1][0]
total = conf_matrix.sum()

fp_rate = fp / total * 100
fn_rate = fn / total * 100

print(f'False Positive Rate: {fp_rate}%')
print(f'False Negative Rate: {fn_rate}%')
