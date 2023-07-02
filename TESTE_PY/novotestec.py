"""
Novo teste => Redes Neurais autoencoder
dividndo a base em normal para treinamento e test-0 para teste
Realizando tambe´m o balanceamento dos dados
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
from sklearn.model_selection import train_test_split

# Carregando os dados
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Transformando a coluna categórica "role" em numérica
df['role'] = df['role'].map({'normal': 1, 'test-0': -1})

# Separando o target do resto dos dados
X = df.drop('role', axis=1)
y = df['role']

# Normalizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separando os dados de treinamento e teste
X_normal = X_scaled[y == 1]
X_test0 = X_scaled[y == -1]

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

# Treinando o autoencoder
autoencoder.fit(X_normal, X_normal, epochs=100, batch_size=32, shuffle=True)

# Obtendo a reconstrução dos dados
predictions_normal = autoencoder.predict(X_normal)
predictions_test0 = autoencoder.predict(X_test0)

# Calculando o erro de reconstrução
mse_normal = np.mean(np.power(X_normal - predictions_normal, 2), axis=1)
mse_test0 = np.mean(np.power(X_test0 - predictions_test0, 2), axis=1)

# Concatenando os erros de reconstrução e os rótulos verdadeiros
mse = np.concatenate((mse_normal, mse_test0))
y_true = np.concatenate((np.ones(len(mse_normal)), -1*np.ones(len(mse_test0))))

# Criando o dataframe de erros
error_df = pd.DataFrame({'reconstruction_error': mse, 'true_class': y_true})

# Definindo um threshold e calculando a matriz de confusão
threshold = np.quantile(error_df.reconstruction_error, 0.95)
y_pred = [1 if e > threshold else -
          1 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)

# Printando a matriz de confusão
print("Confusion Matrix:")
print(conf_matrix)

# Printando o relatório de classificação
print("Classification Report:")
print(classification_report(y_true, y_pred))

# Plotando a curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_true, y_pred)
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
