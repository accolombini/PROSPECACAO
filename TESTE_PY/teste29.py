""" 
Vamos agora retirar a concatenção e separar completamente as bases

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Carregar os dados
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Separar os dados de entrada (X) e saída (y)
X = df.drop(columns=['role'])
y = df['role']

# Codificar as classes de saída
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Padronizar os dados de entrada
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separar os dados de treinamento e teste de acordo com as classes
X_train = X[y == 0]  # Classe "normal" para treinamento
y_train = y[y == 0]
X_test = X[y == 1]  # Classe "test-0" para teste/validação
y_test = y[y == 1]

# Converter as classes de saída em one-hot encoding
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# Reshape dos dados de entrada para serem compatíveis com a camada Conv1D
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Criar o modelo da rede neural
model = Sequential()
model.add(Conv1D(128, kernel_size=7, activation='relu',
          input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
# Alteração: 1 neurônio para classificação binária
model.add(Dense(1, activation='sigmoid'))

# Compilar o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
              'accuracy'])  # Alteração: loss='binary_crossentropy'

# Treinar o modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Avaliar o modelo no conjunto de teste
y_pred_proba = model.predict(X_test)
# Alteração: arredondar as previsões para 0 ou 1
y_pred_classes = np.round(y_pred_proba).flatten()
y_true_classes = y_test

# Calcular a acurácia
accuracy = np.mean(y_pred_classes == y_true_classes)
print(f"Accuracy: {accuracy}")

# Exibir relatório de classificação
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))

# Definir limite de probabilidade para classificar como anomalia
anomaly_threshold = 0.5

# Converter as probabilidades de predição em classes (0 ou 1) usando o limite de probabilidade
y_pred_classes = (y_pred_proba > anomaly_threshold).astype(int)

# Calcular a acurácia
accuracy = np.mean(y_pred_classes == y_true_classes)
print(f"Accuracy: {accuracy}")

# Exibir relatório de classificação
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, zero_division=1))
