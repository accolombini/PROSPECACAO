""" 
Queremos agora trabalhar com o mesmo algoritmo do teste26.py, mas agora dividiremos treinamento e teste com base nas classes normal e test-0

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# Carregar os dados
df = pd.read_csv('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Separar os dados de entrada (X) e saída (y)
X = df.drop(columns=['role'])
y = df['role']

# Codificar as classes de saída
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Padronizar os dados de entrada
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Divisão dos dados para treinamento e teste
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

# Filtrar apenas os dados da classe "normal" para treinamento
X_train_normal = X_train_orig[y_train_orig == 0]
y_train_normal = y_train_orig[y_train_orig == 0]

# Filtrar apenas os dados da classe "test-0" para teste
X_test_test0 = X_test_orig[y_test_orig == 1]
y_test_test0 = y_test_orig[y_test_orig == 1]

# Concatenar os dados filtrados
X_train = np.concatenate((X_train_normal, X_test_test0), axis=0)
y_train = np.concatenate((y_train_normal, y_test_test0), axis=0)

# Converter as classes de saída em one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test_orig)

# Reshape dos dados de entrada para serem compatíveis com a camada Conv1D
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test_orig.reshape(X_test_orig.shape[0], X_test_orig.shape[1], 1)

# Criar o modelo da rede neural
model = Sequential()
model.add(Conv1D(128, kernel_size=7, activation='relu',
          input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compilar o modelo
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Avaliar o modelo no conjunto de teste
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy: {accuracy}")

# Prever as classes para os dados de teste
y_pred = np.argmax(model.predict(X_test), axis=-1)

# Converter as classes previstas para suas categorias originais
y_test_orig = np.argmax(y_test, axis=-1)

# Exibir relatório de classificação
print("Classification Report:")
print(classification_report(y_test_orig, y_pred))

# Definir um limiar para classificação
threshold = 0.5

# Prever probabilidades para os dados de teste
y_pred_proba = model.predict(X_test)

# Aplicar o limiar de classificação
y_pred_threshold = (y_pred_proba[:, 1] > threshold).astype(int)

# Calcular a acurácia considerando o limiar
accuracy_threshold = np.mean(y_pred_threshold == y_test_orig)
print(f"Accuracy with threshold: {accuracy_threshold}")

# Exibir relatório de classificação
print("Classification Report:")
print(classification_report(y_test_orig, y_pred_threshold))
