""" 
Neste teste vamos explorar uma solução que leve em conta as variáveis mais siginificativas para a solução do problema. As variáveis de vibração ................

Resultado atingido

O resultado obtido mostra uma alta precisão e acurácia do modelo. Aqui estão algumas métricas importantes do relatório de classificação:

- Precision (precisão): A precisão média para ambas as classes é alta, com valores de 0,98 para a classe 0 (funcionamento normal) e 1,00 para a classe 1 (anomalia). Isso indica que o modelo tem uma alta taxa de verdadeiros positivos.

- Recall (revocação): O recall médio também é alto, com valores de 0,99 para a classe 0 e 0,99 para a classe 1. Isso indica que o modelo tem uma alta taxa de detecção de verdadeiros positivos e minimiza os falsos negativos.

- F1-score: O F1-score é uma medida que combina precisão e recall em uma única métrica. Para ambas as classes, o F1-score é alto, com valores de 0,99 para a classe 0 e 1,00 para a classe 1.

- Acurácia: A acurácia geral do modelo é de 0,99, o que indica uma alta taxa de classificação correta para as amostras de teste.

Em geral, esses resultados mostram que o modelo é altamente preciso na detecção de falhas nos equipamentos e possui um desempenho muito bom na classificação das amostras de teste.

Este resultado foi obtido utilizando para treinamento e teste o seguinte proposito

No código fornecido, dividimos os dados em conjunto de treinamento e conjunto de teste usando a função train_test_split do scikit-learn. A proporção padrão é 75% para treinamento e 25% para teste.

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

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Converter as classes de saída em one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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
