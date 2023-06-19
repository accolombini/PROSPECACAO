""" 
Trabalhando com Redes neurais convolucionais e DeepLake e a biblioteca Keras

importante, pode ser necessário algo como => Uma abordagem para lidar com o desequilíbrio de classes é aplicar técnicas de reamostragem, como oversampling da classe minoritária ou undersampling da classe majoritária. Isso pode ajudar o modelo a aprender de forma mais equilibrada e melhorar os resultados.

Precisão de 87%
"""


from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Conv1D
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from imblearn.over_sampling import RandomOverSampler

# Carregar o arquivo CSV
df = pd.read_csv('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Pré-processamento dos dados
# Remover colunas desnecessárias, como 'offset_seconds'
df = df.drop('offset_seconds', axis=1)

# Separar as features (X) e os rótulos (y)
X = df.drop('role', axis=1)
y = df['role']

# Codificar as variáveis categóricas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Aplicar oversampling na classe minoritária
oversampler = RandomOverSampler(random_state=42)
X_train_oversampled, y_train_oversampled = oversampler.fit_resample(
    X_train, y_train)

# Redimensionar as features para um formato adequado para CNN (por exemplo, imagens)
# Aqui, é preciso adaptar o redimensionamento de acordo com a natureza das suas variáveis
# Reshape os dados de entrada para adicionar a dimensão do canal
X_train_oversampled = np.expand_dims(X_train_oversampled, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Converter os rótulos para one-hot encoding
y_train_oversampled = to_categorical(y_train_oversampled)
y_test = to_categorical(y_test)

# Criar o modelo
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu',
          input_shape=(X_train_oversampled.shape[1], 1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(y_train_oversampled)), activation='softmax'))

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo com os dados oversampled
model.fit(X_train_oversampled, y_train_oversampled,
          epochs=10, batch_size=32, verbose=1)

# Avaliar o modelo nos dados de teste
loss, accuracy = model.evaluate(X_test, y_test)

# Exibir a acurácia do modelo
print(f"Accuracy: {accuracy}")

# Fazer previsões nos dados de teste
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Converter as previsões de volta para as classes originais
y_pred = label_encoder.inverse_transform(y_pred)
y_test_orig = label_encoder.inverse_transform(np.argmax(y_test, axis=1))

# Calcular a matriz de confusão correta
confusion = confusion_matrix(y_test_orig, y_pred)

# Exibir matriz de confusão
print("Confusion Matrix:")
print(confusion)

# Exibir relatório de classificação
print("Classification Report:")
print(classification_report(y_test_orig, y_pred))
