"""
Neste algoritmo, vamos tentar uma nova abordagem que reune o apredizado das anteriores. Vamos seguir os seguintes
passos:

    Dividir a base de dados com base na coluna 'role': Vamos dividir o DataFrame original em dois: um DataFrame de
    treinamento que consiste apenas em dados 'normais' e um DataFrame de teste que consiste em dados 'test-0'.
    Vamos usar o DataFrame de treinamento para treinar nosso autoencoder e o DataFrame de teste para avaliar seu
    desempenho.

    Treinar o Autoencoder: Com base no DataFrame de treinamento, vamos treinar nosso autoencoder. O objetivo é permitir
    que o autoencoder aprenda a reconstruir as operações normais com a menor perda possível.

    Testar o Autoencoder: Após o treinamento, vamos aplicar o autoencoder aos dados de teste para gerar previsões. Vamos
     calcular o erro de reconstrução para cada observação e marcar as observações com erro de reconstrução acima de um
     certo limiar como anomalias.

    Avaliar o Desempenho: Como temos uma mistura de operações normais e anomalias no conjunto de teste, podemos calcular
     a matriz de confusão, bem como as taxas de falsos positivos e falsos negativos para avaliar o desempenho do nosso
     modelo.

    Ajustar o Modelo: Se as taxas de falsos positivos e falsos negativos forem muito altas, poderemos ajustar nosso
    modelo ou estratégia. Isso pode envolver a alteração do limiar que usamos para marcar as observações como anomalias,
    a alteração da arquitetura do autoencoder ou o uso de uma abordagem de detecção de anomalias completamente diferente.

"""

import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.metrics import confusion_matrix, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar dados
df = pd.read_excel('DADOS/OUTLIERS_FREE.xlsx')

# Remover a coluna 'offset_seconds'
df = df.drop(['offset_seconds'], axis=1)

# Dividir a base de dados em treino e teste baseado na coluna 'role'
df_train = df[df['role'] == 'normal'].drop(['role'], axis=1)
df_test = df[df['role'] == 'test-0'].drop(['role'], axis=1)

# Definir a estrutura do autoencoder
input_dim = df_train.shape[1]  # Número de features
encoding_dim = 14  # Tamanho das representações codificadas
input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compilar o autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o autoencoder com os dados de treino
autoencoder.fit(df_train, df_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Usar o autoencoder para reconstruir os dados de teste
df_test_pred = autoencoder.predict(df_test)

# Calcular o erro de reconstrução
reconstruction_error = np.mean(np.power(df_test - df_test_pred, 2), axis=1)

# Encontrar o limiar ótimo para minimizar falsos positivos e falsos negativos
# Aqui, precisamos fazer uma suposição sobre quais observações são anomalias
df_test['anomaly'] = 1  # Assumindo que todas as entradas 'test-0' são anomalias
precision, recall, thresholds = precision_recall_curve(df_test['anomaly'], reconstruction_error)
threshold_optimal = thresholds[np.argmax(precision + recall)]

# Classificar as observações como normal ou anomalia com base no erro de reconstrução e limiar ótimo
df_test['anomaly_pred'] = (reconstruction_error > threshold_optimal).astype(int)

# Calcular a matriz de confusão
cm = confusion_matrix(df_test['anomaly'], df_test['anomaly_pred'])
tn, fp, fn, tp = cm.ravel()

# Calcular as taxas de falsos positivos e falsos negativos
epsilon = 1e-7  # pequena constante para evitar divisão por zero
fp_rate = fp / (fp + tn + epsilon)
fn_rate = fn / (fn + tp + epsilon)


print(f'Verdadeiros negativos: {tn}')
print(f'Falsos positivos: {fp}')
print(f'Falsos negativos: {fn}')
print(f'Verdadeiros positivos: {tp}')
print(f'Taxa de falsos positivos: {fp_rate * 100:.2f}%')
print(f'Taxa de falsos negativos: {fn_rate * 100:.2f}%')

# Matriz de confusão gráfica
sns.heatmap(cm, annot=True, fmt="d")
plt.title('Matriz de confusão')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
