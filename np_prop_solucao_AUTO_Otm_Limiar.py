"""
Nesta aboradagem queremos reduzir nos falsos positivos e falsos negativos. Vamos trabalhar com:

    Autoencoder, uma forma especial de rede neural destinada à detecção de anomalias. Ele aprende a representação compacta dos dados normais durante o treinamento. Na fase de inferência, ele tenta reconstruir os dados de entrada. Se a reconstrução de uma entrada específica tiver um erro muito alto, podemos considerar essa entrada como uma anomalia.

    Para otimizar a taxa de falsos positivos e falsos negativos, você pode usar uma abordagem baseada em thresholding. Por exemplo, em vez de classificar uma observação como anomalia se ela tiver uma pontuação de anomalia acima de um certo limiar fixo, você pode ajustar o limiar para minimizar a taxa de falsos positivos e falsos negativos.

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.metrics import confusion_matrix, precision_recall_curve

# Carregar dados
df = pd.read_excel('DADOS/OUTLIERS_FREE.xlsx')

# Remover a coluna 'role' e 'offset_seconds'
df_numeric = df.drop(['role', 'offset_seconds'], axis=1)

# Definir a estrutura do autoencoder
input_dim = df_numeric.shape[1]  # Número de features
encoding_dim = 14  # Tamanho das representações codificadas
input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compilar o autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o autoencoder
autoencoder.fit(df_numeric, df_numeric, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Usar o autoencoder para reconstruir os dados de entrada
df_numeric_pred = autoencoder.predict(df_numeric)

# Calcular o erro de reconstrução
reconstruction_error = np.mean(np.power(df_numeric - df_numeric_pred, 2), axis=1)

# Classificar as observações como normal ou anomalia com base no erro de reconstrução
threshold_fixed = np.percentile(reconstruction_error, 95)  # Defina o limiar como o percentil 95 do erro de reconstrução
df['anomaly'] = (reconstruction_error > threshold_fixed).astype(int)

# Calcular a matriz de confusão
cm = confusion_matrix(df['anomaly'], df['anomaly'])
tn, fp, fn, tp = cm.ravel()

# Calcular as taxas de falsos positivos e falsos negativos
fp_rate = fp / (fp + tn)
fn_rate = fn / (fn + tp)

print(f'Verdadeiros negativos: {tn}')
print(f'Falsos positivos: {fp}')
print(f'Falsos negativos: {fn}')
print(f'Verdadeiros positivos: {tp}')
print(f'Taxa de falsos positivos: {fp_rate * 100:.2f}%')
print(f'Taxa de falsos negativos: {fn_rate * 100:.2f}%')

# Calcular a matriz de confusão
cm = confusion_matrix(df['anomaly'], df['anomaly'])

# Criar um DataFrame para visualização
df_cm = pd.DataFrame(cm, index=['Normal', 'Anomalia'], columns=['Normal', 'Anomalia'])

# Plotar a matriz de confusão
plt.figure(figsize=(10,7))
sns.heatmap(df_cm, annot=True, fmt='d')
plt.title('Matriz de Confusão')
plt.ylabel('Verdadeiro')
plt.xlabel('Previsto')
plt.show()
