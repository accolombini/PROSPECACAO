""" 
Vamos agora explorar redes neurais iniciemos com --> redes neurais autoencoder
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Carregar o arquivo CSV
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Filtrar os dados da classe 'normal' para treinamento
df_train = df[df['role'] == 'normal'].copy()

# Filtrar os dados da classe 'test-0' para teste
df_test = df[df['role'] == 'test-0'].copy()

# Selecionar as features
features = df_train.columns.drop(['role'])

# Padronização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train[features])
X_test = scaler.transform(df_test[features])

# Definir a arquitetura do Autoencoder
input_dim = X_train.shape[1]
encoding_dim = 64

input_layer = tf.keras.Input(shape=(input_dim,))
encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)

# Compilar o modelo
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
epochs = 100
batch_size = 32

autoencoder.fit(X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_test, X_test))

# Avaliar o modelo nos dados de treinamento
X_train_pred = autoencoder.predict(X_train)
mse_train = np.mean(np.power(X_train - X_train_pred, 2), axis=1)
threshold = np.percentile(mse_train, 95)

# Identificar anomalias nos dados de treinamento
y_train_pred_labels = [1 if mse > threshold else 0 for mse in mse_train]

# Calcular as métricas nos dados de treinamento
false_positives_train = None
false_negatives_train = None

# Criando a matriz de confusão
role_counts = df_train['role'].value_counts()
if len(role_counts) > 1:
    confusion_train = pd.crosstab(
        index=df_train['role'], columns=np.array(df_train['role']))
    false_positives_train = confusion_train[1][0] / len(df_train) * 100
    false_negatives_train = confusion_train[0][1] / len(df_train) * 100
else:
    print("A coluna 'role' contém apenas um valor único.")

# Exibir as métricas nos dados de treinamento
if false_positives_train is not None and false_negatives_train is not None:
    print(
        f"Porcentagem de falsos positivos (dados de treinamento): {false_positives_train:.2f}%")
    print(
        f"Porcentagem de falsos negativos (dados de treinamento): {false_negatives_train:.2f}%")
    print("Matriz de Confusão (dados de treinamento):")
    print(confusion_train)


# Identificar anomalias nos dados de teste
X_test_pred = autoencoder.predict(X_test)
mse_test = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
y_test_pred_labels = [1 if mse > threshold else 0 for mse in mse_test]

# Calcular as métricas nos dados de teste
false_positives_test = None
false_negatives_test = None

role_counts_test = df_test['role'].value_counts()
if len(role_counts_test) > 1:
    confusion_test = pd.crosstab(index=df_test['role'], columns=np.array(
        y_test_pred_labels), rownames=['True'], colnames=['Predicted'])
    false_positives_test = confusion_test[0][1] / len(df_test) * 100
    false_negatives_test = confusion_test[1][0] / len(df_test) * 100
else:
    print("A coluna 'role' nos dados de teste contém apenas um valor único.")

# Exibir as métricas nos dados de teste
if false_positives_test is not None and false_negatives_test is not None:
    print(
        f"Porcentagem de falsos positivos (dados de teste): {false_positives_test:.2f}%")
    print(
        f"Porcentagem de falsos negativos (dados de teste): {false_negatives_test:.2f}%")
    print("Matriz de Confusão (dados de teste):")
    print(confusion_test)
