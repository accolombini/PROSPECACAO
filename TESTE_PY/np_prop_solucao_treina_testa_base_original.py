"""
Neste algoritmo, vamos tentar uma nova abordagem que reune o apredizado das anteriores. Vamos seguir os seguintes
passos, agora com a base original:

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
    a alteração da arquitetura do autoencoder ou o uso de uma abordagem de detecção de anomalias completamente diferente

    Obs.: o que estamos fazendo neste algoritmo <=>

        Essa nova versão do código:

        Usa a estrutura de modelo Sequential do Keras, que é mais simples e intuitiva.
        Não remove os outliers dos dados de treinamento, o que pode melhorar a capacidade do modelo de detectar
        anomalias.
        Classifica uma observação como uma anomalia se seu erro de reconstrução é maior do que três vezes o desvio
        padrão do erro de reconstrução em todos os dados. Este é um método comum para detectar outliers baseado na
        suposição de que os dados seguem uma distribuição normal.
        Conta o número de anomalias na base de treinamento e de teste.
        Imprime uma descrição estatística para cada variável nas observações que foram classificadas como anomalias.
        Isso pode nos dar algumas indicações sobre quais variáveis contribuíram para as anomalias.

    Importante: Esse código assume que seu conjunto de treinamento não contém anomalias (como normalmente é o caso ao
    treinar um autoencoder para detecção de anomalias). Se você sabe que seu conjunto de treinamento contém anomalias
    e quer que seu modelo seja capaz de detectá-las, você teria que ajustar a maneira como o limiar é calculado.

    Nota: O cálculo de anomalia para cada variável é baseado em uma abordagem comum de classificar um valor como anômalo
    se ele estiver a mais de 3 desvios padrões da média.

"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar dados
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Tratar os valores 99999,99
df = df.replace(99999.99, np.nan)

# Substituir NaN por a média da coluna
for column in df.select_dtypes(include=[np.number]).columns:
    df[column].fillna(df[column].mean(), inplace=True)

# Ignorar a coluna 'offset_seconds'
df = df.drop(columns=['offset_seconds'])

# Dividir a base de dados em treino e teste baseado na coluna 'role'
df_train = df[df['role'] == 'normal'].select_dtypes(include=[np.number])
df_test = df[df['role'] == 'test-0'].select_dtypes(include=[np.number])

# Definir a estrutura do autoencoder
input_dim = df_train.shape[1]  # Número de features
encoding_dim = 14  # Tamanho das representações codificadas

model = Sequential()
model.add(Dense(encoding_dim, activation="relu", input_shape=(input_dim,)))
model.add(Dense(input_dim, activation='sigmoid'))

# Compilar o autoencoder
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o autoencoder com os dados de treino
model.fit(df_train, df_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Usar o autoencoder para reconstruir os dados de treino e teste
df_train_pred = model.predict(df_train)
df_test_pred = model.predict(df_test)

# Calcular o erro de reconstrução
reconstruction_error_train = np.mean(np.power(df_train - df_train_pred, 2), axis=1)
reconstruction_error_test = np.mean(np.power(df_test - df_test_pred, 2), axis=1)

# Definir o limiar como a média mais 3 vezes o desvio padrão do erro de reconstrução do treino
threshold = np.mean(reconstruction_error_train) + 3 * np.std(reconstruction_error_train)

# Classificar as observações como normal (0) ou anomalia (1) com base no erro de reconstrução e limiar
df_train['anomaly'] = (reconstruction_error_train > threshold).astype(int)
df_test['anomaly'] = (reconstruction_error_test > threshold).astype(int)

# Contar o número de anomalias na base de treinamento e teste
print(f"Número de anomalias na base de treinamento: {df_train['anomaly'].sum()}")
print(f"Número de anomalias na base de teste: {df_test['anomaly'].sum()}")

# Criar DataFrame para salvar no Excel
report = pd.DataFrame(columns=['Instante', 'Variável', 'Valor'])

# Para cada instante em que uma anomalia é detectada, exibir as variáveis que apresentam um comportamento
anomalies = df_test[df_test['anomaly'] == 1]
for index, row in anomalies.iterrows():
    print(f"\nRegistro de anomalia detectada no instante {index}:")
    for column in [c for c in anomalies.columns if c != 'anomaly']:
        if row[column] > df_train[column].mean() + 3 * df_train[column].std():
            print(f"A variável {column} apresentou um comportamento anômalo com valor {row[column]}")
            report_temp = pd.DataFrame([[index, column, row[column]]], columns=['Instante', 'Variável', 'Valor'])
            report = pd.concat([report, report_temp], ignore_index=True)

# Salvar DataFrame como arquivo Excel
report.to_excel('DADOS/FINAL_back_30_6.xlsx', index=False)
