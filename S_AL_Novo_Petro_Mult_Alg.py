"""
Neste algoritmo vamos explorar outros algoritmos como o autoencoder; o One-Class SVM e o Isoaltiron Forest:

    Este algoritmo foi direcionado para trabalhar com a base Adendo A.2_Conjunto de Dados_DataSet.csv
    Dividir a base de dados com base na coluna 'role': Vamos dividir o DataFrame original em dois: um DataFrame de
    treinamento que consiste apenas em dados 'normais' e um DataFrame de teste que consiste em dados 'test-0'.
    Vamos usar o DataFrame de treinamento para treinar nosso autoencoder e o DataFrame de teste para avaliar seu
    desempenho.

    Treinar o Autoencoder: Com base no DataFrame de treinamento, vamos treinar nosso autoencoder. O objetivo é permitir
    que o autoencoder aprenda a reconstruir as operações normais com a menor perda possível.
        A opção pelo uso do autoencoder --> se deve ao fato de ser um algoritmo propício para detecção de anomalias
        (há outros) mas vamos começar por aqui. Outro fato importante é que se trata de um algoritmo de aprendizagem
        profunda e que não demanda que os dados sigam uma distribuição normal.

    Testar o Autoencoder: Após o treinamento, vamos aplicar o autoencoder aos dados de teste para gerar previsões. Vamos
    calcular o erro de reconstrução para cada observação e marcar as observações com erro de reconstrução acima de um
    certo limiar como anomalias. (isto pode ser ajustado vamos iniciar com +/- 3 sigmas).

    Obs.: o que estamos fazendo neste algoritmo <=>

        Essa nova versão do código:

        Estamos usando o teste de Shapiro-Wilk pode ser usado para verificar a normalidade dos dados.
        Este método não atendeu.
        Vamos testar também sar o teste de normalidade de Anderson-Darling que é uma versão modificada do teste de
        Kolmogorov-Smirnov e tem mais poder para uma ampla gama de alternativas.

        Usa a estrutura de modelo Sequential do Keras, que é mais simples e intuitiva (há outras para serem testadas).
        Não remove os outliers dos dados de treinamento, o que pode melhorar a capacidade do modelo de detectar
        anomalias.
        Classifica =>> observação será considerado como uma anomalia se seu erro de reconstrução for maior do que três
        vezes o desvio padrão (+/-) 3 sigmas.
        Este é um método muito comum utilizado para detectar outliers baseado na suposição de que os dados seguem uma
        distribuição normal.
        Contar o número de anomalias na base de treinamento e de teste.
        Imprimir uma descrição estatística para cada variável nas observações que foram classificadas como anomalias.
        Isso pode nos dar algumas indicações sobre quais variáveis contribuíram para as anomalias. Foi suprimido na
        entrega, pois neste apenas sinalizamos na saída os pontos sem anomalia (0) e com anomalia (1) no arquivo de
        saída.

    Importante: Esse código assume que seu conjunto de treinamento não contém anomalias (como normalmente é o caso ao
    treinar um autoencoder para detecção de anomalias). Caso o conjunto de treinamento contenha anomalias
    e quer que o modelo seja capaz de detectá-las, é preciso ajustar a maneira como o limiar é calculado.

"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def load_and_prepare_data():
    df = pd.read_csv('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')
    df = df.replace(99999.99, np.nan)

    for column in df.select_dtypes(include=[np.number]).columns:
        df[column].fillna(df[column].mean(), inplace=True)

    df_train = df[df['role'] == 'normal'].select_dtypes(include=[np.number])
    df_test = df[df['role'] == 'test-0'].select_dtypes(include=[np.number])

    return df, df_train, df_test

def create_autoencoder_model(input_dim, encoding_dim=14):
    model = Sequential()
    model.add(Dense(encoding_dim, activation="relu", input_shape=(input_dim,)))
    model.add(Dense(input_dim, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_autoencoder(model, df_train):
    model.fit(df_train, df_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)
    return model

def calculate_autoencoder_anomalies(df, model):
    df_pred = model.predict(df)
    mse = np.mean(np.power(df - df_pred, 2), axis=1)
    error_df = pd.DataFrame({'reconstruction_error': mse}, index=df.index)
    error_df['anomaly'] = (error_df['reconstruction_error'] > np.quantile(error_df['reconstruction_error'], 0.995)).astype(int)
    df = pd.concat([df, error_df['anomaly']], axis=1)
    return df

def create_and_train_IsolationForest(df_train):
    isoforest = IsolationForest(contamination=0.01)
    isoforest.fit(df_train)
    return isoforest

def create_and_train_OneClassSVM(df_train):
    one_class_svm = OneClassSVM(nu=0.01)
    one_class_svm.fit(df_train)
    return one_class_svm

def calculate_anomalies(df, model):
    df_pred = model.predict(df)
    df_pred = pd.DataFrame(df_pred, columns=['anomaly'], index=df.index)
    df_pred['anomaly'] = (df_pred['anomaly'] == -1).astype(int)
    df = pd.concat([df, df_pred['anomaly']], axis=1)
    return df

def main():
    df, df_train, df_test = load_and_prepare_data()
    input_dim = df_train.shape[1]

    # Autoencoder
    model_autoencoder = create_autoencoder_model(input_dim)
    model_autoencoder = train_autoencoder(model_autoencoder, df_train)
    df_train = calculate_autoencoder_anomalies(df_train, model_autoencoder)
    df_test = calculate_autoencoder_anomalies(df_test, model_autoencoder)
    print(f"Autoencoder: Número de anomalias na base de treinamento: {df_train['anomaly'].sum()}")
    print(f"Autoencoder: Número de anomalias na base de teste: {df_test['anomaly'].sum()}")

    # Isolation Forest
    model_isoforest = create_and_train_IsolationForest(df_train.drop(columns='anomaly'))
    df_train = calculate_anomalies(df_train.drop(columns='anomaly'), model_isoforest)
    df_test = calculate_anomalies(df_test.drop(columns='anomaly'), model_isoforest)
    print(f"Isolation Forest: Número de anomalias na base de treinamento: {df_train['anomaly'].sum()}")
    print(f"Isolation Forest: Número de anomalias na base de teste: {df_test['anomaly'].sum()}")

    # One-Class SVM
    model_oneclasssvm = create_and_train_OneClassSVM(df_train.drop(columns='anomaly'))
    df_train = calculate_anomalies(df_train.drop(columns='anomaly'), model_oneclasssvm)
    df_test = calculate_anomalies(df_test.drop(columns='anomaly'), model_oneclasssvm)
    print(f"One-Class SVM: Número de anomalias na base de treinamento: {df_train['anomaly'].sum()}")
    print(f"One-Class SVM: Número de anomalias na base de teste: {df_test['anomaly'].sum()}")

if __name__ == "__main__":
    main()
