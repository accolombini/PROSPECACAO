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
    treinar um autoencoder para detecção de anomalias). Sabemos que o conjunto de treinamento contém anomalias
    e queremos que nosso modelo seja capaz de detectá-las, vamos ajustar a maneira como o limiar é calculado.

    Nota: O cálculo de anomalia para cada variável é baseado em uma abordagem comum de classificar um valor como anômalo
    se ele estiver a mais de 3 desvios padrões da média.

"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


class AnomalyDetector:
    def __init__(self, filepath, encoding_dim=14):
        self.filepath = filepath
        self.encoding_dim = encoding_dim
        self.df, self.df_train, self.df_test = self.load_and_prepare_data()
        self.input_dim = self.df_train.shape[1]
        self.model = self.create_model()

    def load_and_prepare_data(self):
        df = pd.read_csv(self.filepath)
        df = df.replace(99999.99, np.nan)

        for column in df.select_dtypes(include=[np.number]).columns:
            df[column].fillna(df[column].mean(), inplace=True)

        df = df.drop(columns=['offset_seconds'])
        df_train = df[df['role'] == 'normal'].select_dtypes(include=[np.number])
        df_test = df[df['role'] == 'test-0'].select_dtypes(include=[np.number])
        return df, df_train, df_test

    def create_model(self):
        model = Sequential()
        model.add(Dense(self.encoding_dim, activation="relu", input_shape=(self.input_dim,)))
        model.add(Dense(self.input_dim, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_and_predict(self):
        self.model.fit(self.df_train, self.df_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)
        self.df_train_pred = self.model.predict(self.df_train)
        self.df_test_pred = self.model.predict(self.df_test)

    def calculate_reconstruction_error(self):
        reconstruction_error_train = np.mean(np.power(self.df_train - self.df_train_pred, 2), axis=1)
        reconstruction_error_test = np.mean(np.power(self.df_test - self.df_test_pred, 2), axis=1)
        threshold = np.mean(reconstruction_error_train) + 3 * np.std(reconstruction_error_train)
        self.df_train['anomaly'] = (reconstruction_error_train > threshold).astype(int)
        self.df_test['anomaly'] = (reconstruction_error_test > threshold).astype(int)
        print(f"Número de anomalias na base de treinamento: {self.df_train['anomaly'].sum()}")
        print(f"Número de anomalias na base de teste: {self.df_test['anomaly'].sum()}")

    def report_anomalies(self):
        self.report = pd.DataFrame(columns=['Instante', 'Variável', 'Valor'])
        anomalies = self.df_test[self.df_test['anomaly'] == 1]
        for index, row in anomalies.iterrows():
            print(f"\nRegistro de anomalia detectada no instante {index}:")
            for column in [c for c in anomalies.columns if c != 'anomaly']:
                if row[column] > self.df_train[column].mean() + 3 * self.df_train[column].std():
                    print(f"A variável {column} apresentou um comportamento anômalo com valor {row[column]}")
                    report_temp = pd.DataFrame([[index, column, row[column]]],
                                               columns=['Instante', 'Variável', 'Valor'])
                    self.report = pd.concat([self.report, report_temp], ignore_index=True)

    def run(self):
        self.train_and_predict()
        self.calculate_reconstruction_error()
        self.report_anomalies()
        self.report.to_excel('DADOS/FINAL_back_30_6.xlsx', index=False)


if __name__ == "__main__":
    detector = AnomalyDetector('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')
    detector.run()
