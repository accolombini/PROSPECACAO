"""
Neste algoritmo, vamos tentar uma nova abordagem que reune o apredizado das anteriores. Vamos seguir os seguintes
passos, agora com a base original, isto é sem a remoção dos Outliers.:

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
        Classifica =>> uma observação como uma anomalia se seu erro de reconstrução é maior do que três vezes o desvio
        padrão do erro de reconstrução em todos os dados. Este é um método comum para detectar outliers baseado na
        suposição de que os dados seguem uma distribuição normal.
        Conta o número de anomalias na base de treinamento e de teste.
        Imprime uma descrição estatística para cada variável nas observações que foram classificadas como anomalias.
        Isso pode nos dar algumas indicações sobre quais variáveis contribuíram para as anomalias.

    Importante: Esse código assume que seu conjunto de treinamento não contém anomalias (como normalmente é o caso ao
    treinar um autoencoder para detecção de anomalias). Caso o conjunto de treinamento contenha anomalias
    e quer que o modelo seja capaz de detectá-las, é preciso ajustar a maneira como o limiar é calculado.

    Nota: O cálculo de anomalia para cada variável é baseado em uma abordagem comum de classificar um valor como anômalo
    se ele estiver a mais de 3 desvios padrões da média.

"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


class AnomalyDetector:

    def __init__(self, data_file, output_file):
        self.data_file = data_file
        self.output_file = output_file
        self.df_train = None
        self.df_test = None
        self.model = None

    def load_and_prepare_data(self):
        df = pd.read_csv(self.data_file)
        df = df.replace(99999.99, np.nan)

        for column in df.select_dtypes(include=[np.number]).columns:
            df[column].fillna(df[column].mean(), inplace=True)

        self.df_train = df[df['role'] == 'normal'].select_dtypes(include=[np.number])
        self.df_test = df[df['role'] == 'test-0'].select_dtypes(include=[np.number])

    def create_model(self, encoding_dim=14):
        input_dim = self.df_train.shape[1]
        self.model = Sequential()
        self.model.add(Dense(encoding_dim, activation="relu", input_shape=(input_dim,)))
        self.model.add(Dense(input_dim, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_and_predict(self):
        self.model.fit(self.df_train, self.df_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)
        df_train_pred = self.model.predict(self.df_train)
        df_test_pred = self.model.predict(self.df_test)
        return df_train_pred, df_test_pred

    def calculate_reconstruction_error(self, df_train_pred, df_test_pred):
        reconstruction_error_train = np.mean(np.power(self.df_train - df_train_pred, 2), axis=1)
        reconstruction_error_test = np.mean(np.power(self.df_test - df_test_pred, 2), axis=1)
        threshold = np.mean(reconstruction_error_train) + 3 * np.std(reconstruction_error_train)
        self.df_train['anomaly'] = (reconstruction_error_train > threshold).astype(int)
        self.df_test['anomaly'] = (reconstruction_error_test > threshold).astype(int)
        print(f"Número de anomalias na base de treinamento: {self.df_train['anomaly'].sum()}")
        print(f"Número de anomalias na base de teste: {self.df_test['anomaly'].sum()}")

    def report_anomalies(self):
        report = pd.DataFrame(columns=['offset_seconds', 'Registro'] + list(self.df_test.columns[:-1]) + ['TOTAL_Anom'])
        for index, row in self.df_test.iterrows():
            anomaly_row = [index, index]
            total_anom = 0
            for column in self.df_test.columns[:-1]:  # Excluding 'anomaly' column
                if row[column] > self.df_train[column].mean() + 3 * self.df_train[column].std():
                    anomaly_row.append(1)
                    total_anom += 1
                else:
                    anomaly_row.append(0)
            anomaly_row.append(total_anom)
            report_temp = pd.DataFrame([anomaly_row], columns=report.columns)
            report = pd.concat([report, report_temp], ignore_index=True)
        report.to_excel(self.output_file, index=False, sheet_name='RELATÓRIO')

    def run(self):
        self.load_and_prepare_data()
        self.create_model()
        df_train_pred, df_test_pred = self.train_and_predict()
        self.calculate_reconstruction_error(df_train_pred, df_test_pred)
        self.report_anomalies()


if __name__ == "__main__":
    ad = AnomalyDetector('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv', 'DADOS/FINAL.xlsx')
    ad.run()
