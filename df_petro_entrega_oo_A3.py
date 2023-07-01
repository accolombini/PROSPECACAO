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
    def __init__(self, data_path, encoding_dim=14):
        self.data_path = data_path
        self.encoding_dim = encoding_dim
        self.model = None
        self.df = None
        self.df_train = None
        self.df_test_0 = None
        self.df_test_1 = None

    def load_and_prepare_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.replace(99999.99, np.nan)

        for column in self.df.select_dtypes(include=[np.number]).columns:
            self.df[column].fillna(self.df[column].mean(), inplace=True)

        self.df_train = self.df[self.df['role'] == 'normal'].select_dtypes(include=[np.number])
        self.df_test_0 = self.df[self.df['role'] == 'test-0'].select_dtypes(include=[np.number])
        self.df_test_1 = self.df[self.df['role'] == 'test-1'].select_dtypes(include=[np.number])

    def data_analysis(self, df, df_name):
        print(f"Análise da base de dados: {df_name}")
        print(f"Quantidade de linhas: {df.shape[0]}")
        print(f"Quantidade de colunas: {df.shape[1]}")
        print("Tipo de dados em cada coluna:")
        print(df.dtypes)
        print("Quantidade de dados faltantes em cada coluna:")
        print(df.isnull().sum())
        print("\n")

    def create_model(self, input_dim):
        self.model = Sequential()
        self.model.add(Dense(self.encoding_dim, activation="relu", input_shape=(input_dim,)))
        self.model.add(Dense(input_dim, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_and_predict(self, df_train, df_test):
        self.model.fit(df_train, df_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)
        return self.model.predict(df_test)

    def report_anomalies(self, df_train, df_test):
        report = pd.DataFrame(columns=['offset_seconds', 'Registro'] + list(df_test.columns) + ['TOTAL_Anom'])
        total_anom = 0
        for index, row in df_test.iterrows():
            anomaly_row = [index, index]
            row_anom = 0
            for column in df_test.columns:
                if row[column] > df_train[column].mean() + 3 * df_train[column].std():
                    anomaly_row.append(1)
                    row_anom += 1
                else:
                    anomaly_row.append(0)
            anomaly_row.append(row_anom)
            if row_anom > 0:
                total_anom += 1
            report_temp = pd.DataFrame([anomaly_row], columns=report.columns)
            report = pd.concat([report, report_temp], ignore_index=True)
        return report, total_anom

    def run(self):
        self.load_and_prepare_data()
        self.data_analysis(self.df, "Total")
        self.data_analysis(self.df_train, "Treinamento")
        self.data_analysis(self.df_test_0, "Teste-0")
        self.data_analysis(self.df_test_1, "Teste-1")

        self.create_model(self.df_train.shape[1])

        df_test_0_pred = self.train_and_predict(self.df_train, self.df_test_0)
        report_0, total_anom_0 = self.report_anomalies(self.df_train, self.df_test_0)

        df_test_1_pred = self.train_and_predict(self.df_train, self.df_test_1)
        report_1, total_anom_1 = self.report_anomalies(self.df_train, self.df_test_1)

        # Escreve relatórios para abas separadas em uma única planilha Excel
        with pd.ExcelWriter('DADOS/FINAL_A3.xlsx') as writer:
            report_0.to_excel(writer, index=False, sheet_name='RELAT_T_0')
            report_1.to_excel(writer, index=False, sheet_name='RELAT_T1')

        # Relatório de anomalias na tela
        print(f"Total de anomalias na base Teste-0: {total_anom_0}")
        print(f"Total de anomalias na base Teste-1: {total_anom_1}")


if __name__ == "__main__":
    detector = AnomalyDetector('DADOS/Adendo A.3_Conjunto de Dados_DataSet.csv')
    detector.run()
