"""
Neste algoritmo temos o seguinte:

    Este algoritmo foi direcionado para trabalhar com a base Adendo A.2_Conjunto de Dados_DataSet.csv
    Dividir a base de dados com base na coluna 'role': Vamos dividir o DataFrame original em dois: um DataFrame de
    treinamento que consiste apenas em dados 'normais' e um DataFrame de teste que consiste em dados 'test-0'.
    Vamos usar o DataFrame de treinamento para treinar nosso Isolation Forest e o DataFrame de teste para avaliar seu
    desempenho.

    Treinar o Isolation Forest: Com base no DataFrame de treinamento, vamos treinar nosso Isolation Forest. O objetivo é
    permitir que o Isolation Forest aprenda a reconstruir as operações normais com a menor perda possível.
        A opção pelo uso do Isolation Forest --> se deve ao fato de ser um algoritmo propício para detecção de anomalias
        (há outros), como o autoenconder e o One-Class SVM, mas vamos começar por aqui (existem ainda outros além dos
        três utilizados nestes testes.. Outro fato importante é que se trata de um algoritmo de aprendizagem profunda
        e que não demanda que os dados sigam uma distribuição normal.

    Testar o Isolation Forest: Após o treinamento, vamos aplicar o Isolation Forest aos dados de teste para gerar
    previsões. Vamos calcular o erro de reconstrução para cada observação e marcar as observações com erro de
    reconstrução acima de um certo limiar como anomalias. (isto pode ser ajustado vamos iniciar com +/- 3 sigmas).

    Obs.: o que estamos fazendo neste algoritmo:

        Estamos usando o teste de Shapiro-Wilk pode ser usado para verificar a normalidade dos dados.
        Este método não atendeu, pois se restringe a arquivos com reduzido número de registros, abaixo de 5.000.
        Vamos usar o teste de normalidade de Anderson-Darling que é uma versão modificada do teste de
        Kolmogorov-Smirnov e tem mais poder para uma ampla gama de alternativas.

        Utilizar a estrutura de modelo Sequential do Keras, que é mais simples e intuitiva (há outras para serem
        testadas).
        Não remove os outliers dos dados de treinamento, o que pode melhorar a capacidade do modelo de detectar
        anomalias.
        Classifica =>> observação será considerado como uma anomalia se seu erro de reconstrução for maior do que três
        vezes o desvio padrão (+/-) 3 sigmas. Este é um método muito comum utilizado para detectar outliers.
        Contar o número de anomalias na base de treinamento e de teste.
        Imprimir uma descrição estatística para cada variável nas observações que foram classificadas como anomalias.
        Isso pode nos dar algumas indicações sobre quais variáveis contribuíram para as anomalias. Foi suprimido na
        entrega, pois neste apenas sinalizamos na saída os pontos sem anomalia (0) e com anomalia (1) no arquivo de
        saída.

    Importante: Esse código assume que seu conjunto de treinamento não contém anomalias (como normalmente é o caso ao
    treinar um Isolation Forest para detecção de anomalias). Caso o conjunto de treinamento contenha anomalias
    e quer que o modelo seja capaz de detectá-las, é preciso ajustar a maneira como o limiar é calculado.
"""

# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Definindo a classe AnomalyDetectorIsolationForest para detecção de anomalias usando Isolation Forest
class AnomalyDetectorIsolationForest:
    def __init__(self, file_path):
        self.file_path = file_path
        # Preparando os dados ao instanciar o objeto
        self.df, self.df_train, self.df_test = self.load_and_prepare_data()

    # Carregando e preparando os dados
    def load_and_prepare_data(self):
        # Carregando o conjunto de dados
        df = pd.read_csv(self.file_path)
        # Substituindo valores desconhecidos por NaN
        df = df.replace(99999.99, np.nan)

        # Preenchendo NaNs com a média da coluna
        for column in df.select_dtypes(include=[np.number]).columns:
            df[column].fillna(df[column].mean(), inplace=True)

        # Separando os dados em treinamento e teste com base na coluna 'role'
        df_train = df[df['role'] == 'normal'].select_dtypes(include=[np.number])
        df_test = df[df['role'] == 'test-0'].select_dtypes(include=[np.number])
        # Assegurando que as colunas de treino e teste são as mesmas
        df_test.columns = df_train.columns

        return df, df_train, df_test

    # Treinando o modelo e fazendo as previsões
    def train_and_predict(self):
        # Criando e treinando o modelo Isolation Forest
        model = IsolationForest(contamination=0.001)
        model.fit(self.df_train.values)
        df_train_pred = model.predict(self.df_train.values)
        df_test_pred = model.predict(self.df_test.values)
        return df_train_pred, df_test_pred

    # Calculando o erro de reconstrução e marcando anomalias
    def calculate_reconstruction_error(self, df_train_pred, df_test_pred):
        self.df_train['anomaly'] = (df_train_pred == -1).astype(int)
        self.df_test['anomaly'] = (df_test_pred == -1).astype(int)
        print(f"Isolation Forest: Número de anomalias na base de treinamento: {self.df_train['anomaly'].sum()}")
        print(f"Isolation Forest: Número de anomalias na base de teste: {self.df_test['anomaly'].sum()}")

    # Gerando relatório de anomalias
    def report_anomalies(self):
        report = pd.DataFrame(columns=['offset_seconds', 'Registro'] + list(self.df_test.columns[:-1]) + ['TOTAL_Anom'])
        anom_stats = pd.DataFrame(columns=['Registro', 'Variable', 'Mean', 'Found Value', 'Deviation'])

        # Verificando cada linha dos dados de teste
        for index, row in self.df_test.iterrows():
            anomaly_row = [index, index]
            total_anom = 0
            for column in self.df_test.columns[:-1]:  # Excluindo coluna 'anomaly'
                if row[column] > self.df_train[column].mean() + 3 * self.df_train[column].std():
                    anomaly_row.append(1)
                    total_anom += 1
                    found_value = row[column]
                    mean_value = self.df_train[column].mean()
                    deviation = (found_value - mean_value) / self.df_train[column].std()
                    anom_stats.loc[len(anom_stats)] = [index, column, mean_value, found_value, deviation]
                else:
                    anomaly_row.append(0)
            anomaly_row.append(total_anom)
            report_temp = pd.DataFrame([anomaly_row], columns=report.columns)
            report = pd.concat([report, report_temp], ignore_index=True)

        anom_stats.to_excel('DADOS_SIMUL/EST_ANOM_ISOForest.xlsx', index=False)
        return report

    # Função principal para executar o detector de anomalias
    def run(self):
        df_train_pred, df_test_pred = self.train_and_predict()
        self.calculate_reconstruction_error(df_train_pred, df_test_pred)
        report = self.report_anomalies()
        report.to_excel('DADOS_SIMUL/FINAL_A2.xlsx', index=False, sheet_name='RELATÓRIO')


# Executando o detector de anomalias usando Isolation Forest
if __name__ == "__main__":
    # Instanciando o objeto
    detector = AnomalyDetectorIsolationForest('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')
    detector.run()
