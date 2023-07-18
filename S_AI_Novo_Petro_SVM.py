"""
Neste algoritmo temos o seguinte:

    Este algoritmo foi direcionado para trabalhar com a base Adendo A.2_Conjunto de Dados_DataSet.csv
    Dividir a base de dados com base na coluna 'role': Vamos dividir o DataFrame original em dois: um DataFrame de
    treinamento que consiste apenas em dados 'normais' e um DataFrame de teste que consiste em dados 'test-0'.
    Vamos usar o DataFrame de treinamento para treinar nosso autoencoder e o DataFrame de teste para avaliar seu
    desempenho.

    Treinar o One-Class SVM: Com base no DataFrame de treinamento, vamos treinar nosso autoencoder. O objetivo é
    permitir
    que o One-Class SVM aprenda a reconstruir as operações normais com a menor perda possível.
        A opção pelo uso do One-Class SVM --> se deve ao fato de ser um algoritmo propício para detecção de anomalias
        (há outros) mas vamos começar por aqui. Outro fato importante é que se trata de um algoritmo de aprendizagem
        profunda e que não demanda que os dados sigam uma distribuição normal.

    Testar o One-Class SVM: Após o treinamento, vamos aplicar o autoencoder aos dados de teste para gerar previsões. Vamos
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
from sklearn import svm
from scipy.stats import anderson
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data():
    df = pd.read_csv('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')
    df = df.replace(99999.99, np.nan)

    for column in df.select_dtypes(include=[np.number]).columns:
        df[column].fillna(df[column].mean(), inplace=True)

    df_train = df[df['role'] == 'normal'].select_dtypes(include=[np.number])
    df_test = df[df['role'] == 'test-0'].select_dtypes(include=[np.number])

    return df, df_train, df_test


def check_normality(df):
    for column in df.select_dtypes(include=[np.number]).columns:
        result = anderson(df[column])
        stat = result.statistic
        critical_values = result.critical_values
        if stat > critical_values[-1]:
            print(f'Coluna {column} não segue uma distribuição normal')
        else:
            print(f'Coluna {column} segue uma distribuição normal')


def train_and_predict(df_train, df_test):
    scaler = StandardScaler()
    df_train_scaled = scaler.fit_transform(df_train)
    df_test_scaled = scaler.transform(df_test)

    model = svm.OneClassSVM(nu=0.001, kernel="rbf", gamma=0.01)
    model.fit(df_train_scaled)

    df_train_pred = model.predict(df_train_scaled)
    df_test_pred = model.predict(df_test_scaled)

    df_train_pred = [1 if x == -1 else 0 for x in df_train_pred]
    df_test_pred = [1 if x == -1 else 0 for x in df_test_pred]

    return df_train_pred, df_test_pred


def report_anomalies(df_train, df_test):
    report = pd.DataFrame(columns=['offset_seconds', 'Registro'] + list(df_test.columns[:-1]) + ['TOTAL_Anom'])
    anom_stats = pd.DataFrame(columns=['Registro', 'Variable', 'Mean', 'Found Value', 'Deviation'])

    for index, row in df_test.iterrows():
        anomaly_row = [index, index]
        total_anom = 0
        for column in df_test.columns[:-1]:  # Excluding 'anomaly' column
            if row[column] > df_train[column].mean() + 3 * df_train[column].std():
                anomaly_row.append(1)
                total_anom += 1
                found_value = row[column]
                mean_value = df_train[column].mean()
                deviation = (found_value - mean_value) / df_train[column].std()
                anom_stats.loc[len(anom_stats)] = [index, column, mean_value, found_value, deviation]
            else:
                anomaly_row.append(0)
        anomaly_row.append(total_anom)
        report_temp = pd.DataFrame([anomaly_row], columns=report.columns)
        report = pd.concat([report, report_temp], ignore_index=True)

    anom_stats.to_excel('DADOS_SIMUL/EST_ANOM_SVM.xlsx', index=False)
    return report


def main():
    df, df_train, df_test = load_and_prepare_data()
    check_normality(df)
    df_train_pred, df_test_pred = train_and_predict(df_train, df_test)
    print(f"One-Class SVM: Número de anomalias na base de treinamento: {sum(df_train_pred)}")
    print(f"One-Class SVM: Número de anomalias na base de teste: {sum(df_test_pred)}")

    df_train['anomaly'] = df_train_pred
    df_test['anomaly'] = df_test_pred
    report = report_anomalies(df_train, df_test)
    report.to_excel('DADOS_SIMUL/FINAL_A2.xlsx', index=False, sheet_name='RELATÓRIO')


if __name__ == "__main__":
    main()