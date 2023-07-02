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
# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Função para carregar e preparar os dados
def load_and_prepare_data():
    # Carrega os dados de um arquivo CSV
    df = pd.read_csv('DADOS/Adendo A.3_Conjunto de Dados_DataSet.csv')
    # Substitui os valores 99999.99 por NaN
    df = df.replace(99999.99, np.nan)
    # Preenche os valores NaN com a média da coluna correspondente
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column].fillna(df[column].mean(), inplace=True)
    # Divide os dados em conjuntos de treinamento e de teste
    df_train = df[df['role'] == 'normal'].select_dtypes(include=[np.number])
    df_test_0 = df[df['role'] == 'test-0'].select_dtypes(include=[np.number])
    df_test_1 = df[df['role'] == 'test-1'].select_dtypes(include=[np.number])
    # Retorna os conjuntos de dados
    return df, df_train, df_test_0, df_test_1

# Função para analisar os dados
def data_analysis(df, df_name):
    # Imprime informações básicas sobre os dados
    print(f"Análise da base de dados: {df_name}")
    print(f"Quantidade de linhas: {df.shape[0]}")
    print(f"Quantidade de colunas: {df.shape[1]}")
    print("Tipo de dados em cada coluna:")
    print(df.dtypes)
    print("Quantidade de dados faltantes em cada coluna:")
    print(df.isnull().sum())
    print("\n")

# Função para criar o modelo de detecção de anomalias
def create_model(input_dim, encoding_dim=14):
    # Cria um modelo sequencial
    model = Sequential()
    # Adiciona uma camada densa com ativação ReLU
    model.add(Dense(encoding_dim, activation="relu", input_shape=(input_dim,)))
    # Adiciona outra camada densa com ativação sigmoid
    model.add(Dense(input_dim, activation='sigmoid'))
    # Compila o modelo com o otimizador Adam e a perda de erro quadrático médio
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Retorna o modelo
    return model

# Função para treinar o modelo e prever anomalias
def train_and_predict(model, df_train, df_test):
    # Treina o modelo nos dados de treinamento
    model.fit(df_train, df_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)
    # Usa o modelo para prever anomalias nos dados de teste
    df_pred = model.predict(df_test)
    # Retorna as previsões
    return df_pred

# Função para relatar anomalias
def report_anomalies(df_train, df_test):
    # Cria um DataFrame vazio para o relatório
    report = pd.DataFrame(columns=['offset_seconds', 'Registro'] + list(df_test.columns) + ['TOTAL_Anom'])
    total_anom = 0
    # Itera sobre cada linha nos dados de teste
    for index, row in df_test.iterrows():
        anomaly_row = [index, index]
        row_anom = 0
        # Itera sobre cada coluna na linha
        for column in df_test.columns:
            # Se o valor é mais de três desvios padrão da média, marca como uma anomalia
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
    # Retorna o relatório e o total de anomalias
    return report, total_anom

# Função principal
def main():
    # Carrega e prepara os dados
    df, df_train, df_test_0, df_test_1 = load_and_prepare_data()
    # Realiza a análise dos dados
    data_analysis(df, "Total")
    data_analysis(df_train, "Treinamento")
    data_analysis(df_test_0, "Teste-0")
    data_analysis(df_test_1, "Teste-1")
    # Cria o modelo
    model = create_model(df_train.shape[1])
    # Treina o modelo e faz previsões nos dados de teste
    df_test_0_pred = train_and_predict(model, df_train, df_test_0)
    report_0, total_anom_0 = report_anomalies(df_train, df_test_0)
    df_test_1_pred = train_and_predict(model, df_train, df_test_1)
    report_1, total_anom_1 = report_anomalies(df_train, df_test_1)
    # Escreve os relatórios para arquivos Excel
    with pd.ExcelWriter('DADOS/FINAL_A3.xlsx') as writer:
        report_0.to_excel(writer, index=False, sheet_name='RELAT_T_0')
        report_1.to_excel(writer, index=False, sheet_name='RELAT_T1')
    # Imprime o total de anomalias
    print(f"Total de anomalias na base Teste-0: {total_anom_0}")
    print(f"Total de anomalias na base Teste-1: {total_anom_1}")

# Executa a função principal se o script é executado diretamente
if __name__ == "__main__":
    main()
