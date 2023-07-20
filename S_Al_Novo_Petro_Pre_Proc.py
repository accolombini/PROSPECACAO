"""
Este algoritmo é uma forma de iteirar com a base de dados.

    O que ele se propõem a fazer:

    1- Acesse o diretório chamado "DADOS".
    2- Liste todos os arquivos .csv e .xlsx presentes neste diretório.
    3- Permita ao usuário selecionar um desses arquivos.
    4- Carregue o arquivo selecionado e realize uma pré-análise (verificando tipos de dados, variáveis e dimensão).
    5- Calcule estatísticas descritivas do conjunto de dados.
    6- Gere um mapa de calor mostrando as correlações entre as variáveis.
    7- Exiba tudo isso em um relatório visual na tela.
    8- Correlações: identificará variáveis com correlações fortes (>= 0.75) e fracas (<= 0.25) e imprimirá essa
    informação.
    9- Normalidade: o teste de Anderson-Darling para testar a normalidade dos dados. O teste de
    Anderson-Darling é preferido em relação ao de Shapiro-Wilk para amostras grandes, uma vez que o último pode se
    tornar muito lento e até mesmo inaplicável.
    10- O método chamado test_normality verificará a normalidade para cada variável numérica e imprimirá os resultados.
"""

# Importar bibliotecas

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import anderson


# Carregar os dados com base na extensão do arquivo
def load_data(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)


# Mostrar informações básicas sobre o arquivo selecionado
def show_file_info(df):
    # Separando as variáveis numéricas
    numeric_df = df.select_dtypes(include=[float, int])

    print("\nInformações sobre o arquivo:")
    print(df.info())
    print("\nDescrição estatística (apenas variáveis numéricas):")
    print(numeric_df.describe())


# Plota o mapa de calor para mostrar correlação entre as variáveis numéricas
def plot_heatmap(df):
    numeric_df = df.select_dtypes(include=[float, int])
    correlation = numeric_df.corr()
    plt.figure(figsize=(24, 18))
    sns.heatmap(correlation, annot=True, cmap="coolwarm")
    plt.title("Mapa de Calor de Correlação (apenas variáveis numéricas)")
    plt.show()


# Reporta variáveis com alta e baixa correlação
def report_high_low_correlations(df):
    numeric_df = df.select_dtypes(include=[float, int])
    correlation = numeric_df.corr()

    # Variáveis com alta correlação
    high_corr = [(var1, var2) for var1 in correlation for var2 in correlation if 0.75 <= correlation[var1][var2] < 1]
    # Variáveis com baixa correlação
    low_corr = [(var1, var2) for var1 in correlation for var2 in correlation if correlation[var1][var2] <= 0.25]

    print("\nVariáveis com alta correlação (>= 0.75):")
    for var1, var2 in high_corr:
        print(f"{var1} e {var2} : {correlation[var1][var2]:.2f}")

    print("\nVariáveis com baixa correlação (<= 0.25):")
    for var1, var2 in low_corr:
        print(f"{var1} e {var2} : {correlation[var1][var2]:.2f}")


# Verifica a normalidade das variáveis numéricas usando o teste de Anderson-Darling
def test_normality(df):
    numeric_df = df.select_dtypes(include=[float, int])

    print("\nTeste de Normalidade (Anderson-Darling):")
    for column in numeric_df.columns:
        result = anderson(numeric_df[column])
        if result.statistic < result.critical_values[2]:  # Comparação no nível de significância de 5%
            print(f"{column} parece ter uma distribuição normal.")
        else:
            print(f"{column} não parece ter uma distribuição normal.")


# Função principal para execução do programa
def main():
    directory = "DADOS"
    files = [f for f in os.listdir(directory) if f.endswith(('.csv', '.xlsx'))]

    if not files:
        print("Nenhum arquivo .csv ou .xlsx foi encontrado no diretório.")
        return

    print("Arquivos disponíveis:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")

    while True:
        try:
            choice = int(input("\nEscolha o número do arquivo que deseja trabalhar (ou digite 0 para sair): "))
            if 0 <= choice <= len(files):
                break
        except ValueError:
            pass
        print("Escolha inválida. Tente novamente.")

    if choice == 0:
        return

    file_path = os.path.join(directory, files[choice - 1])
    df = load_data(file_path)

    show_file_info(df)
    plot_heatmap(df)
    report_high_low_correlations(df)
    test_normality(df)


# Executa a função principal se o script for executado como principal
if __name__ == "__main__":
    main()
