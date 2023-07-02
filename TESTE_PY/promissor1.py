"""
    Refinando o algogitmo --> hoje 12/06 => vamos explorar o cluster 5 na busca de possível
    solução para o problema
    Este algoritmo tem como referência para rollback o algoritmo teste10.py
    Problema encontrado o Cluster 5 esta vazio - vamos tentar uma nova abordagem. Uso do GMM levou a muitos problemas.Vamos seguir continuando com novos testes.

"""

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(df, numeric_columns):
    # Verificar a presença de valores nulos
    if df[numeric_columns].isnull().values.any():
        raise ValueError(
            "Existem registros com valores nulos nas colunas numéricas.")

    # Verificar a presença de colunas faltantes
    missing_columns = set(numeric_columns) - set(df.columns)
    if missing_columns:
        raise ValueError("Existem colunas faltantes no DataFrame.")

    # Pré-processamento e análise estatística descritiva das colunas numéricas
    df_selected = df[numeric_columns]
    df_selected_scaled = StandardScaler().fit_transform(df_selected)

    return df_selected_scaled


def determine_clusters(df_selected_scaled, n_clusters):
    gmm_scores = []  # Armazenar os resultados de pontuação do GMM

    for n in n_clusters:
        gmm = GaussianMixture(n_components=n)
        gmm.fit(df_selected_scaled)
        gmm_scores.append(gmm.score(df_selected_scaled))

    return gmm_scores


def plot_elbow_curve(n_clusters, gmm_scores):
    # Plotar o gráfico do método Elbow
    plt.plot(n_clusters, gmm_scores, marker='o')
    plt.xlabel("Número de Clusters")
    plt.ylabel("Pontuação do GMM")
    plt.title("Método Elbow para determinação do número ideal de clusters")
    plt.show()


def perform_clustering(df_selected_scaled, best_n_clusters, numeric_columns):
    gmm = GaussianMixture(n_components=best_n_clusters)
    gmm.fit(df_selected_scaled)
    labels_gmm = gmm.predict(df_selected_scaled)
    df_selected = pd.DataFrame(df_selected_scaled, columns=numeric_columns)
    df_selected['cluster_label'] = labels_gmm
    return df_selected


def analyze_clusters(df, numeric_columns):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numeric_columns])

    # Generate correlation heatmap
    correlation_matrix = pd.DataFrame(
        df_scaled, columns=numeric_columns).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

    # Select variables with correlation > 0.5
    correlated_variables = correlation_matrix[correlation_matrix > 0.5].stack(
    ).reset_index()
    correlated_variables.columns = ['Variable 1', 'Variable 2', 'Correlation']
    correlated_variables = correlated_variables[correlated_variables['Variable 1']
                                                != correlated_variables['Variable 2']]

    # Filter variables based on correlation > 0.5
    selected_variables = correlated_variables[correlated_variables['Correlation'] > 0.5]['Variable 1'].unique(
    )

    # Preprocess data with selected variables
    df_selected_scaled = preprocess_data(df, selected_variables)

    # Determine the optimal number of clusters
    n_clusters = range(2, 11)
    gmm_scores = determine_clusters(df_selected_scaled, n_clusters)

    # Plot the elbow curve
    plot_elbow_curve(n_clusters, gmm_scores)

    # Perform clustering with the selected variables
    best_n_clusters = np.argmax(gmm_scores) + 2
    clustered_data = perform_clustering(
        df_selected_scaled, best_n_clusters, selected_variables)

    # Add cluster labels to the original DataFrame
    df['cluster_label'] = clustered_data['cluster_label']

    # Display statistics for each cluster
    for label in df['cluster_label'].unique():
        cluster_data = df[df['cluster_label'] == label]
        print(f'GMM Cluster {label}: {len(cluster_data)} amostra(s)')
        print(
            f'Estatísticas descritivas das colunas numéricas no Cluster {label}:')
        print(cluster_data[numeric_columns].describe())

    # Plot scatter plots for each pair of columns in the numeric columns
    for i in range(len(numeric_columns) - 1):
        for j in range(i + 1, len(numeric_columns)):
            x = numeric_columns[i]
            y = numeric_columns[j]
            if x in df.columns and y in df.columns:
                plot_scatter_clusters(df, x, y)
            else:
                print(
                    f"KeyError: One or both of the columns '{x}' and '{y}' does not exist in the DataFrame.")
                print("Available columns in the DataFrame:")
                print(df.columns)


def plot_scatter_clusters(df, x, y):
    plt.figure(figsize=(8, 6))
    for label in df['cluster_label'].unique():
        cluster_data = df[df['cluster_label'] == label]
        plt.scatter(cluster_data[x], cluster_data[y], label=f'Cluster {label}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Clusters GMM')
    plt.legend()
    plt.show()


def main():
    # Carregar o arquivo CSV
    df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

    # Selecionar apenas as colunas numéricas relevantes para análise
    numeric_columns = ['LIT_802', 'LIT_806', 'PDIT_830', 'PIC_807_IN', 'TE_820', 'TE_830', 'TE_831', 'TE_832',
                       'TE_833', 'TE_834', 'TE_835', 'TIT_806', 'TIT_809', 'VT_812X', 'VT_812Y', 'VT_813X',
                       'VT_813Y', 'VT_814X', 'VT_814Y']

    # Preprocess data and generate correlation heatmap
    df_selected_scaled = preprocess_data(df, numeric_columns)
    correlation_matrix = pd.DataFrame(
        df_selected_scaled, columns=numeric_columns).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

    # Select variables with correlation > 0.5
    correlated_variables = correlation_matrix[correlation_matrix > 0.5].stack(
    ).reset_index()
    correlated_variables.columns = ['Variable 1', 'Variable 2', 'Correlation']
    correlated_variables = correlated_variables[correlated_variables['Variable 1']
                                                != correlated_variables['Variable 2']]
    selected_variables = correlated_variables[correlated_variables['Correlation'] > 0.5]['Variable 1'].unique(
    )

    # Preprocess data with selected variables
    df_selected_scaled = preprocess_data(df, selected_variables)

    # Determine the optimal number of clusters
    n_clusters = range(2, 11)
    gmm_scores = determine_clusters(df_selected_scaled, n_clusters)

    # Plot the elbow curve
    plot_elbow_curve(n_clusters, gmm_scores)

    # Perform clustering with the selected variables
    best_n_clusters = np.argmax(gmm_scores) + 2
    clustered_data = perform_clustering(
        df_selected_scaled, best_n_clusters, selected_variables)

    # Add cluster labels to the original DataFrame
    df['cluster_label'] = clustered_data['cluster_label']

    # Display statistics for each cluster
    for label in df['cluster_label'].unique():
        cluster_data = df[df['cluster_label'] == label]
        print(f'GMM Cluster {label}: {len(cluster_data)} amostra(s)')
        print(
            f'Estatísticas descritivas das colunas numéricas no Cluster {label}:')
        print(cluster_data[numeric_columns].describe())

    # Plot scatter plots for each pair of columns in the numeric columns
    for i in range(len(numeric_columns) - 1):
        for j in range(i + 1, len(numeric_columns)):
            x = numeric_columns[i]
            y = numeric_columns[j]
            if x in df.columns and y in df.columns:
                plot_scatter_clusters(df, x, y)
            else:
                print(
                    f"KeyError: One or both of the columns '{x}' and '{y}' does not exist in the DataFrame.")
                print("Available columns in the DataFrame:")
                print(df.columns)


def plot_scatter_clusters(df, x, y):
    plt.figure(figsize=(8, 6))
    for label in df['cluster_label'].unique():
        cluster_data = df[df['cluster_label'] == label]
        plt.scatter(cluster_data[x], cluster_data[y], label=f'Cluster {label}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Clusters GMM')
    plt.legend()
    plt.show()


def main():
    # Carregar o arquivo CSV
    df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

    # Selecionar apenas as colunas numéricas relevantes para análise
    numeric_columns = ['LIT_802', 'LIT_806', 'PDIT_830', 'PIC_807_IN', 'TE_820', 'TE_830', 'TE_831', 'TE_832',
                       'TE_833', 'TE_834', 'TE_835', 'TIT_806', 'TIT_809', 'VT_812X', 'VT_812Y', 'VT_813X',
                       'VT_813Y', 'VT_814X', 'VT_814Y']

    # Preprocess data and generate correlation heatmap
    df_selected_scaled = preprocess_data(df, numeric_columns)
    correlation_matrix = pd.DataFrame(
        df_selected_scaled, columns=numeric_columns).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

    # Select variables with correlation > 0.5
    correlated_variables = correlation_matrix[correlation_matrix > 0.5].stack(
    ).reset_index()
    correlated_variables.columns = ['Variable 1', 'Variable 2', 'Correlation']
    correlated_variables = correlated_variables[correlated_variables['Variable 1']
                                                != correlated_variables['Variable 2']]
    selected_variables = correlated_variables[correlated_variables['Correlation'] > 0.5]['Variable 1'].unique(
    )

    # Preprocess data with selected variables
    df_selected_scaled = preprocess_data(df, selected_variables)

    # Determine the optimal number of clusters
    n_clusters = range(2, 11)
    gmm_scores = determine_clusters(df_selected_scaled, n_clusters)

    # Plot the elbow curve
    plot_elbow_curve(n_clusters, gmm_scores)

    # Perform clustering with the selected variables and optimal number of clusters
    best_n_clusters = np.argmax(gmm_scores) + 2
    clustered_data = perform_clustering(
        df_selected_scaled, best_n_clusters, selected_variables)

    # Add cluster labels to the original DataFrame
    df['cluster_label'] = clustered_data['cluster_label']

    # Display statistics for each cluster
    for label in df['cluster_label'].unique():
        cluster_data = df[df['cluster_label'] == label]
        print(f'GMM Cluster {label}: {len(cluster_data)} amostra(s)')
        print(
            f'Estatísticas descritivas das colunas numéricas no Cluster {label}:')
        print(cluster_data[numeric_columns].describe())

    # Plot scatter plots for each pair of columns in the numeric columns
    for i in range(len(numeric_columns) - 1):
        for j in range(i + 1, len(numeric_columns)):
            x = numeric_columns[i]
            y = numeric_columns[j]
            if x in df.columns and y in df.columns:
                plot_scatter_clusters(df, x, y)
            else:
                print(
                    f"KeyError: One or both of the columns '{x}' and '{y}' does not exist in the DataFrame.")
                print("Available columns in the DataFrame:")
                print(df.columns)


def plot_elbow_curve(n_clusters, gmm_scores):
    # Plot the elbow curve
    plt.plot(n_clusters, gmm_scores, marker='o')
    plt.xlabel("Número de Clusters")
    plt.ylabel("Pontuação do GMM")
    plt.title("Método Elbow para determinação do número ideal de clusters")
    plt.show()


def perform_clustering(df_selected_scaled, best_n_clusters, selected_variables):
    # Perform clustering with the selected variables and optimal number of clusters
    gmm = GaussianMixture(n_components=best_n_clusters)
    gmm.fit(df_selected_scaled)
    labels_gmm = gmm.predict(df_selected_scaled)
    clustered_data = pd.DataFrame(
        df_selected_scaled, columns=selected_variables)
    clustered_data['cluster_label'] = labels_gmm
    return clustered_data


def determine_clusters(df_selected_scaled, n_clusters):
    # Determine the optimal number of clusters using Gaussian Mixture Models
    gmm_scores = []
    for n in n_clusters:
        gmm = GaussianMixture(n_components=n)
        gmm.fit(df_selected_scaled)
        gmm_scores.append(gmm.score(df_selected_scaled))
    return gmm_scores


def main():
    # Carregar o arquivo CSV
    df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

    # Selecionar apenas as colunas numéricas relevantes para análise
    numeric_columns = ['LIT_802', 'LIT_806', 'PDIT_830', 'PIC_807_IN', 'TE_820', 'TE_830', 'TE_831', 'TE_832',
                       'TE_833', 'TE_834', 'TE_835', 'TIT_806', 'TIT_809', 'VT_812X', 'VT_812Y', 'VT_813X',
                       'VT_813Y', 'VT_814X', 'VT_814Y']

    # Preprocess data and generate correlation heatmap
    df_selected_scaled = preprocess_data(df, numeric_columns)
    correlation_matrix = pd.DataFrame(
        df_selected_scaled, columns=numeric_columns).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

    # Select variables with correlation > 0.5
    correlated_variables = correlation_matrix[correlation_matrix > 0.5].stack(
    ).reset_index()
    correlated_variables.columns = ['Variable 1', 'Variable 2', 'Correlation']
    correlated_variables = correlated_variables[correlated_variables['Variable 1']
                                                != correlated_variables['Variable 2']]
    selected_variables = correlated_variables[correlated_variables['Correlation'] > 0.5]['Variable 1'].unique(
    )

    # Preprocess data with selected variables
    df_selected_scaled = preprocess_data(df, selected_variables)

    # Determine the optimal number of clusters
    n_clusters = range(2, 11)
    gmm_scores = determine_clusters(df_selected_scaled, n_clusters)

    # Plot the elbow curve
    plot_elbow_curve(n_clusters, gmm_scores)

    # Perform clustering with the selected variables and optimal number of clusters
    best_n_clusters = np.argmax(gmm_scores) + 2
    clustered_data = perform_clustering(
        df_selected_scaled, best_n_clusters, selected_variables)

    # Add cluster labels to the original DataFrame
    df['cluster_label'] = clustered_data['cluster_label']

    # Display statistics for each cluster
    for label in df['cluster_label'].unique():
        cluster_data = df[df['cluster_label'] == label]
        print(f'GMM Cluster {label}: {len(cluster_data)} amostra(s)')
        print(
            f'Estatísticas descritivas das colunas numéricas no Cluster {label}:')
        print(cluster_data[numeric_columns].describe())

    # Plot scatter plots for each pair of columns in the numeric columns
    for i in range(len(numeric_columns) - 1):
        for j in range(i + 1, len(numeric_columns)):
            x = numeric_columns[i]
            y = numeric_columns[j]
            if x in df.columns and y in df.columns:
                plot_scatter_clusters(df, x, y)
            else:
                print(
                    f"KeyError: One or both of the columns '{x}' and '{y}' does not exist in the DataFrame.")
                print("Available columns in the DataFrame:")
                print(df.columns)


# Executar a função main()
if __name__ == "__main__":
    main()
