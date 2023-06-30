"""
A partir desse algoritmo estamos considerando que conhecemos a base.

Vamos atacar o problema inicialmente usando ISOLATIONFOREST ou LOF

Seguindo os seguintes passos:

    1- Carregar os dados e remover a coluna 'role', pois não acreditamos que seja relevante para a análise.
    2- Verificar o tipo de dados da coluna 'offset_seconds' e converter para datetime se necessário.
    3- Ordenar o dataframe pela coluna 'offset_seconds' para garantir que os dados estão em ordem cronológica.
    4- Criar novas features que capturam a média móvel e/ou o desvio padrão móvel para cada variável ao longo do tempo.
    5- Aplicar um algoritmo de detecção de anomalias não supervisionado, como o Isolation Forest ou o LOF,
    para classificar os dados como "normal" ou "anomalia".
    6- Avaliar o desempenho do modelo e ajustar conforme necessário.

    Nota: Vamos incluir uma análise de falsos positivos e falsos negativos. No entanto, devemos notar que, como estamos lidando com um  problema de aprendizado de máquina não supervisionado, não temos rótulos verdadeiros para comparar as previsões do modelo. Portanto,    essa análise pode ser um pouco desafiadora.

        Uma possibilidade seria considerar as anomalias detectadas pelo modelo como "previsões de falha" e as não-anomalias como "previsões de operação normal". Neste caso, poderíamos assumir temporariamente que todas as previsões de falha são corretas (verdadeiros positivos) e que todas as previsões de operação normal são corretas (verdadeiros negativos).

        Em seguida, poderíamos introduzir uma certa taxa de erro em nossas previsões para simular falsos positivos (falhas previstas que não ocorreram) e falsos negativos (falhas que não foram previstas). Isso nos permitiria ajustar nosso modelo para minimizar esses erros. No entanto, é importante ressaltar que esta abordagem seria apenas uma aproximação e os resultados podem não ser totalmente precisos sem rótulos verdadeiros.

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix

# Carregar dados
df = pd.read_excel('DADOS/OUTLIERS_FREE.xlsx')

# Remover a coluna 'role'
df = df.drop(['role'], axis=1)

# Verificar e converter 'offset_seconds' para datetime se necessário
# ...

# Ordenar o dataframe por 'offset_seconds'
df = df.sort_values('offset_seconds')

# Criar novas features para capturar a média móvel e/ou o desvio padrão móvel
# ...

# Aplicar o Isolation Forest para detecção de anomalias
iso = IsolationForest(contamination=0.1)
df['anomaly'] = iso.fit_predict(df.drop(['offset_seconds'], axis=1).values) # Convertendo para numpy array

# Converter os rótulos de anomalia de -1/1 para 0/1
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

# Fazer uma cópia dos rótulos de anomalia
df['true_anomaly'] = df['anomaly'].copy()

# Simular falsos positivos e negativos
fp_rate = 0.05  # Taxa de falsos positivos
fn_rate = 0.05  # Taxa de falsos negativos

fp = np.random.choice([0, 1], size=df.shape[0], p=[1 - fp_rate, fp_rate])
fn = np.random.choice([0, 1], size=df.shape[0], p=[1 - fn_rate, fn_rate])

df['anomaly'] = np.where(df['anomaly'] == 1, fp, df['anomaly'])
df['anomaly'] = np.where(df['anomaly'] == 0, fn, df['anomaly'])

# Calcular falsos positivos e negativos
cm = confusion_matrix(df['true_anomaly'], df['anomaly']) # Comparando com os rótulos verdadeiros
tn, fp, fn, tp = cm.ravel()

# Calcular as porcentagens de falsos positivos e falsos negativos
fp_rate = fp / (fp + tn)
fn_rate = fn / (fn + tp)

print(f'Verdadeiros negativos: {tn}')
print(f'Falsos positivos: {fp}')
print(f'Falsos negativos: {fn}')
print(f'Verdadeiros positivos: {tp}')
print(f'Taxa de falsos positivos: {fp_rate * 100:.2f}%')
print(f'Taxa de falsos negativos: {fn_rate * 100:.2f}%')

# Plotar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Falha'],
            yticklabels=['Normal', 'Falha'])
plt.xlabel('Predição')
plt.ylabel('Verdade')
plt.show()
