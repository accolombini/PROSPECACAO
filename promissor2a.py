"""
Alteração do algoritmo para melhorar a busca por defeitos -> OneClassSVM
Este algoritmo parte do algoritmo teste17.py (ter isso como referência para rollback)
dado que nosso primeiro sucesso aconteceu no teste16, vamos seguir a partir dele, pois os resultados estão longe da precisão desejada
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o arquivo CSV
df = pd.read_csv('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Dividir o conjunto de dados em treinamento e teste
train = df[df['role'] == 'normal']
test = df[df['role'] == 'test-0']

# Verificar o número de exemplos normais e anômalos
num_normals_train = train.shape[0]
num_anomalies_test = test.shape[0]

# Amostrar aleatoriamente a mesma quantidade de exemplos anômalos da base de teste
test_sampled = test.sample(n=num_normals_train, random_state=42)

# Combinar os exemplos normais do treinamento com os exemplos amostrados anômalos do teste
combined_data = pd.concat([train, test_sampled])

# Remover as colunas "role" e "offset_seconds" do conjunto de dados
combined_data = combined_data.drop(['role', 'offset_seconds'], axis=1)
test_sampled = test_sampled.drop(['role', 'offset_seconds'], axis=1)

# Separar os dados novamente em treinamento e teste
train = combined_data[:num_normals_train]
test = combined_data[num_normals_train:]

# Treinamento do modelo One-Class SVM
# Nu é um parâmetro de controle da taxa de anomalias esperada
model = OneClassSVM(nu=0.795)
model.fit(train)

# Predição no conjunto de teste
y_test_pred = model.predict(test_sampled)

# Transformar as predições em rótulos (1 para anomalia, -1 para normal)
y_test_pred_labels = [1 if pred == -1 else 0 for pred in y_test_pred]

# Criar as verdadeiras classes do conjunto de teste
y_test_true_labels = [0] * num_normals_train

# Cálculo da matriz de confusão
confusion = confusion_matrix(y_test_true_labels, y_test_pred_labels)

# Exibir a matriz de confusão usando Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Verdadeiro")
plt.show()

# Exibição da matriz de confusão textual
true_negative = confusion[0][0]
false_positive = confusion[0][1]
false_negative = confusion[1][0]
true_positive = confusion[1][1]

print("Matriz de Confusão:")
print("===================")
print("Verdadeiros Negativos (Normal):", true_negative)
print("Falsos Positivos (Anomalia):", false_positive)
print("Falsos Negativos (Normal classificado como Anomalia):", false_negative)
print("Verdadeiros Positivos (Anomalia):", true_positive)

# Cálculo da taxa de acerto
accuracy = (true_negative + true_positive) / (true_negative +
                                              false_positive + false_negative + true_positive)
print("Taxa de Acerto:", accuracy)

# Cálculo da taxa de detecção de anomalias (Recall)
if true_positive + false_negative != 0:
    recall = true_positive / (true_positive + false_negative)
else:
    recall = 0.0
print("Taxa de Detecção de Anomalias (Recall):", recall)

# Cálculo da taxa de precisão
if true_positive + false_positive != 0:
    precision = true_positive / (true_positive + false_positive)
else:
    precision = 0.0
print("Taxa de Precisão:", precision)

# Cálculo da medida F1
if precision + recall != 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0.0
print("Medida F1:", f1_score)
