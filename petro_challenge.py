""" 
Este projeto tem por objetivo atender ao desafio da Petrobras visando atender a seguinte demanda:

"A  busca  é  por  uma  ferramenta  capaz  de  realizar  predição  de  falhasem equipamentosutilizando  modelos computacionais  baseados  emtécnicas modernas e disruptivas, como Inteligência Artificial, de forma integrada aos processos   e   softwares   de   gestão   da   companhia,   e   que   forneça   as informações  com  antecedência  adequadapara  ação, com  boa  acurácia(poucos falsospositivos/negativos), indicando o diagnóstico e ações claras e efetivas para que as equipes de operação e manutenção possam evitar ou  dirimir  falhas  potenciais, além  de otimizar  os  planos de  manutenção preventivados equipamentos e sistemas"

"""

# Importe as bibliotecas necessárias

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Carregando os dados

data = pd.read_csv('DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Manipulando os dados conforme necessidade

# Exibir as primeiras 5 linhas do DataFrame
print(f"As 5 primeiras linhas do arquivo \n{data.head()} \n")

# Verificar o número de linhas e colunas do DataFrame
print(f"O formato dos dados -> {data.shape} \n")

# Verificar os nomes das colunas
print(f"O nome das colunas \n{data.columns}\n")
