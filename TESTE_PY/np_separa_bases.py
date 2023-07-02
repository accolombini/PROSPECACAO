"""
Separando o arquivo em normal e test-0 em duas abas diferentes da mesma planilha

"""

import pandas as pd
import xlsxwriter

# Carregando os dados
df = pd.read_csv('../DADOS/Adendo A.2_Conjunto de Dados_DataSet.csv')

# Separando os dados pelas categorias 'normal' e 'test-0'
df_normal = df[df['role'] == 'normal']
df_test = df[df['role'] == 'test-0']

# Criando uma nova instância do Workbook e adicionando as folhas
workbook = xlsxwriter.Workbook('../DADOS/dataset_separado.xlsx')
worksheet_normal = workbook.add_worksheet('Normal')
worksheet_test = workbook.add_worksheet('Test-0')

# Escrita de dados na folha 'Normal'
for i, col in enumerate(df_normal.columns):
    worksheet_normal.write(0, i, col)
    for j, item in enumerate(df_normal[col]):
        worksheet_normal.write(j+1, i, item)

# Escrita de dados na folha 'Test-0'
for i, col in enumerate(df_test.columns):
    worksheet_test.write(0, i, col)
    for j, item in enumerate(df_test[col]):
        worksheet_test.write(j+1, i, item)

# Fechando o Workbook
workbook.close()
