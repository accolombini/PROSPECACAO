"""
Agora queremos ler nosso arquivo em uma planilha com duas abas e salvá-lo em duas planilhas separadas

"""

import pandas as pd
import os

# Ler o arquivo Excel
df = pd.read_excel('DADOS/saida.xlsx', sheet_name=['Normal', 'Teste-0'])

# Extrair as abas "Normal" e "Teste-0"
df_normal = df['Normal']
df_teste = df['Teste-0']

# Definir os nomes das abas
nome_normal = 'Normal'
nome_teste = 'Test-0'

# Criar o diretório se não existir
os.makedirs('../DADOS', exist_ok=True)

# Salvar as abas em arquivos separados com os nomes correspondentes
df_normal.to_excel(f'DADOS/saida_{nome_normal}.xlsx',
                   sheet_name=nome_normal, index=False)
df_teste.to_excel(f'DADOS/saida_{nome_teste}.xlsx',
                  sheet_name=nome_teste, index=False)
