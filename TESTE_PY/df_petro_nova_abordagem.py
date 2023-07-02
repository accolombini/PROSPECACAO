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

    Threshold adaptativo: Atualmente, você está usando um limite fixo para todas as amostras. No entanto, este limite
    pode não ser ideal para todas as amostras, especialmente se a distribuição do erro de reconstrução for
    significativamente não-normal. Uma maneira de lidar com isso é usar um limite adaptativo que depende da própria
    amostra (por exemplo, um valor que depende do valor de uma variável específica para a amostra).

Erro de reconstrução ponderado: Atualmente, todas as variáveis contribuem igualmente para o erro de reconstrução. No
entanto, pode ser que algumas variáveis sejam mais importantes para a detecção de anomalias do que outras.
Uma maneira de lidar com isso é ponderar o erro de reconstrução com a importância da variável
(por exemplo, com base na variância da variável).

Regularização: Adicionar regularização (como L1 ou L2) ao seu Autoencoder pode ajudar a melhorar o desempenho.
A regularização pode forçar o Autoencoder a aprender representações mais robustas dos dados.

"""