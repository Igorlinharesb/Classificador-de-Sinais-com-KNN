# Classificador de Sinais com KNN e NPC

### Objetivo:
  Implementar os algoritmos de classificação K-Nearest Neighbors e Nearest Prototype para classificar sinais de ECG e sinais de áudio, testando diferentes valores de k e usar o valor que fornecer a melhor taxa de acerto. Além disso, deve-se avaliar os algoritmos com Cross-Validation. E ainda realizar a extração de pelo menos 5 atributos dos sinais que se mostrem relevantes para a classificação.
  
A implementação foi feita no software MATLAB e não foram utilizadas nenhuma das funções prontas para os classificadores, nem para o cross validation.

### Base de dados
A base de dados foi fornecida pelo professor da disciplina, consiste em 2 arquivos .mat localizados no diretório data. Cada arquivo contém 500 amostras de cada sinal, em que no arquivo Classe1.mat contém 500 amostras de sinais ECG e em Classe2.mat contém 500 amostras de sinais de áudio. A imagem abaixo mostra a visualização de algumas dessas amostras, onde os sinais em azul são de ECG e os vermelhos são de áudios.

![Exemplos de amostras](https://github.com/Igorlinharesb/Classificador-de-Sinais-com-KNN/blob/master/images/signals.png)

Observando as amostras acima, nota-se que sinais de áudio possuem apresentam maior amplitude e variações mais bruscas que os sinais de ECG.

Para realizar a classificação foram extraídos 5 atributos das amostras: Média, Desvio Padrão, Curtose, Assimetria e Amplitude máxima. O gráfico de dispersão abaixo ajuda visualizar como os atributos estão relacionados dentre as classes.

![Gráficos de Dispersão e Histogramas](https://github.com/Igorlinharesb/Classificador-de-Sinais-com-KNN/blob/master/images/paiplots.png)

Observando o gráfico acima é possível inferir que as amostras possuem atributos são fácilmente separáveis com o KNN, principalmente se tratando da curtose e da assimetria. Isso devido ao fato das nuvens serem densas e distantes entre si, e desde que o valor de K seja menor que a quantidade de atributos de determinada classe, é possível conseguir 100% de precisão nas previsões.

### Resultados

Abaixo são mostrados os resultados para cada um dos algoritmos.

![Resultados KNN](https://github.com/Igorlinharesb/Classificador-de-Sinais-com-KNN/blob/master/images/results_knn.png)

Para o KNN foram experimentados valores de K entre 0 e 80 com cross validation com 10-folds. No gráfico acima é possível notar que o algoritmo não atinge precisão máxima apenas com um valor de k muito grande (a partir de 40). 

O NPC também atingiu precisão máxima na classificação dos sinais, com a vantagem de ser menos custoso computacionalmente.
