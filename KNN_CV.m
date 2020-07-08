% Universidade Federal do Cear� - UFC
% Reconhecimento de  Padr�es - 2020.1
% Francisco Igor Fel�cio Linhares - 374874

% K Nearest Neighbors(KNN) e Nearest Prototype Classifier(NPC)
% implementados e avaliados com Cross-Validation

% Comando para ver tempo de execu��o do script
tic;

% Inicializando o ambiente de trabalho
clc;
clear;
close all;

% Carregando a base de dados
load('data/Classe1.mat');
load('data/Classe2.mat');

Classe1 = Classe1';
Classe2 = Classe2';

% Inicializando a matriz de caracter�sticas.
features1 = zeros(length(Classe1(:,1)), 6);
features2 = ones(length(Classe2(:,1)), 6);

% A �ltima coluna representa a classe do sinal: 
% 0 - Classe1  e 1 - Classe2

% Visualizando as amostras
figure('Name', 'Visualizando sinais')
for i=1:4
    subplot(2,2, i)
    plot(Classe1(i,:))
    hold on
    plot(Classe2(i,:))
end

% Extraindo atributos de cada amostra
for i=1:length(Classe1(:,1))
    % Extraindo a m�dia
    features1(i,1) = mean(Classe1(i,:));
    features2(i,1) = mean(Classe2(i,:));
        
    % Extraindo o desvio padr�o
    features1(i,2) = std(Classe1(i,:));
    features2(i,2) = std(Classe2(i,:));
   
    % Extraindo a kurtose
    features1(i,3) = kurtosis(Classe1(i,:));
    features2(i,3) = kurtosis(Classe2(i,:));
    
    % Extraindo a assimetria
    features1(i,4) = skewness(Classe1(i,:));
    features2(i,4) = skewness(Classe2(i,:));
    
    % Extraindo a amplitude
    features1(i,5) = max(Classe1(i,:)) - min(Classe1(i,:));
    features2(i,5) = max(Classe2(i,:)) - min(Classe2(i,:));
end

% Nomalizando os atributos de forma que tenham m�dia 0 e vari�ncia 1 (z-score)
normalized_features = normalize([features1(:,1:end-1);features2(:,1:end-1)]);

% Adicionando a coluna target
normalized_features = [normalized_features [features1(:,end);features2(:,end)]];

% Plotando os dados
% Nome dos eixos/atributos
feature_names = {'Mean', 'Standard Deviation', 'Kurtosis', 'Skewness', 'Amplitude'};

% Gerando scatterplot em pares
figure('Name', 'Gr�ficos de Dispers�o e Histogramas atributos normalizados');
pairplot(normalized_features, feature_names)

% Testando os classificadores com os dados normalizados
[results_knn, results_npc] = cross_validation(normalized_features, 10, 80);

% Results knn � uma matriz, onde as linhas s�o os resultados em cada 'fold'
% e as colunas s�o os resultados para cada valor k, ent�o utilizarei o
% resultado m�dio para os resultados de cada valor de k
results_knn = mean(results_knn);

% Plotando acur�cias do KNN para diferentes valores de k
figure('Name', 'Resultados KNN');
plot(results_knn)
title('Taxa de acertos nas previs�es com KNN')
ylabel('Precis�o')
xlabel('Quantidade de vizinhos')

% Plotando os resultados do NPC para cada 'fold'
figure('Name', 'Resultados NPC');
plot(results_npc)
title('Taxa de acertos com NPC nas previs�es em cada divis�o')
ylabel('Precis�o')
xlabel('Fold')

% Parando a verifica��o do tempo de execu��o
toc;

% --------------------------- FUN��ES -------------------------------------

% Fun��o que faz a plotagem em pares, recebe como par�metro a matriz de
% atributos e os nomes dos atributos.
function pairplot(dataset, feature_names)
    m = length(dataset(1,:))-1; % N�mero de atributos
    % La�o que povoa os subplots
    for i=1:m
        for j=1:m
            % Condicional que assegura que n�o ter�o gr�ficos repetidos 
            if i <= j
                % Condicional que plota os  scatterplots dos atributos ixj
                if i ~= j
                    subplot(m, m, m*(j-1)+i);
                    final = length(dataset)/2;
                    plot(dataset(1:final, i), dataset(1:final, j), '.')
                    hold on;
                    begin = final+1;
                    final = length(dataset);
                    plot(dataset(begin:final, i), dataset(begin:final, j), '.')
                else
                    % No caso de colunas com i igual � j, � plotado o
                    % histograma daquele atributo
                    subplot(m, m, m*(j-1)+i);
                    final = length(dataset)/2;
                    histogram(dataset(1:final, i), 10);
                    hold on;
                    begin = final+1;
                    final = length(dataset);
                    histogram(dataset(begin:final, i), 10);
                end
                % Adiciona a label no eixo vertical apenas na primeira
                % coluna de gr�ficos
                if i == 1
                    ylabel(feature_names(j)) 
                end
            end
        end
        % Adiciona label no eixo horizontal apenas na �ltima linha de gr�ficos
        if j == 5
            xlabel(feature_names(i))
        end
    end
    hold off;
end


% Fun��o que calcula a dist�ncia euclidiana entre 2 vetores de atributos
function dist = euclidean_distance(v1, v2)
    dist = sqrt(sum((v1-v2).^2));
end


% Fun��o que retorna o vetor de previs�es e o zre obtido do dataset de
% teste com base no dataset de treino avaliando os k vizinhos mais pr�ximos
function score = knn(train, test, k)
    
    % inicializando vetor de previs�es
    predictions = zeros(length(test), 1);
    
    % Calculando a dist�ncia de cada vetor de atributos de teste para todos
    % os vetores de atributos de treino
    for i=1: length(test)    
        classe1 = 0;
        classe2 = 0;
        dists = zeros(length(train), 1);
        for j=1: length(train)
            % Calcula a dist�ncia euclidiana
            dists(j) = euclidean_distance(test(i,1:end-1), train(j,1:end-1));
            % Ordena o vetor de dist�ncias e retorna as dist�ncias
            % ordenadas e os �ndices de ordena��o
            [~ , Indexes_sorted] = sort(dists);
            
            % Conta as classes dos k vizinhos mais pr�ximos
            classe1 = sum(train(Indexes_sorted(1:k), end) == 0);
            classe2 = sum(train(Indexes_sorted(1:k), end) == 1);
            
            % Certifica-se que n�o houve empate entre as classes
            if classe1 == classe2
                classe1 = sum(train(Indexes_sorted(1:k-1), end) == 0);
                classe2 = sum(train(Indexes_sorted(1:k-1), end) == 1);
            end
        end
        
        % Atribui a classe mais frequente entre os k vizinhos ao vetor de
        % teste
        if classe1 > classe2
            predictions(i) = 0;
        else
            predictions(i) = 1;                
        end
    end
    
    % Calcula a precis�o das previs�es
    hits = 0;
    for i=1:length(predictions)
        if predictions(i) == test(i, end)
            hits = hits + 1;
        end
    end
    score = hits/length(predictions);
    
end

% Fun��o que retorna o vetor de previs�es e o score obtido do dataset de
% teste com base no dataset de treino avaliando o centr�ide mais pr�ximo.

function score = npc(train, test)
    
    % Inicializando o vetor de previs�es
    predictions = zeros(length(test), 1);
    
    % Calculando os centr�ides
    
    % Pegando os �ndices de todas as amostras de treino de cada classe
    ids1 = find(train(:, end) == 0);
    ids2 = find(train(:, end) == 1);
    
    % Encontrando os centr�ides
    c1 = mean(train(ids1, 1:end-1));
    c2 = mean(train(ids2, 1:end-1));
    
    % Essa linha serve apenas para os valores dos centr�ides serem
    % serem mostrados na janela de comando
    % [c1;c2]
    
    % Encontrando o centr�ide mais pr�ximo de cada amostra de teste
    for i=1:length(test)
       dist1 = euclidean_distance(test(i, 1:end-1), c1);
       dist2 = euclidean_distance(test(i, 1:end-1), c2);

       if dist1 < dist2
           predictions(i) = 0;
       else
           predictions(i) = 1;
       end
    end
    
    
    % Calculando a precis�o das previs�es
    hits = 0;
    for i=1:length(predictions)
        if predictions(i) == test(i, end)
            hits = hits + 1;
        end
    end
    score = hits/length(predictions);
end

% Fun��o que faz o cross validation e retorna vetor com percentual de
% acertos
function [cv_scores_knn, cv_scores_npc] = cross_validation(dataset, kfolds, max_neighbors)

    % Inicializando os vetores de scores
    cv_scores_knn = zeros(kfolds,max_neighbors);
    cv_scores_npc = zeros(kfolds, 1);
    
    % Tamanho de cada 'fold'
    fold_size = length(dataset)/kfolds;
    
    % O intuito de gerar os �ndices aleat�rios por cada classe � manter as
    % classes balanceadas, de forma que em todas as divis�es de treino e
    % teste sempre seja mantida a propor��o 50-50 das classes.
    middle = length(dataset)/2;
    indexes1 = randperm(middle, middle);
    indexes2 = randperm(middle, middle)+middle;  
    % class_size � a quantidade de amostras por classe em cada k
    class_size = fold_size/2;
    
    % La�o que faz a divis�o dos sets de treino e teste, previs�o com knn e
    % avalia��o da acur�cia
    for fold=1:kfolds
        %  Juntando os �ndices em um vetor tempor�rio de forma que os 500
        %  primeiros �ndices pertecem � classe1 e o restante � classe 2
        temp_indexes = [indexes1 indexes2];
        
        % Gera a lista de �ndices a serem utilizados para teste.
        test_id1 = temp_indexes(class_size*(fold-1) + 1:fold*class_size); % �ndices da classe 1
        test_id2 = temp_indexes(class_size*(fold-1) + middle+1:middle+fold*class_size); % �ndices da classe 2
        test_indexes = [test_id1 test_id2];
        % Separa o dataset de teste
        test = dataset(test_indexes, :);
        
        % Remove os �ndices utilizados para o dataset de teste dos �ndices
        % tempor�rios, restando apenas os �ndices que n�o foram utilizados
        % no teste
        temp_indexes([class_size*(fold-1) + 1:fold*class_size class_size*(fold-1) + middle+1:middle+fold*class_size]) = [];
        
        % Adiciona a parte do dataset n�o utilizado para teste em um
        % dataset de treino
        train = dataset(temp_indexes, :);

        % Classifica o dataset de treino com o knn com os valores de k
        % variando entre 1 e k_max e retorna a acur�cia para cada valor de
        % k
        for k=1: max_neighbors
            % O vetor abaixo foi colocado apenas para companhamento
            % da execu��o do algoritmo pela janela de comando
            [fold k]
            
            cv_scores_knn(fold, k) = knn(train, test, k);    
        end
        
        % Calcula a acur�cia obtida com o npc
        cv_scores_npc(fold) = npc(train, test);
    end 
end