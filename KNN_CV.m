% Universidade Federal do Ceará - UFC
% Reconhecimento de  Padrões - 2020.1
% Francisco Igor Felício Linhares - 374874

% K Nearest Neighbors(KNN) e Nearest Prototype Classifier(NPC)
% implementados e avaliados com Cross-Validation

% Comando para ver tempo de execução do script
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

% Inicializando a matriz de características.
features1 = zeros(length(Classe1(:,1)), 6);
features2 = ones(length(Classe2(:,1)), 6);

% A última coluna representa a classe do sinal: 
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
    % Extraindo a média
    features1(i,1) = mean(Classe1(i,:));
    features2(i,1) = mean(Classe2(i,:));
        
    % Extraindo o desvio padrão
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

% Nomalizando os atributos de forma que tenham média 0 e variância 1 (z-score)
normalized_features = normalize([features1(:,1:end-1);features2(:,1:end-1)]);

% Adicionando a coluna target
normalized_features = [normalized_features [features1(:,end);features2(:,end)]];

% Plotando os dados
% Nome dos eixos/atributos
feature_names = {'Mean', 'Standard Deviation', 'Kurtosis', 'Skewness', 'Amplitude'};

% Gerando scatterplot em pares
figure('Name', 'Gráficos de Dispersão e Histogramas atributos normalizados');
pairplot(normalized_features, feature_names)

% Testando os classificadores com os dados normalizados
[results_knn, results_npc] = cross_validation(normalized_features, 10, 80);

% Results knn é uma matriz, onde as linhas são os resultados em cada 'fold'
% e as colunas são os resultados para cada valor k, então utilizarei o
% resultado médio para os resultados de cada valor de k
results_knn = mean(results_knn);

% Plotando acurácias do KNN para diferentes valores de k
figure('Name', 'Resultados KNN');
plot(results_knn)
title('Taxa de acertos nas previsões com KNN')
ylabel('Precisão')
xlabel('Quantidade de vizinhos')

% Plotando os resultados do NPC para cada 'fold'
figure('Name', 'Resultados NPC');
plot(results_npc)
title('Taxa de acertos com NPC nas previsões em cada divisão')
ylabel('Precisão')
xlabel('Fold')

% Parando a verificação do tempo de execução
toc;

% --------------------------- FUNÇÕES -------------------------------------

% Função que faz a plotagem em pares, recebe como parâmetro a matriz de
% atributos e os nomes dos atributos.
function pairplot(dataset, feature_names)
    m = length(dataset(1,:))-1; % Número de atributos
    % Laço que povoa os subplots
    for i=1:m
        for j=1:m
            % Condicional que assegura que não terão gráficos repetidos 
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
                    % No caso de colunas com i igual à j, é plotado o
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
                % coluna de gráficos
                if i == 1
                    ylabel(feature_names(j)) 
                end
            end
        end
        % Adiciona label no eixo horizontal apenas na última linha de gráficos
        if j == 5
            xlabel(feature_names(i))
        end
    end
    hold off;
end


% Função que calcula a distância euclidiana entre 2 vetores de atributos
function dist = euclidean_distance(v1, v2)
    dist = sqrt(sum((v1-v2).^2));
end


% Função que retorna o vetor de previsões e o zre obtido do dataset de
% teste com base no dataset de treino avaliando os k vizinhos mais próximos
function score = knn(train, test, k)
    
    % inicializando vetor de previsões
    predictions = zeros(length(test), 1);
    
    % Calculando a distância de cada vetor de atributos de teste para todos
    % os vetores de atributos de treino
    for i=1: length(test)    
        classe1 = 0;
        classe2 = 0;
        dists = zeros(length(train), 1);
        for j=1: length(train)
            % Calcula a distância euclidiana
            dists(j) = euclidean_distance(test(i,1:end-1), train(j,1:end-1));
            % Ordena o vetor de distâncias e retorna as distâncias
            % ordenadas e os índices de ordenação
            [~ , Indexes_sorted] = sort(dists);
            
            % Conta as classes dos k vizinhos mais próximos
            classe1 = sum(train(Indexes_sorted(1:k), end) == 0);
            classe2 = sum(train(Indexes_sorted(1:k), end) == 1);
            
            % Certifica-se que não houve empate entre as classes
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
    
    % Calcula a precisão das previsões
    hits = 0;
    for i=1:length(predictions)
        if predictions(i) == test(i, end)
            hits = hits + 1;
        end
    end
    score = hits/length(predictions);
    
end

% Função que retorna o vetor de previsões e o score obtido do dataset de
% teste com base no dataset de treino avaliando o centróide mais próximo.

function score = npc(train, test)
    
    % Inicializando o vetor de previsões
    predictions = zeros(length(test), 1);
    
    % Calculando os centróides
    
    % Pegando os índices de todas as amostras de treino de cada classe
    ids1 = find(train(:, end) == 0);
    ids2 = find(train(:, end) == 1);
    
    % Encontrando os centróides
    c1 = mean(train(ids1, 1:end-1));
    c2 = mean(train(ids2, 1:end-1));
    
    % Essa linha serve apenas para os valores dos centróides serem
    % serem mostrados na janela de comando
    % [c1;c2]
    
    % Encontrando o centróide mais próximo de cada amostra de teste
    for i=1:length(test)
       dist1 = euclidean_distance(test(i, 1:end-1), c1);
       dist2 = euclidean_distance(test(i, 1:end-1), c2);

       if dist1 < dist2
           predictions(i) = 0;
       else
           predictions(i) = 1;
       end
    end
    
    
    % Calculando a precisão das previsões
    hits = 0;
    for i=1:length(predictions)
        if predictions(i) == test(i, end)
            hits = hits + 1;
        end
    end
    score = hits/length(predictions);
end

% Função que faz o cross validation e retorna vetor com percentual de
% acertos
function [cv_scores_knn, cv_scores_npc] = cross_validation(dataset, kfolds, max_neighbors)

    % Inicializando os vetores de scores
    cv_scores_knn = zeros(kfolds,max_neighbors);
    cv_scores_npc = zeros(kfolds, 1);
    
    % Tamanho de cada 'fold'
    fold_size = length(dataset)/kfolds;
    
    % O intuito de gerar os índices aleatórios por cada classe é manter as
    % classes balanceadas, de forma que em todas as divisões de treino e
    % teste sempre seja mantida a proporção 50-50 das classes.
    middle = length(dataset)/2;
    indexes1 = randperm(middle, middle);
    indexes2 = randperm(middle, middle)+middle;  
    % class_size é a quantidade de amostras por classe em cada k
    class_size = fold_size/2;
    
    % Laço que faz a divisão dos sets de treino e teste, previsão com knn e
    % avaliação da acurácia
    for fold=1:kfolds
        %  Juntando os índices em um vetor temporário de forma que os 500
        %  primeiros índices pertecem à classe1 e o restante à classe 2
        temp_indexes = [indexes1 indexes2];
        
        % Gera a lista de índices a serem utilizados para teste.
        test_id1 = temp_indexes(class_size*(fold-1) + 1:fold*class_size); % índices da classe 1
        test_id2 = temp_indexes(class_size*(fold-1) + middle+1:middle+fold*class_size); % índices da classe 2
        test_indexes = [test_id1 test_id2];
        % Separa o dataset de teste
        test = dataset(test_indexes, :);
        
        % Remove os índices utilizados para o dataset de teste dos índices
        % temporários, restando apenas os índices que não foram utilizados
        % no teste
        temp_indexes([class_size*(fold-1) + 1:fold*class_size class_size*(fold-1) + middle+1:middle+fold*class_size]) = [];
        
        % Adiciona a parte do dataset não utilizado para teste em um
        % dataset de treino
        train = dataset(temp_indexes, :);

        % Classifica o dataset de treino com o knn com os valores de k
        % variando entre 1 e k_max e retorna a acurácia para cada valor de
        % k
        for k=1: max_neighbors
            % O vetor abaixo foi colocado apenas para companhamento
            % da execução do algoritmo pela janela de comando
            [fold k]
            
            cv_scores_knn(fold, k) = knn(train, test, k);    
        end
        
        % Calcula a acurácia obtida com o npc
        cv_scores_npc(fold) = npc(train, test);
    end 
end