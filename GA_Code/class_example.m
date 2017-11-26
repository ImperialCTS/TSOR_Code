%% GENETIC ALGORITHM SCRIPT
% generic genetic algorithm with different stages separated into their own
% methods - stand-alone version

clear;

% initialises problem parameter
params = InitialiseParameters(); 

% initialise variables to save best solutions
history.sol = cell(params.population,1);
history.fit = zeros(params.population,1);

best.sol = cell(1,1);
best.fit = 0;

%j = figure;
%hold on;
%title('Fitness Value at ');
%xlabel('Generations');
%ylabel('Fitness Value');
%xlim([0, params.generations]);
%hold off;

% generate initial population
populationCurrent = GeneratePopulation(params);

% repeat for all specified
for i = 1 : params.generations
    
    fitness = ObjectiveFunction(populationCurrent, params);
          
    [best, history] = SaveBestSolution(populationCurrent, fitness, best, history, i);
    
    populationNew = Selection(populationCurrent, fitness, params);
    populationNew = Crossover(populationNew, params);
    populationNew = Mutation(populationNew, params);
    
    populationCurrent = populationNew;
end

best_x1 = best.sol{1}(1);
best_x2 = best.sol{1}(2);
best_x3 = best.sol{1}(3);
best_x4 = best.sol{1}(4);

bestFitness = best.fit;

%end

%%  ------------------- INITIALISATION

function [p] = InitialiseParameters()
% initialises parameters for the genetic algorithm to function. Uses a
% struct mechanism, in which all variables are stored into a main struct.

p.population = 200;                 % size of solution population
p.genes = 4;                       % size of inidividual solution
p.generations = 5000;              % number of generations

p.elitesize = p.population/5;       % number of children produced directly 
                                   % from parents
p.upBound = 0;                     % upper bound for number generator
p.lowBound = 100;                 % lower bound for number generator
p.crossoverProb = 0.7;             % crossover probability
p.mutationProb = 0.2;              % mutation probability

end

%% ------------------- GENERATION

function [pop] = GeneratePopulation(params)
% generates randomised population of solution - requires population size
% (popSize) and chromosome size (chromSize) as inputs. The method assumes 
% the chromosome is 1 dimensional.

popNum = params.population;
pop = cell(popNum, 1);

for i = 1 : popNum
    pop{i} = GenerateChromosome(params.genes, params);
    
end

end

function [gene] = GenerateChromosome(geneCount, params)

gene = params.lowBound + rand(geneCount,1) * (params.upBound - params.lowBound);

end

%%  ------------------- EVALUATION

function [fit] = ObjectiveFunction(pop, params)
% calculates the "goodness" of the solution. In this case, a value is added
% if two consecutive numbers are presented. Inputs are the population
% analaysed (pop) and the GA parameters (params).
fit = zeros(params.population,1);
for i = 1 : params.population
    
    thisChromo = pop{i};
    x1 = thisChromo(1);
    x2 = thisChromo(2);
    x3 = thisChromo(3);
    x4 = thisChromo(4);

    fit(i) = 4*x1^5+2*x2*log(x3)+x4;
    
    if ~(x1+x2<=40)
        fit(i) = 0;
    end 
    
    if ~(x3+x4<=30)
       fit(i) =0;
   end 
    
end

end

function [best, bestGen] = SaveBestSolution(pop, fit, best, bestGen, gen)
% extracts solution of higher fitness from the population. Inputs are the
% population analised (pop), the fitness of the population (fit), the
% current best solution (best), the best solution per generation (bestGen)
% and the current generation (gen).

% Find solution with lowest fitness in the generation
popSize = length(pop);
fitness = -Inf;

for i = 1 : popSize
    if (fitness <= fit(i))
        fitness = fit(i);
        bestInGen = pop(i);
    end
end

% Replaces best solution if generation's best performes better
bestGen.sol(gen) = bestInGen;
bestGen.fit(gen) = fitness;

if (best.fit <= fitness)
    best.fit = fitness;
    best.sol = bestInGen;
end

end

%%  ------------------- SELECTION METHOD

function [new_population] = Selection(population, pop_fitnesses, params)
% extracts solutions to be carried over for manipulation. Uses a roullete
% method approach. Calculates cumulative fitness of the solutions, then a
% random number is assigned and the solution that corresponds to this
% value is extracted. Higher fitness solutions have higher probability of
% being extracted. Inputs required are the population (pop) and the fitness
% (fit).

%sort the population
[sorted_fitness, sort_sequence] = sort(pop_fitnesses, 'descend');
sorted_population = population(sort_sequence,:);

new_population = sorted_population;

for i = 1 : params.elitesize
 new_population(i) = sorted_population(i);

end

for i = params.elitesize : params.population
    new_population{i} = GenerateChromosome(params.genes, params);
end
end

%%  ------------------- CROSSOVER FUNCTION

function [population] = Crossover(population, params)
% selects another chromosome at random and carries out a uniform crossover.
% Each gene has a 50% probability of coming from the mother or the father.
% Two children are created, each having opposite genes (ie. if one has a
% gene from mother, the other has a gene from father). Inputs are
% parent population (pop) and genetic parameters (params).

for i = params.elitesize : params.population
    if (rand > params.crossoverProb)
        continue;
    end
    
%     grab initial solution to manipulate
    father = population{i};
    
%     find second solution to carry out crossover with
    indexMother = i;
    while indexMother == i
        indexMother = randi(params.population);
    end
    mother = population{indexMother};
    
%     create offsprings
    son = father;
    daug = mother;
    
%     at every gene, there is a probability of 50% that the gene comes from
%     mother or the father
    for j = 1 : params.genes
        
%         since "son" is created from "father" and "daughter" is created
%         from "mother", skip step if change is not necessary.
        if (rand < 0.5)
            continue;
        end
%         son gets a gene from mother, and daughter from father
        son(j) = mother(j);
        daug(j) = son(j);
    end
%     update both solutions in main population
    population{i} = son;
    population{indexMother} = daug;
end
 
end

%%  ------------------- MUTATION FUNCTION

function [pop] = Mutation(pop, params)
% carries out uniform mutation. A random number of genes and random values
% are generated and the population changed. Inputs are population solution
% (pop) and GA parameters (params).
for i = params.elitesize : params.population
%     if random value is greater than the probability, skip population
    if (rand > params.mutationProb)
        continue;
    end
    
%     find gene to mutate
    geneMutationNum = randi(params.genes);
    geneToMutate = randi(params.genes, 1, geneMutationNum);
%     generate new value at gene
    geneValues = GenerateChromosome(geneMutationNum, params);
    
%     substitute gene and update population
    popToMutate = pop{i};
    popToMutate(geneToMutate) = geneValues;
    pop{i} = popToMutate;
end

end



function [plot2] = UpdateGraphs(bests, gen, fig2)

set(0,'CurrentFigure',fig2)
hold on

plot2{1,1} = plot(1:gen,bests.fit(1:gen),'r', 'LineWidth', 2); 
fitness = bests.fit(gen);
title(strcat('Fitness Value = ', num2str(fitness)));
hold off
drawnow;

end
