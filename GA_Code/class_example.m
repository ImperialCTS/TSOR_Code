%% GENETIC ALGORITHM SCRIPT
% Authors:  Jose Escribano, Panagiotis Angeloudis
% Date:     27/11/2017
%
% A simple Genetic Algorithm to demonstrate the various stages of the search
% process genetic. 

% Clear previous results and any open windows
clear;close all;clc;

% Initialise GA parameters
params = InitialiseParameters(); 

% Initialise structures that will contain info on best solutions
history.sol = cell(params.populationsize,1);
history.fit = zeros(params.populationsize,1);
best.sol = cell(1,1);
best.fit = 0;

% Initialise GA parameters
plotinfo = PlotInitialise(params);

% generate initial population
population = GeneratePopulation(params);

% repeat for all specified
for i = 1 : params.generations
    
    % Fitness Assesment 
    population_fitness = FitnessMeasure(population, params);     
    [best, history] = SaveBestSolution(population, population_fitness, best, history, i);           
    
    % Visualisation
    plotinfo = DrawSolution(history, plotinfo, i);
    
    % Genetic Operators
    population = Selection(population, population_fitness, params);
    population = Crossover(population, params);
    population = Mutation(population, params);
end

% Console Outputs
best_x1 = best.sol{1}(1);
best_x2 = best.sol{1}(2);
best_x3 = best.sol{1}(3);
best_x4 = best.sol{1}(4);

best_fitness = best.fit;

fprintf('The optimum value is %4.2f \n',best_fitness);
fprintf('x1 = %4.5f \n',best_x1);
fprintf('x2 = %4.5f \n',best_x2);
fprintf('x3 = %4.5f \n',best_x3);
fprintf('x4 = %4.5f \n',best_x4);

%end

%% ---------- STEP 0: INITIALISATION ----------------

function [params] = InitialiseParameters()
% initialises parameters for the genetic algorithm to function. Uses a
% struct mechanism, in which all variables are stored into a main struct.

% The upper bound and lower bound values in this example apply to all
% variables that are encoded in the gene.
params.upBound = 0;                         % upper bound for number generator
params.lowBound = 100;                      % lower bound for number generator
params.genes = 4;                           % size of inidividual solution

params.populationsize = 50;                         % size of solution population
params.elitesize = round(params.populationsize/3);  % elite population size  
params.generations = 1000;                          % number of generations
params.crossoverProb = 0.5;                         % crossover probability
params.mutationProb = 0.2;                          % mutation probability

end

%% ---------- STEP 1: GENERATION ----------------

function [population] = GeneratePopulation(params)
% This function generates a randomised population of chromosomes.
% Requires a defined population size, and assumes that the chromosome is 
% a one-dimensional string of real values

population = cell(params.populationsize, 1);

for i = 1 : params.populationsize
    population{i} = GenerateChromosome(params);   
end

end

function [chromosome] = GenerateChromosome(params)
% This function generates a single random chromosome.
% Important to ensure that the upper and lower bound values have been 
% defined correctly. 

chromosome = params.lowBound + rand(params.genes,1) * (params.upBound - params.lowBound);

end

%% ---------- STEP 2: EVALUATION ----------------

function [population_fitness] = FitnessMeasure(population, params)
% Calculates the fitness of all chromosomes in the population.
% In this example, we also check whether the constraints are met.  

population_fitness = zeros(params.populationsize,1);
for i = 1 : params.populationsize
    
    chromosome = population{i};
    
    % for legibility, we copy the values of the decision variables into
    % distinct variables that we can use with the objective and constraint
    % formulas
    
    x1 = chromosome(1);
    x2 = chromosome(2);
    x3 = chromosome(3);
    x4 = chromosome(4);

    fitness = 4 * x1^5 + 2 * x2 * log(x3) + x4;
    
    % we now check whether the constraints are met. if a constraint is
    % found to be violated, we assign a value of zero to the fitness
    
    if ~(x1 + x2 <= 40)
        fitness = 0;
        
    end 
    
    if ~(x3 + x4 <= 30)
       fitness = 0;
    end 
   
    population_fitness(i) = fitness;
    
end

end

function [best, history] = SaveBestSolution(population, population_fitness, best, history, gen)
% extracts solution of higher fitness from the population. Inputs are the
% population analised (pop), the fitness of the population (fit), the
% current best solution (best), the best solution per generation (bestGen)
% and the current generation (gen).

% Find solution with lowest fitness in the generation
popSize = length(population);
fitness_max = -Inf;

for i = 1 : popSize
    if (fitness_max <= population_fitness(i))
        fitness_max = population_fitness(i);
        bestInGen = population(i);
    end
end

% Replaces best solution if generation's best performes better
history.sol(gen) = bestInGen;
history.fit(gen) = fitness_max;

if (best.fit <= fitness_max)
    best.fit = fitness_max;
    best.sol = bestInGen;
end

end

%% ---------- STEP 3: SELECTION ----------------

function [new_population] = Selection(population, pop_fitnesses, params)
% This function selects the chromosomes to be carried over to the next 
% generation of the search, using the elitist approach. 
% It requires a degree of elitism to have been defined in the parameters


%produce a vector with the rankings of the fitness values in the population
[~, sort_sequence] = sort(pop_fitnesses, 'descend');

%create a new, sorted population, based on the sequence obtained above
new_population = population(sort_sequence,:);

%overwrite any popupulation member that does not belong to the elite class 
% with a new random chromosome.
for i = params.elitesize : params.populationsize
    new_population{i} = GenerateChromosome(params);
end

end

%% ---------- STEP 4: CROSSOVER ----------------


function [population] = Crossover(population, params)
% For every member of the non-elite class, select a random chromosome to 
% pair with. 
% Two offspring chromosomes are created, each having opposite genes 
% (ie. if one has the first parent, the other will have a gene from the 
% second parent)  
% The crossover operator in this example will manipulate every single gene
% in the chromosomes.

for i = params.elitesize : params.populationsize
    if (rand > params.crossoverProb)
        continue;
    end
    
%   The first parent is the current chromosome that we are examining
    parent_1 = population{i};
    
%   The second parent could be any other chromosome in the population  
    parent_2_index = i;
    while parent_2_index == i
        parent_2_index = randi(params.populationsize);
    end
    parent_2 = population{parent_2_index};
    
%   We initialise the offspring chromosomes, which initially are identical 
%   to their parents.

    offspring_1 = parent_1;
    offspring_2 = parent_2;
    
%   There is a 50% probability that each gene in the offspring will be 
%   obtained from a certain parent.  

    for j = 1 : params.genes
        
        if (rand < 0.5)
            continue;
        end
        
        offspring_1(j) = parent_2(j);
        offspring_2(j) = offspring_1(j);
    end
    
%   The offspring REPLACE their parents in the population

    population{i} = offspring_1;
    population{parent_2_index} = offspring_2;
end
 
end

%% ---------- STEP 5: MUTATION ----------------

function [population] = Mutation(population, params)
% Carries out uniform mutation. A new random chromosome is created and 
% replaces a random number of genes in the current chromosome. 

for i = params.elitesize : params.populationsize
    if (rand > params.mutationProb)
        continue;
    end

    chromosome    = population{i};
    chromo_random = GenerateChromosome(params);
    
    % select a gene to mutate
    mutation_point =  randi(params.genes);          
    chromosome(mutation_point) = chromo_random(mutation_point);

    % release the mutant to the population
    population{i} = chromosome;
end

end

%% ---------- VISUALISATION  ----------------
function [plotinfo] = DrawSolution(history, plotinfo, generation_index)

% Updates figure with GA progress. As inputs it requires the struct of all
% solutions (bests), the plot (p), and the generation number (gen).
if ~isempty(plotinfo{1,1})
    delete(plotinfo{1,1})
end

hold on
plotinfo{1,1} = plot(1:generation_index,history.fit(1:generation_index),'r', 'LineWidth', 2); 
fitness = round(history.fit(generation_index));
title(strcat('Fitness Value = ', num2str(fitness)));
hold off
drawnow;

end

function [plotinfo] = PlotInitialise(params)
%Initialise plot
plotinfo = cell(1,1);
figure;
hold on;
xlabel('Generations');
ylabel('Fitness Value');
xlim([0, params.generations]);
hold off;
end

