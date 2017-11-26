%% GENETIC ALGORITHM SCRIPT
% generic genetic algorithm with different stages separated into their own
% methods - stand-alone version

% initialises problem parameter
paramsGA = InitialiseParameters(); 

% initialise variables to save best solutions
bestPerGen.sol = cell(paramsGA.population,1);
bestPerGen.fit = zeros(paramsGA.population,1);
best.sol = cell(1,1);
best.fit = 0;


j = figure;
hold on;
title('Fitness Value at ');
xlabel('Generations');
ylabel('Fitness Value');
xlim([0, paramsGA.generations]);
hold off;

% generate initial population
parentPop = GenerationChromosome(paramsGA);

% repeat for all specified
for i = 1 : paramsGA.generations
    
    fitness = ObjectiveFunction(parentPop, paramsGA);
    
   % DrawSolution(bestPerGen,paramsGA, j);
    
    [best, bestPerGen] = SaveBestSolution(parentPop, fitness, best,bestPerGen, i);
    newPop = Selection(parentPop, fitness, paramsGA.population);
    childPop = Crossover(newPop, paramsGA);
    childPop = Mutation(childPop, paramsGA);
    
    parentPop = childPop;
end

best_x1 = best.sol{1}(1);
best_x2 = best.sol{1}(2);
bestFitness = best.fit;


%end

%%  ------------------- INITIALISATION

function [p] = InitialiseParameters()
% initialises parameters for the genetic algorithm to function. Uses a
% struct mechanism, in which all variables are stored into a main struct.

p.population = 50;                 % size of solution population
p.genes = 2;                        % size of inidividual solution
p.generations = 5000;               % number of generations

p.childNum = p.population/5;          % number of children produced directly 
                                    % from parents
p.upBound = 0;                     % upper bound for number generator
p.lowBound = 1000;                   % lower bound for number generator
p.crossoverProb = 0.7;              % crossover probability
p.mutationProb = 0.2;               % mutation probability

end

%% ------------------- GENERATION

function [pop] = GenerationChromosome(params)
% generates randomised population of solution - requires population size
% (popSize) and chromosome size (chromSize) as inputs. The method assumes 
% the chromosome is 1 dimensional.

popNum = params.population;
pop = cell(popNum, 1);

for i = 1 : popNum
    pop{i} = GenerateGene(params.genes, params);
    
end

end

function [gene] = GenerateGene(chromSize, params)

gene = params.lowBound + rand(chromSize,1) * (params.upBound - params.lowBound);

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

    fit(i) = 2* x1 + 4*x2;
    
    if ~(3*x1+4*x2<=1700)
        fit(i) = -Inf;
    end 
    
    if ~(2*x1+5*x2<=1600)
        fit(i) = -Inf;
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

function [newPop] = Selection(pop, fit, popSize)
% extracts solutions to be carried over for manipulation. Uses a roullete
% method approach. Calculates cumulative fitness of the solutions, then a
% random number is assigned and the solution that corresponds to this
% value is extracted. Higher fitness solutions have higher probability of
% being extracted. Inputs required are the population (pop) and the fitness
% (fit).

lastFit = -Inf;
minFit = min(fit);
fit = fit - minFit; % make the worst solution = 0, for readability
newPop = pop;
totFit = sum(fit);
cumulativeFit = zeros(length(pop),1);

% calculate cumulative fitness values for solutions. i.e. given two
% solutions of fitness 2 and 3 respectively, the resultant cumulative
% fitness will be 2/5 and 5/5.
for i = 1 : popSize
    cumulativeFit(i) = fit(i)/totFit + lastFit;
    lastFit = cumulativeFit(i);
end

% following the previous example, if the random number "diceRoll" is
% smaller than 2/5, the first solution is selected, if greater than 2/5 and
% smaller or equal to 1, then second solution is selected.
for i = 1 : popSize
    diceRoll = rand;
    for j = 1 : popSize
        if (diceRoll <= cumulativeFit(j))
            newPop(i) = pop(j);
            break;
        end
    end
end
end

%%  ------------------- CROSSOVER FUNCTION

function [pop] = Crossover(pop, params)
% selects another population at random and carries out a uniform crossover.
% Each gene has a 50% probability of coming from the mother or the father.
% Two children are created, each having opposite genes (ie. if one has a
% gene from mother, the other has a gene from father). Inputs are
% parent population (pop) and genetic parameters (params).

for i = 1 : params.population
    if (rand > params.crossoverProb)
        continue;
    end
    
%     grab initial solution to manipulate
    father = pop{i};
    
%     find second solutoin to carry out crossover with
    indexMother = i;
    while indexMother == i
        indexMother = randi(params.population);
    end
    mother = pop{indexMother};
    
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
    pop{i} = son;
    pop{indexMother} = daug;
end
 
end

%%  ------------------- MUTATION FUNCTION

function [pop] = Mutation(pop, params)
% carries out uniform mutation. A random number of genes and random values
% are generated and the population changed. Inputs are population solution
% (pop) and GA parameters (params).
for i = 1 : params.population
%     if random value is greater than the probability, skip population
    if (rand > params.mutationProb)
        continue;
    end
    
%     find gene to mutate
    geneMutationNum = randi(params.genes);
    geneToMutate = randi(params.genes, 1, geneMutationNum);
%     generate new value at gene
    geneValues = GenerateGene(geneMutationNum, params);
    
%     substitute gene and update population
    popToMutate = pop{i};
    popToMutate(geneToMutate) = geneValues;
    pop{i} = popToMutate;
end

end

function [plot2] = DrawSolution(bests, gen, fig2)
% draws solutions in two graphs. Do not close any of the graphs or else
% the program stops!


set(0,'CurrentFigure',fig2)
hold on

plot2{1,1} = plot(1:gen,bests.fit(1:gen),'r', 'LineWidth', 2); 
fitness = bests.fit(gen);
title(strcat('Fitness Value = ', num2str(fitness)));
hold off
drawnow;

end
