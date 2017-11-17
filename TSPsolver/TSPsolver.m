%   X = VRPsolver(DEPOT,CUSTOMERS) 
%   solves the Classic VRP problem based on distance travelled. The
%   algorithm returns the final best solution and each best solution for
%   every generation. Each solution contains the related fitness (or cost
%   of travel), and an ordered list of each node visited by each
%   vehicle(separated into different row arrays).
%   Inputs:
%   DEPOT - row vector with X and Y coordinates. 
%   CUSTOMERS - 2D array with 1st column as X and 2nd as Y coordinates.
%   DEMANDS - 1D array with total customer demands.
%   VEHICLES - total number of vehicles taking part in the problem.
%   VEHICLECAPACITY - total units of cargo that a single vehicle can carry
% 
%   X = VRPsolver(DEPOT,CUSTOMERS,LINK) 
%   solves the Classic VRP problem based given a link cost.
%   LINK - 2D square matrix with travel costs - last entry is depot.

function [best, bestPerGen] = TSPsolver(depot, customer, linkCosts)

params = InitialiseProblem(depot, customer);

CheckForErrors(params);

if nargin < 3
    [best, bestPerGen] = TSPnoLinks(params);
else
    params.links = linkCosts;

    CheckForLinkErrors(params);
    
    [best, bestPerGen] = GeneticAlgorithm(params);
end

end

function [] = CheckForLinkErrors(params)
if (size(params.links,1) ~= params.nodeNum + 1)
    error('Link array not of equal size of total node number');
end
if (size(params.links,1) ~= size(params.links,2))
    error('Link matrix is not a square matrix');
end

end

function [] = CheckForErrors(params)
if (length(params.depot) ~= 2)
    error('Depot is not a row vector of 2 elements');
end
if (size(params.nodes, 2) ~= 2)
    error('Nodes must have only two columns');
end

end

function [best, bestPerGen] = TSPnoLinks(params)
% Inputs: Depot - row vector with X and Y coordinates. 
% Customer - 2D array with 1st column as X and 2nd as Y coordinates.
% Demand - 1D array with total customer demands.
% Vehicles - total number of vehicles taking part in the problem.
% VehicleCapacity - total units of cargo that a single vehicle can carry

params = InitialiseLinks(params);

[best, bestPerGen] = GeneticAlgorithm(params);

end

function [params] = InitialiseProblem(depot, customers)

params.depot = depot;
params.nodes = customers;
params.nodeNum = length(customers);
params.nodeId = 1 : params.nodeNum;
params.allNodes = [customers;depot];
params.allNodesId = 1 : params.nodeNum + 1;
params.colours = ['b','k','r','g','y','c','m'];
params.linkCost = 1;

end

function [params] = InitialiseLinks(params)

linksNum = size(params.depot,1) + params.nodeNum;
links = zeros(linksNum, linksNum);

for i = 1 : linksNum
    for j = 1 : linksNum
        if (i <= params.nodeNum)
            from = params.nodes(i,:);
        else
            from = params.depot;
        end
        if (j <= params.nodeNum)
            to = params.nodes(j,:);
        else
            to = params.depot;
        end
        links(i,j) = GetDistance(from, to);
    end
end
    
params.links = links;
params.linkNum = linksNum;

end

function [dist] = GetDistance(from, to)

dist = sqrt(sum((to - from) .* (to - from)));

end

function [best, bestPerGen] = GeneticAlgorithm(params)
% Main method - calls the rest of the function in the following order:
% generate initial solution, calulate fitness, crossover, mutate, and
% feasibility stage, then repeat until generations have run out.

paramsGA = InitialiseParameters(params);

bestPerGen.sol = cell(paramsGA.generations,1);
bestPerGen.fit = ones(paramsGA.generations,1) * Inf;
bestList.sol = bestPerGen.sol;
bestList.fit = bestPerGen.fit;
best.sol = cell(1,1);
best.fit = Inf;

parentPop = Generation(params, paramsGA);
p = cell(1,1);
q = cell(1,1);

h = figure;
hold on
scatter(params.nodes(:,1), params.nodes(:,2), 'k');
scatter(params.depot(:,1), params.depot(:,2), 'r', 'LineWidth', 2);
xlabel('X (km)');
ylabel('Y (km)');
xlim([0, ceil(max(params.allNodes(:,1)))]);
ylim([0, ceil(max(params.allNodes(:,2)))]);
hold off;

j = figure;
hold on;
title('Fitness Value at ');
xlabel('Generations');
ylabel('Fitness Value');
xlim([0, paramsGA.generations]);
hold off;

for i = 1 : paramsGA.generations
    
    fitness = FitnessCalculation(parentPop, paramsGA, params);
    [best,bestList,bestPerGen] = SaveBestSolution(parentPop, fitness,...
        best, bestList, bestPerGen, i);
    [p,q] = DrawSolution(best,bestList,params, p, q, i, h, j);
    newPop = Selection(parentPop, fitness, paramsGA, params);
    childPop = Crossover(newPop, paramsGA);
    childPop = Mutation(childPop, paramsGA);
    parentPop = childPop;
end

% PrintSolution(best);

close(h);

end

function [] = PrintSolution(best)

sol = best.sol{1,1};

fprintf('The best solution found is: \n')
   
fprintf(' Vehicle Trajectory: ');
text = '';
for i = 1 : length(sol)
    if (i == 1)
        text = strcat(text, 'depot-');
    end
    text = strcat(text, num2str(sol(i)), '-');
    if (i == length(sol))
        text = strcat(text, 'depot');
    end
end
fprintf(text);
fprintf('\n');

end

function [plot1,plot2] = DrawSolution(best, bests ,params, plot1, plot2,...
    gen,fig1, fig2)
% draws solutions in two graphs. Do not close any of the graphs or else
% the program stops!
set(0,'CurrentFigure',fig1)
hold on;
route = best.sol{1};
title(strcat('Generation = ', num2str(gen)));

route = [params.allNodesId(end),route,params.allNodesId(end)];
X = params.allNodes(route,1);
Y = params.allNodes(route,2);
if ~isempty(plot1{1})
    delete(plot1{1});
end
plot1{1} = plot(X,Y,params.colours(1)); 

hold off

set(0,'CurrentFigure',fig2)
hold on
if ~isempty(plot2{1,1})
    delete(plot2{1,1});
end
plot2{1,1} = plot(1:gen,bests.fit(1:gen),'r', 'LineWidth', 2); 
fitness = bests.fit(gen);
title(strcat('Fitness Value = ', num2str(fitness)));
hold off
drawnow;

end

function [p] = InitialiseParameters(params)
% initialises parameters for the genetic algorithm to function. Uses a
% struct mechanism, in which all variables are stored into a main struct.

p.population = params.nodeNum * 5;                  % size of solution popualtion
p.chromNum = ceil(params.nodeNum);                  % max size of inidividual solution
p.generations = 3000;                               % number of generations

p.childNum = ceil(p.population * 6 / 10);           % number of children produced directly 
                                                    % from parents
p.crossoverProb = 0.7;                              % crossover probability
p.mutationProb = 0.4;                               % mutation probability
p.nearestNeighProb = 0.5;                           % proability to undergo nearest neighbour

end

function [pop] = Generation(params, paramsGA)
% generates randomised population of solution - requires population size
% (popSize) and chromosome size (chromSize) as inputs. The method assumes 
% the chromosome is 1 dimensional.

popNum = paramsGA.population;
pop = cell(popNum, 1);

for i = 1 : popNum
    pop{i} = GenerateChromosome(params);
end

end

function [gene] = GenerateChromosome(params)

gene = shuffle(params.nodeId);

end

function [v] = shuffle(v)

v = v(randperm(length(v)));

end

function [fit] = FitnessCalculation(pop, paramsGA, params)
% calculates the "goodness" of the solution. In this case, a value is added
% if two consecutive numbers are presented. Inputs are the population
% analaysed (pop) and the GA parameters (params).
fit = zeros(paramsGA.population, 1);
for i = 1 : paramsGA.population
    fit(i) = ObjectiveFunction(pop{i}, params);
end

end

function [fit] = ObjectiveFunction(gene, params)

fit = 0;

visitsNum = length(gene);
for i = 1 : visitsNum
    
    if i == 1
        fit = fit + params.links(end, gene(i)) * params.linkCost;
    end
    if i == visitsNum
        fit = fit + params.links(gene(i), end) * params.linkCost;
        continue;
    end
    fit = fit + params.links(gene(i), gene(i + 1)) * params.linkCost;
end

end

function [best, bests, bestGen] = SaveBestSolution(pop, fit, best,bests, bestGen, gen)
% extracts solution of higher fitness from the population. Inputs are the
% population analised (pop), the fitness of the population (fit), the
% current best solution (best), the best solution per generation (bestGen)
% and the current generation (gen).
popSize = length(pop);
fitness = Inf;
for i = 1 : popSize
    if (fitness >= fit(i))
        fitness = fit(i);
        bestInGen = pop(i);
    end
end

bestGen.sol(gen) = bestInGen;
bestGen.fit(gen) = fitness;
if (best.fit >= fitness)
    best.fit = fitness;
    best.sol = bestInGen;
end

bests.sol(gen) = best.sol;
bests.fit(gen) = best.fit;

end

function [newPop] = Selection(pop, fit, p, params)
% extracts solutions to be carried over for manipulation. Uses a roullete
% method approach. Calculates cumulative fitness of the solutions, then a
% random number is assigned and the solution that corresponds to this
% value is extracted. Higher fitness solutions have higher probability of
% being extracted. Inputs required are the population (pop) and the fitness
% (fit).

lastFit = 0;
minFit = min(fit);
fit = fit - minFit;
maxFit = max(fit);
fit = maxFit - fit;
newPop = pop;
totFit = sum(fit);
cumulativeFit = zeros(length(pop),1);

for i = 1 : p.population
    cumulativeFit(i) = fit(i)/totFit + lastFit;
    lastFit = cumulativeFit(i);
end

for i = 1 : p.childNum
    diceRoll = rand;
    for j = 1 : p.population
        if (diceRoll < cumulativeFit(j))
            newPop(i) = pop(j);
            break;
        end
    end
end

for i = p.childNum : p.population
    newPop{i} = GenerateChromosome(params);
end
end

function [pop] = Crossover(pop, paramsGA)
% selects another population at random and carries out a uniform crossover.
% Each gene has a 50% probability of coming from the mother or the father.
% Two children are created, each having opposite genes (ie. if one has a
% gene from mother, the other has a gene from father). Inputs are
% parent population (pop) and genetic parameters (params).

for i = 1 : paramsGA.population
    if (rand > paramsGA.crossoverProb)
        continue;
    end
    
    father = pop{i};
    
    indexMother = i;
    while indexMother == i
        indexMother = randi(paramsGA.population);
    end
    mother = pop{indexMother};
          
    [dadRange, momRange, dadId, momId] = CrossoverRange(father,...
        mother);
    
    [son, daug] = SwapGenes(father, mother, dadRange,momRange,...
        dadId, momId);
        
    pop{i} = son;
    pop{indexMother} = daug;
end
 
end

function [dadRange, momRange, dadLower, momLower] = CrossoverRange(f, m)

[dadUpper, dadLower, momUpper, momLower] = CrossoverBounds(f, m);

dadRange = f(dadLower : dadUpper);
momRange = m(momLower : momUpper);

end

function [fUpBound, fLowBound, mUpBound, mLowBound] = CrossoverBounds(f, m)

[fUpBound, fLowBound] = GetBounds(f);
[mUpBound, mLowBound] = GetBounds(m);

end

function [upBound, lowBound] = GetBounds(array)

bounds = randperm(length(array));
if (length(bounds) == 1)
    upBound = bounds;
    lowBound = bounds;
else
    if bounds(1) > bounds(2)
        upBound = bounds(1);
        lowBound = bounds(2);
    else
        lowBound = bounds(1);
        upBound = bounds(2);
    end
end

end

function [s, d] = SwapGenes(f, m, fRange, mRange, fId, mId)

[s, fId] = DeleteGenes(f, mRange, fId);
[d, mId] = DeleteGenes(m, fRange, mId);

s = InsertGenes(s, mRange, fId);
d = InsertGenes(d, fRange, mId);

end

function [array] = InsertGenes(array, range, point)

bef = randi(2) - 1;

if (isempty(range))
    return
end

if (isempty(array))
    array = range;
else
    if (point >= length(array))
        array = [array(1:end),range];
    else
        array = [array(1:point - bef),range,array(point+1 - bef:end)];
    end
end

end

function [array, point] = DeleteGenes(array, range, point)

index = 1;
while index <= length(array)
    if any(array(index) == range)
        array(index) = [];
        if (index < point)
            point = point - 1;
        end
    else
        index = index + 1;
    end
end

end

function [pop] = Mutation(pop, paramsGA)
% carries out uniform mutation. A random number of genes and random values
% are generated and the population changed. Inputs are population solution
% (pop) and GA parameters (params).
for i = 1 : paramsGA.population
    if (rand <= paramsGA.mutationProb)
        diceRoll = rand;
        if (diceRoll < 0.4)
            [pop{i}] = IntraSwap(pop{i});
        elseif (diceRoll >= 0.4 && diceRoll < 0.7)
            [pop{i}] = IntraShuffle(pop{i});
        else
            [pop{i}] = IntraReverse(pop{i});        
        end
    end
end
    
end

function [chrom] = IntraSwap(chrom)

bounds = randperm(length(chrom));
chrom([bounds(1),bounds(2)]) = chrom([bounds(2),bounds(1)]);

end

function [chrom] = IntraShuffle(chrom)

visitsNum = length(chrom);
change = randperm(visitsNum);
if (change(1) > change(2))
    upBound = change(1);
    lowBound  = change(2);
else
    lowBound = change(1);
    upBound  = change(2);
end

newroute = chrom(lowBound : upBound);

newroute = shuffle(newroute);

chrom(lowBound : upBound) = newroute;

end

function [chrom] = IntraReverse(chrom)

route = randperm(length(chrom));

if (route(1) > route(2))
    upBound = route(1);
    lowBound = route(2);
else
    upBound = route(2);
    lowBound = route(1);
end

chrom(lowBound : upBound) = chrom(upBound : -1 : lowBound);

end