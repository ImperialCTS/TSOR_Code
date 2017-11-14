%   X = VRPsolver(DEPOT,CUSTOMERS,DEMANDS,VEHICLES,VEHICLECAPACITY) 
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
%   X = VRPsolver(DEPOT,CUSTOMERS,DEMANDS,VEHICLES,VEHICLECAPACITY,LINK) 
%   solves the Classic VRP problem based given a link cost.
%   LINK - 2D square matrix with travel costs - last entry is depot.

function [best, bestPerGen] = VRPsolver(depot, customer, demands, vehicles,...
vehicleCapacity, linkCosts)

params = InitialiseProblem(depot, customer, demands, vehicles, ...
    vehicleCapacity);

CheckForErrors(params);

if nargin < 6
    [best, bestPerGen] = VRPnoLinks(params);
else
    params.links = linkCosts;

    CheckForLinkErrors(params);
    
    [best, bestPerGen] = GeneticAlgorithm(params);
end
% PrintResults(best, bestPerGen)

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
if (sum(params.demand) / params.vehCap > params.veh)
    error('Total Vehicles not sufficient to satisfy all demand');
end
if (length(params.depot) ~= 2)
    error('Depot is not a row vector of 2 elements');
end
if (size(params.nodes, 2) ~= 2)
    error('Nodes must have only two columns');
end
if (length(params.demand) ~= params.nodeNum)
    error('Demand array not of same size to nodes');
end
if (length(params.veh) ~= 1)
    error('Vehicle parameter must be only a single value');
end
if (length(params.vehCap) ~= 1)
    error('Vehicle capacity parameter must be only a single value');
end

end

function [best, bestPerGen] = VRPnoLinks(params)
% Inputs: Depot - row vector with X and Y coordinates. 
% Customer - 2D array with 1st column as X and 2nd as Y coordinates.
% Demand - 1D array with total customer demands.
% Vehicles - total number of vehicles taking part in the problem.
% VehicleCapacity - total units of cargo that a single vehicle can carry

params = InitialiseLinks(params);

[best, bestPerGen] = GeneticAlgorithm(params);

% PrintResults(results) TODO

end

function [params] = InitialiseProblem(depot, customers, demands, ...
    vehicles, vehicleCapacity)

params.depot = depot;
params.nodes = customers;
params.nodeNum = length(customers);
params.nodeId = 1 : params.nodeNum;
params.demand = demands;
params.veh = vehicles;
params.vehCap = vehicleCapacity;
params.penalty = 10000;
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
p = cell(params.veh, 1);
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
    childPop = Mutation(childPop, paramsGA, params);
    parentPop = childPop;
end

PrintSolution(best);

close(h);

end

function [] = PrintSolution(best)

sol = best.sol{1,1};

fprintf('The best solution found is: \n')

for i = 1 : length(sol)
    
    fprintf(' Vehicle %g: ', i);
    text = '';
    for j = 1 : length(sol{i})
        if (j == 1)
            text = strcat(text, 'depot-');
        end
        text = strcat(text, num2str(sol{i}(j)), '-');
        if (j == length(sol{i}))
            text = strcat(text, 'depot');
        end
    end
    fprintf(text);
    fprintf('\n');
end

end

function [plot1,plot2] = DrawSolution(best, bests ,params, plot1, plot2,...
    gen,fig1, fig2)
% draws solutions in two graphs. Do not close any of the graphs or else
% the program stops!
set(0,'CurrentFigure',fig1)
hold on;
route = best.sol{1};
coloursNum = length(params.colours);
title(strcat('Generation = ', num2str(gen)));

for i = 1:length(route)
    if isempty(route{i})
        continue;
    end
    cId = i;
    while cId > coloursNum
        cId = cId - coloursNum;
    end
    route{i} = [params.allNodesId(end),route{i},params.allNodesId(end)];
    X = params.allNodes(route{i},1);
    Y = params.allNodes(route{i},2);
    if ~isempty(plot1{i})
        delete(plot1{i});
    end
    plot1{i} = plot(X,Y,params.colours(cId)); 
end
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

p.population = params.nodeNum * 4;                  % size of solution popualtion
p.chromNum = ceil(params.nodeNum/(params.veh/2));   % max size of inidividual solution
p.generations = 500;                               % number of generations

p.childNum = ceil(p.population * 6 / 10);           % number of children produced directly 
                                                    % from parents
p.crossoverProb = 0.7;                              % crossover probability
p.mutationProb = 0.5;                               % mutation probability
p.nearestNeighProb = 0.5;                           % proability to undergo nearest neighbour

end

function [gene] = GenerateChromosome(params)

nodes = shuffle(params.nodeId);
vehicles = randi(params.veh, 1, params.nodeNum);

gene = cell(params.veh, 1);
for i = 1 : params.veh
   gene{i} = nodes(vehicles == i);
end

end

function [v] = shuffle(v)

    v = v(randperm(length(v)));
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
currentCap = ones(params.veh, 1) * params.vehCap;

for i = 1 : params.veh
    visitsNum = length(gene{i});
    if (visitsNum == 0)
        continue;
    end
    for j = 1 : visitsNum
        currentCap(i) = currentCap(i) - params.demand(gene{i}(j));
        if j == 1
            fit = fit + params.links(end, gene{i}(j)) * params.linkCost;
        end
        if j == visitsNum
            fit = fit + params.links(gene{i}(j), end) * params.linkCost;
            continue;
        end
        fit = fit + params.links(gene{i}(j), gene{i}(j + 1)) * params.linkCost;
    end
end

currentCap(currentCap > 0) = 0;

fit = fit + sum(currentCap) * -params.penalty;

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
    vehicles = length(pop{i});
    
    indexMother = i;
    while indexMother == i
        indexMother = randi(paramsGA.population);
    end
    mother = pop{indexMother};
       
    dadVeh = randi(vehicles);
    momVeh = randi(vehicles);
    
    [dadRange, momRange, dadId, momId] = CrossoverRange(father{dadVeh},...
        mother{momVeh});
    
    [son, daug] = SwapGenes(father, mother, dadVeh, momVeh, dadRange,...
        momRange, dadId, momId);
        
    pop{i} = son;
    pop{indexMother} = daug;
end
 
end

function [s, d] = SwapGenes(f, m, fVeh, mVeh, fRange, mRange, fId, mId)

[s, fId] = DeleteGenes(f, fVeh, mRange, fId);
[d, mId] = DeleteGenes(m, mVeh, fRange, mId);

[s{fVeh}] = InsertGenes(s{fVeh}, mRange, fId);
[d{mVeh}] = InsertGenes(d{mVeh}, fRange, mId);

end

function [array, point] = InsertGenes(array, range, point)

bef = randi(2) - 1;

if (isempty(range))
    
else
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

end

function [array, point] = DeleteGenes(array, row, range, point)

for i = 1 : length(array)
    if (isempty(array{i}))
        continue;
    end
    index = 1;
    while index <= length(array{i})
        if any(array{i}(index) == range)
            array{i}(index) = [];
            if (i == row)
                if (index < point)
                    point = point - 1;
                end
            end
        else
            index = index + 1;
        end
    end
end

end

function [dadRange, momRange, dadLower, momLower] = CrossoverRange(f, m)

[dadUpper, dadLower, momUpper, momLower] = CrossoverBounds(f, m);

if dadUpper == 0
    dadRange = [];
else
    dadRange = f(dadLower : dadUpper);
end
if momUpper == 0
    momRange = [];
else
    momRange = m(momLower : momUpper);
end

end

function [fUpBound, fLowBound, mUpBound, mLowBound] = CrossoverBounds(f, m)

    if (isempty(f))
        fUpBound = 0;
        fLowBound = 0;
    else
        bounds = randperm(length(f));
        if (length(bounds) == 1)
            fUpBound = bounds;
            fLowBound = bounds;
        else
            if bounds(1) > bounds(2)
                fUpBound = bounds(1);
                fLowBound = bounds(2);
            else
                fLowBound = bounds(1);
                fUpBound = bounds(2);
            end
        end
    end
    if (isempty(m))
        mUpBound = 0;
        mLowBound = 0;
    else
        bounds = randperm(length(m));
        if (length(bounds) == 1)
            mUpBound = bounds;
            mLowBound = bounds;
        else
            if bounds(1) > bounds(2)
                mUpBound = bounds(1);
                mLowBound = bounds(2);
            else
                mLowBound = bounds(1);
                mUpBound = bounds(2);
            end
        end
    end
end

function [pop] = Mutation(pop, paramsGA, params)
% carries out uniform mutation. A random number of genes and random values
% are generated and the population changed. Inputs are population solution
% (pop) and GA parameters (params).
for i = 1 : paramsGA.population
    if (rand <= paramsGA.mutationProb)
        diceRoll = rand;
        if (diceRoll < 0.2)
            [pop{i}] = IntraSwap(pop{i});
        elseif (diceRoll >= 0.2 && diceRoll < 0.4)
            [pop{i}] = IntraShuffle(pop{i});
        elseif (diceRoll >= 0.4 && diceRoll < 0.6)
            [pop{i}] = IntraReverse(pop{i});        
        elseif (diceRoll >= 0.6 && diceRoll < 0.8)
            [pop{i}] = InterSwap(pop{i});
        else
            [pop{i}] = InterBest(pop{i}, params);
        end
    end
    
    if (rand <= paramsGA.nearestNeighProb)
        [pop{i}] = IntraNearestNeigh(pop{i}, params);
    end
end
    
end

function [chrom] = IntraSwap(chrom)

veh = randi(length(chrom));

if (~isempty(chrom{veh}) && length(chrom{veh}) > 1)

    bounds = randperm(length(chrom{veh}));
    chrom{veh}([bounds(1),bounds(2)]) = chrom{veh}([bounds(2),bounds(1)]);
end

end

function [chrom] = IntraShuffle(chrom)

veh = randi(length(chrom));

if (~isempty(chrom{veh}) && length(chrom{veh}) > 1)

    visitsNum = length(chrom{veh});
    change = randperm(visitsNum);
    if (change(1) > change(2))
        upBound = change(1);
        lowBound  = change(2);
    else
        lowBound = change(1);
        upBound  = change(2);
    end
    
    newroute = chrom{veh}(lowBound : upBound);
    
    newroute = shuffle(newroute);
    
    chrom{veh}(lowBound : upBound) = newroute;
end

end

function [chrom] = IntraReverse(chrom)

veh = randi(length(chrom));

if (~isempty(chrom{veh})&&length(chrom{veh}) > 1)
    route = randperm(length(chrom{veh}));
    
    if (route(1) > route(2))
        upBound = route(1);
        lowBound = route(2);
    else
        upBound = route(2);
        lowBound = route(1);
    end
    
    chrom{veh}(lowBound : upBound) = chrom{veh}(upBound : -1 : lowBound);
end

end

function [chrom] = InterSwap(chrom)

veh1 = randi(length(chrom));
veh2 = veh1;

while (veh1 == veh2)
   veh2 =  randi(length(chrom));
end

if (~isempty(chrom{veh1}))
    gene1 = randi(length(chrom{veh1}));    
end
if (~isempty(chrom{veh2}))
    gene2 = randi(length(chrom{veh2}));    
end

if (isempty(chrom{veh2}) && ~isempty(chrom{veh1}))
    chrom{veh2} = chrom{veh1}(gene1);
elseif (~isempty(chrom{veh2}) && isempty(chrom{veh1}))
    chrom{veh1} = chrom{veh2}(gene2);
elseif (~isempty(chrom{veh2}) && ~isempty(chrom{veh1}))
    tmp = chrom{veh1}(gene1);
    chrom{veh1}(gene1) = chrom{veh2}(gene2);
    chrom{veh2}(gene2) = tmp;
end

end

function [chrom] = InterBest(chrom, params)

veh1 = randi(length(chrom));
veh2 = veh1;

while (veh1 == veh2)
   veh2 =  randi(length(chrom));
end

if (~isempty(chrom{veh1}))
    visitsNum = length(chrom{veh1});
    change = randi(visitsNum);
    
    if(~isempty(chrom{veh2}))
        visitsNum = length(chrom{veh2});
        dist = zeros(1,visitsNum);
        for j = 1 : visitsNum
            if j == 1
                dist(j) = params.links(end, chrom{veh2}(j));
            else
                dist(j) = params.links(chrom{veh2}(j - 1), chrom{veh2}(j));
            end
            if j == visitsNum
                dist(j) = params.links(chrom{veh2}(j), end);
            else
                dist(j) = params.links(chrom{veh2}(j), chrom{veh2}(j + 1));
            end
        end
        [~, minId] = min(dist);
        tmp = chrom{veh1}(change);
        chrom{veh1}(change) = chrom{veh2}(minId);
        chrom{veh2}(minId) = tmp;
    else
        chrom{veh2} = chrom{veh1}(change);
        chrom{veh1}(change) = [];
    end
end

end

function [chrom] = IntraNearestNeigh(chrom, params)

veh = randi(length(chrom));
route = chrom{veh};
revRoute = fliplr(route);

if (~isempty(chrom{veh}) && length(chrom{veh}) > 1)

    [newRoute,cost] = NearestNeighbourAlgorithm(route,params);
    [newRevRoute,revCost] = NearestNeighbourAlgorithm(revRoute,params);
    
    if (cost > revCost)
        chrom{veh} = newRevRoute;
    else
        chrom{veh} = newRoute;
    end
end

end

function [newRoute, cost] = NearestNeighbourAlgorithm(route, params)
newRoute = zeros(size(route));
cost = 0;
count = 1;
while (count <= length(newRoute))
    
    if (count == 1)
        origin = params.nodeNum + 1;
    else
        origin = newRoute(count - 1);
    end
    
    [minDist, minNode] = min(params.links(origin, route));
    cost = cost + minDist;
    newRoute(count) = route(minNode);
    route(minNode) = [];
    count = count + 1;
end

cost = cost + params.links(newRoute(count - 1),params.nodeNum + 1);

end 