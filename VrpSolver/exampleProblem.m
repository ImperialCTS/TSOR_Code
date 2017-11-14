%% Example Problem to use the Vehicle Routing Problem Solver

% Clean Matlab
clear all;
close all;
clc;

% Load problem parameters
% Problem parameters include: depot location, customer locations (both in X
% Y kilometers), unit demand for the nodes, number of vehicles, and vehicle
% Capacities).
load('exampleProblem.mat');

% Call the VRP solver and provide this information. Use the command 'help
% VRPsolver' to learn the structure of each parameter.
% The function returns the best solution as a struct file.
[best, bestPerGen] = VRPsolver(depot, customers, demand, veh, vehCap);

% accesses the cell inside best.sol
sol = best.sol{1,1};


fprintf('The best solution found is: \n')

% run through each vector in "sol" and prints the node index of each member
% visited by the vehicle.
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

% draw final solution

colours =  ['b','k','r','g','y','c','m'];
coloursNum = length(colours);

figure;
hold on
title('Final Route');
scatter(customers(:,1), customers(:,2), 'k');
scatter(depot(:,1), depot(:,2), 'r', 'LineWidth', 2);
xlabel('X (km)');
ylabel('Y (km)');

% plot each revery route for each vehicle
for i = 1:length(sol)
    if isempty(sol{i})
        continue;
    end
    cId = i;
    while cId > coloursNum
        cId = cId - coloursNum;
    end
   
    %plot
    numVisits = length(sol{i});
    X = zeros(1, numVisits);
    Y = X;
    
    id = 1;
    for j = 1 : numVisits
        if (j == 1)
            X(id) = depot(1);
            Y(id) = depot(2);
            id = id + 1;
        end
        X(id) = customers(sol{i}(j),1);
        Y(id) = customers(sol{i}(j),2);
        id = id + 1;
        if (j == numVisits)
            X(id) = depot(1);
            Y(id) = depot(2);
        end
    end
    plot(X,Y,colours(cId)); 
end
hold off