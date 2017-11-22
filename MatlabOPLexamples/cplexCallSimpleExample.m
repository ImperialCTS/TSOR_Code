%% cplexCallSimpleExample
% Matlab file that can interact with a CPLEX OPL model. The file takes 11
% seconds to run. Data is obtained from an excel spreasheet, manipulated 
% using matlab, and used to generate custom data files. CPLEX is then called 
% using the cplex_call function. The cplex model includes a postprocessing script
% that generates a csv file, which is then used by matlab to read the results.  

clear all;
close all;
clc;

%% Load data
links = xlsread('Data.xlsx',1); %read sheet in file

%% Call Cplex
% Data and Model files must be located in the matlab path.
% Move cplex files (.mod, and .dat files) to the folder
% containing the matlab script calling cplex.

% Create command that starts with "oplrun" (this will execute cplex)
% followed by the model file (*.mod) and the data files associated (*.dat)
command = createCplexCommand('mcf.mod','mcf.dat','links.dat');

% We call cplex - this creates a tuple data file automatically
cost = cplex_call(command, '', 'Costs.csv', 'Links', links);
% the cplex model creates a csv file with the results in the same folder as
% it is located in. We now extract the solution using matlab.

%% Results
% read the csv file
cargo = xlsread('Links.csv');
nodes = xlsread('Nodes.csv');
clc; % clean command line

count = 0;

% finally, we can display the solution into the matlab console
for i = 1 : length(cargo)
   fprintf('Cargo moved in link %g ', i);
   fprintf('equates to %g\n', cargo(i));
end

for i = 1 : length(nodes)
   
    if (nodes(i) > 0)
        count = count + 1;
        fprintf('Node %g has not delivered/recieved %g cargo!', i, nodes(i));
        fprintf('\n');
    end
end

if count == 0
    fprintf('All cargo has been delivered! \n');
end
%% Functions

function [command] = createCplexCommand(modelfile, varargin)
%cplex_call calls cplex given a modelfile and datafiles. MODELFILE is a
%string that must end in .mod (i.e. must be a cplex model file) and
%VARGARING accepts multiple strings, all strings must finish in .dat.

% Exception checks
if (length(modelfile) > 4)
    if (~strcmp(modelfile(end-3:end),'.mod'))
        error('Modelfile must be a *.mod file');
    end
else
    error('Modelfile must be a *.mod file');
end

if (size(varargin,1) > 1 && size(varargin, 2) > 1)
    error('Must be provided data files as row vector or separately');
end

for i = 1 : length(varargin)
    if (length(varargin{i}) > 4)
        if (~strcmp(varargin{i}(end-3:end),'.dat'))
            error('Data files must use a *.dat format');
        end
    else
        error('Data files must use a *.dat format');
    end
end

% combine strings to create command
command = ['oplrun',' ',modelfile];

for i = 1 : length(varargin)
    command = [command, ' ', varargin{i}];
end

end