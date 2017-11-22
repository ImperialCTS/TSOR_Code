%% cplexCallComplexExample
% Matlab file that can interact with a CPLEX OPL model. The file takes 50
% seconds to run. We run the model iteratively, editing the data files to get 
% different results from the same model file. We delete a single link in 
% every iteration. Finally we save the results and identify the most 
% important link in the network based on the added cost with respect to the
% original problem instance.

clear all;
close all;
clc;

%% Add path of Cplex model files
% Two methods to achieve this:
% 1. Locate the file and add the path to matlab using
% addpath("filelocation").
% 2. (Easier) Move cplex files (.mod, and .dat files) to the folder
% containing the matlab script calling cplex.

links = xlsread('Data.xlsx',1); %read first sheet in file

% Create command that starts with "oplrun" (this will execute cplex)
% followed by the model file (*.mod) and the data files associated (*.dat).
% This is done by the cplex_call command in this file.
command = createCplexCommand('mcf.mod','mcf.dat','links.dat');

% We call cplex - this creates a tuple data file automatically for the
% links
cost = cplex_call(command, '', 'Costs.csv', 'Links', links);
% the model files creates a csv file with the results in the same folder as
% it is located in. We now extract the solution using matlab.

%% Results
% read the csv file
cargo = xlsread('Links.csv');
nodes = xlsread('Nodes.csv');

%% Iterative cplex calls
% now we eliminate destroy one link for every iteration and call cplex
% again. This way we analyse the potential economical consequences of any
% disruption in the links.

numLinks = size(links, 1);
newcargo = cell(numLinks,1);
newnodes = cell(numLinks,1);
newcosts = zeros(numLinks,1);
mainLinks = links;

% creates a copy of the links data file
copyfile('links.dat','mainLinks.dat','f');

for i = 1 : numLinks
    % command line clear
    clc;
    
    %% METHOD 1: create a new .dat file to and call cplex only using the
    % command.
    % copies the data file except for the link specified by "usedLinks"
    CopyFileButOneLine('mainLinks.dat','links.dat', i)
    % calls cplex
    newcosts(i) = cplex_call(command, '', 'Costs.csv');
    
    %% METHOD 2: update data in matlab and call cplex with new data
% %     UNCOMMENT THIS SECTION TO RUN
%     links = mainLinks;
%     links(i, :) = [];
%     newcosts(i) = cplex_call(command, '', 'Costs.csv', 'Links', links);
% %  

    % reads the shortest paths found
    results = xlsread('Nodes.csv');
    newcargo{i} = xlsread('Links.csv');
    newnodes{i} = xlsread('Nodes.csv');
end

%% Output variables
[maxCost, maxCostId] = max(newcosts);

fprintf(['The most vulnerable link is %g, which incurs an additional',...
    'cost of %g, resulting in a total of %g units, from the original',...
    ' %g \n'],maxCostId, maxCost - cost, maxCost, cost);
fprintf('A total of %g units of cargo are left undelivered \n',...
    sum(newnodes{maxCostId})/2);
    
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

function [] = CopyFileButOneLine(fromFile, toFile, line)
% copies FROMFILE to TOFILE except for row "LINE". FROMFILE and TOFILE must
% be strings, and line is an integer
if (~ischar(fromFile))
    error('FROMFILE must be a string');
end
if (~ischar(toFile))
    error('TOFILE must be a string');
end
if (~isnumeric(line))
    error('LINE must be a numeric value');
end
if (length(line) > 1)
    error('Only a single line is allowed in this function');
end

fold = fopen(fromFile,'r');
fnew = fopen(toFile,'w');

% copy all lines before the link we dont want to copy
for j = 1 : line
    writeline = fgets(fold);
    if ~ischar(writeline)
        break % if no string, possibly end of file
    end
    fwrite(fnew,writeline); % check if to add to a string?
end

writeline = fgets(fold); % get new line but don't do anything

% copy rest of the lines after 'line'
while ~isempty(writeline) && ischar(writeline)
    writeline = fgets(fold);
    if ischar(writeline)   % checks if we have arrived to end of file
        fwrite(fnew, writeline);
    end
end

% MUST CLOSE FILES, otherwise they may become corrupted
fclose(fnew);
fclose(fold);

end