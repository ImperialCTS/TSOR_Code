function fit_value = cplex_call(command,maindataFile,resultFile,varargin)
% cplex_call(COMMAND,MAINDATAFILE,RESULTFILE,VARIABLENAME,VARIABLE) calls
% cplex to solve a linear problem specified by the user. The COMMAND takes 
% a string with the specified command to call cplex. The main data file is 
% a string with the name of the data file of the core information of your 
% model. Note that the order of variable declaration in CPLEX is important.
% All non-tuples will go in this data file by default, so call this data 
% file first in the command. The RESULTFILE is a string that indicate the 
% csv file from which results can be read. VARIABLENAME is a string, and  
% value can be a 2D, 1D or single element array, or cell array. 2-D array 
% data are converted to tuples. Tuples are assigned to a separate data file
% with the variable name. i.e. if linkData was a tuple, it will be 
% allocated to 'links.dat'.
% cplex_call(C,M,R,VARIABLENAME,VARIABLE,VARIABLENAME2,VARIABLE2)
% cplex_call(C,M,R,VNAME1,V1,VNAME2,V2,VNAME3,V3...) 
% Function accepts any number of variables. A name and variable must be
% provided.


setGlobal();
% Using global, these variables can be accessed everywhere in the script

disp(' ');

[variables] = cplex_datprepare(varargin);    % Prepare model parameters
cplex_datwrite(variables, maindataFile);     % Write model parameters to OPL .dat file
cplex_executemodel(command);                 % Call CPLEX and execute model
fit_value = cplex_datreader(resultFile);     % Read CPLEX result file and get fitness value
end

function setGlobal()

global s;
s.newline = '\n';
s.tab = '\t';
s.scolon = ';';
s.comma = ',';
s.leftsqbrac = '[';
s.rightsqbrac = ']';
s.leftcubrac = '{';
s.rightcubrac = '}';
s.greaterthan = '>';
s.lowerthan = '<';

end

function [variables] = cplex_datprepare(variables)
% This fucntion prepares tuple data as a matlab 2-D array.

varlength = length(variables);
if mod(varlength,2) ~= 0
    error('variables must be presented by a name first (string format) and the variable after');
end

squarearrays = [];

for i = 1 : 2 : varlength
    if (size(variables{i+1},1) == size(variables{i+1},2))
        squarearrays(end+1) = i;
    end
end

if (isempty(squarearrays))
    return;
end

for i = 1 : length(squarearrays)
    variables{squarearrays(i)} = SquareArrayToTuple(variables...
        {squarearrays(i)});
end

end

function [linkTable] = SquareArrayToTuple(data)

% OUTPUT 1 - Travel Time Link Table
% 1 = Link ID;
% 2 = From Node ID
% 3 = To Node ID
% 4 = Data

id = 0;

if (iscell(data))
    if size(data{1,1},1) > 1 && size(data{1,1},2) > 1
        error('Data in cell must be in 1D array format!');
    end
    
    for i = 1 : size(data,1)
        for j = 1 : size(data,2)

            if  (isempty(data{i,j}))
                continue;
            end
            id = id + 1;

            linkTable(id,1) = id;
            linkTable(id,2) = i;
            linkTable(id,3) = j;
            
            for t = 1 : length(data{i,j})
                linkTable(id,3+t) = data{i,j}(t);
            end
        end
    end
else
    for i = 1 : size(data,1)
        for j = 1 : size(data,2)

            if (isempty(data(i,j) == 0))
                continue;
            end
            id = id + 1;

            linkTable(id,1) = id;
            linkTable(id,2) = i;
            linkTable(id,3) = j;
            linkTable(id,4) = data(i,j);

        end
    end
end
    
end

function fitvalue = cplex_datreader(file)
% Reads results from OPL and extracts scalar objective value (fitvalue)
disp('');
disp('Interpreting results');
fitvalue = xlsread(file);
end

function [] = cplex_datwrite(variables, filename)

text = GetFileString(variables);   

if (isempty(text) || isempty(filename))
    return;
end

f = fopen(filename,'wt');
fprintf(f,text);
fclose(f);

end

function [text] = GetFileString(variables)

text = '';

for i = 1 : 2 : length(variables)
    text = strcat(text,VariableToCplexFormat(variables{i},variables{i+1}));
end

end

function [text] = VariableToCplexFormat(name,values)

if size(values,1) > 1 && size(values,2) > 1
   
    if size(values,1) ~= size(values,2)
        
        CreateTuple(name,values);
        text = '';
        return;
    end
    
    text = CreateArray(name,values);
    return;
end

if size(values,1) > 1 || size(values,2) > 1
    text = CreateArray(name,values);
    return;
end

text = CreateSingleValue(name,values);

end

function [text] = CreateArray(name,values)

global s;

rows = size(values,1);
cols = size(values,2);

text = strcat(name,'=',s.leftsqbrac);

if rows > 1 && cols > 1
    for i = 1 : rows
        text = strcat(text,s.leftsqbrac);
        for j = 1 : cols
            if j == size(link_table,2)
                text = strcat(text,num2str(values(i,j)));
            else
                text = strcat(text,num2str(values{i,j}),s.comma);
            end
        end
        
        if i == size(values,1)
            text = strcat(text,s.rightsqbrac,s.newline);
        else
            text = strcat(text,s.rightsqbrac,s.comma,s.newline);
        end
    end
    text = strcat(text,s.rightsqbrac,s.scolon,s.newline);
    return;
end
    
text = strcat(name, '=', s.leftsqbrac);

if rows > cols
    vlength = rows;
else
    vlength = cols;
end

for i = 1 : vlength
    if i == vlength
        text = strcat(text, num2str(values(i)));
    else
        text = strcat(text, num2str(round(values(i))), s.comma);
    end
end

text = strcat(text,s.rightsqbrac,s.scolon,s.newline);

end

function [text] = CreateSingleValue(name,values)

global s;

text = strcat(name, ' = ', num2str(values),s.scolon,s.newline);

end

function [] = CreateTuple(name, value)

global s;

rows = size(value,1); % number of rows
cols = size(value,2); % number of columns

text = strcat(name, '=', s.leftcubrac, s.newline);
for i = 1 : rows 
% writes existing text and a < - i.e. "name = { <"
    text = strcat(text, s.lowerthan);
    
% writes data at every column individually
    for j = 1 : cols
        if (j == cols)
            text = strcat(text, num2str(value(i,j)));
        else
            text = strcat(text, num2str(value(i,j)), s.comma);
        end
    end
% writes "> and starts next line"
    if (i == rows)
        text = strcat(text, s.greaterthan, s.newline);
    else
        text = strcat(text, s.greaterthan, s.comma, s.newline);
    end
end
% writes "}"
text = strcat(text, s.rightcubrac, s.scolon);

f = fopen(strcat(name,'.dat'),'wt');

% writes all text
fprintf(f,text);
fclose(f);

end

function cplex_executemodel(command)
fprintf('Calling CPLEX ');
[~,~]=system(command);
fprintf('  Done!\n');
end
