function [pe, opt] = SIMBaOptions(path_to_python_env, path_to_simba)
% Loading default options
% Run a python session with the right environment (from poetry in this case)
pe = pyenv('Version', [path_to_python_env], "ExecutionMode","OutOfProcess");
% Load the base parameters of SIMBa
if not(isfolder('python_compatibility'))
    mkdir('python_compatibility')
end
pyrunfile(fullfile(path_to_simba, 'get_parameters_for_matlab.py'));
opt = load('python_compatibility/parameters.mat');
end