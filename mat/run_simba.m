function [sys] = run_simba(parameters, data, data_val, data_test, Ts)
% Call SIMBa from matlab
% Send everything to python and run it
U = data.u;
Y = data.y;
U_val = data_val.u;
Y_val = data_val.y;
U_test = data_test.u;
Y_test = data_test.y;
save(fullfile('python_compatibility', 'parameters.mat'), 'parameters');
save(fullfile('python_compatibility', 'U.mat'), 'U');
save(fullfile('python_compatibility', 'Y.mat'), 'Y');
save(fullfile('python_compatibility', 'U_val.mat'), 'U_val');
save(fullfile('python_compatibility', 'Y_val.mat'), 'Y_val');
save(fullfile('python_compatibility', 'U_test.mat'), 'U_test');
save(fullfile('python_compatibility', 'Y_test.mat'), 'Y_test');

pyrunfile(fullfile('..','call_from_matlab.py'));

% Load the result
load(fullfile(parameters.directory,parameters.save_name,'matrices.mat'));
sys = ss(A,B,C,D,Ts);
end