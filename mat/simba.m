clear
clc
clear classes

%% Prepare the data
Ts = 1228.8;
load([fullfile('data', 'Daisy_1234567_3.mat')]);
data = iddata(Y, U, Ts);
data_val = iddata(Y_val, U_val, Ts);
data_test = data_val;

%% Launch python and load default options
path_to_python_env = fullfile('..', '.venv', 'bin', 'python'); % Poetry env
path_to_simba = '..';
[pe, opt] = SIMBaOptions(path_to_python_env,path_to_simba);

opt.path_to_matlab = 'None';

%% USER modifications
% Needed parameters
opt.directory = 'saves';
opt.save_name = 'Test_matlab';
opt.nx = 5;
opt.dt = Ts;
opt.delta = 'None';
opt.input_output = true;

% The loss can only be 'mse' or 'mae' for now, it needs to be a string
opt.train_loss = 'mse';
opt.val_loss = 'mse';
opt.init_loss = 'mse';

% How to treat x0
opt.x0 = 'None';
opt.x0_val = 'None';
opt.x0_test = 'None';
opt.learn_x0 = true;

% Additional simulation parameters 
% (see '../simba/parameters' for all the available options)
opt.seed = 12345678;
opt.max_epochs = 100;
opt.print_each = 100;
opt.init_epochs = 100;
opt.init_print_each = 100;

opt.init_from_matlab_or_ls = true;

opt.C_init = ones(size(Y,2), opt.nx).*0.5;
opt.learn_C = false;

%% Run SIMBa
sys = run_simba(opt, data, data_val, data_test, Ts);
