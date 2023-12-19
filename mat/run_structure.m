function [m_times, m_train, m_val, m_test] = run_baselines(m_A, m_B, m_C, m_D, m_mask_A, m_mask_B, m_mask_C, m_mask_D, m_U, m_Y, m_U_val, m_Y_val, m_U_test, m_Y_test, Ts, m_enforce_stability)

nx = size(m_mask_A,1);

data = iddata(m_Y, m_U, Ts);
data_val = iddata(m_Y_val, m_U_val, Ts);
data_test = iddata(m_Y_test, m_U_test, Ts);

m_times = [];
m_train = [];
m_val = [];
m_test = [];

digitsOld = digits(64);

%% 
opt = n4sidOptions;
opt = n4sidOptions('InitialState', 'zero', 'Focus', 'simulation', 'EnforceStability', m_enforce_stability);
[sys, m_x0] = n4sid(data, nx, opt);
init_sys = idss(sys);

init_sys.D = init_sys.D .* m_mask_D;
init_sys.K = zeros(size(init_sys.K));

init_sys.Structure.A.Free = true;
init_sys.Structure.B.Free = true;
init_sys.Structure.C.Free = true;
init_sys.Structure.D.Free = m_mask_D;
init_sys.Structure.K.Free = false;

opt = ssestOptions('InitialState', 'zero', 'Focus', 'simulation', 'EnforceStability', m_enforce_stability);
tic;
sys = ssest(data, init_sys, opt);
m_times = [m_times, toc];
ssmodel=idss(sys);

OPT = simOptions('InitialCondition','z');
y_train = sim(ssmodel, data, OPT).OutputData;
m_train = [m_train, y_train];
y_val = sim(ssmodel, data_val, OPT).OutputData;
m_val = [m_val, y_val];
y_test = sim(ssmodel, data_test, OPT).OutputData;
m_test = [m_test, y_test];


%%
init_sys.D = m_D;
init_sys.Structure.D.Free = false;

tic;
sys = ssest(data, init_sys, opt);
m_times = [m_times, toc];
ssmodel=idss(sys);
y_train = sim(ssmodel, data, OPT).OutputData;
m_train = [m_train, y_train];
y_val = sim(ssmodel, data_val, OPT).OutputData;
m_val = [m_val, y_val];
y_test = sim(ssmodel, data_test, OPT).OutputData;
m_test = [m_test, y_test];


%%
init_sys.C = init_sys.C .* m_mask_C;
init_sys.Structure.C.Free = m_mask_C;

tic;
sys = ssest(data, init_sys, opt);
m_times = [m_times, toc];
ssmodel=idss(sys);
y_train = sim(ssmodel, data, OPT).OutputData;
m_train = [m_train, y_train];
y_val = sim(ssmodel, data_val, OPT).OutputData;
m_val = [m_val, y_val];
y_test = sim(ssmodel, data_test, OPT).OutputData;
m_test = [m_test, y_test];

%%
init_sys.C = m_C;
init_sys.Structure.C.Free = false;

tic;
sys = ssest(data, init_sys, opt);
m_times = [m_times, toc];
ssmodel=idss(sys);
y_train = sim(ssmodel, data, OPT).OutputData;
m_train = [m_train, y_train];
y_val = sim(ssmodel, data_val, OPT).OutputData;
m_val = [m_val, y_val];
y_test = sim(ssmodel, data_test, OPT).OutputData;
m_test = [m_test, y_test];


%%
init_sys.B = init_sys.B .* m_mask_B;
init_sys.Structure.B.Free = m_mask_B;

tic;
sys = ssest(data, init_sys, opt);
m_times = [m_times, toc];
ssmodel=idss(sys);
y_train = sim(ssmodel, data, OPT).OutputData;
m_train = [m_train, y_train];
y_val = sim(ssmodel, data_val, OPT).OutputData;
m_val = [m_val, y_val];
y_test = sim(ssmodel, data_test, OPT).OutputData;
m_test = [m_test, y_test];

%%
init_sys.B = m_B;
init_sys.Structure.B.Free = false;

tic;
sys = ssest(data, init_sys, opt);
m_times = [m_times, toc];
ssmodel=idss(sys);
y_train = sim(ssmodel, data, OPT).OutputData;
m_train = [m_train, y_train];
y_val = sim(ssmodel, data_val, OPT).OutputData;
m_val = [m_val, y_val];
y_test = sim(ssmodel, data_test, OPT).OutputData;
m_test = [m_test, y_test];


%%
init_sys.A = init_sys.A .* m_mask_A;
init_sys.Structure.A.Free = m_mask_A;

tic;
sys = ssest(data, init_sys, opt);
m_times = [m_times, toc];
ssmodel=idss(sys);
y_train = sim(ssmodel, data, OPT).OutputData;
m_train = [m_train, y_train];
y_val = sim(ssmodel, data_val, OPT).OutputData;
m_val = [m_val, y_val];
y_test = sim(ssmodel, data_test, OPT).OutputData;
m_test = [m_test, y_test];

digits(digitsOld);
end