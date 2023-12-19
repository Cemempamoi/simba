function [A, B, C, D, x0, m_time, m_train, m_val, m_test] = run_baselines(nx, m_A, m_B, m_C, m_D, m_mask_A, m_mask_B, m_mask_C, m_mask_D, m_U, m_Y, m_U_val, m_Y_val, m_U_test, m_Y_test, Ts, m_enforce_stability, m_fit_x0)

data = iddata(m_Y, m_U, Ts);
data_val = iddata(m_Y_val, m_U_val, Ts);
data_test = iddata(m_Y_test, m_U_test, Ts);

digitsOld = digits(64);

%% 
opt = n4sidOptions;
opt = n4sidOptions('InitialState', m_fit_x0, 'Focus', 'simulation', 'EnforceStability', m_enforce_stability);
[sys, m_x0] = n4sid(data, nx, opt);
init_sys = idss(sys);
init_sys.K = zeros(size(init_sys.K));
init_sys.Structure.K.Free = false;

if ~islogical(m_A)
    init_sys.A = m_A;
    init_sys.Structure.A.Free = false;
elseif ~islogical(m_mask_A)
    init_sys.A = init_sys.A .* m_mask_A;
    init_sys.Structure.A.Free = m_mask_A;
end

if ~islogical(m_B)
    init_sys.B = m_B;
    init_sys.Structure.B.Free = false;
elseif ~islogical(m_mask_B)
    init_sys.B = init_sys.B .* m_mask_B;
    init_sys.Structure.B.Free = m_mask_B;
end

if ~islogical(m_C)
    init_sys.C = m_C;
    init_sys.Structure.C.Free = false;
elseif ~islogical(m_mask_C)
    init_sys.C = init_sys.C .* m_mask_C;
    init_sys.Structure.C.Free = m_mask_C;
end

if ~islogical(m_D)
    init_sys.D = m_D;
    init_sys.Structure.D.Free = false;
elseif ~islogical(m_mask_D)
    init_sys.D = init_sys.D .* m_mask_D;
    init_sys.Structure.D.Free = m_mask_D;
end


opt = ssestOptions('InitialState', m_fit_x0, 'Focus', 'simulation', 'EnforceStability', m_enforce_stability);
tic;
[sys, x0] = ssest(data, init_sys, opt);
m_time = toc;
ssmodel=idss(sys);

if isequal(m_fit_x0,'estimate')
    x0 = findstates(sys,data);
    OPT = simOptions('InitialCondition',x0);
else
    OPT = simOptions('InitialCondition','z');
end
m_train = sim(ssmodel, data, OPT).OutputData;

if isequal(m_fit_x0,'estimate')
    x0 = findstates(sys,data_val);
    OPT = simOptions('InitialCondition',x0);
else
    OPT = simOptions('InitialCondition','z');
end
m_val = sim(ssmodel, data_val, OPT).OutputData;

if isequal(m_fit_x0,'estimate')
    x0 = findstates(sys,data_test);
    OPT = simOptions('InitialCondition',x0);
else
    OPT = simOptions('InitialCondition','z');
end
m_test = sim(ssmodel, data_test, OPT).OutputData;

A = sys.A;
B = sys.B;
C = sys.C;
D = sys.D;

digits(digitsOld);
end