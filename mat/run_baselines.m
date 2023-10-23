function [A, B, C, D, m_x0, m_arx_time, m_n4sid_time, m_pem_time, m_arx_train, m_n4sid_train, m_pem_train, m_arx_val, m_n4sid_val, m_pem_val, m_arx_test, m_n4sid_test, m_pem_test] = run_baselines(m_U, m_Y, m_U_val, m_Y_val, m_U_test, m_Y_test, Ts, m_nxs, m_enforce_stability, m_fit_x0)
%runs MATLAB sysid

nu = size(m_U,2);
ny = size(m_Y,2);

data = iddata(m_Y, m_U, Ts);
data_val = iddata(m_Y_val, m_U_val, Ts);

m_arx_time = [];
m_n4sid_time = [];
m_pem_time = [];
m_arx_train = [];
m_n4sid_train = [];
m_pem_train = [];
m_arx_val = [];
m_n4sid_val = [];
m_pem_val = [];
m_arx_test = [];
m_n4sid_test = [];
m_pem_test = [];

digitsOld = digits(64);

for i = 1 : numel(m_nxs)
    nx = m_nxs(i);

    na = ones(ny) * nx;
    nb = ones(ny,nu) * nx;
    nk = zeros(ny,nu);
    tic;
    sys = arx(data, [na,nb,nk]);
    m_arx_time = [m_arx_time, toc];
    x0 = findstates(sys, data);
    OPT = simOptions('InitialCondition',x0);
    ssmodel = idss(sys);
    y_train = sim(ssmodel,data,OPT).OutputData;
    m_arx_train = [m_arx_train, y_train]; 
    
    x0_val = findstates(sys,data_val);
    OPT = simOptions('InitialCondition',x0_val);
    ssmodel = idss(sys);
    y_val = sim(ssmodel,data_val,OPT).OutputData;
    m_arx_val = [m_arx_val, y_val];

    test = [];
    for t = 1 : numel(m_Y_test)
        data_test = iddata(m_Y_test(t), m_U_test(t), Ts);
        x0_test = findstates(sys,data_test);
        OPT = simOptions('InitialCondition',x0_test);
        ssmodel = idss(sys);
        y_test = sim(ssmodel,data_test,OPT).OutputData;
        test = [test, y_test];
    end
    m_arx_test = [m_arx_test, test];

    opt = n4sidOptions;
    opt = n4sidOptions('InitialState', m_fit_x0, 'Focus','simulation', 'EnforceStability', m_enforce_stability);
    tic;
    [sys, m_x0] = n4sid(data, nx, opt);
    m_n4sid_time = [m_n4sid_time, toc];
    ssmodel=idss(sys);
    if isequal(m_fit_x0,'estimate')
        x0 = findstates(sys,data);
        OPT = simOptions('InitialCondition',x0);
    else
        OPT = simOptions('InitialCondition','z');
    end
    y_train = sim(ssmodel, data, OPT).OutputData;
    m_n4sid_train = [m_n4sid_train y_train];
    
    if isequal(m_fit_x0,'estimate')
        x0_val = findstates(sys,data_val);
        OPT = simOptions('InitialCondition',x0_val);
    else
        OPT = simOptions('InitialCondition','z');
    end
    y_val_n = sim(ssmodel, data_val, OPT).OutputData;
    m_n4sid_val = [m_n4sid_val, y_val_n];

    test = [];
    for t = 1 : numel(m_Y_test)
        data_test = iddata(m_Y_test(t), m_U_test(t), Ts);
        if isequal(m_fit_x0,'estimate')
            x0_test = findstates(sys,data_test);
            OPT = simOptions('InitialCondition',x0_test);
        else
            OPT = simOptions('InitialCondition','z');
        end
        y_test = sim(ssmodel, data_test, OPT).OutputData;
        test = [test, y_test];
    end
    m_n4sid_test = [m_n4sid_test, test];

    % Store current matrices
    A = sys.A;
    B = sys.B;
    C = sys.C;
    D = sys.D;

    opt = ssestOptions('InitialState', m_fit_x0, 'Focus', 'simulation', 'EnforceStability', m_enforce_stability);
    tic;
    sys = pem(data, sys, opt);
    m_pem_time = [m_pem_time, toc];
    ssmodel=idss(sys);
    if isequal(m_fit_x0,'estimate')
        x0 = findstates(sys,data);
        OPT = simOptions('InitialCondition',x0);
    else
        OPT = simOptions('InitialCondition','z');
    end
    y_train = sim(ssmodel, data, OPT).OutputData;
    m_pem_train = [m_pem_train y_train];
    
    if isequal(m_fit_x0,'estimate')
        x0_val = findstates(sys,data_val);
        OPT = simOptions('InitialCondition',x0_val);
    else
        OPT = simOptions('InitialCondition','z');
    end
    y_val = sim(ssmodel, data_val, OPT).OutputData;
    m_pem_val = [m_pem_val, y_val];

    test = [];
    for t = 1 : numel(m_Y_test)
        data_test = iddata(m_Y_test(t), m_U_test(t), Ts);
        if isequal(m_fit_x0,'estimate')
            x0_test = findstates(sys,data_test);
            OPT = simOptions('InitialCondition',x0_test);
        else
            OPT = simOptions('InitialCondition','z');
        end
        y_test = sim(ssmodel, data_test, OPT).OutputData;
        test = [test, y_test];
    end
    m_pem_test = [m_pem_test, test];

    % Overwirte matrices if performs better
    if mean(mean((y_val - m_Y_val).^2)) < mean(mean((y_val_n - m_Y_val).^2))
        A = sys.A;
        B = sys.B;
        C = sys.C;
        D = sys.D;
    end

end
digits(digitsOld);
end