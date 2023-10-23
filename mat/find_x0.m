function [x0, y_est] = find_x0(m_U, m_Y, Ts, A, B, C, D)

digitsOld = digits(64);

data = iddata(m_Y, m_U, Ts);
sys = idss(A,B,C,D);
x0 = findstates(sys, data);

OPT = simOptions('InitialCondition',x0);
y_est = sim(sys, data, OPT).OutputData;

digits(digitsOld);
end