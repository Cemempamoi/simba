import numpy as np
import scipy
import torch
import warnings

from simba.util import elapsed_timer, HiddenPrints
from simba.parameters import baselines_to_use, IS_MATLAB, IS_SIPPY

if IS_MATLAB:
    import matlab
    import matlab.engine

if IS_SIPPY:
    from sippy import *
    from sippy import functionset as fset


def identify_baselines(nx, U, U_val, U_test, Y, Y_val, Y_test, x0, x0_val, x0_test, dt, parameters, baselines_to_use=baselines_to_use, id_mat=True):
    # Define orders
    x0 = x0_val = x0_test = torch.zeros((1,1,nx), dtype=torch.float64, device=parameters['device'])

    nu = U.shape[-1]
    ny = Y.shape[-1]
    ordersna = [nx] * ny
    ordersnb = [[nx] * nu] * ny
    ordersnc = [nx] * ny
    ordersnd = [nx] * ny
    ordersnf = [nx] * ny
    theta_list = [[0] * nu] * ny

    names = []
    baselines = []
    times = []
    train_ids = []
    validation_ids = []
    test_ids = []

    if IS_SIPPY:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with HiddenPrints():
                if baselines_to_use['armax_ills']:
                    with elapsed_timer() as elapsed:
                        tf = system_identification(Y[0,:,:], U[0,:,:], 'ARMAX', ARMAX_orders=[ordersna, ordersnb, ordersnc, theta_list], max_iterations=200, centering = 'None')  #
                        validation = fset.validation(tf,U_val[0,:,:],Y_val[0,:,:], np.linspace(0, Y_val.shape[1]-1, Y_val.shape[1]), centering = 'None')
                        test = []
                        for t in range(len(Y_test)):
                            test.append(fset.validation(tf,U_test[t,:,:],Y_test[t,:,:], np.linspace(0, Y_test.shape[1]-1, Y_test.shape[1]), centering = 'None').T)
                        baselines.append(tf)
                        train_ids.append(tf.Yid.T)
                        validation_ids.append(validation.T)
                        test_ids.append(np.stack(test))
                        names.append('ARMAX-ILLS')
                        times.append(elapsed())

                if baselines_to_use['armax_rlls']:
                    with elapsed_timer() as elapsed:
                        tf = system_identification(Y[0,:,:], U[0,:,:], 'ARMAX', ARMAX_orders=[ordersna, ordersnb, ordersnc, theta_list], ARMAX_mod = 'RLLS', max_iterations=200, centering = 'None')  #
                        validation = fset.validation(tf,U_val[0,:,:],Y_val[0,:,:], np.linspace(0, Y_val.shape[1]-1, Y_val.shape[1]), centering = 'None')
                        test = []
                        for t in range(len(Y_test)):
                            test.append(fset.validation(tf,U_test[t,:,:],Y_test[t,:,:], np.linspace(0, Y_test.shape[1]-1, Y_test.shape[1]), centering = 'None').T)
                        baselines.append(tf)
                        train_ids.append(tf.Yid.T)
                        validation_ids.append(validation.T)
                        test_ids.append(np.stack(test))
                        names.append('ARMAX-RLLS')
                        times.append(elapsed())

                if baselines_to_use['armax_opt']:
                    with elapsed_timer() as elapsed:
                        tf = system_identification(Y[0,:,:], U[0,:,:], 'ARMAX', ARMAX_orders=[ordersna, ordersnb, ordersnc, theta_list], ARMAX_mod = 'OPT', max_iterations=200, centering = 'None')  #
                        validation = fset.validation(tf,U_val[0,:,:],Y_val[0,:,:], np.linspace(0, Y_val.shape[1]-1, Y_val.shape[1]), centering = 'None')
                        test = []
                        for t in range(len(Y_test)):
                            test.append(fset.validation(tf,U_test[t,:,:],Y_test[t,:,:], np.linspace(0, Y_test.shape[1]-1, Y_test.shape[1]), centering = 'None').T)
                        baselines.append(tf)
                        train_ids.append(tf.Yid.T)
                        validation_ids.append(validation.T)
                        test_ids.append(np.stack(test))
                        names.append('ARMAX-OPT')
                        times.append(elapsed())

                if baselines_to_use['arx_ills']:
                    with elapsed_timer() as elapsed:
                        tf = system_identification(Y[0,:,:], U[0,:,:], 'ARX', ARX_orders=[ordersna, ordersnb, theta_list], max_iterations=200, centering = 'None')  #
                        validation = fset.validation(tf,U_val[0,:,:],Y_val[0,:,:], np.linspace(0, Y_val.shape[1]-1, Y_val.shape[1]), centering = 'None')
                        test = []
                        for t in range(len(Y_test)):
                            test.append(fset.validation(tf,U_test[t,:,:],Y_test[t,:,:], np.linspace(0, Y_test.shape[1]-1, Y_test.shape[1]), centering = 'None').T)
                        baselines.append(tf)
                        train_ids.append(tf.Yid.T)
                        validation_ids.append(validation.T)
                        test_ids.append(np.stack(test))
                        names.append('ARX-ILLS')
                        times.append(elapsed())

                if baselines_to_use['arx_rlls']:
                    with elapsed_timer() as elapsed:
                        tf = system_identification(Y[0,:,:], U[0,:,:], 'ARX', ARX_orders=[ordersna, ordersnb, theta_list], ARX_mod = 'RLLS', max_iterations=200, centering = 'None')  #
                        validation = fset.validation(tf,U_val[0,:,:],Y_val[0,:,:], np.linspace(0, Y_val.shape[1]-1, Y_val.shape[1]), centering = 'None')
                        test = []
                        for t in range(len(Y_test)):
                            test.append(fset.validation(tf,U_test[t,:,:],Y_test[t,:,:], np.linspace(0, Y_test.shape[1]-1, Y_test.shape[1]), centering = 'None').T)
                        baselines.append(tf)
                        train_ids.append(tf.Yid.T)
                        validation_ids.append(validation.T)
                        test_ids.append(np.stack(test))
                        names.append('ARX-RLLS')
                        times.append(elapsed())

                if baselines_to_use['arx_opt']:
                    with elapsed_timer() as elapsed:
                        tf = system_identification(Y[0,:,:], U[0,:,:], 'ARX', ARX_orders=[ordersna, ordersnb, theta_list], ARX_mod = 'OPT', max_iterations=200, centering = 'None')  #
                        validation = fset.validation(tf,U_val[0,:,:],Y_val[0,:,:], np.linspace(0, Y_val.shape[1]-1, Y_val.shape[1]), centering = 'None')
                        test = []
                        for t in range(len(Y_test)):
                            test.append(fset.validation(tf,U_test[t,:,:],Y_test[t,:,:], np.linspace(0, Y_test.shape[1]-1, Y_test.shape[1]), centering = 'None').T)
                        baselines.append(tf)
                        train_ids.append(tf.Yid.T)
                        validation_ids.append(validation.T)
                        test_ids.append(np.stack(test))
                        names.append('ARX-OPT')
                        times.append(elapsed())

                if baselines_to_use['oe_rlls']:
                    with elapsed_timer() as elapsed:
                        tf = system_identification(Y[0,:,:], U[0,:,:], 'OE', OE_orders=[ordersnb, ordersnf, theta_list], OE_mod = 'RLLS', max_iterations=200, centering = 'None')  #
                        validation = fset.validation(tf,U_val[0,:,:],Y_val[0,:,:], np.linspace(0, Y_val.shape[1]-1, Y_val.shape[1]), centering = 'None')
                        test = []
                        for t in range(len(Y_test)):
                            test.append(fset.validation(tf,U_test[t,:,:],Y_test[t,:,:], np.linspace(0, Y_test.shape[1]-1, Y_test.shape[1]), centering = 'None').T)
                        baselines.append(tf)
                        train_ids.append(tf.Yid.T)
                        validation_ids.append(validation.T)
                        test_ids.append(np.stack(test))
                        names.append('OE-ILLS')
                        times.append(elapsed())

                if baselines_to_use['oe_opt']:
                    with elapsed_timer() as elapsed:
                        tf = system_identification(Y[0,:,:], U[0,:,:], 'OE', OE_orders=[ordersnb, ordersnf, theta_list], OE_mod = 'OPT', max_iterations=200, centering = 'None')  #
                        validation = fset.validation(tf,U_val[0,:,:],Y_val[0,:,:], np.linspace(0, Y_val.shape[1]-1, Y_val.shape[1]), centering = 'None')
                        test = []
                        for t in range(len(Y_test)):
                            test.append(fset.validation(tf,U_test[t,:,:],Y_test[t,:,:], np.linspace(0, Y_test.shape[1]-1, Y_test.shape[1]), centering = 'None').T)
                        baselines.append(tf)
                        train_ids.append(tf.Yid.T)
                        validation_ids.append(validation.T)
                        test_ids.append(np.stack(test))
                        names.append('OE-OPT')
                        times.append(elapsed())

                if baselines_to_use['bj']:
                    with elapsed_timer() as elapsed:
                        tf = system_identification(Y[0,:,:], U[0,:,:], 'BJ', BJ_orders=[ordersnb, ordersnc, ordersnd, ordersnf, theta_list], max_iterations=200, centering = 'None')  #
                        validation = fset.validation(tf,U_val[0,:,:],Y_val[0,:,:], np.linspace(0, Y_val.shape[1]-1, Y_val.shape[1]), centering = 'None')
                        test = []
                        for t in range(len(Y_test)):
                            test.append(fset.validation(tf,U_test[t,:,:],Y_test[t,:,:], np.linspace(0, Y_test.shape[1]-1, Y_test.shape[1]), centering = 'None').T)
                        baselines.append(tf)
                        train_ids.append(tf.Yid.T)
                        validation_ids.append(validation.T)
                        test_ids.append(np.stack(test))
                        names.append('BJ')
                        times.append(elapsed())
                
                if baselines_to_use['gen']:
                    with elapsed_timer() as elapsed:
                        tf = system_identification(Y[0,:,:], U[0,:,:], 'GEN', GEN_orders=[ordersna, ordersnb, ordersnc, ordersnd, ordersnf, theta_list], max_iterations=200, centering = 'None')  #
                        validation = fset.validation(tf,U_val[0,:,:],Y_val[0,:,:], np.linspace(0, Y_val.shape[1]-1, Y_val.shape[1]), centering = 'None')
                        test = []
                        for t in range(len(Y_test)):
                            test.append(fset.validation(tf,U_test[t,:,:],Y_test[t,:,:], np.linspace(0, Y_test.shape[1]-1, Y_test.shape[1]), centering = 'None').T)
                        baselines.append(tf)
                        train_ids.append(tf.Yid.T)
                        validation_ids.append(validation.T)
                        test_ids.append(np.stack(test))
                        names.append('GEN')
                        times.append(elapsed())

                if baselines_to_use['n4sid']:
                    with elapsed_timer() as elapsed:
                        ss = system_identification(Y[0,:,:], U[0,:,:], id_method='N4SID', SS_fixed_order=nx, SS_D_required=parameters['id_D'], SS_A_stability=parameters['stable_A'])  #
                        lti = scipy.signal.dlti(ss.A, ss.B, ss.C, ss.D, dt=dt)
                        _,y_train,_ = lti.output(U[0,:,:], np.arange(Y.shape[1])*dt, x0[0,:,:])
                        _,y_val,_ = lti.output(U_val[0,:,:], np.arange(Y_val.shape[1])*dt, x0_val[0,:,:])
                        y_test = []
                        for t in range(len(Y_test)):
                            _,y,_ = lti.output(U_test[t,:,:], np.arange(Y_test.shape[1])*dt, x0_test[t,:,:])
                            y_test.append(y)
                        baselines.append(ss)
                        train_ids.append(y_train)
                        validation_ids.append(y_val)
                        test_ids.append(np.stack(y_test))
                        names.append('N4SID')
                        times.append(elapsed())

                if baselines_to_use['moesp']:
                    with elapsed_timer() as elapsed:
                        ss = system_identification(Y[0,:,:], U[0,:,:], id_method='MOESP', SS_fixed_order=nx, SS_D_required=parameters['id_D'], SS_A_stability=parameters['stable_A'])  #
                        lti = scipy.signal.dlti(ss.A, ss.B, ss.C, ss.D, dt=dt)
                        _,y_train,_ = lti.output(U[0,:,:], np.arange(Y.shape[1])*dt, x0[0,:,:])
                        _,y_val,_ = lti.output(U_val[0,:,:], np.arange(Y_val.shape[1])*dt, x0_val[0,:,:])
                        y_test = []
                        for t in range(len(Y_test)):
                            _,y,_ = lti.output(U_test[t,:,:], np.arange(Y_test.shape[1])*dt, x0_test[t,:,:])
                            y_test.append(y)
                        baselines.append(ss)
                        train_ids.append(y_train)
                        validation_ids.append(y_val)
                        test_ids.append(np.stack(y_test))
                        names.append('MOESP')
                        times.append(elapsed())

                if baselines_to_use['cva']:
                    with elapsed_timer() as elapsed:
                        ss = system_identification(Y[0,:,:], U[0,:,:], id_method='CVA', SS_fixed_order=nx, SS_D_required=parameters['id_D'], SS_A_stability=parameters['stable_A'])  #
                        lti = scipy.signal.dlti(ss.A, ss.B, ss.C, ss.D, dt=dt)
                        _,y_train,_ = lti.output(U[0,:,:], np.arange(Y.shape[1])*dt, x0[0,:,:])
                        _,y_val,_ = lti.output(U_val[0,:,:], np.arange(Y_val.shape[1])*dt, x0_val[0,:,:])
                        y_test = []
                        for t in range(len(Y_test)):
                            _,y,_ = lti.output(U_test[t,:,:], np.arange(Y_test.shape[1])*dt, x0_test[t,:,:])
                            y_test.append(y)
                        baselines.append(ss)
                        train_ids.append(y_train)
                        validation_ids.append(y_val)
                        test_ids.append(np.stack(y_test))
                        names.append('CVA')
                        times.append(elapsed())

                if baselines_to_use['parsim_k']:
                    with elapsed_timer() as elapsed:
                        ss = system_identification(Y[0,:,:], U[0,:,:], id_method='PARSIM-K', SS_fixed_order=nx, SS_D_required=parameters['id_D'], SS_A_stability=parameters['stable_A'])  #
                        lti = scipy.signal.dlti(ss.A, ss.B, ss.C, ss.D, dt=dt)
                        _,y_train,_ = lti.output(U[0,:,:], np.arange(Y.shape[1])*dt, x0[0,:,:])
                        _,y_val,_ = lti.output(U_val[0,:,:], np.arange(Y_val.shape[1])*dt, x0_val[0,:,:])
                        y_test = []
                        for t in range(len(Y_test)):
                            _,y,_ = lti.output(U_test[t,:,:], np.arange(Y_test.shape[1])*dt, x0_test[t,:,:])
                            y_test.append(y)
                        baselines.append(ss)
                        train_ids.append(y_train)
                        validation_ids.append(y_val)
                        test_ids.append(np.stack(y_test))
                        names.append('PARSIM-K')
                        times.append(elapsed())

                if baselines_to_use['parsim_s']:
                    with elapsed_timer() as elapsed:
                        ss = system_identification(Y[0,:,:], U[0,:,:], id_method='PARSIM-S', SS_fixed_order=nx, SS_D_required=parameters['id_D'], SS_A_stability=parameters['stable_A'])  #
                        lti = scipy.signal.dlti(ss.A, ss.B, ss.C, ss.D, dt=dt)
                        _,y_train,_ = lti.output(U[0,:,:], np.arange(Y.shape[1])*dt, x0[0,:,:])
                        _,y_val,_ = lti.output(U_val[0,:,:], np.arange(Y_val.shape[1])*dt, x0_val[0,:,:])
                        y_test = []
                        for t in range(len(Y_test)):
                            _,y,_ = lti.output(U_test[t,:,:], np.arange(Y_test.shape[1])*dt, x0_test[t,:,:])
                            y_test.append(y)
                        baselines.append(ss)
                        train_ids.append(y_train)
                        validation_ids.append(y_val)
                        test_ids.append(np.stack(y_test))
                        names.append('PARSIM-S')
                        times.append(elapsed())

                if baselines_to_use['parsim_p']:
                    with elapsed_timer() as elapsed:
                        ss = system_identification(Y[0,:,:], U[0,:,:], id_method='PARSIM-P', SS_fixed_order=nx, SS_D_required=parameters['id_D'], SS_A_stability=parameters['stable_A'])  #
                        lti = scipy.signal.dlti(ss.A, ss.B, ss.C, ss.D, dt=dt)
                        _,y_train,_ = lti.output(U[0,:,:], np.arange(Y.shape[1])*dt, x0[0,:,:])
                        _,y_val,_ = lti.output(U_val[0,:,:], np.arange(Y_val.shape[1])*dt, x0_val[0,:,:])
                        y_test = []
                        for t in range(len(Y_test)):
                            _,y,_ = lti.output(U_test[t,:,:], np.arange(Y_test.shape[1])*dt, x0_test[t,:,:])
                            y_test.append(y)
                        baselines.append(ss)
                        train_ids.append(y_train)
                        validation_ids.append(y_val)
                        test_ids.append(np.stack(y_test))
                        names.append('PARSIM-P')
                        times.append((elapsed()))

    if IS_MATLAB and id_mat:
        _, names, times, train_ids, validation_ids,test_ids = matlab_baselines(parameters['path_to_matlab'], names, times, train_ids, validation_ids, test_ids,
                                                                                        nx=nx, U=U, U_val=U_val, U_test=U_test, Y=Y, Y_val=Y_val, Y_test=Y_test, x0=x0,
                                                                                        dt=dt, stable_A=parameters['stable_A'], learn_x0=parameters['learn_x0'])

    return names, baselines, times, train_ids, validation_ids, test_ids


def matlab_baselines(path_to_matlab, names, times, train_ids, validation_ids, test_ids,
                     nx, U, U_val, U_test, Y, Y_val, Y_test, x0, stable_A, learn_x0, dt=None):
    
    eng = matlab.engine.start_matlab()
    if path_to_matlab is not None:
        eng.cd(path_to_matlab, nargout=0)

    Ts = dt if dt is not None else 1.
    nxs = [nx] if isinstance(nx, int) else nx

    m_U = matlab.double(U[0,:,:])  if isinstance(U, np.ndarray) else matlab.double(U[0,:,:].cpu().detach().numpy())
    m_Y = matlab.double(Y[0,:,:]) if isinstance(Y, np.ndarray) else matlab.double(Y[0,:,:].cpu().detach().numpy())
    m_U_val = matlab.double(U_val[0,:,:]) if isinstance(U_val, np.ndarray) else matlab.double(U_val[0,:,:].cpu().detach().numpy())
    m_Y_val = matlab.double(Y_val[0,:,:]) if isinstance(Y_val, np.ndarray) else matlab.double(Y_val[0,:,:].cpu().detach().numpy())
    m_U_test = [matlab.double(U_test[t,:,:]) if isinstance(U_test, np.ndarray) else matlab.double(U_test[t,:,:].cpu().detach().numpy()) for t in range(len(Y_test))]
    m_Y_test = [matlab.double(Y_test[t,:,:]) if isinstance(Y_test, np.ndarray) else matlab.double(Y_test[t,:,:].cpu().detach().numpy()) for t in range(len(Y_test))]
    m_nxs = matlab.double(nxs)

    m_enforce_stability = matlab.logical(stable_A)
    m_fit_x0 = 'estimate' if learn_x0 else 'zero'

    matlab_A, matlab_B, matlab_C, matlab_D, matlab_x0, m_arx_time, m_n4sid_time, m_pem_time, m_arx_train, m_n4sid_train, m_pem_train, m_arx_val, m_n4sid_val, m_pem_val, m_arx_test, m_n4sid_test, m_pem_test = eng.run_baselines(m_U, m_Y, m_U_val, m_Y_val, m_U_test, m_Y_test, Ts, m_nxs, m_enforce_stability, m_fit_x0, nargout=17)

    eng.quit() 

    matrices = [np.array(matlab_A), np.array(matlab_B), np.array(matlab_C), np.array(matlab_D), np.array(matlab_x0)]
    names += ['mat-ARX', 'mat-N4SID', 'mat-PEM']
    times += [m_arx_time, m_n4sid_time, m_pem_time]
    train_ids += [np.array(m_arx_train), np.array(m_n4sid_train), np.array(m_pem_train)]
    validation_ids += [np.array(m_arx_val), np.array(m_n4sid_val), np.array(m_pem_val)]
    ny = Y_test.shape[-1]
    test_ids += [np.stack([np.array(m_arx_test)[:,i*ny:(i+1)*ny] for i in range(Y_test.shape[0])]), np.stack([np.array(m_n4sid_test)[:,i*ny:(i+1)*ny] for i in range(Y_test.shape[0])]), np.stack([np.array(m_pem_test)[:,i*ny:(i+1)*ny] for i in range(Y_test.shape[0])])]

    return matrices, names, times, train_ids, validation_ids, test_ids

def matlab_sub(eng, U, U_val, U_test, X, X_val, X_test):

    m_U = matlab.double(U.squeeze().detach().numpy().T)
    m_U_val = matlab.double(U_val.squeeze().detach().numpy().T)
    m_U_test = matlab.double(U_test.squeeze().detach().numpy().T)
    m_X = matlab.double(X.squeeze().detach().numpy().T)
    m_X_val = matlab.double(X_val.squeeze().detach().numpy().T)
    m_X_test = matlab.double(X_test.squeeze().detach().numpy().T)

    A_ls, B_ls, A_SUB, B_SUB = eng.SUB(m_U, m_U_val, m_U_test, m_X, m_X_val, m_X_test, nargout=4)    

    A_ls = torch.tensor(np.array(A_ls), dtype=torch.float64)
    B_ls = torch.tensor(np.array(B_ls), dtype=torch.float64)
    A_sub = torch.tensor(np.array(A_SUB), dtype=torch.float64)
    B_sub = torch.tensor(np.array(B_SUB), dtype=torch.float64)

    return A_ls, B_ls, A_sub, B_sub

def findstate(eng, U, Y, simba):
    m_U = matlab.double(U.squeeze().detach().numpy())
    m_Y = matlab.double(Y.squeeze().detach().numpy())
    A = simba.A.detach().numpy()
    B = simba.B.detach().numpy()
    C = simba.C.detach().numpy()
    D = simba.D.detach().numpy()
    Ts = 1
    x0, y_est = eng.find_x0(m_U, m_Y, Ts, A, B, C, D, nargout=2)
    x0 = torch.tensor(np.array(x0), dtype=torch.float64)
    y_est = torch.tensor(np.array(y_est), dtype=torch.float64).unsqueeze(0)
    return x0, y_est