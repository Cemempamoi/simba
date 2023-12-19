import numpy as np
import scipy
import torch
import warnings

from simba.util import generate_A_Hurwitz, voss_noise, elapsed_timer, HiddenPrints
from simba.parameters import baselines_to_use, IS_MATLAB, IS_SIPPY

if IS_MATLAB:
    import matlab
    import matlab.engine

if IS_SIPPY:
    from sippy import *
    from sippy import functionset as fset

def generate_random_system(nx, nu, ny, N, stable_A=True, min_eigenvalue=None):
    
    if stable_A:
        A = generate_A_Hurwitz(nx)
        if min_eigenvalue is not None:
            for _ in range(100):
                if np.min(np.abs(np.linalg.eigvals(A))) > min_eigenvalue:
                    break
                A = generate_A_Hurwitz(nx)
    else:
        A = np.random.normal(np.random.uniform(-.5,.5), 1, (nx,nx))
    
    B = np.random.normal(np.random.uniform(-.5,.5), 1, (nx,nu))
    C = np.random.normal(np.random.uniform(-.5,.5), 1, (ny,nx))
    D = np.random.normal(np.random.uniform(-.5,.5), 1, (ny,nu))
    
    return A, B, C, D

def generate_data(A, B, C, D, N, id_D, process_noise_scale, U=None, x0=None, gaussian_U=False, low_limit=-1, high_limit=1, mean=0, scale=1, random_x0=False, dt=0.05):
    
    if not id_D:
        lti = scipy.signal.dlti(A, B, C, np.zeros(D.shape), dt=dt)
    else:
        lti = scipy.signal.dlti(A, B, C, D, dt=dt)
    
    t = np.arange(N) * dt
    if U is None:
        if gaussian_U:
            U = np.random.normal(mean, scale, (N, B.shape[1]))
        else:
            U = np.random.uniform(low_limit, high_limit, (N, B.shape[1]))
    if x0 is None:
        if random_x0:
            x0 = np.random.random((1, A.shape[1]))
        else:
            x0 = np.zeros((1, A.shape[1]))

    if process_noise_scale > 0:
        U_ = add_noise(U, scale=process_noise_scale)
        _, Y, X = lti.output(U_, t, x0)
    else:
        _, Y, X = lti.output(U, t, x0)
    
    return U, Y, X
    

def add_noise(*args, voss=False, colored=False, scale=1, ncols=16):
    args_ = []
    for arg in args:
        if voss:
            noise = np.empty(arg.shape)
            for i in range(arg.shape[1]):
                noise[:,i] = voss_noise(arg.shape[0], ncols=ncols)
        elif colored:
            A, B, C, D = generate_random_system(nx=arg.shape[2], nu=arg.shape[2], ny=arg.shape[2], N=arg.shape[1], stable_A=True)
            _, noise, _ = generate_data(A, B, C, D, id_D=True, N=arg.shape[1], U=None, x0=None, gaussian_U=True, process_noise_scale=1)
            noise = np.expand_dims(noise, 0)
        else:
            noise = np.random.normal(0, 1, size=arg.shape)
        args_.append(arg + noise * scale)
    return args_ if len(args_) > 1 else args_[0]

def get_noise(X, nx, nu, ny, N, colored, scale, gaussian_U=True, process_noise_scale=1):
    if colored:
        A, B, C, D = generate_random_system(nx, nu, ny, N=N, stable_A=True)
        _, noise, _ = generate_data(A, B, C, D, id_D=True, N=N, U=None, x0=None, gaussian_U=gaussian_U, process_noise_scale=process_noise_scale)
        noise = noise.T / scale
    else:
        noise = np.concatenate([np.random.normal(loc=0., scale=np.std(X[i,:].flatten()) / scale, size=(1, X.shape[1]+1)) for i in range(X.shape[0])], axis=0)
    return noise

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
                                                                                        nx=nx, U=U, U_val=U_val, U_test=U_test, Y=Y, Y_val=Y_val, Y_test=Y_test,
                                                                                        dt=dt, stable_A=parameters['stable_A'], learn_x0=parameters['learn_x0'])

    return names, baselines, times, train_ids, validation_ids, test_ids


def matlab_baselines(path_to_matlab, names, times, train_ids, validation_ids, test_ids,
                     nx, U, U_val, U_test, Y, Y_val, Y_test, stable_A, learn_x0, dt=None):
    
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

def matlab_init(parameters, nx, U, U_val, U_test, Y, Y_val, Y_test, dt=None):
    
    eng = matlab.engine.start_matlab()
    if parameters['path_to_matlab'] is not None:
        eng.cd(parameters['path_to_matlab'], nargout=0)

    m_A = matlab.double(parameters['A_init']) if parameters['A_init'] is not None else matlab.logical(False)
    m_B = matlab.double(parameters['B_init']) if parameters['B_init'] is not None else matlab.logical(False)
    m_C = matlab.double(parameters['C_init']) if parameters['C_init'] is not None else matlab.logical(False)
    m_D = matlab.double(parameters['D_init']) if parameters['D_init'] is not None else matlab.logical(False)
    
    m_mask_A = matlab.double(parameters['mask_A']*1.) if parameters['mask_A'] is not None else matlab.logical(False)
    m_mask_B = matlab.double(parameters['mask_B']*1.) if parameters['mask_B'] is not None else matlab.logical(False)
    m_mask_C = matlab.double(parameters['mask_C']*1.) if parameters['mask_C'] is not None else matlab.logical(False)
    m_mask_D = matlab.double(parameters['mask_D']*1.) if parameters['mask_D'] is not None else matlab.logical(False)

    m_U = matlab.double(U[0,:,:]) if isinstance(U, np.ndarray) else matlab.double(U[0,:,:].cpu().detach().numpy())
    m_Y = matlab.double(Y[0,:,:]) if isinstance(Y, np.ndarray) else matlab.double(Y[0,:,:].cpu().detach().numpy())
    m_U_val = matlab.double(U_val[0,:,:]) if isinstance(U_val, np.ndarray) else matlab.double(U_val[0,:,:].cpu().detach().numpy())
    m_Y_val = matlab.double(Y_val[0,:,:]) if isinstance(Y_val, np.ndarray) else matlab.double(Y_val[0,:,:].cpu().detach().numpy())
    m_U_test = [matlab.double(U_test[t,:,:]) if isinstance(U_test, np.ndarray) else matlab.double(U_test[t,:,:].cpu().detach().numpy()) for t in range(len(Y_test))]
    m_Y_test = [matlab.double(Y_test[t,:,:]) if isinstance(Y_test, np.ndarray) else matlab.double(Y_test[t,:,:].cpu().detach().numpy()) for t in range(len(Y_test))]

    m_enforce_stability = matlab.logical(parameters['stable_A'])
    Ts = dt if dt is not None else 1.
    m_fit_x0 = 'estimate' if parameters['learn_x0'] else 'zero'
    nx = matlab.double(nx)

    matlab_A, matlab_B, matlab_C, matlab_D, matlab_x0, m_time, m_train, m_val, m_test = eng.init_simba(nx, m_A, m_B, m_C, m_D, m_mask_A, m_mask_B, m_mask_C, m_mask_D, m_U, m_Y, m_U_val, m_Y_val, m_U_test, m_Y_test, Ts, m_enforce_stability, m_fit_x0, nargout=9)
    eng.quit() 
    matrices = [np.array(matlab_A), np.array(matlab_B), np.array(matlab_C), np.array(matlab_D), np.array(matlab_x0)]
    return  matrices, m_time, np.array(m_train), np.array(m_val), np.array(m_test)

def matlab_structure(eng, A, B, C, D, mask_A, mask_B, mask_C, mask_D, U, U_val, U_test, Y, Y_val, Y_test, stable_A, dt=None):
    
    m_A = matlab.double(A)
    m_B = matlab.double(B)
    m_C = matlab.double(C)
    m_D = matlab.double(D)
    ny = C.shape[0]
    
    m_mask_A = matlab.double(mask_A)
    m_mask_B = matlab.double(mask_B)
    m_mask_C = matlab.double(mask_C)
    m_mask_D = matlab.double(mask_D)

    m_U = matlab.double(U[0,:,:])  if isinstance(U, np.ndarray) else matlab.double(U[0,:,:].cpu().detach().numpy())
    m_Y = matlab.double(Y[0,:,:]) if isinstance(Y, np.ndarray) else matlab.double(Y[0,:,:].cpu().detach().numpy())
    m_U_val = matlab.double(U_val[0,:,:]) if isinstance(U_val, np.ndarray) else matlab.double(U_val[0,:,:].cpu().detach().numpy())
    m_Y_val = matlab.double(Y_val[0,:,:]) if isinstance(Y_val, np.ndarray) else matlab.double(Y_val[0,:,:].cpu().detach().numpy())
    m_U_test = [matlab.double(U_test[t,:,:]) if isinstance(U_test, np.ndarray) else matlab.double(U_test[t,:,:].cpu().detach().numpy()) for t in range(len(Y_test))]
    m_Y_test = [matlab.double(Y_test[t,:,:]) if isinstance(Y_test, np.ndarray) else matlab.double(Y_test[t,:,:].cpu().detach().numpy()) for t in range(len(Y_test))]

    m_enforce_stability = matlab.logical(stable_A)
    Ts = dt if dt is not None else 1.

    m_times, m_train, m_val, m_test = eng.run_structure(m_A, m_B, m_C, m_D, m_mask_A, m_mask_B, m_mask_C, m_mask_D, m_U, m_Y, m_U_val, m_Y_val, m_U_test, m_Y_test, Ts, m_enforce_stability, nargout=4)

    train = [np.array(m_train)[:,i*ny: (i+1)*ny] for i in range(7)]
    val = [np.array(m_val)[:,i*ny: (i+1)*ny] for i in range(7)]
    test = [np.array(m_test)[:,i*ny: (i+1)*ny] for i in range(7)]
    return  list(np.array(m_times).flatten()), train, val, test

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