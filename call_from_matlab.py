import os
import scipy
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
import sys

# To see the simba module
sys.path.append('..')

from simba.model import Simba
from simba.util import fix_seed, load_mat

if __name__=="__main__":

    parameters = load_mat(os.path.join('python_compatibility', 'parameters.mat'))['parameters']
    U = scipy.io.loadmat(os.path.join('python_compatibility', 'U.mat'))['U']
    Y = scipy.io.loadmat(os.path.join('python_compatibility', 'Y.mat'))['Y']
    U_val = scipy.io.loadmat(os.path.join('python_compatibility', 'U_val.mat'))['U_val']
    Y_val = scipy.io.loadmat(os.path.join('python_compatibility', 'Y_val.mat'))['Y_val']
    U_test = scipy.io.loadmat(os.path.join('python_compatibility', 'U_test.mat'))['U_test']
    Y_test = scipy.io.loadmat(os.path.join('python_compatibility', 'Y_test.mat'))['Y_test']

    for k,p in parameters.items():
        if not isinstance(p, np.ndarray):
            if p == 'None':
                parameters[k] = None
            elif isinstance(p, int):
                if p == 1:
                    parameters[k] = True
                elif p == 0:
                    parameters[k] = False

    nu = U.shape[1]
    ny = Y.shape[1]

    # Input-output data
    X = None
    X_val = None
    X_test = None

    directory = parameters['directory']
    name = parameters['save_name']
    nx = parameters['nx']
    x0 = parameters['x0']
    x0_val = parameters['x0_val']
    x0_test = parameters['x0_test']

    fix_seed(parameters['seed'])
    
    simba = Simba(nx=nx, nu=nu, ny=ny, parameters=parameters)
    simba.fit(U, U_val=U_val, U_test=U_test, X=X, X_val=X_val, X_test=X_test, Y=Y, Y_val=Y_val, Y_test=Y_test, x0=x0, x0_val=x0_val, x0_test=x0_test)
    simba.save(directory=os.path.join(directory, name), save_name='SIMBa')
    
    if not parameters['id_D']:
        simba.D_ = nn.Parameter(torch.zeros((ny,nu), dtype=torch.float64, device=simba.device), requires_grad=simba.learn_D)
    scipy.io.savemat(os.path.join(directory, name, f'matrices.mat'), 
                {'A': simba.A.cpu().detach().numpy(), 
                'B': simba.B.cpu().detach().numpy(),
                'C': simba.C.cpu().detach().numpy(), 
                'D': simba.D.cpu().detach().numpy()})

