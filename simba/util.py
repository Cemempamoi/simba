import numpy as np
import os, sys
import torch
import random
from copy import deepcopy

from contextlib import contextmanager
from timeit import default_timer

def format_elapsed_time(diff):
    """
    Small function to print the time elapsed between tic and toc in a nice manner
    """

    hours = int(diff // 3600)
    minutes = int((diff - 3600 * hours) // 60)
    seconds = str(int(diff - 3600 * hours - 60 * minutes))

    # Put everything in strings
    hours = str(hours)
    minutes = str(minutes)

    # Add a zero for one digit numbers for consistency
    if len(seconds) == 1:
        seconds = '0' + seconds
    if minutes == '0':
        return f"{seconds}\""

    if len(minutes) == 1:
        minutes = '0' + minutes
    if hours == '0':
        return f"{minutes}'{seconds}\""
    
    if len(hours) == 1:
        hours = '0' + hours
    return f"{hours}:{minutes}'{seconds}\""

def put_in_batch_form(data, name, verbose):
    if data is not None:
        if len(data.shape) == 2:
            if isinstance(data, np.ndarray):
                data = np.expand_dims(data, axis=0)
            else:
                data = data.unsqueeze(0)
            if verbose > 1:
                print(f'Assuming one batch only, reshaped {name} to {data.shape}')
    return data

def make_tensors(U, X=None, Y=None, x0=None, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = "cpu" 
    if not isinstance(U, torch.DoubleTensor):
        U = torch.tensor(U, dtype=torch.float64).to(device) if U is not None else None
        X = torch.tensor(X, dtype=torch.float64).to(device) if X is not None else None
        Y = torch.tensor(Y, dtype=torch.float64).to(device) if Y is not None else None
        x0 = torch.tensor(x0, dtype=torch.float64).to(device) if x0 is not None else None
    else:
        U = U.to(device) if U is not None else None
        X = X.to(device) if X is not None else None
        Y = Y.to(device) if Y is not None else None
        x0 = x0.to(device) if x0 is not None else None
    return U, X, Y, x0

def check_and_initialize_data(U=None, U_val=None, U_test=None, X=None, X_val=None, X_test=None, 
                              Y=None, Y_val=None, Y_test=None, x0=None, x0_val=None, x0_test=None, 
                              verbose=0, autonomous=False, input_output=False, device='cpu'):
    
    if not autonomous:
        assert U is not None, 'U has to be provided for non-autonomous systems.'
        if U_val is None:
            U_val = U.copy()
            if verbose > 1:
                print('U_val was not provided --> set U_val=U.')
        U = put_in_batch_form(U, 'U', verbose)
        U_val = put_in_batch_form(U_val, 'U_val', verbose)
        U_test = put_in_batch_form(U_test, 'U_val', verbose)

    if input_output:
        assert Y is not None, 'Y must be provided for input output model fitting.'
        if Y_val is None:
            Y_val = Y.copy()
            if verbose > 1:
                print('Y_val was not provided --> set Y_val=Y.')
        Y = put_in_batch_form(Y, 'Y', verbose)
        Y_val = put_in_batch_form(Y_val, 'Y_val', verbose)
        Y_test = put_in_batch_form(Y_test, 'Y_test', verbose)

        if U is None:
            # Trick to pass the batch size and prediction horizon in the forward pass:
            # U is actually never used, but u.shape[0] and u.shape[1] define the shape of the
            # predictions of Simba in the forward pass
            U = np.empty_like(Y, dtype='float64')
            U_val = np.empty_like(Y_val, dtype='float64')
        else:
            assert U.shape[0] == Y.shape[0], f'U and Y must have the same first dimension, got {U.shape[0]} != {Y.shape[0]}.'
            assert U_val.shape[0] == Y_val.shape[0], f'U_val and Y_val must have the same first dimension, got {U_val.shape[0]} != {Y_val.shape[0]}.'
            assert U.shape[1] == Y.shape[1], f'U and Y must have the same number of samples, got {U.shape[1]} != {Y.shape[1]}.'
            assert U_val.shape[1] == Y_val.shape[1], f'U_val and Y_val must have the same number of samples, got {U_val.shape[1]} != {Y_val.shape[1]}.'
    else:
        assert X is not None, 'X must be provided for input state model fitting.'
        if X_val is None:
            X_val = X.copy()
            if verbose > 1:
                print('X_val was not provided --> set X_val=X.')
        X = put_in_batch_form(X, 'X', verbose)
        X_val = put_in_batch_form(X_val, 'X_val', verbose)
        X_test = put_in_batch_form(X_test, 'X_test', verbose)
        
        if U is None:
            # Trick to pass the batch size and prediction horizon in the forward pass:
            # U is actually never used, but u.shape[0] and u.shape[1] define the shape of the
            # predictions of Simba in the forward pass
            U = np.empty_like(Y, dtype='float64')
            U_val = np.empty_like(Y_val, dtype='float64')
        else:
            assert U.shape[0] == X.shape[0], f'U and X must have the same first dimension, got {U.shape[0]} != {X.shape[0]}.'
            assert U_val.shape[0] == X_val.shape[0], f'U_val and X_val must have the same first dimension, got {U_val.shape[0]} != {X_val.shape[0]}.'
            assert U.shape[1] == X.shape[1], f'U and X must have the same number of samples, got {U.shape[1]} != {X.shape[1]}.'
            assert U_val.shape[1] == X_val.shape[1], f'U_val and X_val must have the same number of samples, got {U_val.shape[1]} != {X_val.shape[1]}.'    
        
    if (x0 is not None) and (x0_val is None):
        x0_val = deepcopy(x0)
    if (x0 is not None) and (x0_test is None):
        x0_test = deepcopy(x0_val)
    if x0 is not None:
        x0 = put_in_batch_form(x0, 'x0', verbose)
        x0_val = put_in_batch_form(x0_val, 'x0_val', verbose)
        x0_test = put_in_batch_form(x0_test, 'x0_test', verbose)

    if not isinstance(U, torch.DoubleTensor) or device is None:
        U, X, Y, x0 = make_tensors(U, X, Y, x0, device)
        U_val, X_val, Y_val, x0_val = make_tensors(U_val, X_val, Y_val, x0_val, device)
        U_test, X_test, Y_test, x0_test = make_tensors(U_test, X_test, Y_test, x0_test, device)
    
    if not autonomous:
        assert not torch.isnan(U).any(), f'U contains {torch.isnan(U).sum()} NaN(s), which is not handled currently.'
        assert not torch.isnan(U_val).any(), f'U_val contains {torch.isnan(U_val).sum()} NaN(s), which is not handled currently.'
        
    return U, U_val, U_test, X, X_val, X_test, Y, Y_val, Y_test, x0, x0_val, x0_test
    
def break_trajectories(data, horizon, stride=1):
    """
    Breaks trajectories in overlapping bits:
        Transforms data = (n_traj, N, :) 
        into data = (n_traj * ?, horizon+1, :)
    where '?' is determined by the desired overlapping between the created bits, N, and the horizon.
    """
    data = put_in_batch_form(data, name='', verbose=0)
    assert horizon < data.shape[1], f'Trajectories are not long enough for the desired horizon of {horizon}: at least {horizon + 1} points are required by trajectory, but {data.shape[1]} are provided.'
    n_traj = data.shape[0]
    N = data.shape[1]
    if isinstance(data, np.ndarray):
        out = np.concatenate([
            np.concatenate([
                data[[traj], n:n+horizon+1, :] for n in range(0, N-horizon, stride)
            ], axis=0) for traj in range(n_traj)], axis=0)
    else:
        out = torch.cat([
            torch.cat([
                data[[traj], n:n+horizon+1, :] for n in range(0, N-horizon, stride)
            ], axis=0) for traj in range(n_traj)], axis=0)
    return out

def evaluate(A, B, U, X, noise=None, name='', print_trajs=True, return_mean=False):
    H = X.shape[1]
    if not return_mean:
        print(f'{name}:', end='')
    errors = []
    maes = []
    mapes = []
    fros = []
    for j in range(U.shape[0]):
        x_all = torch.empty((X.shape[1], X.shape[2]), dtype=torch.float64, device=U.device)
        if noise is None:
            x = X[j,[0],:].T
        else:
            x = X[j,[0],:].T - noise[:,[j*H]]
        for i in range(X.shape[1]):
            x_all[[i],:] = x.T
            x = A @ x + B @ U[j,[i],:].T
        if noise is None:
            X_ = X[j,:,:]
        else:
            X_ = X[j,:,:] - noise[:,j*H:(j+1)*H].T
        errors.append(torch.mean((x_all - X_)**2).cpu().detach().numpy())
        maes.append(torch.mean(torch.abs(x_all - X_)).cpu().detach().numpy()) 
        mapes.append(torch.mean(torch.abs((x_all - X_)/X_)).cpu().detach().numpy()) 
        fros.append(torch.linalg.norm(x_all - X_, ord='fro').cpu().detach().numpy()**2/2)
        if print_trajs:
            print(f'\tTr. {j+1}:\t{errors[-1]:.2E}\t{maes[-1]:.2E}\t{mapes[-1]*100:.1f}%\t\t{fros[-1]:.2E}')
    if print_trajs:
        print('\t\t--------------------------------------------------------')
    if return_mean:
        return np.mean(errors)
    else:
        print(f'\t{"Mean:" if print_trajs else""}\t{np.mean(errors):.2E}\t{np.mean(maes):.2E}\t{np.mean(mapes)*100:.1f}%\t\t{np.mean(fros):.2E}')
    if print_trajs:
        print('')

def generate_A_Hurwitz(nx):
    # https://math.stackexchange.com/questions/2674083/randomly-generate-hurwitz-matrices
    while True:
        try:
            W = np.diag(np.random.uniform(-1,1,(nx,)))
            V = np.random.normal(0, 1, (nx,nx))
            A = V.dot(W).dot(np.linalg.inv(V))
            break
        except:
            continue
    return A

def normalize(data, min_=None, diff=None):
    if min_ is None:
        min_ = data.min(axis=0, keepdim=True).values.min(axis=1, keepdim=True).values
        diff = data.max(axis=0, keepdim=True).values.max(axis=1, keepdim=True).values - min_
        return (data - min_) / diff * 0.8 + 0.1, min_, diff
    else:
        return (data - min_) / diff * 0.8 + 0.1

def inverse_normalize(data, min_, diff):
    return (data - 0.1) / 0.8 * diff + min_

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)