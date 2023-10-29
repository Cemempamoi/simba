import os
import numpy as np
from copy import deepcopy

from torch import nn
import torch
from torch import optim
import torch.nn.functional as F

from simba.util import check_and_initialize_data, generate_A_Hurwitz, normalize, inverse_normalize, elapsed_timer, make_tensors, break_trajectories, put_in_batch_form, format_elapsed_time
from simba.functions import matlab_baselines, identify_baselines
from simba.parameters import base_parameters, check_parameters, check_sizes, IS_MATLAB, IS_SIPPY

class SIMBa(nn.Module):
    
    def __init__(self, nx, nu, ny, parameters=base_parameters):
        super().__init__()
        
        self.nx = nx
        self.nu = nu
        self.ny = ny

        check_sizes(nx, nu, ny, parameters)

        # Matrices masks
        self.mask_A = parameters['mask_A']
        self.mask_B = parameters['mask_B']
        self.mask_C = parameters['mask_C']
        self.mask_D = parameters['mask_D']
        # Matricies initilization
        self.A_init = parameters['A_init']
        self.B_init = parameters['B_init']
        self.C_init = parameters['C_init']
        self.D_init = parameters['D_init']
        # Which matricies to learn (typically set to True if A_init is known a priori)
        self.learn_A = parameters['learn_A']
        self.learn_B = parameters['learn_B']
        self.learn_C = parameters['learn_C']
        self.learn_D = parameters['learn_D']
        # Which matrices to identify
        self.autonomous = parameters['autonomous']
        self.input_output = parameters['input_output']
        self.id_D = parameters['id_D']
        # Stability parameters for the matrix A
        self.stable_A = parameters['stable_A']
        self.LMI_A = parameters['LMI_A']
        self.max_eigenvalue = parameters['max_eigenvalue']
        self.naive_A = parameters['naive_A']
        self.tol_A = parameters['tol_A']
        # For input-output models, whether to learn_x0 (as a free parameter) of to learn it from y0
        self.learn_x0 = parameters['learn_x0']
        self.learn_x0_from_y0 = parameters['learn_x0_from_y0']
        self.delta = parameters['delta']

        # Initialize A
        self.mask_A = torch.tensor(self.mask_A, device=self.device, dtype=torch.float64) if self.mask_A is not None else None
        if self.mask_A is not None and self.LMI_A:
            self.N = torch.diag(torch.max(torch.sum(self.mask_A, dim=1), torch.sum(self.mask_A, dim=0)) + self.tol_A)
        if self.stable_A:
            if self.LMI_A:
                self.X = nn.Parameter(torch.randn(2*nx, 2*nx, dtype=torch.float64), True)
                self.V = nn.Parameter(torch.randn(nx, nx, dtype=torch.float64), True)
            elif self.naive_A:
                self.A_ = nn.Parameter(torch.randn(nx, nx, dtype=torch.float64), True)
                self.max_eig_param = nn.Parameter(torch.randn(1,1, dtype=torch.float64), True)
            else:
                self.M = nn.Parameter(torch.randn(nx, nx, dtype=torch.float64), True)
                self.A_ = nn.Parameter(torch.randn(nx, nx, dtype=torch.float64), True)
        else:
            if self.A_init is None:
                self.A_ = nn.Parameter(torch.tensor(generate_A_Hurwitz(nx), dtype=torch.float64), True)
            else:
                self.A_ = nn.Parameter(torch.tensor(self.A_init, dtype=torch.float64), requires_grad=self.learn_A)
        
        # Initialize B
        if not self.autonomous:
            if self.B_init is None:
                self.B_ = nn.Parameter(torch.randn(nx, nu, dtype=torch.float64), True)
            else:
                self.B_ = nn.Parameter(torch.tensor(self.B_init, dtype=torch.float64), requires_grad=self.learn_B)
        else:
            self.B_ = torch.zeros(nx, nu, device=self.device, dtype=torch.float64)
        self.mask_B = torch.tensor(self.mask_B, device=self.device, dtype=torch.float64) if self.mask_B is not None else None
        
        # Initilize C
        if self.input_output:
            if self.C_init is None:
                self.C_ = nn.Parameter(torch.randn(ny, nx, dtype=torch.float64), True)
            else:
                self.C_ = nn.Parameter(torch.tensor(self.C_init, dtype=torch.float64), requires_grad=self.learn_C)
        else:
            self.C_ = torch.zeros(ny, nx, device=self.device, dtype=torch.float64)
        self.mask_C = torch.tensor(self.mask_C, device=self.device, dtype=torch.float64) if self.mask_C is not None else None

        # Initilize D
        if self.id_D and self.input_output:
            if self.D_init is None:
                self.D_ = nn.Parameter(torch.randn(ny, nu, dtype=torch.float64), True)
            else:
                self.D_ = nn.Parameter(torch.tensor(self.D_init, dtype=torch.float64), requires_grad=self.learn_D)
        else:
            self.D_ = torch.zeros(ny, nu, device=self.device, dtype=torch.float64 )
        self.mask_D = torch.tensor(self.mask_D, device=self.device, dtype=torch.float64) if self.mask_D is not None else None
            
        # Initilize x0 if required
        if self.learn_x0:
            self.x0 =  nn.Parameter(torch.zeros(nx,1, dtype=torch.float64), True)
        else:
            self.x0 = None
        if self.learn_x0_from_y0:
            self.map = nn.Sequential(nn.Linear(ny, nx, dtype=torch.float64))

        self.to(self.device)

    @property
    def A(self):
        if self.mask_A is not None:
            if self.stable_A:
                if self.LMI_A:
                    S = self.X.T.clone() @ self.X.clone() + self.tol_A * torch.eye(2*self.nx, requires_grad=False, device=self.device)
                    P = S[:self.nx, :self.nx]
                    G = 0.5 * (S[self.nx:, self.nx:] + P) + self.V - self.V.T
                    A_ = S[:self.nx,self.nx:] @ torch.inverse(self.N * G)
                    masked_A =  self.mask_A * A_
                elif self.naive_A:
                    masked_A = self.mask_A * self.A_
                    masked_A = masked_A / torch.abs(torch.linalg.eigvals(masked_A)).max() * torch.sigmoid(self.max_eig_param) * self.max_eigenvalue
                else:
                    raise ValueError('Either LMI_A or naive_A has to be set to True in the masked case.')
            else:
                if self.delta is None:
                    masked_A = self.mask_A * self.A_
                else:
                    masked_A = torch.eye(self.nx, requires_grad=False, device=self.device) + self.delta * self.mask_A * self.A_
            return masked_A
        else:
            if self.stable_A:
                if self.LMI_A:
                    S = self.X.T.clone() @ self.X.clone() + self.tol_A * torch.eye(2*self.nx, requires_grad=False, device=self.device)
                    if self.delta is None:
                        P = S[:self.nx, :self.nx] / self.max_eigenvalue
                        E = 0.5 * (P / self.max_eigenvalue + S[self.nx:, self.nx:]) + self.V - self.V.T
                        A_ = S[:self.nx, self.nx:] @ torch.inverse(E)
                    else:
                        A_ = torch.eye(self.nx, requires_grad=False, device=self.device) - 2 * torch.inverse(S[:self.nx,:self.nx] + self.V - self.V.T) @ S[:self.nx,self.nx:] @ torch.inverse(S[self.nx:,self.nx:]) @ S[self.nx:,:self.nx]
                elif self.naive_A:
                    A_ = self.A_ / torch.abs(torch.linalg.eigvals(self.A_)).max() * torch.sigmoid(self.max_eig_param) * self.max_eigenvalue
                else:
                    M_ = 1 - 0.1*torch.sigmoid(self.M.clone())
                    A_ = self.A_.clone()
                    for row in range(A_.shape[0]):
                        sum_ = torch.sum(torch.exp(A_[row, :]))
                        A_[row, :] = torch.exp(A_[row, :]) / sum_ * M_[row, :]
            else:
                if self.delta is None:
                    A_ = self.A_
                else:
                    A_ = torch.eye(self.nx, requires_grad=False, device=self.device) + self.delta * self.A_
            return A_

    @property
    def B(self):
        if self.mask_B is not None:
            if self.delta is None:
                return self.mask_B * self.B_
            else:
                return self.delta * self.mask_B * self.B_
        else:
            if self.delta is None:
                return self.B_
            else:
                return self.delta * self.B_

    @property
    def C(self):
        if self.mask_C is not None:
            return self.mask_C * self.C_
        else:
            return self.C_

    @property
    def D(self):
        if self.mask_D is not None:
            return self.mask_D * self.D_
        else:
            return self.D_
                
    def forward(self, u_, x0=None, y0=None, ms_train=False):

        u = u_.clone()
        if len(u.shape) == 1:
            u = u.unsqueeze(0)
            u = u.unsqueeze(-1)
        elif len(u.shape) == 2:
            u = u.unsqueeze(0)

        A = self.A.clone()            
        A = torch.stack([A.clone() for _ in range(u.shape[0])], dim=0)
        
        if not self.autonomous:
            B = self.B.clone()
            B = torch.stack([B.clone() for _ in range(u.shape[0])], dim=0)
        if self.input_output:
            C = self.C.clone()
            C = torch.stack([C.clone() for _ in range(u.shape[0])], dim=0)
        if self.id_D:
            D = self.D.clone()
            D = torch.stack([D.clone() for _ in range(u.shape[0])], dim=0)
                
        if self.learn_x0:
            if self.learn_x0_from_y0:
                assert y0 is not None, 'If x0 is learned from y0, an initial point must be given to the forward pass'
                C_ = torch.linalg.pinv(self.C)
                C_inv = torch.stack([C_.clone() for _ in range(u_.shape[0])], dim=0) 
                if self.id_D:
                    x = torch.bmm(C_inv, y0.unsqueeze(-1) - torch.bmm(D, u[:,[0],:].permute(0,2,1).clone()))
                else:
                    x = torch.bmm(C_inv, y0.unsqueeze(-1))
            else:
                x = torch.stack([self.x0.clone() for _ in range(u_.shape[0])], dim=0)
        else:
            assert x0 is not None, 'If x0 is not learned, an initial point must be given to the forward pass'
            x = x0.clone().permute(0,2,1)
        if (self.ms_horizon is not None) and ms_train:
            x = torch.cat([x[[0],:,:], self.ms_x0.clone().permute(0,2,1)], dim=0)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        # First prediction
        if self.input_output:
            predictions = torch.empty((u.shape[0], u.shape[1], self.ny), dtype=torch.float64).to(self.device)
            if self.id_D:
                predictions[:,0,:] = (torch.bmm(C, x.clone()) + torch.bmm(D, u[:,0,:].unsqueeze(-1))).squeeze(-1).clone()
            else:
                predictions[:,0,:] = torch.bmm(C, x.clone()).squeeze(-1).clone()
        else:
            predictions = torch.empty((u.shape[0], u.shape[1], self.nx), dtype=torch.float64).to(self.device)
            predictions[:,0,:] = x.squeeze(-1).clone()
                    
        for t in range(u.shape[1]-1):
            if self.autonomous:
                x = torch.bmm(A, x)
            else:
                x = torch.bmm(A, x) + torch.bmm(B, u[:,t,:].unsqueeze(-1))
            if self.input_output:
                if self.id_D:
                    predictions[:,t+1,:] = (torch.bmm(C, x.clone()) + torch.bmm(D, u[:,t+1,:].unsqueeze(-1))).squeeze(-1).clone()
                else:
                    predictions[:,t+1,:] = torch.bmm(C, x.clone()).squeeze(-1).clone()
            else:
                predictions[:,t+1,:] = x.squeeze(-1).clone()
            
        return predictions, x.permute(0,2,1)


class SIMBaWrapper(SIMBa):

    def __init__(self, nx, nu, ny, parameters=base_parameters):
        
        # Check and store the parameters
        parameters = check_parameters(parameters)
        self.params = parameters
        
        # Data handling: multiple shooting
        self.ms_horizon = parameters['ms_horizon']
        self.base_lambda = parameters['base_lambda']
        # Data handling: breaking input-to-state trajectories
        self.horizon = parameters['horizon']
        self.stride = parameters['stride']
        self.horizon_val = parameters['horizon_val']
        self.stride_val = parameters['stride_val']
        # Initialization parameters
        self.init_from_matlab_or_ls = parameters['init_from_matlab_or_ls']
        self.init_lr = parameters['init_learning_rate']
        self.init_loss = parameters['init_loss']
        self.init_epochs = parameters['init_epochs']
        self.init_grad_clip = parameters['init_grad_clip']
        self.init_print_each = parameters['init_print_each']
        # Optimization parameters
        self.lr = parameters['learning_rate']
        self.batch_size = parameters['batch_size']
        self.max_epochs = parameters['max_epochs']
        self.train_loss = parameters['train_loss']
        self.val_loss = parameters['val_loss']
        self.dropout = parameters['dropout']
        self.grad_clip = parameters['grad_clip']
        self.shuffle = parameters['shuffle']
        # To use normalized data (Not advised)
        self.norm = parameters['normalize_data']
        # Prints and others
        self.verbose = parameters['verbose']
        self.print_each = parameters['print_each']
        self.return_best = parameters['return_best']
        self.device = parameters['device']

        super().__init__(nx=nx, nu=nu, ny=ny, parameters=parameters)

        self.init_optimizer = optim.AdamW(self.parameters(), lr=self.init_lr)
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        self.init_times = []
        self.times = []
        self.init_losses = []
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.norm_losses = []
        self.norm_losses_val = []
        self.norm_losses_test = []
        self.ms_losses = []
        self.auto_fit = False
        self.data = None
        self.is_initialized = False

    def prepare_data(self, U=None, U_val=None, U_test=None, X=None, X_val=None, X_test=None, Y=None, Y_val=None, Y_test=None, x0=None, x0_val=None, x0_test=None):

        if self.ms_horizon is not None:
            assert U.shape[0] == 1, 'Current implementation only accepts a single trajectory for multiple shooting.'

            U = self.data_to_multiple_shooting(U)
            X = self.data_to_multiple_shooting(X) if X is not None else None
            Y = self.data_to_multiple_shooting(Y) if Y is not None else None
            
            # Can only use all the data at each iteration
            self.batch_size = U.shape[0]
            self.ms_x0 = nn.Parameter(torch.randn(U.shape[0]-1, 1, self.nx, dtype=torch.float64), True).to(self.device)

        if self.horizon is not None:
            U = break_trajectories(U, horizon=self.horizon, stride=self.stride)
            X = break_trajectories(X, horizon=self.horizon, stride=self.stride)
            Y = break_trajectories(Y, horizon=self.horizon, stride=self.stride)
            x0 = X[:,[0],:]
        
        if self.horizon_val is not None:
            U_val = break_trajectories(U_val, horizon=self.horizon_val, stride=self.stride_val)
            X_val = break_trajectories(X_val, horizon=self.horizon_val, stride=self.stride_val)
            Y_val = break_trajectories(Y_val, horizon=self.horizon_val, stride=self.stride_val)
            x0_val = X_val[:,[0],:]
            if U_test is not None:
                U_test = break_trajectories(U_test, horizon=self.horizon_val, stride=self.stride_val)
                X_test = break_trajectories(X_test, horizon=self.horizon_val, stride=self.stride_val)
                Y_test = break_trajectories(Y_test, horizon=self.horizon_val, stride=self.stride_val)
                x0_test = X_test[:,[0],:]

        self.data = (U, U_val, U_test, X, X_val, X_test, Y, Y_val, Y_test, x0, x0_val, x0_test)

        return U, U_val, U_test, X, X_val, X_test, Y, Y_val, Y_test, x0, x0_val, x0_test
                                
    def batch_iterator(self, U, batch_size: int = None, shuffle: bool = True) -> None:

        if batch_size is None:
            batch_size = self.batch_size
            
        indices = np.arange(U.shape[0])
            
        # Shuffle them if wanted
        if shuffle:
            np.random.shuffle(indices)

        # Define the right number of batches according to the wanted batch_size - taking care of the
        # special case where the indicies ae exactly divisible by the batch size, which can induce
        # an additional empty batch breaking the simulation down the line
        n_batches = int(np.ceil(len(indices) / batch_size))

        # Iterate to yield the right batches with the wanted size
        for batch in range(n_batches):
            yield indices[batch * batch_size: (batch + 1) * batch_size]
            
    def build_data(self, U, X=None, Y=None, x0=None, indices=None):
    
        if isinstance(U, np.ndarray):
            U, X, Y, x0 = make_tensors(U, X, Y, x0, device=self.device)
        if len(U.shape) < 3:
            U = put_in_batch_form(U, 'U', self.verbose)
            X = put_in_batch_form(X, 'X', self.verbose)
            Y = put_in_batch_form(Y, 'Y', self.verbose)
            x0 = put_in_batch_form(x0, 'x0', self.verbose)

        if indices is None:
            indices = np.arange(U.shape[0])
                    
        u = U[indices, :, :]

        if self.input_output:
            assert Y is not None, 'Y must be provided for input output identification'
            y = Y[indices, :, :]
            if not self.learn_x0:
                y0 = None
                assert x0 is not None, 'x0 must be provided if not learned'
                if self.ms_horizon is not None:
                    x0 = x0[[0],:,:]
                else:
                    x0 = x0[indices, :, :]
            else:
                x0 = None
                if self.learn_x0_from_y0:
                    y0 = Y[indices, 0, :]
                    if len(y0.shape) == 1:
                        y0 = y0.unsqueeze(0)
                else:
                    y0 = None
            return u, y, x0, y0
                
        else:
            assert X is not None, 'X must be provided for input state identification'
            x = X[indices, :, :]
            if (not self.learn_x0) or self.autonomous:
                x0 = X[indices, [0], :].unsqueeze(1)
                if self.ms_horizon is not None:
                    x0 = x0[[0],:,:]
            else:
                x0 = None
            return u, x, x0, None

    def normalize_all(self, U, U_val, U_test, X, X_val, X_test, Y, Y_val, Y_test, x0, x0_val, x0_test):

        norm_U, self.min_U, self.diff_U = normalize(U) if U is not None else (None, None, None)
        norm_U_val = normalize(U_val, self.min_U, self.diff_U) if U_val is not None else None
        norm_U_test = normalize(U_test, self.min_U, self.diff_U) if U_test is not None else None
        norm_X, self.min_X, self.diff_X = normalize(X) if X is not None else (None, None, None)
        norm_X_val = normalize(X_val, self.min_X, self.diff_X) if X_val is not None else None
        norm_X_test = normalize(X_test, self.min_X, self.diff_X) if X_test is not None else None
        norm_Y, self.min_Y, self.diff_Y = normalize(Y) if Y is not None else (None, None, None)
        norm_Y_val = normalize(Y_val, self.min_Y, self.diff_Y) if Y_val is not None else None
        norm_Y_test = normalize(Y_test, self.min_Y, self.diff_Y) if Y_test is not None else None
        if X is None:
            norm_x0, self.min_X, self.diff_X = normalize(x0) if x0 is not None else None
            norm_x0_val, self.min_X, self.diff_X = normalize(x0_val) if x0_val is not None else None
            norm_x0_test, self.min_X, self.diff_X = normalize(x0_test) if x0_test is not None else None
        else:
            norm_x0 = normalize(x0, self.min_X, self.diff_X) if x0 is not None else None
            norm_x0_val = normalize(x0_val, self.min_X, self.diff_X) if x0_val is not None else None
            norm_x0_test = normalize(x0_test, self.min_X, self.diff_X) if x0_test is not None else None
        return norm_U, norm_U_val, norm_U_test, norm_X, norm_X_val, norm_X_test, norm_Y, norm_Y_val, norm_Y_test, norm_x0, norm_x0_val, norm_x0_test
    
    def data_to_multiple_shooting(self, data):
        if data.shape[1] % self.ms_horizon == 0:
            return torch.stack([data[0, i:i+self.ms_horizon, :] for i in range(0, data.shape[1], self.ms_horizon)], dim=0).to(self.device) 
        else:
            if data.shape[1] > self.ms_horizon:
                return torch.stack([data[0, i:i+self.ms_horizon, :] for i in range(0, data.shape[1] - self.ms_horizon, self.ms_horizon)], dim=0).to(self.device)
            else:
                return torch.stack([data[0, i:i+self.ms_horizon, :] for i in range(0, data.shape[1], self.ms_horizon)], dim=0).to(self.device)

    def _lambda(self, epoch):
        return self.base_lambda #* (epoch // 1000 + 1)

    def norm_loss(self, predictions, y, val=False):
        if self.input_output:
            predictions = inverse_normalize(predictions, self.min_Y, self.diff_Y)
            y = inverse_normalize(y, self.min_Y, self.diff_Y)
        else:
            predictions = inverse_normalize(predictions, self.min_X, self.diff_X)
            y = inverse_normalize(y, self.min_X, self.diff_X)
        if val:
            return self.val_loss(predictions, y).item()
        else:
            return self.train_loss(predictions, y).item()
        
    def fit_A(self, A):

        if not isinstance(A, torch.DoubleTensor):
            A = torch.tensor(A, dtype=torch.float64, device=self.device)

        # The exact inililization only works when A is not sparse (since the A provided
        # by classical methods will be full)
        if self.naive_A and (self.mask_A is None):
            if self.max_eigenvalue > torch.abs(torch.linalg.eigvals(A)).max():
                self.A_ = nn.Parameter(torch.tensor(A, dtype=torch.float64, device=self.device), requires_grad=self.learn_A)
                self.max_eig_param = nn.Parameter(torch.logit(torch.abs(torch.linalg.eigvals(A)).max() / self.max_eigenvalue), True)
                print(f"\nInitilized A exactly!")

        else:
            if (self.verbose > 0) and not self.auto_fit:
                print(f"\nInitilization starts, fitting A!")
                print('Epoch\tFitting loss')

            self.best_init_loss = np.inf
            with elapsed_timer() as init_elapsed:
                for epoch in range(self.init_epochs):

                    loss = self.init_loss(self.A, A)
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.parameters(), self.init_grad_clip)
                    self.init_optimizer.step()

                    self.init_losses.append(float(loss))
                    if (self.verbose > 0) and ((epoch % self.init_print_each == self.init_print_each-1) or (epoch==0)):
                        print(f'{epoch + 1}\t{self.init_losses[-1]:.2E}')

                    if self.init_losses[-1] < self.best_init_loss:
                        self.best_init_loss = self.init_losses[-1]
                        self.save_state()

                # Timing information
                self.init_times.append(init_elapsed())

            if (self.verbose > 0) and not self.auto_fit:
                print(f"Total initialization time:\t{format_elapsed_time(self.init_times[-1])}")
                print(f"Best loss at epoch {np.argmin(self.init_losses)}:\t{np.min(self.init_losses):.2E}")
        self.overwrite_best_performance()
        
    def initialize(self, U, U_val, U_test, X, X_val, X_test, Y, Y_val, Y_test, x0, x0_val, x0_test, baselines_to_use=None):

        # Run least squares
        if not self.input_output:
            Y_ls = torch.cat([X[i,1:,:] for i in range(len(X))], axis=0).T
            X_ls = torch.cat([X[i,:-1,:] for i in range(len(X))], axis=0).T
            if not self.autonomous:
                U_ls = torch.cat([U[i,:-1,:] for i in range(len(U))], axis=0).T
                AB_ls = Y_ls @ torch.linalg.pinv(torch.cat([X_ls, U_ls], axis=0))
                A = AB_ls[:,:self.nx]
                if self.B_init is None:
                    self.B_ = nn.Parameter(AB_ls[:,self.nx:].clone().detach().requires_grad_(self.learn_B).to(self.device))
            else:
                A = Y_ls @ torch.linalg.pinv(X_ls)

        # Run matlab or SIPPY
        else:
            val_err = np.inf
            if IS_MATLAB:
                alg = 'Matlab'
                matrices, names, _, train_ids, validation_ids, test_ids = matlab_baselines(path_to_matlab=self.params['path_to_matlab'],
                                                                                names=[], times=[], train_ids=[], validation_ids=[],
                                                                                nx=self.nx, U=U, U_val=U_val, Y=Y, Y_val=Y_val, x0=x0,
                                                                                U_test=U_test, Y_test=Y_test, test_ids=[],
                                                                                dt=None, stable_A=self.stable_A, learn_x0=self.learn_x0)
                val_err = np.min([self.val_loss(torch.tensor(val_id, dtype=torch.float64, device=self.device), Y_val[0,:,:]) for val_id in validation_ids[1:]])
                totake = np.argmin([self.val_loss(torch.tensor(val_id, dtype=torch.float64, device=self.device), Y_val[0,:,:]) for val_id in validation_ids[1:]]) + 1
                train_id = torch.tensor(train_ids[totake], dtype=torch.float64, device=self.device)
                validation_id = torch.tensor(validation_ids[totake], dtype=torch.float64, device=self.device)
                test_id = torch.tensor(test_ids[totake], dtype=torch.float64, device=self.device)

            if IS_SIPPY:
                if baselines_to_use is None:
                    from simba.parameters import baselines_to_use
                baselines = {k:False if (('arm' in k) or ('arx' in k) or ('oe_' in k) or ('bj' in k) or ('gen' in k)) else v for k,v in baselines_to_use.items()}
                try:
                    names, ss, _, train_ids, validation_ids, test_ids = identify_baselines(nx=self.nx, U=U, U_val=U_val, U_test=U_test, Y=Y, Y_val=Y_val, Y_test=Y_test,
                                                                                        x0=x0, x0_val=x0_val, x0_test=x0_test, dt=1,
                                                                                        parameters=self.params, baselines_to_use=baselines, id_mat=False)
                except:
                    baselines['parsim_s'] = False
                    baselines['parsim_p'] = False
                    names, ss, _, train_ids, validation_ids, test_ids = identify_baselines(nx=self.nx, U=U, U_val=U_val, U_test=U_test, Y=Y, Y_val=Y_val, Y_test=Y_test,
                                                                                        x0=x0, x0_val=x0_val, x0_test=x0_test, dt=1,
                                                                                        parameters=self.params, baselines_to_use=baselines, id_mat=False)
                if np.min([self.val_loss(torch.tensor(val_id, dtype=torch.float64, device=self.device), Y_val[0,:,:]) for val_id in validation_ids]) < val_err:
                    totake = np.argmin([self.val_loss(torch.tensor(val_id, dtype=torch.float64, device=self.device), Y_val[0,:,:]) for val_id in validation_ids])
                    matrices = [ss[totake].A, ss[totake].B, ss[totake].C, ss[totake].D]
                    alg = f'SIPPY-{names[totake]}'
                    train_id = torch.tensor(train_ids[totake], dtype=torch.float64, device=self.device)
                    validation_id = torch.tensor(validation_ids[totake], dtype=torch.float64, device=self.device)
                    test_id = torch.tensor(test_ids[totake], dtype=torch.float64, device=self.device)

            A = matrices[0]
            if self.B_init is None:
                self.B_ = nn.Parameter(torch.tensor(matrices[1], dtype=torch.float64, device=self.device), requires_grad=self.learn_B)
            if self.C_init is None:
                self.C_ = nn.Parameter(torch.tensor(matrices[2], dtype=torch.float64, device=self.device), requires_grad=self.learn_C)
            if self.D_init is None:
                self.D_ = nn.Parameter(torch.tensor(matrices[3], dtype=torch.float64, device=self.device), requires_grad=self.learn_D)
            if self.learn_x0 and (alg == 'Matlab'):
                self.x0 =  nn.Parameter(torch.tensor(matrices[4], dtype=torch.float64, device=self.device), True)

        # Fit A
        self.fit_A(A)

        if (self.verbose > 0) and not self.auto_fit and self.input_output:
            print(f"\n{alg} performance (Train and validation are only measured on the\nfirst trajectory if there are several for now):\nTrain loss\tVal loss\tTest loss")
            print(f"{self.train_loss(train_id,Y[0,:,:]):.2E}\t{self.val_loss(validation_id, Y_val[0,:,:]):.2E}\t{self.val_loss(test_id, Y_test):.2E}")

        self.is_initialized = True

    def fit(self, U=None, U_val=None, U_test=None, X=None, X_val=None, X_test=None, Y=None, Y_val=None, Y_test=None, x0=None, x0_val=None, x0_test=None, baselines_to_use=None):
        
        U, U_val, U_test, X, X_val, X_test, Y, Y_val, Y_test, x0, x0_val, x0_test = check_and_initialize_data(U, U_val, U_test, X, X_val, X_test, Y, Y_val, Y_test, x0, x0_val, x0_test,
                                                                                                            verbose=self.verbose, autonomous=self.autonomous, 
                                                                                                            input_output=self.input_output, device=self.device)

        if self.norm:
            U, U_val, U_test, X, X_val, X_test, Y, Y_val, Y_test, x0, x0_val, x0_test = self.normalize_all(U, U_val, U_test, X, X_val, X_test, Y, Y_val, Y_test, x0, x0_val, x0_test )

        if not self.is_initialized and self.init_from_matlab_or_ls:
            self.initialize(U, U_val, U_test, X, X_val, X_test, Y, Y_val, Y_test, x0, x0_val, x0_test, baselines_to_use=baselines_to_use)
            self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        elif self.stable_A and self.A_init is not None:
            self.fit_A(self.A_init)
            
        U, U_val, U_test, X, X_val, X_test, Y, Y_val, Y_test, x0, x0_val, x0_test = self.prepare_data(U, U_val, U_test, X, X_val, X_test, Y, Y_val, Y_test, x0, x0_val, x0_test)

        if (self.verbose > 0) and not self.auto_fit:
            print(f"\nTraining of SIMBa starts!\nTraining data shape:\t({U.shape[0]}, {U.shape[1]}, *)\nValidation data shape:\t({U_val.shape[0]}, {U_val.shape[1]}, *)\nTest data shape:\t({U_test.shape[0]}, {U_test.shape[1]}, *)\n")
        
        if len(self.val_losses) > 0:
            self.best_loss = np.min(self.val_losses)
        else:
            self.best_loss = np.inf

        exploded = False
        
        with elapsed_timer() as elapsed:
            
            # Assess the number of epochs the model was already trained on to get nice prints
            trained_epochs = len(self.train_losses)

            for epoch in range(trained_epochs, trained_epochs + self.max_epochs):
            
                if (self.verbose > 0) and (epoch == 0):
                    print('Epoch', end='\t')
                    if self.ms_horizon is not None:
                        print('Total loss', end='\t')
                    if self.norm:
                        print('Normed train\tNormed val\tTrain loss\tVal loss')
                    else:
                        print('Train loss\tVal loss\tTest loss')
                
                self.train()
                train_losses = []
                train_sizes = []
                norm_losses = []
                ms_losses = []
                for indices in self.batch_iterator(U=U, shuffle=self.shuffle):

                    self.optimizer.zero_grad()
                    
                    # Compute the loss of the batch and store it
                    batch_u, batch_data, batch_x0, batch_y0 = self.build_data(U, X, Y, x0, indices)
                    mask = ~torch.isnan(batch_data)
                    drop = torch.rand(mask.shape, device=self.device) > self.dropout
                    predictions, x_final = self.forward(x0=batch_x0, u_=batch_u, y0=batch_y0, ms_train=True)
                    loss = self.train_loss(predictions * mask * drop, batch_data * mask * drop)
                    if self.ms_horizon is not None:
                        loss = loss + self._lambda(epoch) * self.train_loss(self.ms_x0, x_final[:-1,:,:])
                    
                    # Compute the gradients and take one step using the optimizer
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.parameters(), self.grad_clip)
                    self.optimizer.step()
                    
                    train_sizes.append(len(batch_u))
                    if self.norm:
                        with torch.no_grad():
                            norm_losses.append(self.norm_loss(predictions * mask * drop, batch_data * mask * drop))
                    if self.ms_horizon is not None:
                        with torch.no_grad():
                            ms_losses.append(float(loss))
                            train_losses.append(float(self.train_loss(predictions * mask * drop, batch_data * mask * drop)))
                    else:
                        train_losses.append(float(loss))

                self.train_losses.append(sum([l*s for l,s in zip(train_losses, train_sizes)]) / sum(train_sizes))
                if self.norm:
                    self.norm_losses.append(sum([l*s for l,s in zip(norm_losses, train_sizes)]) / sum(train_sizes))
                if self.ms_horizon is not None:
                    self.ms_losses.append(sum([l*s for l,s in zip(ms_losses, train_sizes)]) / sum(train_sizes))
                
                self.eval()
                val_losses = []
                val_sizes = []
                norm_losses_val = []
                for indices in self.batch_iterator(U=U_val, shuffle=False):
                
                    batch_u, batch_data, batch_x0, batch_y0 = self.build_data(U_val, X_val, Y_val, x0_val, indices)
                    val_mask = ~torch.isnan(batch_data)
                    predictions, _ = self.forward(x0=batch_x0, u_=batch_u, y0=batch_y0, ms_train=False)
                    loss = self.val_loss(predictions * val_mask, batch_data * val_mask)
                    
                    val_losses.append(float(loss))
                    val_sizes.append(len(batch_u))
                    if self.norm:
                        with torch.no_grad():
                            norm_losses_val.append(self.norm_loss(predictions * val_mask, batch_data * val_mask, val=True))
                    
                self.val_losses.append(sum([l*s for l,s in zip(val_losses, val_sizes)]) / sum(val_sizes))
                if self.norm:
                    self.norm_losses_val.append(sum([l*s for l,s in zip(norm_losses_val, val_sizes)]) / sum(val_sizes))
                
                if Y_test is not None:
                    test_losses = []
                    test_sizes = []
                    norm_losses_test = []
                    for indices in self.batch_iterator(U=U_test, shuffle=False):
                    
                        batch_u, batch_data, batch_x0, batch_y0 = self.build_data(U_test, X_test, Y_test, x0_test, indices)
                        test_mask = ~torch.isnan(batch_data)
                        predictions, _ = self.forward(x0=batch_x0, u_=batch_u, y0=batch_y0, ms_train=False)
                        loss = self.val_loss(predictions * test_mask, batch_data * test_mask)
                        
                        test_losses.append(float(loss))
                        test_sizes.append(len(batch_u))
                        if self.norm:
                            with torch.no_grad():
                                norm_losses_test.append(self.norm_loss(predictions * test_mask, batch_data * test_mask, val=True))
                        
                    self.test_losses.append(sum([l*s for l,s in zip(test_losses, test_sizes)]) / sum(test_sizes))
                    if self.norm:
                        self.norm_losses_test.append(sum([l*s for l,s in zip(norm_losses_test, test_sizes)]) / sum(test_sizes))

                # Compute the average loss of the training epoch and print it
                if (self.verbose > 0) and ((epoch % self.print_each == self.print_each-1) or (epoch==0)):
                    print(f'{epoch + 1}', end='\t')
                    if self.ms_horizon is not None:
                        print(f"{self.ms_losses[-1]:.2E}", end='\t')
                    if self.norm:
                        print(f"{self.train_losses[-1]:.2E}\t{self.val_losses[-1]:.2E}\t{self.norm_losses[-1]:.2E}\t{self.norm_losses_val[-1]:.2E}\t{self.norm_losses_test[-1]:.2E}")
                    else:
                        print(f"{self.train_losses[-1]:.2E}\t{self.val_losses[-1]:.2E}\t{self.test_losses[-1]:.2E}")
                    
                if self.val_losses[-1] < self.best_loss:
                    self.best_loss = self.val_losses[-1]
                    self.save_state()
                
                elif np.isnan(self.val_losses[-1]):
                    print('\nEplosion! Restarting training\n')
                    self.__init__(self.nx, self.nu, self.ny, self.delta, self.params)
                    self.fit(U=U, U_val=U_val, X=X, X_val=X_val, Y=Y, Y_val=Y_val, x0=x0, x0_val=x0_val)
                    if self.auto_fit:
                        raise ValueError('A very specific bad thing happened.')
                    exploded = True
                    break
                    
                # Timing information
                self.times.append(elapsed())

        if not exploded:
            if self.return_best:
                self.overwrite_best_performance()
                
            if self.verbose > 0:
                if len(self.times) > 100:
                    print(f"\nAverage time per 100 epochs:\t{format_elapsed_time(np.mean(np.array(self.times[100:]) - np.array(self.times[:-100])))}")
                else:
                    print(f"")
                print(f"Total training time:\t\t{format_elapsed_time(self.times[-1])}")
                print(f"\nBest model performance:")
                e = np.argmin(self.val_losses)
                print(f'{e + 1}', end='\t')
                if self.ms_horizon is not None:
                    print(f"{self.ms_losses[e]:.2E}", end='\t')
                if self.norm:
                    print(f"{self.train_losses[e]:.2E}\t{self.val_losses[e]:.2E}\t{self.norm_losses[e]:.2E}\t{self.norm_losses_val[e]:.2E}\t{self.norm_losses_test[e]:.2E}")
                else:
                    print(f"{self.train_losses[e]:.2E}\t{self.val_losses[e]:.2E}\t{self.test_losses[e]:.2E}")
                
            if self.verbose > 1:
                print('\nFirst elements of each matrices')
                print(f'A:\n{self.A[:5,:5].detach().numpy()}')
                print(f'B:\n{self.B[:5,:5].detach().numpy()}')
                if self.input_output:
                    print(f'C:\n{self.C[:5,:5].detach().numpy()}')
                if self.id_D:
                    print(f'D:\n{self.D[:5,:5].detach().numpy()}')

    def save_state(self):
        self.best_state = deepcopy(self.state_dict())
        self.best_optimizer = deepcopy(self.optimizer.state_dict())
            
    def overwrite_best_performance(self):
        self.load_state_dict(self.best_state)
        self.optimizer.load_state_dict(self.best_optimizer)

    def save(self, directory, save_name):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        savename = os.path.join(directory, f'{save_name}.pt')
        torch.save(
            {
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'params': self.params,
                'fit_data': self.data,
                'init_losses': self.init_losses,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'test_losses': self.test_losses,
            },
            savename,
        )
    
    def load(self, directory, save_name):
        # Load the checkpoint
        checkpoint = torch.load(os.path.join(directory, f'{save_name}.pt'), map_location=lambda storage, loc: storage)

        # Put it into the model
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if torch.cuda.is_available():
            for state in self.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.tensor):
                        state[k] = v.cuda()
            self.to(self.device)
        self.loaded_params = checkpoint['params']
        self.loaded_data = checkpoint['fit_data']
        self.init_losses = checkpoint['init_losses']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.test_losses = checkpoint['test_losses']
        self.is_initialized = True
        
        # Check parameter copliance
        self.check_loaded_run() 

    def check_loaded_run(self):
        for key, value in self.loaded_params.items():
            if isinstance(value, np.ndarray) or isinstance(self.params[key], np.ndarray):
                if (value != self.params[key]).any():
                    print(f"Warning: loaded value for {key}: {value} doesn't correspond to the current one: {self.params[key]}") 
            elif value != self.params[key]:
                print(f"Warning: loaded value for {key}: {value} doesn't correspond to the current one: {self.params[key]}") 
        if self.data is not None:
            for x, y in zip(self.data, self.loaded_data):
                if isinstance(x, np.ndarray):
                    if not np.allclose(x,y):
                        print(f"Warning: the loaded data is different from the one used by the current version of Simba, shaped {y.shape} instead of {x.shape}")
                elif isinstance(x, torch.Tensor):
                    if not torch.allclose(x,y):
                        print(f"Warning: the loaded data is different from the one used by the current version of Simba, shaped {y.shape} instead of {x.shape}")


def Simba(nx, nu, ny, parameters=base_parameters):
    return SIMBaWrapper(nx=nx, nu=nu, ny=ny, parameters=parameters)


def Simba_auto_fit(nx, nu, ny, parameters=base_parameters,
                   U=None, U_val=None, X=None, X_val=None, Y=None, Y_val=None, x0=None, x0_val=None):

    for lr in [2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
        parameters['learning_rate'] = lr
        try:
            print(f'Trying with learning rate {lr:.0E}\n')
            simba =SIMBaWrapper(nx=nx, nu=nu, ny=ny, parameters=parameters)
            simba.auto_fit = True
            simba.fit(U=U, U_val=U_val, X=X, X_val=X_val, Y=Y, Y_val=Y_val, x0=x0, x0_val=x0_val)
            break
        except ValueError:
            continue
            
    return simba