import torch
import torch.nn.functional as F
import numpy as np

try:
    import matlab
    IS_MATLAB = True
except ImportError:
    print('Matlab could not be loaded')
    IS_MATLAB = False

try:
    import sippy
    IS_SIPPY = True
except ImportError:
    print('SIPPY could not be loaded')
    IS_SIPPY = False

base_parameters = dict(
    ## SIMBa core initialization parameters
    # Matrices masks
    mask_A=None, 
    mask_B=None, 
    mask_C=None, 
    mask_D=None,
    # Matricies initilization
    A_init=None, 
    B_init=None, 
    C_init=None, 
    D_init=None, 
    # Which matricies to learn (typically set to True if A_init is known a priori)
    learn_A=True, 
    learn_B=True, 
    learn_C=True, 
    learn_D=True,
    # Which matrices to identify
    autonomous=False, 
    input_output=False, 
    id_D=False, 
    # Stability parameters for the matrix A
    stable_A=True, 
    LMI_A=True, 
    max_eigenvalue=1.-1e-6,              # The maximum eigenvalue for A
    naive_A=False, 
    tol_A=1e-9, 
    # For input-output models, whether to learn_x0 (as a free parameter) of to learn it from y0
    learn_x0=True, 
    learn_x0_from_y0=False,
    # To use proposition 2 (i.e., serach for A close to identity)
    delta=None,
    
    ## Training parameters
    # Data handling: multiple shooting
    ms_horizon=None,
    base_lambda=1,                  # Weighting parameter to balance the fit to the data and the collocation points loss
    # Data handling: breaking input-to-state trajectories
    horizon=None, 
    stride=1, 
    horizon_val=None, 
    stride_val=1,
    # Initialization parameters
    init_from_matlab_or_ls=True,    # Whether to initialize SIMBa close to the solution from matlab (input-output) or Least Squares (state)
    init_learning_rate=0.001,
    init_loss=F.mse_loss,
    init_epochs=150000,
    init_grad_clip=0.1,             # Clips gradients to stabilize training and avoid too large parameter updates
    init_print_each=50000,
    # Optimization parameters
    learning_rate=0.001,
    batch_size=128, 
    shuffle=True,                   # Shuffle training batches
    max_epochs=25000, 
    train_loss=F.mse_loss, 
    val_loss=F.mse_loss,
    dropout=0.2,                    # Probabibility to randomly discard points during training to avoid overfitting
    grad_clip=100,                  # Clips gradients to stabilize training and avoid too large parameter updates
    # If using normalized data (Not advised)
    normalize_data=False, 
    # Prints and others
    verbose=1, 
    print_each=100, 
    return_best=True,               # Overwrites the best performance at the end of training to return the best mode on the validation set
    device=None,                    # If None, it will detect whether a GPU is available and use it
    path_to_matlab='mat',
    )


def check_parameters(parameters):
    # Check that we only don't learn matrices when an initialization is provided (otherwise would be a random one)
    if not parameters['learn_A']:
        assert parameters['A_init'] is not None, 'learn_A is set to False but no matrix A is given as A_init.'
    if not parameters['learn_B']:
        assert parameters['B_init'] is not None, 'learn_B is set to False but no matrix B is given as B_init.'
    if not parameters['learn_C']:
        assert parameters['C_init'] is not None, 'learn_C is set to False but no matrix C is given as C_init.'
    if not parameters['learn_D']:
        assert parameters['D_init'] is not None, 'learn_D is set to False but no matrix D is given as D_init.'
    
    # Check that only the system is either autonomous, input-to-state, or input-output
    if parameters['id_D'] and (not parameters['input_output']):
        print('Warning: Cannot identify "D" on input-to-state or autonomous models. Set input_output=True if that the desired behavior.')
    if not parameters['input_output']:
        if parameters['C_init'] is not None:
            print('Warning, the system is not input-output but C_init has been provided. It will not be used. Set input_output=True if that the desired behavior.')
        if parameters['mask_C'] is not None:
            print('Warning, the system is not input-output but mask_C has been provided. It will not be used. Set input_output=True if that the desired behavior.')
        if parameters['D_init'] is not None:
            print('Warning, the system is not input-output but D_init has been provided. It will not be used. Set input_output=True if that the desired behavior.')
        if parameters['mask_D'] is not None:
            print('Warning, the system is not input-output but mask_D has been provided. It will not be used. Set input_output=True if that the desired behavior.')
    if parameters['autonomous']:
        if parameters['B_init'] is not None:
            print('Warning, the system is autonomous but B_init has been provided. It will not be used. Set autonomous=False if that the desired behavior.')
        if parameters['mask_B'] is not None:
            print('Warning, the system is autonomous but mask_B has been provided. It will not be used. Set autonomous=False if that the desired behavior.')

    # Check which stability criterion to use
    if parameters['stable_A']:
        assert not (parameters['LMI_A'] and parameters['naive_A']), 'Only one of the LMI or naive approach can be used to stabilize A, modify either naive_A or LMI_A.'
    else:
        if parameters['LMI_A'] or parameters['naive_A']:
            print('Warning, stable_A=False but naive_A or LMI_A is True, they will not be use.')
    assert -1e-6 < parameters['max_eigenvalue'] < 1 + 1e-6, f'The maximum eigenvalue of A, {parameters["max_eigenvalue"]} is greater than 1, which means A can be unstable.'
    
    # Check how x0 should be learned (if at all)
    if parameters['learn_x0']:
        if not parameters['input_output']:
            print('Warning, the system is not input-output but learn_x0=True, this will not be used. Set input_output=True if that the desired behavior.')
    if parameters['learn_x0_from_y0']:
        if not parameters['input_output']:
            print('Warning, the system is not input-output but learn_x0_from_y0=True, this will not be used. Set input_output=True if that the desired behavior.')
        assert parameters['learn_x0'], 'learn_x0_from_y0=True requires learn_x0=True for code compliance.'

    # Check multiple shooting parameters
    if parameters['ms_horizon'] is not None:
        assert parameters['input_output'], 'Multiple shooting is only used for input-output data. For input-to-state, you can use the parameters horizon, horizon_val, stride, and stride_val to break the given trajectories into smaller potentially overlapping pieces.'
    # Check breaking trajectories
    if (parameters['horizon'] is not None) or (parameters['horizon_val'] is not None):
        assert not parameters['input_output'], 'Breaking trajectories cannot be used for input-output data but horizon or horizon_val was provided - but multiple shooting can, through the parameter ms_horizon.'

    # Normalizing data is not robust
    if parameters['normalize_data']:
        print('\n______________\nWarning!\nThis is a temporary test feature.\nThe normalization is currently ONLY applied during training - predictions with simba.forward() after training will fail. If you want to use normalized data, it is better to handle the normalization outside of Simba, before feeding the data to the algorithm, so it always works with normalized data.\n______________\n')

    # Check initialization
    if parameters['init_from_matlab_or_ls']:
        assert not (isinstance(parameters['A_init'], np.ndarray)), 'init_from_matlab_or_ls is set to True, you cannot pass custom A initializations.'
        assert IS_MATLAB or IS_SIPPY, 'Either MATLAB or SIPPY is required to initialize SIMBa from one of their solution'

    # Check training parameters
    assert -1e-5 < parameters['dropout'] < 1., f'dropout is understood as a probability, it has to be between 0 and 1, not {parameters["dropout"]}.'

    # Check if loss is given as a string
    for loss in ['train_loss', 'val_loss', 'init_loss']:
        if isinstance(parameters[loss], str):
            if parameters[loss] == 'mse':
                parameters[loss] = F.mse_loss
            elif parameters[loss] == 'mae':
                parameters[loss] = F.l1_loss

    # Detect if there is a GPU
    if parameters['device'] is None:
        if torch.cuda.is_available():
            parameters['device'] = torch.device("cuda:0")
            print("GPU acceleration on!")
        else:
            parameters['device'] = "cpu" 

    return parameters       


def check_sizes(nx, nu, ny, parameters):
    # Check that initilizations and masks have the right sizes
    if parameters['A_init'] is not None:
        assert parameters['A_init'].shape == (nx, nx), f"A_init has the wrong shape {parameters['A_init'].shape} but nx = {nx}."
    if parameters['B_init'] is not None:
        assert parameters['B_init'].shape == (nx, nu), f"B_init has the wrong shape {parameters['B_init'].shape} but (nx, nu) = ({nx},{nu})."
    if parameters['C_init'] is not None:
        assert parameters['C_init'].shape == (ny, nx), f"C_init has the wrong shape {parameters['C_init'].shape} but (ny, nx) = ({ny},{nx})."
    if parameters['D_init'] is not None:
        assert parameters['D_init'].shape == (ny, nu), f"D_init has the wrong shape {parameters['D_init'].shape} but (ny, nu) = ({ny},{nu})."
    
    if parameters['mask_A'] is not None:
        assert parameters['mask_A'].shape == (nx, nx), f"A_init has the wrong shape {parameters['mask_A'].shape} but nx = {nx}."
    if parameters['mask_B'] is not None:
        assert parameters['mask_B'].shape == (nx, nu), f"B_init has the wrong shape {parameters['mask_B'].shape} but (nx, nu) = ({nx},{nu})."
    if parameters['mask_C'] is not None:
        assert parameters['mask_C'].shape == (ny, nx), f"C_init has the wrong shape {parameters['mask_C'].shape} but (ny, nx) = ({ny},{nx})."
    if parameters['mask_D'] is not None:
        assert parameters['mask_D'].shape == (ny, nu), f"D_init has the wrong shape {parameters['mask_D'].shape} but (ny, nu) = ({ny},{nu})."

baselines_to_use = dict(
    ## Toggle baselines to use from sippy
    # ARMAX
    armax_ills=True,
    armax_rlls=True,
    armax_opt=False,
    # ARX
    arx_ills=True,
    arx_rlls=True,
    arx_opt=False,
    # OE
    oe_rlls=True,
    oe_opt=False,
    # others
    bj=False,
    gen=False,
    # SS methods
    n4sid=True,
    moesp=True,
    cva=True,
    # PARSIM 
    parsim_k=True,
    parsim_s=True,
    parsim_p=True
    )