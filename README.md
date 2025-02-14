# SIMBa
## System Identification Methods with Backpropagation

**SIMBa** (**S**ystem **I**dentification **M**ethods leveraging **B**ackpropagation) is an open-source toolbox leveraging Pytorch's Automatic Differentiation framework for stable state-space linear SysID. It allows the user to incorporate prior knowledge (like sparsity patterns of the state-space matrices) during the identification procedure.

## Intallation
SIMBa is available on pypi, it can be installed with `pip install simbapy`.  
Note that this will NOT install SIPPY to avoid compatibility issues. SIMBa will only try to use Matlab for initilization by default, but you can install SIPPY separately.

Alternatively, you can clone this github repository to use simba locally.

## Compatibility with matlab
If matlab is installed on your machine, you can install `matlabengine`. If you are on the latest version of MATLAB, `pip install matlabengine` works, otherwise you might need to install an older version of matlabengine. See [here](https://pypi.org/project/matlabengine) the supported version of MATLAB.  
SIMBa needs access the `System Identification Toolbox` and `Symbolic Math Toolbox` in MATLAB.  
You can disable the use of matlab by overwriting `IS_MATLAB` in `simba.parameters`

## Project status
SIMBa was first presented in [Stable Linear Subspace Identification: A Machine Learning Approach](https://arxiv.org/pdf/2311.03197.pdf) and subsequently extended in [SIMBa: System Identification Methods leveraging Backpropagation](https://arxiv.org/pdf/2311.13889.pdf).

## Known issue
### For SIPPY users
There seems to be a bug in the `control` library when the dimension of the control input is one, which raises an unwanted exception.  
This does not affect SIMBa in general but can fail is `SIPPY` is required (e.g., for SIMBa's initialization).

To correct it, you'll need to go to `timeresp.py` in the library, typically located in your virtual environment at `.venv/lib/python3.x/site-packages/control/` and add the following transpose at line 1002, before `_check_convert_array`:

    if len(U.shape) > 1:
        U = U.T

## Contact

This project is jointly led by Loris Di Natale and Muhammad Zakwan, with the participation of Bratislav Svetozarevic, Philipp Heer, Giancarlo Ferrari Trecate, and Colin N. Jones.  
Urban Energy Systems Lab, Empa, Switzerland  
Laboratoire d'Automatique, EPFL, Switzerland  

For more information, please contact loris.dinatale@alumni.epfl.ch