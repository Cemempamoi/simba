# SIMBa
## System Identification Methods with Backpropagation

**SIMBa** (**S**ystem **I**dentification **M**ethods leveraging **B**ackpropagation) is an open-source toolbox leveraging Pytorch's Automatic Differentiation framework for stable state-space linear SysID. It allows the user to incorporate prior knowledge (like sparsity patterns of the state-space matrices) during the identification procedure.

It was first presented in [Stable Linear Subspace Identification: A Machine Learning Approach](https://arxiv.org/pdf/2311.03197.pdf) and subsequently extended in [SIMBa: System Identification Methods leveraging Backpropagation](https://arxiv.org/pdf/2311.13889.pdf).

## Compatibility with matlab
If matlab is installed on your machine, the code will try to use it.  
It will ask for the `System Identification Toolbox` and `Symbolic Math Toolbox`.  
You can disable the use of matlab by overwriting `IS_MATLAB` in `simba.parameters`

## Known issue
### For SIPPY users
here seems to be a bug in the `control` library when the dimension of the control input is one, which raises an unwanted exception.  
This does not affect SIMBa in general but can fail is `SIPPY` is required (e.g., for SIMBa's initialization).

To correct it, you'll need to go to `timeresp.py` in the library, typically located in your virtual environment at `.venv/lib/python3.x/site-packages/control/` and add the following transpose at line 1002, before `_check_convert_array`:

    if len(U.shape) > 1:
        U = U.T

## Contact

This project is jointly led by Loris Di Natale and Muhammad Zakwan, with the participation of Bratislav Svetozarevic, Philipp Heer, Giancarlo Ferrari Trecate, and Colin N. Jones.  
Urban Energy Systems Lab, Empa, Switzerland  
Laboratoire d'Automatique, EPFL, Switzerland  

For more information, please contact loris.dinatale@alumni.epfl.ch