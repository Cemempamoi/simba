# SIMBa
## System Identification Methods with Backpropagation

Loris Di Natale, Muhammad Zakwan, Bratislav Svetozarevic, Philipp Heer, Giancarlo Ferrari Trecate, Colin Jones

## Compatibility with matlab
If matlab is installed on your machine, the code will try to use it.  
It will ask for the `System Identification Toolbox` and `Symbolic Math Toolbox`.  
You can disable the use of matlab by overwriting `IS_MATLAB` in `simba.parameters`

## Known issue
There seems to be a bug in the `control` library when the dimension of the control input is one, which raises an unwanted exception.  
This does not affect SIMBa in general but can fail is `SIPPY` is required (e.g., for SIMBa's initialization).

To correct it, you'll need to go to `timeresp.py` in the library, typically located in your virtual environment at `.venv/lib/python3.x/site-packages/control/` and add the following transpose at line 1002, before `_check_convert_array`:

    if len(U.shape) > 1:
        U = U.T

## Contact

For more information, please contact loris.dinatale@empa.ch.