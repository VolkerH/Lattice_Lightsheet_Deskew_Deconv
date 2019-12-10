from functools import partial
import numpy as np
import warnings
from typing import Optional, Callable
import logging

try:
    import pycudadecon
except ImportError:
    warnings.warn("Deconvolution backend pycudadecon selected but required dependency was not found.")
    warnings.warn("Please install pycudadecon into your python environment.")
    exit(-1)


logger = logging.getLogger("lls_dd")

def init_rl_deconvolver(**kwargs):
    """init deconvolver: dummy, no initialization necessary for pycudadecon"""
    return None

def get_deconv_function(
    psf: np.ndarray, deconvolver: object, n_iter: int
) -> Callable:
    """generates a deconvolution function with specified psf, deconvolver and number of iterations
    
    Parameters
    ----------
    psf : np.ndarray
        point spread function
    deconvolver : object
        not needed here, just to provide the same interface as for flowdec
    n_iter : int
        number of iterations
    
    Returns
    -------
    Callable
        deconvolution function that simply takes an input volume and returns
        a deconvolved volume
    """
    deconv_func = partial(pycudadecon.decon, psf=psf, background=0, n_iters=n_iter)
    return deconv_func