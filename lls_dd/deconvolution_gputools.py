import gputools.deconv
from functools import partial
import numpy as np
import warnings
from typing import Optional, Callable

#
# This is an attempt to use gputools to deconvolve
# here, I simply use the same function names as for the flowdec stuff for testing.
# I think medium term I should use an abstract class Deconvolver with init and run methods.
# From this abstract class I can derive concrete implemenations as child classes, e.g.
# FlowdecDeconvolver, GPUtoolsDeconvolver and PyCudaDeconDeconvolver
#
# These tests are to see whether it is worth the effort


def init_rl_deconvolver():
    """ dummy, nothing to initialiaze for the gputools deconv
    Note: maybe one can setup and keep the fft plan, this may require
    changes to gputools code.
    """

    return None


def deconv_volume(
    vol: np.ndarray,
    psf: np.ndarray,
    deconvolver: object,
    n_iter: int,
    observer: Optional[Callable] = None,
) -> np.ndarray:
    """ perform RL deconvolution using deconvolver 
    input:
    vol : input volume
    psf : psf (numpy array) 
    deconvolver : see init_rl_deconvolver
    n_iter: number of iterations

    TODO: for gputools
    """

    if deconvolver:
        warnings.warn("deconvolver not required for gputools deconv. None expected")
    if observer:
        warnings.warn("observer not implemented for gputools deconv")

    return gputools.deconv.deconv_rl(vol, psf, Niter=n_iter)


def get_deconv_function(psf: np.ndarray, deconvolver: object, n_iter: int) -> Callable:
    deconv_func = partial(
        deconv_volume, psf=psf, deconvolver=deconvolver, n_iter=n_iter
    )
    return deconv_func
