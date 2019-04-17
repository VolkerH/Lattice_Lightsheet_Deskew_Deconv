from flowdec import restoration as tfd_restoration
from flowdec import data as fd_data
from functools import partial
import tensorflow as tf
import numpy as np
import warnings
from typing import Optional, Callable
import os
import logging

logger = logging.getLogger("lls_dd")
# suppress tensorflow diagnostic output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def init_rl_deconvolver(**kwargs):
    """initializes the tensorflow-based Richardson Lucy Deconvolver """
    return tfd_restoration.RichardsonLucyDeconvolver(n_dims=3, start_mode="input", **kwargs).initialize()


def deconv_volume(
    vol: np.ndarray,
    psf: np.ndarray,
    deconvolver: tfd_restoration.RichardsonLucyDeconvolver,
    n_iter: int,
    observer: Optional[Callable] = None,
) -> np.ndarray:
    """perform RL deconvolution on volume vol using deconvolver
    
    Parameters
    ----------
    vol : np.ndarray
        input volume
    psf : np.ndarray
        point spread function
    deconvolver : tfd_restoration.RichardsonLucyDeconvolver
        see init_rl_deconvolver
    n_iter : int
        number of RL iterations
    observer : Optional[Callable], optional
        NOT YET IMPLEMENTED
        observer callback so that progress updates for each iteration
        can be displayed. Also, add option to save intermediate results within
        a certain range of iterations.(the default is None)
    
    Returns
    -------
    np.ndarray
        deconvolved volume
    """
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    aq = fd_data.Acquisition(data=vol, kernel=psf)
    if observer is not None:
        warnings.warn("Observer function for iteration not yet implemented.")
    result = deconvolver.run(aq, niter=n_iter, session_config=config)
    logger.debug(f"flowdec info: {result.info}")
    return result.data


def get_deconv_function(
    psf: np.ndarray, deconvolver: tfd_restoration.RichardsonLucyDeconvolver, n_iter: int
) -> Callable:
    """generates a deconvolution function with specified psf, deconvolver and number of iterations
    
    Parameters
    ----------
    psf : np.ndarray
        point spread function
    deconvolver : tfd_restoration.RichardsonLucyDeconvolver
        see init_rl_deconvolver
    n_iter : int
        number of iterations
    
    Returns
    -------
    Callable
        deconvolution function that simply takes an input volume and returns
        a deconvolved volume
    """
    deconv_func = partial(deconv_volume, psf=psf, deconvolver=deconvolver, n_iter=n_iter)
    return deconv_func
