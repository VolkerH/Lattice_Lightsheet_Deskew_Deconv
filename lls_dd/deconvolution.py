from flowdec import restoration as tfd_restoration
from flowdec import data as fd_data
from functools import partial
import tensorflow as tf
import numpy as np
import warnings
from typing import Optional, Callable

def init_rl_deconvolver():
    """ initializes the tensorflow-based Richardson Lucy Deconvolver """

    return tfd_restoration.RichardsonLucyDeconvolver(n_dims=3, start_mode="input").initialize()


def deconv_volume(vol: np.ndarray,
                  psf: np.ndarray,
                  deconvolver: tfd_restoration.RichardsonLucyDeconvolver,
                  n_iter: int,
                  observer: Optional[Callable] = None) -> np.ndarray:
    """ perform RL deconvolution using deconvolver 
    input:
    vol : input volume
    psf : psf (numpy array) 
    deconvolver : see init_rl_deconvolver
    n_iter: number of iterations

    TODO: add observer callback so that progress updates for each iteration
    can be displayed. Also, add option to save intermediate results within
    a certain range of iterations.
    """

    # TODO: this is a quick test whether tensorflow session configs can be used to limit the memory use
    # if it works, add an option to pass in a tensorflow session config.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)
    config.gpu_options.allow_growth = True

    # Given an acquisition "acq" defined somewhere
    res = algo.run(acq, niter=10, session_config=config)

    aq = fd_data.Acquisition(data=vol, kernel=psf)
    if observer is not None:
        warnings.warn("Observer function for iteration not yet implemented.")
    return deconvolver.run(aq, niter=n_iter, session_config=config).data


def get_deconv_function(psf: np.ndarray,
                        deconvolver: tfd_restoration.RichardsonLucyDeconvolver,
                        n_iter: int) -> Callable:
    deconv_func = partial(deconv_volume, psf=psf, deconvolver=deconvolver, n_iter=n_iter)
    return deconv_func
