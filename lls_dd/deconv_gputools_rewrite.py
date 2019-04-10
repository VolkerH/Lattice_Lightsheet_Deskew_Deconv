""" Lucy richardson deconvolution

based on code from Martin Weigert's gputools
"""

import os
import numpy as np
from gputools import OCLArray, OCLProgram, get_device
from gputools import convolve, fft_convolve, fft, fft_plan
from gputools import OCLElementwiseKernel
from typing import Optional, Callable
import warnings
import gputools

_multiply_inplace = OCLElementwiseKernel(
    "float *a, float * b", "a[i] = a[i] * b[i]", "mult_inplace"
)

_divide_inplace = OCLElementwiseKernel(
    "float *a, float * b", "b[i] = a[i]*b[i]/(b[i]*b[i]+0.001f)", "divide_inplace"
)

_complex_multiply = OCLElementwiseKernel(
    "cfloat_t *a, cfloat_t * b,cfloat_t * res", "res[i] = cfloat_mul(a[i],b[i])", "mult"
)

_complex_multiply_inplace = OCLElementwiseKernel(
    "cfloat_t *a, cfloat_t * b", "a[i] = cfloat_mul(a[i],b[i])", "mult_inplace"
)

_complex_divide = OCLElementwiseKernel(
    "cfloat_t *a, cfloat_t * b,cfloat_t * res",
    "res[i] = cfloat_divide(b[i],a[i])",
    "div",
)

_complex_divide_inplace = OCLElementwiseKernel(
    "cfloat_t *a, cfloat_t * b", "b[i] = cfloat_divide(a[i],b[i])", "divide_inplace"
)


class Deconvolver_RL_gputools(object):
    """ fft deconvolver based on Martin Weigert's gputools fft-based RL implementation
        breaks it into two stages to avoid unnecessary processig and allocation
    """

    def __init__(self, psf: np.ndarray, psf_is_fftshifted: bool = False, n_iter=10):
        """ setup deconvolution for a given shape """
        self.shape = psf.shape
        if not psf_is_fftshifted:
            psf = np.fft.fftshift(psf)

        self.n_iter = n_iter
        # What happens here? Indices are being flipped ? Why. What if it is 3D?
        psfflip = psf[::-1, ::-1]

        self.psf_g = OCLArray.from_array(psf.astype(np.complex64))
        self.psfflip_f_g = OCLArray.from_array(psfflip.astype(np.complex64))
        self.plan = fft_plan(self.shape)

        # transform psf
        fft(self.psf_g, inplace=True)
        fft(self.psfflip_f_g, inplace=True)

        # get temp
        self.tmp_g = OCLArray.empty(psf.shape, np.complex64)

    def run(self, data: np.ndarray):
        if data.shape != self.shape:
            raise ValueError("data and h have to be same shape")

        # set up some gpu buffers
        data64 = data.astype(np.complex64)
        y_g = OCLArray.from_array(data64)
        u_g = OCLArray.from_array(data64)

        # hflipped_g = OCLArray.from_array(h.astype(np.complex64))

        for i in range(self.n_iter):
            # logger.info("Iteration: {}".format(i))
            fft_convolve(
                u_g, self.psf_g, plan=self.plan, res_g=self.tmp_g, kernel_is_fft=True
            )

        _complex_divide_inplace(y_g, self.tmp_g)

        fft_convolve(
            self.tmp_g,
            self.psfflip_f_g,
            plan=self.plan,
            inplace=True,
            kernel_is_fft=True,
        )

        _complex_multiply_inplace(u_g, self.tmp_g)

        # can abs be calculated on the gpu ?
        return np.abs(u_g.get())


## below are simple wrappers to make the interface compatible with the functions
## that were originally written for deconvolution with flowdec
## eventually all of those should be refactured into a classs
def init_rl_deconvolver(**kwargs):
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
    decon = Deconvolver_RL_gputools(psf=psf, n_iter=n_iter)
    return decon.run
