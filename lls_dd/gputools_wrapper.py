from builtins import tuple

# this import is failing
# from gputools.convolve import gaussian_filter
from gputools import affine
import numpy as np
import warnings
from typing import Iterable, Optional, Any, Union, Sequence


def affine_transform_gputools(
    input_data: np.ndarray,
    matrix: np.ndarray,
    offset: Union[float, Iterable[float]] = 0.0,
    output_shape: Optional[Iterable[int]] = None,
    output: Optional[np.ndarray] = None,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: Optional[Any] = None,
):
    """ affine_transform_gputools
    Wraps affine transform from GPUtools such that
    it can be used as a drop in replacement for 
    scipy.ndimage.affine_transform
    
    Pads and crops the array to output shape as necessary.
    mode specifies boundary treatment and is passed on to np.pad and gputools.affine.
    As gputools.affine does not handle all of the modes that np.pad does, it will fall back 
    to constant if necessary.

    Note: requires gputools >= 0.2.8
    """

    if matrix.shape != (4, 4):
        warnings.warn(
            "Wrapper has only been tested for 4x4 affine matrices. Behaviour for other shapes unknown/untested."
        )

    if prefilter is not None:
        warnings.warn("Prefilter is not available in the gputools wrapper. Argument is ignored")

    if np.any(offset):
        warnings.warn("Offset argument not implemented in gputools wrapper. Will be ignored.")

    if order == 0:
        interpolation = "nearest"
    elif order == 1:
        interpolation = "linear"
    else:
        interpolation = "linear"
        warnings.warn("interpolation order >1 not supported, defaulting to linear.")

    if output is not None:
        # TODO : can we support in-place?
        warnings.warn("inplace operation not yet implemented in gputools wrapper")

    # pad input array for output shape
    # see np.pad if you wonder about the strange padding
    needs_crop = False
    if output_shape is not None:
        assert isinstance(input_data.shape, tuple)
        padding = np.array(output_shape) - np.array(input_data.shape)
        if np.any(padding < 0):
            needs_crop = True
        padding = [(0, max(i, 0)) for i in padding]
        # TODO check whether np.pad supports the same modes as gputools.affine
        if mode == "constant":
            input_data = np.pad(input_data, padding, mode=mode, constant_values=cval)
        else:
            input_data = np.pad(input_data, padding, mode=mode)

    if mode not in ("edge", "constant", "wrap"):
        warnings.warn("Mode " + mode + " not supported by gputools.constant. Falling back to constant")
        mode = "constant"

    if mode == "constant" and cval != 0.0:
        warnings.warn(
            f"cval of {cval} provided. cval not supported by gputools and will default to zero."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = affine(data=input_data, mat=matrix, mode=mode, interpolation=interpolation)

    if needs_crop and output_shape is not None:
        i, j, k = output_shape
        result = result[0:i, 0:j, 0:k]

    return result


def gaussian_gputools(
    image,
    sigma=1,
    output=None,
    mode="nearest",
    cval=0,
    multichannel=None,
    preserve_range=False,
    truncate=4,
):
    """ 
    scipy.filters.gaussian compatible wrapper around gputools gaussian
    NOT yet working !
    """

    if mode != "nearest" or cval != 0 or multichannel or preserve_range:
        warnings.warn(
            f"gputools wrapper for gaussian currently doesn't support options"
            "mode, cval, multichannel or preserve_range. These are simply ignored"
        )

    if output:
        warnings.warn(f"passing an output variable in has not been tested with gputools wrapper")
    # TODO: check what happens if a numpy array is passed into res_g (expects OCLArray)
    return gaussian_filter(data=image, sigma=sigma, truncate=truncate, normalize=True, res_g=output)

