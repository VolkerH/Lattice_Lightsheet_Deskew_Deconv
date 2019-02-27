import transforms
from transform_helpers import calc_deskew_factor
from skimage.feature import peak_local_max
from skimage.filters import gaussian

# from scipy.ndimage import affine_transform
from gputools_wrapper import affine_transform_gputools as affine_transform
from numpy.linalg import inv
import warnings
import pathlib
import tifffile
import numpy as np
from typing import Tuple, Union, Optional, Iterable


def psf_find_maximum(psf: np.ndarray, maxiter: int = 20, gauss_sigma: float = 1.5):
    """
    Tries to find a single maxium in the numpy volume psf.
    Uses peak_local_max to find the maximum. Iteratively
    smooths the original volume with a gaussian, until only
    one maximum is left (threshold-free). Gaussian smoothing
    should not shift the location of the maximum."""

    smoothed = psf.copy()
    while len(peak_local_max(smoothed)) > 1 and maxiter > 0:
        maxiter -= 1
        smoothed = gaussian(smoothed, gauss_sigma)

    centre = peak_local_max(smoothed)
    if len(centre) > 1:
        warnings.warn(
            f"No single PSF maximum found after {maxiter}" + "iterations of smoothing. Returning first maximum"
        )
    return centre[0]


def psf_background_subtraction(psf: np.ndarray) -> Tuple[np.ndarray, float]:
    """ 
    Estimates ans substracts the background fluorescence intensity.
    assumes that first and last slice of the stack contain mostly background
    and takes the median grey value of these slices as background
    Returns a tuple (psf_bgcorr, bg_estimate)
    """

    bgval = np.median(psf[(0, -1), :, :])
    psf_bgcorr = np.clip(psf - bgval, 0, np.max(psf))
    return psf_bgcorr, float(bgval)


def psf_rescale_centre_skew_pad(psf: np.ndarray,
                                dz_ratio_galvo_stage: float,
                                centre: Iterable[float],
                                output_shape: Iterable[int],
                                deskewfactor: Optional[float] = None,
                                interpolation: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a 
    * psf: volume (numpy array)
    * centre: centre coordinate of bead in volume
    * dz_ratio_galvo_stage: scaling factor along z axis (this accounts for different z scaling between galvo
    and stage)
    * output_shape: desired output shape of array
    * deskewfactor: if not None (default), will skew the psf for direct deconvolution on the skewed data.
    
    Returns: tuple (processed_psf, transform_matrix)
    """
    scale_psf = transforms.scale_pixel_z(dz_ratio_galvo_stage)
    shift = transforms.shift_centre(2 * np.array(centre))
    unshift = transforms.unshift_centre(output_shape)
    if deskewfactor:
        skew = inv(transforms.deskew_mat(deskewfactor))
    else:
        skew = np.eye(4)  # no skew, identity
    combined_transform = unshift @ skew @ scale_psf @ shift
    processed_psf = affine_transform(psf, inv(combined_transform), output_shape = output_shape, order=interpolation)
    return processed_psf, combined_transform


def psf_normalize_intensity(psf: np.ndarray) -> np.ndarray:
    """
    Given 
    * psf: a numpy array
    Returns 
    * normalized_psf,
    which is the input mulitplied with a scalar value such that 
    the sum of all elements is 1.0.
    (The assumption is that the PSF does not diminish intensity, 
    but rather spatially redistributes intensity.)
    """

    sum_all = psf.sum()
    if sum_all != 0.0:
        return psf / sum_all
    else:
        warnings.warn("sum of PSF pixel is zero, cannot rescale")
        return psf


def generate_psf(psffile: Union[pathlib.Path, str],
                 output_shape: Iterable[int],
                 dz_stage: float,
                 dz_galvo: float,
                 xypixelsize: float,
                 angle: float,
                 subtract_bg: bool = True,
                 normalize_intensity: bool = True) -> np.ndarray:
    """
    Generate a PSF for use with flowdec or Deconvolutionlab2.
    Finds maximum of bead, centres it in the volume.
    The unskewed (because it is galvo-scanned) PSF volume is skewed so it
    can be used directly on the stage-scanned volumes. Finally the volume 
    is padded with zeros to match output_shape and normalized such that the 
    sum of all voxels adds up to 1.0
     
    such that it has the final shape output_shape to output_shape.

    Input: 
    psffile: filename of a file containing a volume of a single bead
    output_shape: desired shape of the output volume
    dz_stage: z step of stage for volume to be deconvolved 
    dz_galvo: z step of galvo scanner with which PSF was captured 
    xypixelsize: x,y pixel spacing 
    angle: lightsheet angle
    subtract_bg: estimate and subtract bg from PSF. 

    Typically  dz_stage, dz_galvo, xypixelsize will be in um, but it is
    only the ratios that matter, so as long as the units are consistent you can use any.
    """

    if isinstance(psffile, pathlib.Path):
        psffile = str(psffile)
    psf_orig = tifffile.imread(psffile)

    # assert output shape >= input shape, otherwise we'd have to crop
    assert np.all(np.array(output_shape) >= np.array(psf_orig.shape))

    bead_centre = psf_find_maximum(psf_orig)
    dz_ratio_galvo_stage = dz_galvo / dz_stage
    deskewfactor = calc_deskew_factor(dz_stage, xypixelsize, angle)

    psf, transform = psf_rescale_centre_skew_pad(
        psf_orig, dz_ratio_galvo_stage, bead_centre, output_shape, deskewfactor, interpolation=1
    )
    if subtract_bg:
        psf, bgval = psf_background_subtraction(psf)
    if normalize_intensity:
        psf = psf_normalize_intensity(psf)

    return psf
