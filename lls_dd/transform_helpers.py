import numpy as np
import matplotlib.pyplot as plt
import warnings
from numpy.linalg import inv
from functools import partial
from typing import Union, Iterable, Callable, Optional, Collection
import logging

logger = logging.getLogger("lls_dd")

# from scipy.ndimage import affine_transform
from lls_dd.gputools_wrapper import affine_transform_gputools as affine_transform
from lls_dd.transforms import rot_around_y, deskew_mat, shift_centre, unshift_centre, scale_pixel_z


def ceil_to_mulitple(x, base: int = 4):
    """rounds up to the nearest integer multiple of base
    
    Parameters
    ----------
    x : scalar or np.array
        value/s to round up from
    base : int, optional
        round up to multiples of base (the default is 4)

    Returns
    -------

    scalar or np.array:
        rounded up value/s
    
    """

    return (np.int(base) * np.ceil(np.array(x).astype(np.float) / base)).astype(np.int)


def get_transformed_corners(
    aff: np.ndarray, vol_or_shape: Union[np.ndarray, Iterable[float]], zeroindex: bool = True
):
    """ Input
    aff: an affine transformation matrix 
    vol_or_shape: a numpy volume or shape of a volume. 
    
    This function will return the positions of the corner points of the volume (or volume with
    provided shape) after applying the affine transform.
    """
    # get the dimensions of the array.
    # see whether we got a volume
    if np.array(vol_or_shape).ndim == 3:
        d0, d1, d2 = np.array(vol_or_shape).shape
    elif np.array(vol_or_shape).ndim == 1:
        d0, d1, d2 = vol_or_shape
    else:
        raise ValueError
    # By default we calculate where the corner points in
    # zero-indexed (numpy) arrays will be transformed to.
    # set zeroindex to False if you want to perform the calculation
    # for Matlab-style arrays.
    if zeroindex:
        d0 -= 1
        d1 -= 1
        d2 -= 1
    # all corners of the input volume (maybe there is
    # a more concise way to express this with itertools?)
    corners_in = [
        (0, 0, 0, 1),
        (d0, 0, 0, 1),
        (0, d1, 0, 1),
        (0, 0, d2, 1),
        (d0, d1, 0, 1),
        (d0, 0, d2, 1),
        (0, d1, d2, 1),
        (d0, d1, d2, 1),
    ]
    corners_out = list(map(lambda c: aff @ np.array(c), corners_in))
    corner_array = np.concatenate(corners_out).reshape((-1, 4))
    # print(corner_array)
    return corner_array


def get_output_dimensions(aff: np.ndarray, vol_or_shape: Union[np.ndarray, Iterable[float]]):
    """ given an 4x4 affine transformation matrix aff and 
    a 3d input volume (numpy array) or volumen shape (iterable with 3 elements)
    this function returns the output dimensions required for the array after the
    transform. Rounds up to create an integer result.
    """
    corners = get_transformed_corners(aff, vol_or_shape, zeroindex=True)
    # +1 to avoid fencepost error
    dims = np.max(corners, axis=0) - np.min(corners, axis=0) + 1
    dims = ceil_to_mulitple(dims, 2)
    logger.debug(f"get_output_dimensions: output dimensions: {dims[:3]}")
    return dims[:3].astype(np.int)


def get_projections(in_array: np.ndarray, fun: Callable = np.max) -> Iterable[np.ndarray]:
    """given an array, projects along each axis using the function fun (defaults to np.max).
    
    Parameters
    ----------
    in_array : np.ndarray
        input array
    fun : Callable, optional
        function to use for projecting along axis (the default is np.max)
    
    Returns
    -------
    Iterable[np.ndarray]
        Returns a mapping (iterator) of projections
    
    """
    projections = map(lambda ax: fun(in_array, axis=ax), range(in_array.ndim))
    return projections


def plot_all(imlist: Iterable[np.ndarray], backend: str = "matplotlib"):
    """ given an iterable of 2d numpy arrays (images),
        plots all of them in order.
        Will add different backends (Bokeh) later """
    if backend == "matplotlib":
        for im in imlist:
            plt.imshow(im)
            plt.show()
    else:
        warnings.warn(f"backend {backend} not yet implemented")


def imprint_coordinate_system(volume, origin=(0, 0, 0), l=100, w=5, vals=(6000, 10000, 14000)):
    """ imprints coordinate system axes in a volume at origin
    axes imprints have length l and width w and intensity values in val.
    This can be quite helpful for debugging affine transforms to see how the coordinate axes are mapped."""
    o = origin
    volume[o[0] : o[0] + l, o[1] : o[1] + w, o[2] : o[2] + w] = vals[0]
    volume[o[0] : o[0] + w, o[1] : o[1] + l, o[2] : o[2] + w] = vals[1]
    volume[o[0] : o[0] + w, o[1] : o[1] + w, o[2] : o[2] + l] = vals[2]


def get_projection_montage(
    vol: np.ndarray, gap: int = 10, proj_function: Callable = np.max
) -> np.ndarray:
    """ given a volume vol, creates a montage with all three projections (orthogonal views)

    
    Parameters
    ----------
    vol : np.ndarray
        input volume
    gap : int, optional
        gap between projections in montage (the default is 10 pixels)
    proj_function : Callable, optional
        function to create the projection (the default is np.max, which performs maximum projection)
    
    Returns
    -------
    np.ndarray
        the montage of all projections
    """

    assert len(vol.shape) == 3, "only implemented for 3D-volumes"
    nz, ny, nx = vol.shape
    m = np.zeros((ny + nz + gap, nx + nz + gap), dtype=vol.dtype)
    m[:ny, :nx] = proj_function(vol, axis=0)
    m[ny + gap :, :nx] = np.max(vol, axis=1)
    m[:ny, nx + gap :] = np.max(vol, axis=2).transpose()
    return m


def calc_deskew_factor(dz_stage: float, xypixelsize: float, angle: float) -> float:
    """calculates the deskew factor for the affine transform
    
    Parameters
    ----------
    dz_stage : float
        stage step size in z direction
    xypixelsize : float
        pixel size in object space (use same units as for dz_stage)
    angle : float
        light sheet angle relative to stage in degrees
    
    Returns
    -------
    float
        deskew factor
    """

    deskewf = np.cos(angle * np.pi / 180.0) * dz_stage / xypixelsize
    logger.debug(f"deskew factor caclulated as {deskewf}")
    return deskewf


def get_deskew_function(
    input_shape: Iterable[int], dz_stage: float, xypixelsize: float, angle: float, interp_order: int = 1
) -> Callable:

    """Generate a deskew function for processing raw volumes with the provided parameters
    
    Parameters
    ----------
    input_shape: Iterable[int]
        raw input volume shape (typically tuple)
    dz_stage : float
        stage step size in z direction
    xypixelsize : float
        pixel size in object space (use same units as for dz_stage)
    angle : float
        light sheet angle relative to stage in degrees
    interp_order : int
        interpolation order to use for affine transform (default is 1)

    Returns
    -------
    Callable
        function f that deskews an input volume f(np.array: vol) -> np.array
    
    
    Notes
    -----
    use functools.partial to achieve this
    """

    # TODO: also read up on toolz and a function currying here
    # https://toolz.readthedocs.io/en/latest/curry.html
    # https://github.com/elegant-scipy/elegant-scipy/blob/2e65e7fe0fbf69e9fb45e0cf4d90e85b7a0a7ae4/markdown/ch8.markdown

    skew = deskew_mat(calc_deskew_factor(dz_stage, xypixelsize, angle))
    output_shape = get_output_dimensions(skew, input_shape)
    logger.debug(f"deskew function: skew matrix: {skew}")
    logger.debug(f"deskew function: output shape: {output_shape}")
    deskew_func = partial(
        affine_transform, matrix=inv(skew), output_shape=output_shape, order=interp_order
    )
    return deskew_func


def _twostep_affine(
    vol: np.array,
    mat1: np.array,
    outshape1: Iterable[int],
    mat2: np.array,
    outshape2: Iterable[int],
    order: int = 1,
) -> np.array:
    """performs two affine transforms in succession
    
    Parameters
    ----------
    vol : np.array
        input array
    mat1 : np.array
        first affine matrix
    outshape1 : Iterable[int]
        output shape for first transform
    mat2 : np.array
        second affine matrix
    outshape2 : Interable[int]
        output shape for second matrix
    order : Optional[int], optional
        interpolation order (default is 1 = linear)
    
    Returns:

    np.array:
        transformed volume
    """
    # TODO: deal with "mode" ... maybe pass varargs
    step1 = affine_transform(vol, mat1, output_shape=outshape1, order=order)
    step2 = affine_transform(step1, mat2, output_shape=outshape2, order=order)
    return step2


def get_rotate_to_coverslip_function(
    orig_shape: Collection[int], dz_stage: float, xypixelsize: float, angle: float, interp_order: int = 1
) -> Callable:
    """Generate a function that rotates a deskewed volume to coverslip coordinates
    
    Parameters
    ----------
    orig_shape : Iterable[int]
        shape of the original raw volume (not the shape of the deskewed volume!)
    dz_stage : float, optional
        stage step size in z direction
    xypixelsize : float, optional
        pixel size in object space (use same units as for dz_stage)
    angle : float, optional
        light sheet angle relative to stage in degrees
    interp_order : int, optional
        interpolation order to use for affine transform (default is 1)

    
    Returns
    -------
    Callable
        function f that rotates a deskewed volume to coverslip coordinates f(np.array: vol) -> np.array
    
    Notes
    -----
        The returned function performs the rotation in two seperate affine transformation steps to avoid
        aliasing (see [1]_).

    References
    ----------
        [1] https://github.com/VolkerH/Lattice_Lightsheet_Deskew_Deconv/issues/22
    """

    dz = np.sin(angle * np.pi / 180.0) * dz_stage
    dx = xypixelsize
    deskewfactor = np.cos(angle * np.pi / 180.0) * dz_stage / dx
    dzdx_aspect = dz / dx

    logger.debug(f"rotate function: dx: {dx}")
    logger.debug(f"rotate function: dz: {dz}")
    logger.debug(f"rotate function: deskewfactor: {deskewfactor}")
    logger.debug(f"rotate function: dzdx_aspect: {dzdx_aspect}")

    # shift volume such that centre is at (0,0,0) for rotations

    # Build deskew matrix
    skew = deskew_mat(deskewfactor)
    shape_after_skew = get_output_dimensions(skew, orig_shape)
    # matrix to scale z to obtain isotropic pixels
    scale = scale_pixel_z(dzdx_aspect)
    shape_after_scale = get_output_dimensions(scale, shape_after_skew)
    shift_scaled = shift_centre(shape_after_scale)
    # rotation matrix
    rot = rot_around_y(-angle)

    # determine final output shape for an all-in-one (deskew/scale/rot) transform
    # (which is not actually applied)
    shift = shift_centre(orig_shape)
    combined = rot @ scale @ skew @ shift
    shape_final = get_output_dimensions(combined, orig_shape)
    # determine shape after scale/rot on deskewed, this is larger than final due to
    # fill pixels
    shape_scalerot = get_output_dimensions(rot @ shift_scaled @ scale, shape_after_skew)

    # calc rotshift

    logger.debug(f"shape_scalerot {shape_scalerot}")
    logger.debug(f"shape_final {shape_final}")
    _tmp = unshift_centre(shape_final)
    diff = (shape_scalerot[0] - shape_final[0]) / 2
    logger.debug(f"diff {diff}")
    # _tmp[0,3] -= diff
    unshift_final = _tmp
    rotshift = unshift_final @ rot @ shift_scaled

    logger.debug(f"rotate to coverslip: scale matrix: {scale}")
    logger.debug(f"rotate to coverslip: outshape1: {shape_after_scale}")
    logger.debug(f"rotate to coverslip: rotshift: {rotshift}")
    logger.debug(f"rotate to coverslip: outshape2: {shape_final}")
    rotate_func = partial(
        _twostep_affine,
        mat1=inv(scale),
        outshape1=shape_after_scale,
        mat2=inv(rotshift),
        outshape2=shape_final,
        order=interp_order,
    )
    return rotate_func


def get_rotate_function_all_in_one(
    input_shape: Collection[Union[int]],
    dz_stage: float = 0.299_401,
    xypixelsize: float = 0.1040,
    angle: float = 31.8,
    interp_order: int = 1,
) -> Callable:
    """ returns an all-in-one rotate function
    that takes a raw volume and deskews and rotates 
    it to coverslip coordinates using a single affine transform.
    This is more efficient, but as the data are not isotropic 
    this causes severe aliasing as interpolation between x/y and
    z takes place.
    Do not use !
    For discussion see
    https://github.com/VolkerH/Lattice_Lightsheet_Deskew_Deconv/issues/22
    """
    dz = np.sin(angle * np.pi / 180.0) * dz_stage
    dx = xypixelsize
    deskewfactor = np.cos(angle * np.pi / 180.0) * dz_stage / dx
    dzdx_aspect = dz / dx

    logger.debug(f"rotate function: dx: {dx}")
    logger.debug(f"rotate function: dz: {dz}")
    logger.debug(f"rotate function: deskewfactor: {deskewfactor}")
    logger.debug(f"rotate function: dzdx_aspect: {dzdx_aspect}")

    # shift volume such that centre is at (0,0,0) for rotations
    shift = shift_centre(input_shape)
    logger.debug(f"rotate function: shift matrix {shift}")
    # Build deskew matrix
    skew = deskew_mat(deskewfactor)
    logger.debug(f"")
    # scale z to obtain isotropic pixels
    scale = scale_pixel_z(dzdx_aspect)
    # rotate
    rot = rot_around_y(-angle)

    # determine output shape for combined transform (except unshift)
    combined = rot @ scale @ skew @ shift
    output_shape = get_output_dimensions(combined, input_shape)

    # add unshift to bring 0,0,0 to centre of output volume from centre
    unshift_final = unshift_centre(output_shape)
    logger.debug(f"rotate function: unshift matrix: {unshift_final}")
    logger.debug(f"rotate function: output shape: {output_shape}")
    all_in_one = unshift_final @ rot @ scale @ skew @ shift
    logger.debug(f"rotate function: all in one: {all_in_one}")
    logger.debug(f"rotate function: all in one: {inv(all_in_one)}")
    rotate_func = partial(
        affine_transform, matrix=inv(all_in_one), output_shape=output_shape, order=interp_order
    )
    return rotate_func
