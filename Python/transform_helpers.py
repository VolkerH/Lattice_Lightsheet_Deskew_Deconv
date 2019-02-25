import numpy as np
import matplotlib.pyplot as plt
from gputools_wrapper import affine_transform_gputools as affine_transform
from numpy.linalg import inv
from functools import partial
from transforms import rot_around_y, deskew_mat, shift_centre, unshift_centre, scale_pixel_z
from typing import Union, Iterable, Callable
import warnings

def ceil_to_mulitple(x, base: int = 4):
    """ rounds up to the nearest multiple of base
    input can be a numpy array or any scalar
    """
    return (np.int(base) * np.ceil(np.array(x).astype(np.float) / base)).astype(np.int)


def get_transformed_corners(aff: np.ndarray, vol_or_shape: Union[np.ndarray, Iterable[float]], zeroindex: bool = True):
    """ Input
    aff: an affine transformation matrix 
    vol_or_shape: a numpy volume or shape of a volume. 
    
    This function will return the positions of the corner points of the volume (or volume with
    provided shape) after applying the affine transform.
    """
    # get the dimensions of the array.
    # see whether we got a volume
    if np.array(vol_or_shape).ndim == 3:
        d0, d1, d2 = vol_or_shape.shape
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
    return dims[:3].astype(np.int)


def get_projections(in_array: np.ndarray, fun: Callable = np.max) -> Iterable[np.ndarray]:
    """ given an array, projects along each axis using the function fun (defaults to np.max).
    Returns a mapping (iterator) of projections """
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


def get_projection_montage(vol: np.ndarray, gap: int = 10, proj_function: Callable = np.max) -> np.ndarray:
    """ 
    given a spim volume vol, creates a montage with all three projections
    (orthogonal views)
    gap specifies the number of zero fill pixels to seperate the projections
    proj_function: allows to pass in other functions to create the projections (e.g. np.median)
    """
    nz, ny, nx = vol.shape
    m = np.zeros((ny + nz + gap, nx + nz + gap), dtype=vol.dtype)
    m[:ny, :nx] = proj_function(vol, axis=0)
    m[ny + gap :, :nx] = np.max(vol, axis=1)
    m[:ny, nx + gap :] = np.max(vol, axis=2).transpose()
    return m


def calc_deskew_factor(dz_stage: float, xypixelsize: float, angle: float) -> float:
    return np.cos(angle * np.pi / 180.0) * dz_stage / xypixelsize


def get_deskew_function(input_shape: Iterable[int],
                        dz_stage: float =0.299401,
                        xypixelsize: float = 0.1040,
                        angle: float = 31.8,
                        interp_order: int = 1) -> Callable:
    """ 
    returns a ready to use deskew function using partial function evaluation
    
    use functools.partial to achieve this
    TODO: also read up on toolz and a function currying here
    https://toolz.readthedocs.io/en/latest/curry.html
    https://github.com/elegant-scipy/elegant-scipy/blob/2e65e7fe0fbf69e9fb45e0cf4d90e85b7a0a7ae4/markdown/ch8.markdown
    """

    skew = deskew_mat(calc_deskew_factor(dz_stage, xypixelsize, angle))
    output_shape = get_output_dimensions(skew, input_shape)
    deskew_func = partial(affine_transform, matrix=inv(skew), output_shape=output_shape, order=interp_order)
    return deskew_func


def get_rotate_function(input_shape: Iterable[int],
                        dz_stage: float = 0.299401,
                        xypixelsize: float = 0.1040,
                        angle: float = 31.8,
                        interp_order: int = 1) -> Callable:
    dz = np.sin(angle * np.pi / 180.0) * dz_stage
    dx = xypixelsize
    deskewfactor = np.cos(angle * np.pi / 180.0) * dz_stage / dx
    dzdx_aspect = dz / dx

    # shift volume such that centre is at (0,0,0) for rotations
    shift = shift_centre(input_shape)
    # Build deskew matrix
    skew = deskew_mat(deskewfactor)
    # scale z to obtain isotropic pixels
    scale = scale_pixel_z(dzdx_aspect)
    # rotate
    rot = rot_around_y(-angle)

    # determine output shape for combined transform (except unshift)
    combined = rot @ scale @ skew @ shift
    output_shape = get_output_dimensions(combined, input_shape)

    # add unshift to bring 0,0,0 to centre of output volume from centre
    unshift_final = unshift_centre(output_shape)
    all_in_one = unshift_final @ rot @ scale @ skew @ shift
    rotate_func = partial(affine_transform, matrix=inv(all_in_one), output_shape=output_shape, order=interp_order)
    return rotate_func
