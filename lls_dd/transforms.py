# transformation matrices for lattice lightsheet deskewing and rotation
#
# Assumption:
# Works on numpy arrays with dimension order (z,y,x)
# Author:
# Volker Hilsenstein at monash dot edu


from typing import Collection
import numpy as np


def scale_pixel_z(zscale: float) -> np.ndarray:
    """create affine matrix for scaling along the z axis
    
    Parameters
    ----------
    zscale : float
        scale factor
    
    Returns
    -------
    np.ndarray
        4x4 affine matrix
    """
    s = np.eye(4)
    s[0, 0] = zscale
    return s


def shift_centre(matrix_shape: Collection[float], direction=-1.0) -> np.ndarray:
    """create an affine matrix for shifting centre of a volume to the origin
    
    Parameters
    ----------
    matrix_shape : Collection[float]
        shape of the volume to be shifted (typically a 3-tuple)
    direction : float, optional
        [description] (the default is -1.0, which [default_description])
    
    Returns
    -------
    np.ndarray
        4x4 affine translation matrix
    """
    assert(len(matrix_shape)==3)
    centre = np.array(matrix_shape) / 2
    shift = np.eye(4)
    shift[:3, 3] = direction * centre
    return shift


def unshift_centre(matrix_shape: Collection[float]) -> np.ndarray:
    """like shift_centre but with inverse tranlation direction"""
    return shift_centre(matrix_shape, 1.0)


def deskew_mat(deskew_factor: float) -> np.ndarray:
    """create a shear matrix for deskewing a lattice light sheet volume
    
    Parameters
    ----------
    deskew_factor : float
        deskew factor 
    
    Returns
    -------
    np.ndarray
        4x4 affine matrix for shearing 
    """
    deskew = np.eye(4)
    deskew[2, 0] = deskew_factor
    return deskew


def rot_around_y(angle_deg: float) -> np.ndarray:
    """create affine matrix for rotation around y axis
    
    Parameters
    ----------
    angle_deg : float
        rotation angle in degrees
    
    Returns
    -------
    np.ndarray
        4x4 affine rotation matrix
    """
    arad = angle_deg * np.pi / 180.0
    roty = np.array(
        [
            [np.cos(arad), 0, np.sin(arad), 0],
            [0, 1, 0, 0],
            [-np.sin(arad), 0, np.cos(arad), 0],
            [0, 0, 0, 1],
        ]
    )
    return roty
