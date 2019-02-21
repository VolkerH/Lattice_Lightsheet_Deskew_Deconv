# transformation matrices for lattice lightsheet deskewing and rotation
#
# Assumption:
# Works on numpy arrays with dimension order (z,y,x)
# Author:
# Volker Hilsenstein at monash dot edu

import numpy as np


def scale_pixel_z(zscale):
    """ returns a 4x4 affine transformation matrix that 
    scales the zaxis with factor zscale"""
    s = np.eye(4)
    s[0, 0] = zscale
    return s


def shift_centre(matrix_shape, direction=-1.0):
    """ returns a 4x4 affine matrix which translates
    a volume of shape matrix_shape (3-tuple) such
    that its centre falls onto the origin 
    """
    centre = np.array(matrix_shape) / 2
    shift = np.eye(4)
    shift[:3, 3] = direction * centre
    return shift


def unshift_centre(matrix_shape):
    """ like shift_centre but with inverse tranlation direction"""
    return shift_centre(matrix_shape, 1.0)


def deskew_mat(deskew_factor):
    """ returns a 4x4 affine matrix for LLS deskewing by 
    factor deskew_factor (shearing)"""
    deskew = np.eye(4)
    deskew[2, 0] = deskew_factor
    return deskew


def rot_around_y(angle_deg):
    """returns an affine matrix that rotates around the y-axis in clockwise
    direction by angle_deg degrees"""
    arad = angle_deg * np.pi / 180.0
    roty = np.array(
        [[np.cos(arad), 0, np.sin(arad), 0], [0, 1, 0, 0], [-np.sin(arad), 0, np.cos(arad), 0], [0, 0, 0, 1]]
    )
    return roty
