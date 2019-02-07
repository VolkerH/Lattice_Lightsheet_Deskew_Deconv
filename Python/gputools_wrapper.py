import gputools
import numpy as np
import warnings 


def affine_transform_gputools(input, matrix, offset=0.0, output_shape=None, output=None, order=1, mode='constant', cval=0.0, prefilter=None):
    ''' affine_transform_gputools
    Wraps affine transform from GPUtools such that
    it can be used as a drop in replacement for 
    scipy.ndimage.affine_transform
    
    Pads and crops the array to output shape as necessary.
    mode specifies boundary treatment and is passed on to np.pad and gputools.affine.
    As gputools.affine does not handle all of the modes that np.pad does, it will fall back 
    to constant if necessary.

    Note: requires gputools >= 0.2.8
    ''' 

    if(matrix.shape!=(4,4)):
        warnings.warn("Wrapper has only been tested for 4x4 affine matrices. Behaviour for other shapes unknown/untested.")

    if prefilter is not None:
        warnings.warn('Prefilter is not available in the gputools wrapper. Argument is ignored')

    if np.any(offset):
        warnings.warn('Offset argument not implemented in gputools wrapper. Will be ignored.')

    if order == 0: 
        interpolation = 'nearest'
    elif order == 1:
        interpolation = 'linear'
    else:
        warnings.warn("interpolation order >1 not supported, defaulting to linear.")

    # pad input array for output shape
    # see np.pad if you wonder about the strange padding
    needs_crop = False
    if output_shape is not None:
        padding = np.array(output_shape) - np.array(input.shape)
        if np.any(padding < 0):
            needs_crop = True
        padding = [(0, max(i,0)) for i in padding]
        # TODO check whether np.pad supports the same modes as gputools.affine
        if mode == 'constant':
            input = np.pad(input, padding, mode=mode, constant_values=cval)
        else:
            input = np.pad(input, padding, mode=mode)
            
    if not mode in ('edge', 'constant', 'wrap'):
        warnings.warn('Mode ' + mode + ' not supported by gputools.constant. Falling back to constant')
        mode = 'constant'

    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore")
        result = gputools.affine(data=input, mat=matrix, mode=mode, interpolation=interpolation)
    
    if needs_crop:
        i, j, k = output_shape
        result = result[0:i, 0:j, 0:k]
        
    return result
