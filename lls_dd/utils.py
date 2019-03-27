import pathlib
import warnings
import logging
import numpy as np
from typing import Union

from lls_dd.imsave import imsave

logging.getLogger("tifffile").setLevel(logging.ERROR)

# write in a separate thread
# https://www.geeksforgeeks.org/writing-files-background-python/


def write_tiff_createfolder(
    path: Union[str, pathlib.Path], nparray: np.ndarray, **opt_kwargs
):
    """ 
    given a
    path: to a tiff file of type pathlib.Path (or str), and
    nparray: numpy array holding an image/volume
    
    creates all necessary folders in the path (in case they don't exist already)
    and writes the numpy array as a tiff
    """
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(str(path), nparray, **opt_kwargs)
