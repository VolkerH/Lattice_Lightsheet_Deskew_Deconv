import pathlib
import tifffile
import warnings

import logging
logging.getLogger("tifffile").setLevel(logging.ERROR)

def write_tiff_createfolder(path, nparray):
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
    #print(f"writing {str(path)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tifffile.imsave(str(path), nparray)