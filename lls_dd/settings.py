import json
from typing import Dict, Any


def create_fixed_settings(file: str):
    """creates a minimalist .json file with a dictionary fixed settings
    
    Parameters
    ----------
    file : str
            path to the file

    Notes
    -----
    the fixed settings file contains settings pertaining to 
    a particular microscope that are known not to change between
    experiments. The settings below are specific to our lls at Monash
    but probably apply to most lls microscopes based on the original
    Janelia design.
    """
    fs = {}
    fs["xypixelsize"] = 0.1040
    fs["angle_fixed"] = 31.8
    fs["sytems_magnification"] = 62.5
    fs["pixel_pitch_um"] = 6.5
    with open(file, "w") as f:
        json.dump(fs, f)


def read_fixed_settings(file: str) -> Dict[str, Any]:
    """reads a fixed settings .json file and returns the settings as a dictionary
    
    Parameters
    ----------
    file : str
            Path to fixed settings file
    
    Returns
    -------
    Dict[str, Any]
            settings dictionary
    """
    with open(file, "r") as fp:
        return json.load(fp)
