# Extract some relevant metadata from Janelia Output
#
# This file contains code for extracting metadata from
# LLS settings file as produced by the Janelia Lab View acquisition
# Software.
#
# Author: Volker dot Hilsenstein at Monash dot edu
#


import re
import pandas as pd


def extract_lls_metadata(settingsfilepath: str, verbose: bool = False) -> pd.DataFrame:
    """Extract LLS metadata from settingsfile
    
    Parameters
    ----------
    settingsfilepath : str
        settings file path
    verbose : bool, optional
        if True, prints some debugging output to the console
    
    Returns
    -------
    pd.DataFrame
        data frame containing settings

    Notes
    -----
    If everything worked well, each row should corrspond to one channel
    Follow the links to regex101.com to see some sample lines.

    TODO:
    -----
    1. there is plenty more metadata we could  extract
    2. error handling
    """

    res = {}

    # https://regex101.com/r/PnABLk/4
    res["ZPZT"] = re.compile(
        r"Z PZT Offset, .*\((?P<channel>\d+)\)\s*:\s*(?P<galvoscan_zoffset>[\d.]+)\s*(?P<galvoscan_interval>[\d.]+)\s*(?P<galvoscan_n_tiff>\d+)\s*"
    )
    # https://regex101.com/r/T0eS35/3
    res["SPZT"] = re.compile(
        r"S PZT Offset, .*\((?P<channel>\d+)\)\s*:\s*(?P<samplescan_zoffset>[\d.]+)\s*(?P<dz_stage>[\d.]+)\s*(?P<samplescan_n_tiff>\d+)"
    )
    # https://regex101.com/r/pjsZXC/3
    res["Laser"] = re.compile(
        r"Excitation Filter, Laser,.*\((?P<channel>\d+)\)\s+:\s+(?P<filter>.+)\s+(?P<lambda>[\d.]+)\s(?P<power>[\d.]+)\s(?P<exposure>[\d.]+).*"
    )
    # https://regex101.com/r/eGnfJz/2
    res["Zmotion"] = re.compile(r"Z motion\s+:\s+(?P<zmotion>.+)")
    # https://regex101.com/r/3OeXLm/1
    res["angle"] = re.compile(r"Angle between stage.*=\s*(?P<angledeg>.+)")

    with open(settingsfilepath, "r") as f:
        text = f.read()

    dfs = {}
    for key in res.keys():
        if verbose:
            print("### Extracting metadata pattern " + key)
        matchdict = [m.groupdict() for m in res[key].finditer(text)]
        dfs[key] = pd.DataFrame(matchdict)

    # Merge multiple-row tables beased on channel
    metadata = dfs["SPZT"].merge(dfs["ZPZT"], on="channel").merge(dfs["Laser"], on="channel")
    # Add columns with single values
    metadata["angle"] = dfs["angle"].iloc[0][0]
    metadata["zmotion"] = dfs["Zmotion"].iloc[0][0]
    # Change columns to numeric where possible
    metadata = metadata.apply(pd.to_numeric, errors="ignore")
    return metadata


def _test_extraction():
    filename = "c:/Users/Volker/Dropbox/Gitlab/lattice-lightsheet/ExampleSettings/488_300mW_642_350mW_Settings.txt"
    return extract_lls_metadata(filename)
