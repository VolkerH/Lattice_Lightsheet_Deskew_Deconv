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

def extract_lls_metadata(settingsfilepath, verbose=False):
    ''' Extract LLS Metadata from settingsfile
    Given a settingsfilepath open the file and extract 
    the metadata we need for deskew, deconvolve
    
    Returns the metadata as a pandas dataframe.

    If everything worked well, each row should corrspond to one channel
    Follow the links to regex101.com to see some sample lines.

    TODO:
    1. there is plenty more metadata we could  extract
    2. error handling
    '''
    res = {}

    # https://regex101.com/r/PnABLk/4
    res["ZPZT"] = re.compile("Z PZT Offset, .*\((?P<channel>\d+)\)\s*:\s*(?P<galvoscan_zoffset>[\d.]+)\s*(?P<galvoscan_interval>[\d.]+)\s*(?P<galvoscan_n_tiff>\d+)\s*")
    # https://regex101.com/r/T0eS35/3
    res["SPZT"] = re.compile("S PZT Offset, .*\((?P<channel>\d+)\)\s*:\s*(?P<samplescan_zoffset>[\d.]+)\s*(?P<dz_stage>[\d.]+)\s*(?P<samplescan_n_tiff>\d+)")
    #https://regex101.com/r/pjsZXC/3
    res["Laser"] = re.compile("Excitation Filter, Laser,.*\((?P<channel>\d+)\)\s+:\s+(?P<filter>.+)\s+(?P<lambda>[\d.]+)\s(?P<power>[\d.]+)\s(?P<exposure>[\d.]+).*")
    #https://regex101.com/r/eGnfJz/2
    res["Zmotion"] = re.compile("Z motion\s+:\s+(?P<zmotion>.+)")
    #https://regex101.com/r/3OeXLm/1
    res["angle"] = re.compile('Angle between stage.*=\s*(?P<angledeg>.+)')

    with open(settingsfilepath , 'r') as f:
        text = f.read()

    dfs = {}
    for key in res.keys():
        if verbose:
            print("### Extracting metadata pattern " +key)
        matchdict = [m.groupdict() for m in res[key].finditer(text)]
        dfs[key] = pd.DataFrame(matchdict)
        
    # Merge multiple-row tables beased on channel
    metadata = dfs["SPZT"].merge(dfs["ZPZT"], on="channel").merge(dfs["Laser"], on="channel")    
    # Add columns with single values
    metadata["angle"]=dfs["angle"].iloc[0][0]
    metadata["zmotion"]=dfs["Zmotion"].iloc[0][0]
    # Change columns to numeric where possible
    metadata = metadata.apply(pd.to_numeric, errors='ignore')
    return metadata

def test_extraction():
    filename = 'c:/Users/Volker/Dropbox/Gitlab/lattice-lightsheet/ExampleSettings/488_300mW_642_350mW_Settings.txt'
    return(extract_lls_metadata(filename))