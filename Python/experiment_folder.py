import pathlib
import re
import pandas as pd
from extract_metadata import extract_lls_metadata
from settings import read_fixed_settings
from typing import Union, List, Any


class Experimentfolder(object):
    """
    Represents data relating to an experiment folder
    """

    regex_PSF = re.compile(r".*PSF[/\\](?P<wavelength>\d+)[/\\](?P<scantype>.*)[/\\].*_(?P<abssec>\d+)msecAbs\.tif")
    regex_Stackfiles = re.compile(
        r".*[/\\](?P<prefix>.+)_ch(?P<channel>\d+)_[^\d]*(?P<zslice>\d+)_(?P<wavelength>\d+)nm_(?P<reltime_ms>\d+)msec_(?P<abstime_ms>\d+)msec*"
    )

    def __init__(self, f: Union[str, pathlib.Path], fixed_settings_file: str = "fixed_settings.json"):
        if not isinstance(f, pathlib.Path):
            # try to convert into pathlib Path if something else has been passed in
            f = pathlib.Path(f)
        self.folder: pathlib.Path = f
        assert f.exists()

        self.stacks: Union[List[str], None] = None
        # all image files in the stack
        self.stackfiles: Union[pd.DataFrame, None] = None
        self.PSFs: Union[pd.DataFrame, None] = None
        self.settings: Union[pd.DataFrame, None] = None
        self.psf_settings: Union[pd.DataFrame, None] = None
        self.fixed_settings_file: str = fixed_settings_file
        self.defaultPSFs = None  # TODO

        self.scan_folder()
        # self.print_diagnostics()

    def print_diagnostics(self):
        print("Summary of experiment folder " + str())
        print("Stacks:")
        print(self.stacks)
        print("Stack files:")
        print(self.stacks)
        print("PSFs:")
        print(self.PSFs)
        print("Settings:")
        print(self.settings)

    def scan_folder(self):
        self.stackfiles = self.find_stacks()
        self.stacks = list(pd.unique(self.stackfiles.stack_name))
        self.PSFs = self.find_PSFs
        self.settings = self.find_settings()
        self.psf_settings = self.find_PSF_settings()
        self._apply_fixed_settings()

    def find_PSFs(self) -> pd.DataFrame:
        """ finds and parses file names of PSF
        """

        files = (self.folder / "PSF").rglob("*.tif")
        files = map(str, files)  # type: ignore

        # This complicated list comprehension
        # extracts some fields from the Path using
        # a regular expression

        # TODO: what happens if an unexpected tiff file is present?
        matchdict = [{**self.regex_PSF.match(f).groupdict(), **{"file": f}} for f in files]  # type: ignore
        df = pd.DataFrame(matchdict)
        return df

    def find_OTFs(self):
        """ checks whether OTFs exist and finds and parses filenames of OTFs"""
        pass

    def find_stacks(self) -> pd.DataFrame:
        """ finds all the stacks in stacks and creates a data table with metadata
        """
        # Glob folders below stack first, want to avoid
        # recursive glob on .tifs because this will also
        # find all the deskewed stuff

        allfiles = (self.folder / "Stacks").glob("*")
        stackfolders = list(filter(lambda x: x.is_dir(), allfiles))
        # stacknames = list(map(lambda f: str(f.name), stackfolders))
        matched_stacks: List[Any] = []
        for sf in stackfolders:
            stackname = sf.name
            stackfiles = list(sf.glob("*.tif"))
            stackfiles = map(str, stackfiles)  # type: ignore
            matched_stacks += [
                {**self.regex_Stackfiles.match(f).groupdict(), **{"file": f, "stack_name": stackname}}  # type: ignore
                for f in stackfiles
            ]
        return pd.DataFrame(matched_stacks)

    def find_settings(self) -> pd.DataFrame:
        """ reads and parses the settings files for all the stacks in order to extract
        data such as dz step """
        sfiles = list((self.folder / "Stacks").rglob("*Settings.txt"))

        def process_settings(sfile: pathlib.Path) -> pd.DataFrame:
            settings = extract_lls_metadata(str(sfile))
            settings["stack_folder"] = str(sfile.parent)
            settings["stack_name"] = str(sfile.parent.name)
            settings["settingsfile"] = sfile
            return settings

        settings = pd.concat(map(process_settings, sfiles))
        return settings

    def find_PSF_settings(self) -> pd.DataFrame:
        """
        Similar to find_settings but for the PSFs rather than the stacks
        currently as stub TODO
        """
        sfiles = list((self.folder / "PSF").rglob("*Settings.txt"))

        def process_psf_settings(sfile):
            settings = extract_lls_metadata(str(sfile))
            settings["psf_folder"] = str(sfile.parent)
            settings["scantype"] = str(sfile.parent.name)
            settings["psf_settingsfile"] = sfile.name
            return settings

        settings = pd.concat(map(process_psf_settings, sfiles))
        return settings

    def _apply_fixed_settings(self):
        """ this will append fixed settings to the settings data frame.
        These are settings given due to the microscope hardware (camera, stage angle)
        that should not change between experiments for a given microscope).
        """
        fixed_settings = read_fixed_settings(str(self.fixed_settings_file))
        # turn into a pd.series
        fs = pd.Series(fixed_settings)
        # repeat as many times as we have rows in the overall settings
        fs_df = pd.concat([fs] * len(self.settings), axis=1).transpose()
        # and add to settings

        self.settings = self.settings.reset_index()

        # TODO: deal with case where settings name appers both
        # in settings file and in fixed settings
        self.settings = pd.concat([self.settings, fs_df], axis=1)
