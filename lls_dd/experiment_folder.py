import pathlib
import re
import pandas as pd
from natsort import natsorted
from typing import Union, List, Any
from lls_dd.extract_metadata import extract_lls_metadata
from lls_dd.settings import read_fixed_settings


class Experimentfolder(object):
    """
    Represents data relating to an experiment folder
    """

    regex_PSF = re.compile(
        r".*PSF[/\\](?P<wavelength>\d+)[/\\](?P<scantype>.*)[/\\].*_(?P<abssec>\d+)msecAbs\.tif"
    )
    regex_Stackfiles = re.compile(
        r".*[/\\](?P<prefix>.+)_ch(?P<channel>\d+)_[^\d]*(?P<stack_nr>\d+)_(?P<wavelength>\d+)nm_(?P<reltime_ms>\d+)msec_(?P<abstime_ms>\d+)msec*"
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
        self.defaultPSFs = None  # TODO implement default PSFs

        self.scan_folder()

    def __repr__(self):
        return f"""
        folder: {self.folder}
        fixed_settings_file: {self.fixed_settings_file}
        settings: {self.settings}
        PFSs: {self.PSFs}
        defaultPSFs: {self.defaultPSFs}
        stacks: {self.stacks}
        stackfiles: {self.stackfiles}
        """

    def _str_stacks(self):
        msg = ["Stacks:"]
        for i, stack in enumerate(self.stacks):
            msg.append(f"[{i}] {stack}")
        return "\n".join(msg)

    def _str_PSFs(self):
        msg = ["PSFs:"]
        msg.append(str(self.PSFs[["name", "wavelength", "scantype"]]))
        return "\n".join(msg)

    def __str__(self):
        msg = ["Summary of Experimentfolder:"]
        msg.append("============================\n")
        msg += [f"folder path: {str(self.folder)}", ""]
        msg.append(self._str_stacks())
        msg.append("")
        msg.append(self._str_PSFs())
        return "\n".join(msg)

    def scan_folder(self):
        """ scans the experimentfolder and finds stack folder, stack files, PSFs and parses settings """
        self.stackfiles = self.find_stacks()
        self.stacks = natsorted(list(pd.unique(self.stackfiles.stack_name)))
        self.PSFs = self.find_PSFs()
        self.settings = self.find_settings()
        self.psf_settings = self.find_PSF_settings()
        self._apply_fixed_settings()

    def find_PSFs(self) -> pd.DataFrame:
        """ finds and parses file names of PSFs
        """
        files = (self.folder / "PSF").rglob("*.tif")

        # This complicated list comprehension
        # extracts some fields from the Path using
        # a regular expression

        # TODO: what happens if unexpected tiff files are present?
        # matchdict = [
        #    {**self.regex_PSF.match(str(f)).groupdict(), **{"file": str(f)}} for f in files
        # ]
        matchdict = []
        for f in files:
            match = self.regex_PSF.match(str(f))
            if match:
                matchdict.append({**match.groupdict(), **{"file": str(f)}})
        df = pd.DataFrame(matchdict)
        df["name"] = df.file.apply(lambda x: pathlib.Path(x).name)
        return df

    def find_OTFs(self):
        """ checks whether OTFs exist and finds and parses filenames of OTFs
        (this is currently not used... placeholder in case we are using cudaDeconv)."""
        pass

    def find_stacks(self) -> pd.DataFrame:
        """ finds all the tiff-volumes in stacks and creates a data table with metadata
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
            for f in stackfiles:
                match = self.regex_Stackfiles.match(str(f))
                if match:
                    matched_stacks += [
                        {**match.groupdict(), **{"file": str(f), "stack_name": stackname}}
                    ]
        # TODO: (flagged for removal) matched_stacks = natsorted(matched_stacks)
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
