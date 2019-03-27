import warnings
import logging
import pathlib
import tifffile
import tqdm
import numpy as np
import os
from typing import Iterable, Callable, Optional, Union, Any, Dict, DefaultDict
from collections import defaultdict
from .experiment_folder import Experimentfolder
from .psf_tools import generate_psf
from .transform_helpers import (
    get_deskew_function,
    get_rotate_to_coverslip_function,
    get_projections,
    get_projection_montage,
)
from .utils import write_tiff_createfolder

# from scipy.ndimage.filters import gaussian_filter
# note: deconvolution functions will be imported according to chosen backend later

# tifffile produces too many warnings for the Labview-generated tiffs. Silence it:
logging.getLogger("tifffile").setLevel(logging.ERROR)
logger = logging.getLogger("lls_dd")
logger.setLevel(logging.ERROR) # TODO: combine with verbose in ExperimentProcessor ?


# TODO: change terminology: stacks -> timeseries ?
#     :                     single timepoint -> stack
# TODO: discuss with David


class ExperimentProcessor(object):
    def __init__(
        self,
        ef: Experimentfolder,
        skip_existing: bool = True,
        skip_files_regex: Optional[Iterable[str]] = None,
        exp_outfolder: Optional[Union[str, pathlib.Path]] = None,
    ):
        """Initialize an ExperimentProcessor object
        
        Parameters
        ----------
        object : [type]
            self
        ef : Experimentfolder
            an initiliazed Experimentfolder object
        skip_existing : bool, optional
            don't process files for which the output files already exists (the default is True)
        skip_files_regex : Optional[Iterable[str]], optional
            a regular expression that matches files which are not to be processed (not yet implemented!)
        exp_outfolder : Optional[Union[str, pathlib.Path]], optional
            path to an output folder (the default is None, which means processed volumes are stored in 
            subfolders of the original Experimentfolder)
        
        """
      
        self.ef: Experimentfolder = ef

        self.skip_files_regex: Optional[Iterable[str]] = skip_files_regex
        self.processedPSFCache = (
            None
        )  # TODO: this is not currently used. Seem to reprocess PSFs for each stack?
        self.skip_existing = skip_existing

        if exp_outfolder is None:
            self.exp_outfolder = ef.folder
        else:
            self.exp_outfolder = pathlib.Path(exp_outfolder)

        # Set processing options
        # camera
        self.bg_subtract_value: int = 100  # Value to be subtracted from each input file ~200 for Orca

        # output options
        self.do_MIP: bool = True  # if set, MIPs will be generated for all of the operations
        self.do_deskew: bool = True
        self.do_rotate: bool = True
        self.MIP_method: str = "montage"  # one of ["montage", "multi"]
        self._montage_gap: int = 10  # gap between projections in orthogonal view montage
        self.output_imaris: bool = False  # TODO: implement imaris output using Talley's imarispy
        self.output_bdv: bool = False  # TODO:
        self.output_dtype = np.float32  # set the output dtype (no check for overflow)
        self.output_tiff_compress: Union[
            int, str
        ] = 0  # compression level 0-9 or string (see tifffile)

        # deconvolution and deconvolution output options
        self.do_deconv: bool = False  # set to True if performing deconvolution on skewed raw volume
        self.deconv_backend: str = "flowdec"  # can be "flowdec" or "gputools"
        self.do_deconv_deskew: bool = False  # if you want the deconv deskewed
        self.do_deconv_rotate: bool = True  # if you want the deconv rotated
        self.do_deconv = (
            self.do_deconv or self.do_deconv_deskew or self.do_deconv_rotate
        )
        self.deconv_n_iter: int = 10

        # general
        self.verbose: bool = False  # if True, prints diagnostic output

    def _description_variable_pairs(self):
        return [
            ["skip existing output files", self.skip_existing],
            ["Perform max intensity projections", self.do_MIP],
            ["max intensity projection output", self.MIP_method],
            ["Perform deskew", self.do_deskew],
            ["Rotate to coverslip", self.do_rotate],
            ["Perform deconvolution", self.do_deconv],
            ["Devonvolution backend", self.deconv_backend],
            ["Number of iterations for deconvolution", self.deconv_n_iter],
            ["Perform deskew after deconvolution", self.do_deconv_deskew],
            ["Rotate to coverslip after deconvolution", self.do_deconv_rotate],
            ["Background subtraction value for camera", self.bg_subtract_value],
        ]  # TODO: add missing

    def __repr__(self):
        # this should actually be more concise than __str__ but,
        # __str__ will do for now
        return self.__str__()

    def __str__(self):
        msg = ["ExperimentProcessor summary"]
        msg += ["===========================\n"]
        msg += ["Processor for experiment folder:", f"{self.ef.folder}"]
        msg += ["Output folder:"]
        msg += [f"{self.exp_outfolder}"]
        msg += ["Processing options:"]
        msg += ["==================:"]
        pairs = self._description_variable_pairs()
        for pair in pairs:
            msg.append(f"{pair[0]}: {pair[1]}")
        return "\n".join(msg)

    def generate_outputnames(self, infile: pathlib.Path) -> Dict[str, pathlib.Path]:
        """generate full output paths for the processed volumes
        
        Parameters
        ----------
        infile : pathlib.Path
            input file path from which the output paths will be derived from

        Returns
        -------
        Dict[str, pathlib.Path]
            dictionary of Path objects for the various output files
        """

        outfiles = {}
        parents = infile.parents
        suffix = infile.suffix
        stem = infile.stem
        if self.verbose:
            print(f"Experiment outputfolder {self.exp_outfolder}")
        out_base = self.exp_outfolder / parents[1].name / parents[0].name

        outfiles["deskew"] = out_base / "py_deskew" / f"{stem}_deskew{suffix}"
        outfiles["rotate"] = out_base / "py_rotate" / f"{stem}_rotate{suffix}"
        outfiles["deconv"] = (
            out_base / "py_deconv" / f"{stem}_deconv_raw{suffix}"
        )  # TODO: implement or remove
        outfiles["deconv/deskew"] = (
            out_base / "py_deconv" / "deskew" / f"{stem}_deconv_deskew{suffix}"
        )
        outfiles["deconv/rotate"] = (
            out_base / "py_deconv" / "rotate" / f"{stem}_deconv_rotate{suffix}"
        )
        # Maximum intensity projections ...
        outfiles["deskew/MIP"] = (
            out_base / "py_deskew" / "MIP" / f"{stem}_deskew_MIP{suffix}"
        )
        outfiles["rotate/MIP"] = (
            out_base / "py_rotate" / "MIP" / f"{stem}_rotate_MIP{suffix}"
        )
        outfiles["deconv/deskew/MIP"] = (
            out_base
            / "py_deconv"
            / "deskew"
            / "MIP"
            / f"{stem}_deconv_deskew_MIP{suffix}"
        )
        outfiles["deconv/rotate/MIP"] = (
            out_base
            / "py_deconv"
            / "rotate"
            / "MIP"
            / f"{stem}_deconv_rotate_MIP{suffix}"
        )

        return outfiles

    def generate_PSF_name(self, wavelength: Union[str, int]) -> pathlib.Path:
        """generates the output Path object (including subfolders) for the PSF file
        
        Parameters
        ----------
        wavelength : Union[str, int]
            wavelength for this channel
        
        Returns
        -------
        pathlib.Path
            PSF filename
        """
        
        return (
            self.exp_outfolder
            / "PSF_Processed"
            / f"{wavelength}"
            / f"PSF_{wavelength}.tif"
        )

    def create_MIP(
        self, vol: np.array, outfile: pathlib.Path, write_func: Callable = write_tiff_createfolder
    ):  
        """creates an image file with a maximum intensity projection of outfile
        
        Parameters
        ----------
        vol : np.array
            volume for which to create MIP
        outfile : pathlib.Path
            Path for the output file. Suffix will be modified for multi method
        write_func : Callable, optional
            function that handles writing the file
        

        Notes
        -----

        The type of MIP is also affected by the instance variable
        `self.MIP_method`
        """
        assert self.MIP_method in ["montage", "multi"]

        try:
            if (
                self.MIP_method == "montage"
            ):  # montages all three MIPs into a single 2D image
                montage = get_projection_montage(vol, gap=self._montage_gap)
                write_func(str(outfile), montage)
            if self.MIP_method == "multi":  # saves the projections as individual files
                projections = get_projections(vol)
                for i, proj in enumerate(projections):
                    axisfile = outfile.parent / f"{outfile.stem}_{i}{outfile.suffix}"
                    write_func(str(axisfile), proj)
        except:
            warnings.warn(f"Error creating MIP {str(outfile)} ... skipping")

    def process_file(
        self,
        infile: pathlib.Path,
        deskew_func: Optional[Callable] = None,
        rotate_func: Optional[Callable] = None,
        deconv_func: Optional[Callable] = None,
        write_func: Callable = write_tiff_createfolder,
    ):
        """Process a single volume according to the self.do_* flags
        
        This method handles deskewing, rotating, deconvolving and
        for a single volume.

        Parameters
        ----------
        infile : pathlib.Path
            Path to input volume
        deskew_func : Optional[Callable], optional
            function that handles deskewing of a raw volume
        rotate_func : Optional[Callable], optional
            function that handles scaling and coverslip rotation of a deskewed volume
        deconv_func : Optional[Callable], optional
            function that handles deconvolution of a raw volume
        write_func : Callable, optional
            function that handles writing of the images        
        """

        outfiles = self.generate_outputnames(infile)
        # Do we have to do anything? Return otherwise.
        checks = [False, False, False, False]
        if self.skip_existing:
            checks = []
            checks.append(self.do_deskew and outfiles["deskew"].exists())
            checks.append(self.do_rotate and outfiles["rotate"].exists())
            checks.append(self.do_deconv_deskew and outfiles["deconv/deskew"].exists())
            checks.append(self.do_deconv_rotate and outfiles["deconv/rotate"].exists())
        if all(checks):
            if self.verbose:
                warnings.warn(
                    f"nothing to do for {infile}. All outputs already exist. '\
                                 Disable skip_existing to overwrite"
                )
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vol_raw = tifffile.imread(str(infile))
        vol_raw = (
            vol_raw.astype(np.float32) - self.bg_subtract_value
        )  # TODO see issue https://github.com/VolkerH/Lattice_Lightsheet_Deskew_Deconv/issues/13
        # vol_raw = np.clip(vol_raw, a_min=0, a_max=None).astype(np.uint16)  # in-place clipping of negative values

        # The following case handling is ugly 
        # (and got even uglier due to type checking (assert statements)
        # and due to use of multi-step affine transform which requires us
        # to deskew and keep intermediate result for rotations)
        if (self.do_deskew and not checks[0]) or (self.do_rotate and not checks[1]):
            assert(deskew_func is not None)
            deskewed = deskew_func(vol_raw)
            if self.do_deskew and not checks[0]:
                write_func(outfiles["deskew"], deskewed.astype(self.output_dtype))
                if self.do_MIP:
                    self.create_MIP(
                        deskewed.astype(self.output_dtype), outfiles["deskew/MIP"]
                    )
            # TODO write settings/metadata file to subfolder
        if self.do_rotate and not checks[1]:
            assert(rotate_func is not None)
            rotated = rotate_func(deskewed)
            write_func(outfiles["rotate"], rotated.astype(self.output_dtype))
            if self.do_MIP:
                self.create_MIP(
                    rotated.astype(self.output_dtype), outfiles["rotate/MIP"]
                )
            # write settings/metadata file to subfolder
        if self.do_deconv:
            assert(deconv_func is not None)
            deconv_raw = deconv_func(vol_raw)
            # TODO: write deconv settings
            if self.do_deconv_deskew and not checks[2] or self.do_deconv_rotate and not checks[3]:
                assert(deskew_func is not None)
                deconv_deskewed = deskew_func(deconv_raw)
                if self.do_deconv_deskew:
                    write_func(
                        outfiles["deconv/deskew"], deconv_deskewed.astype(self.output_dtype)
                    )
                    if self.do_MIP:
                        self.create_MIP(
                            deconv_deskewed.astype(self.output_dtype),
                            outfiles["deconv/deskew/MIP"],
                        )
            if self.do_deconv_rotate and not checks[3]:
                assert(rotate_func is not None)
                deconv_rotated = rotate_func(deconv_deskewed)
                write_func(
                    outfiles["deconv/rotate"], deconv_rotated.astype(self.output_dtype)
                )
                if self.do_MIP:
                    self.create_MIP(
                        deconv_rotated.astype(self.output_dtype),
                        outfiles["deconv/rotate/MIP"],
                    )

    def process_stack_subfolder(
        self, stack_name: str, write_func: Callable = write_tiff_createfolder
    ):
        """Process all files in a "Stack" folder of an Experiment folder
        
        Parameters
        ----------
        stack_name : str
            name of the Stack folder (just the subfolder name, not the full path)
        write_func : Callable, optional
            function that handles image writing
        """
        warnings.warn("Fix write_func stuff to include compression and units")

        # get subset of files and settings specific to this stack
        assert self.ef.stackfiles is not None
        assert self.ef.settings is not None
        assert self.ef.stackfiles is not None
        subset_files = self.ef.stackfiles[self.ef.stackfiles.stack_name == stack_name]
        subset_files = subset_files.reset_index()
        stack_settings = self.ef.settings[self.ef.settings.stack_name == stack_name]
        stack_settings = stack_settings.reset_index()

        # Take dz_stage setting from first row
        # based on assumption that all channels have same dz_stage

        # calculate deskew factor and create volume deskew and deskew/rotate partial functions
        # we need the input file shape, ... this is not in the metadata, so we read the first volume
        # in the timeseries and get the shape from there

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmp_vol = tifffile.imread(subset_files.file[0])
        dz_stage = stack_settings.dz_stage[0]
        xypixelsize = stack_settings.xypixelsize[0]
        angle = stack_settings.angle_fixed[0]
        if self.verbose:
            print("Generating deskew function")

        deskew_func = get_deskew_function(tmp_vol.shape, dz_stage, xypixelsize, angle)
        if self.verbose:
            print("Generating rotate function")
        rotate_func = get_rotate_to_coverslip_function(tmp_vol.shape, dz_stage, xypixelsize, angle)

        processed_psfs = {}
        deconv_functions: DefaultDict[str, Union[None, Callable]] = defaultdict(
            lambda: None
        )
        if self.do_deconv:
            # import selected backend
            if self.deconv_backend == "gputools_rewrite":
                from .deconv_gputools_rewrite import (
                    init_rl_deconvolver,
                    get_deconv_function,
                )
            elif self.deconv_backend == "gputools":
                from .deconvolution_gputools import (
                    init_rl_deconvolver,
                    get_deconv_function,
                )
            elif self.deconv_backend == "flowdec":
                from .deconvolution import init_rl_deconvolver, get_deconv_function
            else:
                warnings.warn(f"unknown deconvolution backend {self.deconv_backend}")
                exit(-1)
            # Prepare deconvolution:
            # Here we initialize a single deconvolver that gets used
            # for all deconvolutions.
            # I have doubts whether this will work if several threads run in parallel,
            # I assume a deconvolver will have to be initialized for each worker
            # Therefore this may have to be moved into `get_deconv_func` (TODO)
            deconvolver = init_rl_deconvolver()

            # Preprocess PSFs and create deconvolution functions
            wavelengths = (
                subset_files.wavelength.unique()
            )  # find which wavelengths are present in files

            for wavelength in wavelengths:
                # find all PSF files matching this wavelength where scan=='Galvo'
                assert self.ef.PSFs is not None
                assert self.ef.psf_settings is not None
                psf_candidates = self.ef.PSFs[
                    (self.ef.PSFs.scantype == "Galvo")
                    & (self.ef.PSFs.wavelength == wavelength)
                ]
                psf_candidates = psf_candidates.reset_index()
                if len(psf_candidates) == 0:
                    warnings.warn(f"no suitable PSF found for {wavelength}")
                    # TODO: fall back to a default PSF for this case (synthetic? or measurment library? Discuss with David)
                    raise ValueError("No  suitable PSF")
                elif len(psf_candidates) > 1:
                    warnings.warn(f"more than one PSF found. Taking first one")
                    # TODO define rules which PSF to choose (first , last, largest filesize?)
                    # TODO check for unfinished tiff files
                psffile = psf_candidates.file[0]
                # find galvo z-step setting
                tmp = self.ef.psf_settings[
                    (self.ef.psf_settings.scantype == "Galvo")
                    & (self.ef.psf_settings["lambda"] == int(wavelength))
                ]
                tmp.reset_index()
                dz_galvo = tmp.galvoscan_interval[0]
                if self.verbose:
                    print("dz galvo interval", dz_galvo)
                    print(f"processing PSF file {psffile} for wavelength {wavelength}")
                processed_psfs[wavelength] = generate_psf(
                    psffile, tmp_vol.shape, dz_stage, dz_galvo, xypixelsize, angle
                )
                write_func(
                    self.generate_PSF_name(wavelength), processed_psfs[wavelength]
                )
                deconv_functions[wavelength] = get_deconv_function(
                    processed_psfs[wavelength], deconvolver, self.deconv_n_iter
                )

        # Start batch processing
        for index, row in tqdm.tqdm(
            subset_files.iterrows(), total=subset_files.shape[0]
        ):
            if self.verbose:
                print(f"Processing {index}: {row.file}")
            # TODO implement regex check for files to skip
            wavelength = row.wavelength
            # what happens if I use a ThreadPoolExectutor/ProcessPoolExecutor here? How will they share
            # the GPU resources.
            self.process_file(
                pathlib.Path(row.file),
                deskew_func,
                rotate_func,
                deconv_functions[wavelength],
            )

    def process_all(self):
        """Process all time series (stacks) in the Experimentfolder this ExperimentProcessor refers to
        """
        for stack in tqdm.tqdm(self.ef.stacks):
            self.process_stack_subfolder(stack)
