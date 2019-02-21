from experiment_folder import Experimentfolder
from psf_tools import generate_psf
import pathlib
import tifffile
import tqdm
import numpy as np
from settings import read_fixed_settings
from transform_helpers import get_rotate_function, get_deskew_function, get_projections, get_projection_montage
from utils import *
from deconvolution import init_rl_deconvolver, get_deconv_function

import logging
logging.getLogger("tifffile").setLevel(logging.ERROR)

# TODO: change terminology: stacks -> timeseries ?
#     :                     single timepoint -> stack
# TODO: discuss with David

class ExperimentProcessor(object):
    def __init__(self, ef, skip_files_regex=[], skip_existing=True, exp_outfolder = None , dask_settings={}):
        """
        Input:
        ef: Experimentfolder class
        skip_existing: if True, skip processing if the output file already exists
        skip_files_regex: list of regular expressions that are applied
                        to the input files. Any matching file will
                        be disregarded. (useful to not process unwanted wavelengths)
        exp_outfolder: (optional) if the output subfolders are not to be created in the input experiment folder,
                       provide the desired output folder here. Can be string or Pathlib.Path object.
        dask_settings: (not yet implemented)
                      dictionary of settings if dask is to be used to distribute jobs.
        """
        self.ef = ef
       
        self.skip_files_regex = skip_files_regex
        self.processedPSFCache = None
        self.skip_existing = skip_existing

        if exp_outfolder is None:
            self.exp_outfolder = ef.path
        else:
            self.exp_outfolder = pathlib.Path(exp_outfolder)
        # Which processing steps should be performed ?
        # TODO: add functions to set these

        # if do_MIP is set, it will be applied to any of the other processing steps
        # as it is such a cheap operation to perform 
        self.do_MIP = True

        self.do_deskew = True
        self.do_rotate = True

        self.do_deconv = False # set to True if performing deconvolution on skewed raw volume
        self.do_deconv_deskew = False # if you want the deconv deskewed 
        self.do_deconv_rotate = False # if you want the deconv rotated
        self.do_deconv = self.do_deconv or self.do_deconv_deskew or self.do_deconv_rotate
        self.deconv_n_iter = 10

        self._montage_gap = 10 # gap between projections in orthogonal view montage
        self.output_imaris = False # TODO: implement imaris output using Talley's imarispy
        self.output_bdv = False # TODO: 
        self.output_dtype = np.uint16 # set the output dtype
        self.verbose = False # if True, prints diagnostic output

    def generate_outputnames(self, infile):
        """ 
        generates the output paths (including subfolders) for the processed data 
        inputs:
        infile: pathlib.Path object for the input file

        returns: dictionary with pathlib.path commands
        """
        outfiles = {}
        parents = infile.parents
        name = infile.name
        suffix = infile.suffix
        stem = infile.stem
        if self.verbose:
            print(f"Experiment outputfolder {self.exp_outfolder}")
        out_base  = self.exp_outfolder / parents[1].name / parents[0].name
 
        outfiles["deskew"] = out_base / "py_deskew" / f"{stem}_deskew{suffix}"
        outfiles["rotate"] = out_base / "py_rotate" / f"{stem}_rotate{suffix}"
        outfiles["deconv"] = out_base / "py_deconv" / f"{stem}_deconv_raw{suffix}" # TODO: implement or remove
        outfiles["deconv/deskew"] = out_base / "py_deconv" / "deskew" / f"{stem}_deconv_deskew{suffix}"
        outfiles["deconv/rotate"] = out_base / "py_deconv" / "rotate" / f"{stem}_deconv_rotate{suffix}"
        # Maximum intensity projections ... 
        # 
        # (TODO: improve later, too much boiler plate code for my taste)
        outfiles["deskew/MIP"] = out_base / "py_deskew" / "MIP" / f"{stem}_deskew_MIP{suffix}" 
        outfiles["rotate/MIP"] = out_base / "py_rotate" / "MIP" / f"{stem}_rotate_MIP{suffix}"  
        outfiles["deconv/deskew/MIP"] = out_base / "py_deconv" / "deskew" / "MIP" / f"{stem}_deconv_deskew_MIP{suffix}"
        outfiles["deconv/rotate/MIP"] = out_base / "py_deconv" / "rotate" / "MIP" / f"{stem}_deconv_rotate_MIP{suffix}"
        
        return(outfiles)
    
    def generate_PSF_name(self, wavelength):
        """ 
        generates the output path (including subfolders) for the PSF file
        """
        return self.exp_outfolder / "PSF_Processed" / f"{wavelength}"/ f"PSF_{wavelength}.tif"

    def create_MIP(self, vol, outfile, method="montage"):
        """
        Creates a MIP from a volume and saves it to outfile

        input: vol
        outfile: output file path
        method: one of ["montage", "multi", "individual"]
        "montage" montages all three MIPs into a single 2D image
        "individual" saves the projections as individual files. The output file name
            is modified by replacing the suffix (e.g. ".tif") with f"_{a}"+suffix where
            a is the axis number (0,1,2)
        """
        assert(method in ["montage", "individual"])
    
        try:
            if method == "montage":
                montage = get_projection_montage(vol, gap=self._montage_gap)
                write_tiff_createfolder(str(outfile), montage)
            if method == "multi":
                projections = get_projections(vol)
                for i, proj in enumerate(projections):
                    axisfile = outfile.parent / f"{outfile.stem}_{i}{outfile.suffix}"
                    write_tiff_createfolder(str(axisfile), proj)
        except:
            warnings.warn(f"Error creating MIP {str(outfile)} ... skipping")
            
    def process_file(self, infile, deskew_func=None, rotate_func=None, deconv_func=None, bg_subtract=0):
        """ process an individual file 
        file: input file (pathlib object)

        deskew_func: the deskew function
        rotate_func: the rotate function 
        deconv_func: the deconv function

        bg_subtract: constant value to subtract from the background

        n_iter = (optional) number of iterations for Richardson-Lucy. Defaults to 10.
        psf: (optional) psf for this file, only required if deconvolving
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
                warnings.warn(f"nothing to do for {infile}. All outputs already exist. '\
                                 Disable skip-existing to overwrite")
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vol_raw = tifffile.imread(str(infile))
        vol_raw -= bg_subtract # TODO implement proper bg subtraction

        if self.do_deskew and not checks[0]:
            deskewed = deskew_func(vol_raw)
            write_tiff_createfolder(outfiles["deskew"], deskewed.astype(self.output_dtype)) 
            if self.do_MIP:
                 self.create_MIP(deskewed.astype(self.output_dtype), outfiles["deskew/MIP"])
            # write settings/metadata file to subfolder
        if self.do_rotate and not checks[1]:
            rotated = rotate_func(vol_raw)
            write_tiff_createfolder(outfiles["rotate"], rotated.astype(self.output_dtype))
            if self.do_MIP:
                 self.create_MIP(rotated.astype(self.output_dtype), outfiles["rotate/MIP"])
            # write settings/metadata file to subfolder
        if self.do_deconv:
            deconv_raw = deconv_func(vol_raw)
            # TODO: write deconv settings
            if self.do_deconv_deskew:
                deconv_deskewed = deskew_func(deconv_raw)
                write_tiff_createfolder(outfiles["deconv/deskew"], deconv_deskewed.astype(self.output_dtype))
                if self.do_MIP:
                    self.create_MIP(deconv_deskewed.astype(self.output_dtype), outfiles["deconv/deskew/MIP"])               
            if self.do_deconv_rotate:
                deconv_rotated = rotate_func(deconv_raw)
                write_tiff_createfolder(outfiles["deconv/rotate"], deconv_rotated.astype(self.output_dtype))
                if self.do_MIP:
                    self.create_MIP(deconv_rotated.astype(self.output_dtype), outfiles["deconv/rotate/MIP"]) 
                
    def process_stack_subfolder(self, stack_name):
        """ process a timeseries """
        
        # get subset of files and settings specific to this stack 
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
        xypixelsize=stack_settings.xypixelsize[0]
        angle = stack_settings.angle_fixed[0]
        if self.verbose:
            print("Generating deskew function")

        deskew_func = get_deskew_function(tmp_vol.shape, dz_stage, xypixelsize, angle)
        if self.verbose:
            print("Generating rotate function")
        rotate_func = get_rotate_function(tmp_vol.shape, dz_stage, xypixelsize, angle)
        
        if self.do_deconv:
            # Prepare deconvolution:
            
            ## Here we initialize a single deconvolver that gets used
            ## for all deconvolutions. 
            ## I have doubts whether this will work if several threads run in parallel,
            ## I assume a deconvolver will have to be initialized for each worker
            ## Therefore this may have to be moved into `get_deconv_func` (TODO)
            deconvolver = init_rl_deconvolver()
            
            # Preprocess PSFs and create deconvolution functions
            wavelengths = subset_files.wavelength.unique() #find which wavelengths are present in files
            processed_psfs = {}
            deconv_functions = {}
            for wavelength in wavelengths:
                #find all PSF files matching this wavelength where scan=='Galvo'
                psf_candidates = self.ef.PSFs[(self.ef.PSFs.scantype == 'Galvo') & (self.ef.PSFs.wavelength==wavelength)]
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
                tmp = self.ef.psf_settings[(self.ef.psf_settings.scantype == "Galvo") & (self.ef.psf_settings["lambda"]==int(wavelength))]
                tmp.reset_index()
                dz_galvo = tmp.galvoscan_interval[0]
                if self.verbose:
                    print("dz galvo interval", dz_galvo)
                    print(f"processing PSF file {psffile} for wavelength {wavelength}")
                processed_psfs[wavelength] = generate_psf(psffile, tmp_vol.shape, dz_stage, dz_galvo, xypixelsize, angle)
                write_tiff_createfolder(self.generate_PSF_name(wavelength), processed_psfs[wavelength])
                deconv_functions[wavelength] = get_deconv_function(processed_psfs[wavelength], deconvolver, self.deconv_n_iter)
                
        ### Start batch processing 
        for index, row in tqdm.tqdm(subset_files.iterrows(), total=subset_files.shape[0]):
            if self.verbose:
                print(f"Processing {index}: {row.file}")
            # TODO implement regex check for files to skip

            if self.do_deconv:
                # determine wavelength for file and pick corresponding PSF
                self.process_file(pathlib.Path(row.file), deskew_func, rotate_func, deconv_functions[wavelength])
            else:
                self.process_file(pathlib.Path(row.file), deskew_func, rotate_func)

    def process_all(self):
        """
        Process all timeseries (stacks) in experiment folder
        """
        for stack in tqdm.tqdm(self.ef.stacks):
            self.process_stack_subfolder(stack)
