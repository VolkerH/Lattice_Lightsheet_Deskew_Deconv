from experiment_folder import Experimentfolder
from psf_tools import generate_psf
import pathlib
import tifffile
import numpy as np
from settings import read_fixed_settings
from transform_helpers import *
from utils import *


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
        fixed_settings: fixed settings that are not read from the 
                        per-experiment settings file. 
                        (TODO maybe put them in a json file)
        skip_existing: if True, skip processing if the output file already exists
        skip_files_regex: list of regular expressions that are applied
                        to the input files. Any matching file will
                        be disregarded. (useful to not process unwanted wavelengths)
        exp_outfolder: (optional) if the outpust subfolders are not to be created in the input experiment folder,
                       provide a pathlib.Path object for the desired output folder here
        dask_settings: dictionary of settings if dask is to be used to distribute jobs.
        TODO:
        Where to set the process options ?
        
        
        Comment: 
        By moving a lot of the parameters into the instance variables we don't need to
        pass them aorund. 
        However, when batch processing there is a risk that someone may change the parameters
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

        self._montage_gap = 10 # gap between projections in orthogonal view montage
        self.output_imaris = False # Todo, implement imaris output

        self.verbose = True # if True, prints diagnostic output

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
        outfiles["deconv/deskew/MIP"] = out_base / "py_deconv" / "deskew" / f"{stem}_deconv_deskew_MIP{suffix}"
        outfiles["deconv/rotate/MIP"] = out_base / "py_deconv" / "rotate" / f"{stem}_deconv_rotate_MIP{suffix}"
        
        return(outfiles)

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
            
    def process_file(self, infile, deskew_func=None, rotate_func=None, deconv_func=None, bg_subtract=0): #n_iter=10, psf=None):
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
            warnings.warn(f"nothing to do for {infile}. All outputs already exist. '\
                             Disable skip-existing to overwrite")
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vol_raw = tifffile.imread(str(infile))
        vol_raw -= bg_subtract # TODO implement proper bg subtraction

        if self.do_deskew and not checks[0]:
            deskewed = deskew_func(vol_raw)
            write_tiff_createfolder(outfiles["deskew"], deskewed) 
            if self.do_MIP:
                 self.create_MIP(deskewed, outfiles["deskew/MIP"])
            # write settings/metadata file to subfolder
        if self.do_rotate and not checks[1]:
            rotated = rotate_func(vol_raw)
            write_tiff_createfolder(outfiles["rotate"], rotated)
            if self.do_MIP:
                 self.create_MIP(rotated, outfiles["rotate/MIP"])
            # write settings/metadata file to subfolder
        if self.do_deconv:
            deconv_raw = deconv_func(vol_raw)
            # TODO: write deconv settings
            if self.do_deconv_deskew:
                deconv_deskewed = deskew_func(deconv_raw)
                write_tiff_createfolder(outfiles["deconv/deskew"], deconv_deskewed)
                if self.do_MIP:
                    self.create_MIP(deconv_deskewed, outfiles["deconv/deskew/MIP"])               
            if self.do_deconv_rotate:
                deconv_rotated = rotate_func(deconv_raw)
                write_tiff_createfolder(outfiles["deconv/rotate"], deconv_rotated)
                if self.do_MIP:
                    self.create_MIP(deconv_rotated, outfiles["deconv/rotate/MIP"]) 
                
    def process_stack_subfolder(self, stack_name):
        """ process a timeseries """
        
        # get subset of files and settings specific to this stack 
        subset_files = self.ef.stackfiles[self.ef.stackfiles.stack_name == stack_name]
        subset_files = subset_files.reset_index()
        stack_settings =  self.ef.settings[self.ef.settings.stack_name == stack_name]
        stack_settings = stack_settings.reset_index()

        # Take dz_stage setting from first row 
        # based on assumption that all channels have same dz_stage
        
        # calculate deskew factor and create volume deskew and deskew/rotate partial functions
        # we need the input file shape, ... this is not in the metadata, so we read the first volume
        # in the timeseries and get the shape from there

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("subset_files", subset_files)
            print("trying to read file", subset_files.file[0])
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
            # TODO: deal with this later
            # prepare deconv
            # process PSFs
            wavelengths = subset_files.wavelength.unique() #find which wavelengths are present in files
            processed_psfs = {}
            for wavelength in wavelengths:
                #find all PSF files matching this wavelength where scan=='Galvo'
                psf_candidates = self.ef.PSFs[(self.ef.PSFs.scantype == 'Galvo') & (self.ef.PSFs.wavelength==wavelength)]
                psf_candidates = psf_candidates.reset_index()
                if len(psf_candidates) == 0:
                    warnings.warn(f"no suitable PSF found for {wavelength}")
                    # TODO: revert to a default
                    raise ValueError("No  suitable PSF")
                elif len(psf_candidates) > 1:
                    warnings.warn(f"more than one PSF found. Taking first one")
                    # TODO define rules which PSF to choose (first , last, largest filesize?)
                
                psffile = psf_candidates.file[0]
                # find galvo z-step setting

                tmp = self.ef.psf_settings[(self.ef.psf_settings.scantype == "Galvo") & (self.ef.psf_settings["lambda"]==int(wavelength))]
                tmp.reset_index()
                dz_galvo = tmp.galvoscan_interval[0]
                print("dz galvo", dz_galvo)
                processed_psfs[wavelength] = generate_psf(psffile, tmp_vol.shape, dz_stage, dz_galvo, xypixelsize, angle)
                print(f"processing PSF file {psffile} for wavelength {wavelength}")

                # TODO save PSF to disk
                
                # TODO this is debugging code, remove
                print("processed_psf.keys()", processed_psfs.keys())
       
        ### Start batch processing 
        for index, row in subset_files.iterrows():
            
            if self.verbose:
                print(f"Processing {index}: {row.file}")
            # TODO implement regex check for files to skip

            # TODO pass in psf for channel
            self.process_file(pathlib.Path(row.file), deskew_func, rotate_func)

    def process_all(self):
        """
        Process all timeseries (stacks) in experiment folder
        """
        for stack in self.ef.stacks:
            self.process_stack_subfolder(stack)

        



    """
    Original "Pseudo-code" below ...
    remove when done


    given Experimentfolder
    find all stacks in Experimentfolder
    read fixed settings
    for each stack:
        ################################
        # general preparations for stack
        ################################
        read settings for stack
        calculate deskew factor
        calculate deskew transform
        calculate deskew/rotate transform
        ################################
        # deconvolution preparations for stack
        ################################
        if deconvolution_wanted:
            find which wavelengths are present in files
            for each wavelength present:
                find all PSF files matching this wavelength where scan=='galvo'
                select a single PSF file based on ?? (newest file, largest file)
                if no PSF file found:
                    print warning and revert to a default PSF matching the same settings (beam pattern)
                    (TODO create defaults)
                calculate dz ratio
                process PSF for use with deconv and add to dictionary indexed by wavelength
     #######################
     # Batch processing
     #######################
     for each file in stack (sorted by timepoint, wavelength):
         submit job:
             assemble output filenames
             if all output files already exist and skip_existing
                 return
             read file
             if deskew:
                 if deskew does not already exist:
                     deskew file and write to subfolder "deskewed" (create folder if not present)
                     write settings/metadata file to subfolder
             if rotate:
                  if rotate output does not already exist:
                     deskew/rotate file and write to subfolder "rotated"
                     write settings/metadata file to subfolder
             if deconvolution_wanted:
                  deconvolve raw/skewed
                  if deskew:
                      deskew deconvolved
                      write to subfolder "deconv_deskew"
                      write settings/metadata file to subfolder
                  if rotate:
                      deskew/rotate deconvolved
                      write to subfolder "deconv rotate"
                      write settings/metadata file to subfolder
    """
    
