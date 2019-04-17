# `lls_dd` 

## `l`attice `l`ight`s`heet `d`eskew and `d`econvolution with GPU-accelerated Python

## About

This project provides GPU-accelerated python code for post-processing (deconvolution, deskew, rotate to coverslip, MIP projections) of image stacks acquired on a lattice light sheet microscope. If no supported GPU is present, all functions can also be run on a CPU.

The repository encompasses:

* The `lls_dd` command line tool for batch processing of experiments
* The `lls_dd` python package which provides the functions on which the command line tool is built.
* Several Jupyter notebooks that illustrate how to calculate and apply the affine transformations underlying the deskew/rotate  steps, how to set up and apply the deconvolution.

## `lls_dd` command line tool usage

### Prerequisites

* `lls_dd` installed (see **Installation** section below)
* it is assumed that your experiments are organized and named according to the [folder structure outlined in `folder_structure.md`](./folder_structure.md).
* `lls_dd` expects a configuration file that contains some fixed settings. The default location for this configuration file is `$HOME/.lls_dd/fixed_settings.json`. When `lls_dd` does not find this file it will try and create it. Make sure you edit the file to reflect the configuration of your microscope (mainly the light sheet angle and the magnification).

### Command-line options help

Run `lls_dd --help` to get an overview of the command line arguments and the main processing commands:

```console
λ lls_dd --help
Usage: lls_dd [OPTIONS] EXP_FOLDER COMMAND [ARGS]...

  lls_dd: lattice lightsheet deskew and deconvolution utility

Options:
  --home TEXT
  --debug / --no-debug
  -f, --fixed_settings TEXT  .json file with fixed settings
  --help                     Show this message and exit.

Commands:
  process  Processes an experiment folder or individual stacks
           therein.
  psfs     list psfs in experiment folder
  stacks   list stacks in experiment folder
```

* `EXP_FOLDER` is the experiment folder that you wich to process.
* The `process` command takes additional `[ARGS]`

You can see the arguments for `process` by passing `--help` after the experiment folder and process command:

```console
λ lls_dd ..\..\..\Data\Experiment_testing_stacks\ process --help
Usage: lls_dd process [OPTIONS] [OUT_FOLDER]

  experiment folder to process (required) output folder (optional)
  Otherwise same as input

Options:
  -M, --MIP                 calculate maximum intensity projections
  --rot                     rotate deskewed data to coverslip
                            coordinates and save
  --deskew                  save deskewed data
  -b, --backend TEXT        deconvolution backend, either "flowdec"
                            or "gputools"
  -i, --iterations INTEGER  if >0, perform deconvolution this number
                            of Richardson-Lucy iterations
  -r, --decon-rot           if  deconvolution was chosen, rotate
                            deconvolved and deskewed data to
                            coverslip coordinates and save.
  -s, --decon-deskew        if  deconvolution was chosen, rotate
                            deconvolved and deskewed data to
                            coverslip coordinates and save.
  -n, --number TEXT         stack number to process. if not provided,
                            all stacks are processed
  --mstyle [montage|multi]  MIP output style
  --skip_existing           if this opting is given, files for which
                            the output already exists will not be
                            processed
  --lzw INTEGER             lossless compression level for tiff
                            (0-9). 0 is no compression
  --help                    Show this message and exit.
```

## Jupyter notebooks

### Sample image volume and PSF file

The juyter notebook requires a sample image volume  and PSF that is too large for Github. Download them here (dropbox links):

* [sample image file (courtesy of Felix Kraus / Monash University)](https://www.dropbox.com/s/34ei5jj0qgylf8q/drp1_dendra2_test_1_CamA_ch0_stack0000_488nm_0000000msec_0018218290msecAbs.tif?dl=0)
* [corresponding PSF](https://www.dropbox.com/s/39ljascy4vkp0tk/488_PSF_galvo_CamA_ch0_stack0000_488nm_0000000msec_0016836088msecAbs.tif?dl=0)

### Documentation and explanation of the algorithms

The following notebooks illustrate the basic algorithms used and provide examples for batch processing.

* **Deskewing** [Jupyter notebook that illustrates how to deskew and rotate with affine transforms](./examples/00_Lattice_Light_Sheet_Deskew.ipynb)
* **Deconvolution** [Demonstrates deskewing both on raw data with skewed PSF (less voxels, therefore faster) and on deskewed data with unskewed PSF](./examples/01_Lattice_Light_Sheet_Deconvolution.ipynb)

## Installation

### Option 1: `conda` + `pip` 

Install anaconda or miniconda.

Create a `conda` environment `llsdd`:

```console
conda env create -f environment_{...}_.yml
```

where `{environment_{...}_.yml}` stands for one of the two provided environment files. Choose
 `environment-no-CUDA.yml` if you do not have a CUDA-compatible graphics card. Choose 
 `evironment-CUDA.yml` if you do have a CUDA compatible gpu. The latter will install `tensorflow-gpu` which will be essential for fast deconvolution.

Activate the new environment with `conda activate llsdd`.

Download and unzip or `git clone` this repository.

Change to the top-level of the cloned or unzipped repository and type `pip install .`. If this completes successfully you should now be able to use the `lls_dd` command line utility.

If the installation fails for `pyopencl`, see 
the paragraph on `pyopencl` in section Troubleshooting.

### Option 2: Docker container

TODO (the Docker container has not been built yet):
If you have `nvidia-docker` you can run `lls_dd` in a pre-built Docker container. 

## Troubleshooting

### Errors due to lack of GPU memory

The post-processing of lattice light sheet stacks
requires a considerable amount of GPU memory. In particular, the deconvolution is very memory hungry.
See the discussion in the issues [here](https://github.com/VolkerH/Lattice_Lightsheet_Deskew_Deconv/issues/31) and [here](https://github.com/hammerlab/flowdec/issues/19).
If you run out of GPU memory there are several troubleshooting steps you can take:

* it the input arrays are much too large, you may have to leave out deconvolution and run deskew and/or rotate only. A workaround for deconvolving in chunks is [outlined in this notebook](https://github.com/hammerlab/flowdec/blob/master/python/examples/notebooks/Tile-by-tile%20deconvolution%20using%20dask.ipynb) but is not yet implemented in `lls_dd`.
* if the input volume is only a little too large, try running deconvolution with `--decon-deskew` only (the `--decon-rot` option requires more GPU memory for the affine transform).

### `pyopencl` installation and errors 

For GPU accelerated deskew/rotate you need install OpenCL drivers for your GPU.
Getting `pyopencl` (one of the required python dependencies) to work can be tricky. When installing from `conda-forge` it seems to be somewhat of a lottery whether the installed package works. On some windows machines where I did not manage to obtain a working `pyopencl` from `conda-forge` I found that uninstalling the conda-installed `pyopencl` and then  `pip`-installing a [binary wheel from Chris Gohlke](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl) into my conda environment did the trick.

## Roadmap

## Project History

This project was started in late 2018 with the intention to create an open-source solution because the existing GPU-accelarated implementation at the time was only available in binary form after signing a research license agreement with Janelia. Meanwhile, the Janelia code has been [open-sourced and put on Github](https://github.com/dmilkie/cudaDecon).
Many of the features of `lls_dd` overlap with [Talley Lambert's `LLSpy`](https://github.com/tlambert03/LLSpy) which also handles batch processing of experiments and has additional functions such as channel registration and sCMOS sensor corrections that are not (yet) present in `lls_dd`.

## Credits

Currently `lls_dd` is mainly leveraging on two libraries that handle the heavy lifting:

* [`flowdec`](https://github.com/hammerlab/flowdec) by Eric Czech . This is the default library used for deconvolution, based on tensorflow.
* [`gputools`](https://github.com/maweigert/gputools) by Martin Weigert  for affine transformations and optionally also for OpenCL-based deconvolution (experimental).

## License

This library was written by Volker Hilsenstein at Monash Micro Imaging.

The code in this repository is distributed under a BSD-3 license, with the exceptions of parts from other projects (which will
have their respective licenses reproduced in the source files.)