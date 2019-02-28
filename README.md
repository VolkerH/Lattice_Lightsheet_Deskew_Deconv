# Lattice Lightsheet Deskew and Deconvolution with GPU acceleration in Python

## CAVEAT: 

Work in progress with frequent changes. The notebooks referenced below document the basic algorithms 
(this is not going to change). The batch processing framework around it is still under active development.
I aim to make this `pip`-installable and provide a command-line tool for batch processing soon. 

## About - Lattice Lightsheet Deskew/Deconv in Python

This repository provides python based-code for deskewing and deconvolution of lattice light sheet data.
The aim is that the code can run **both** on GPU and on CPU (although deconvolution on CPU will be painfully slow). 

Currently this is mainly leveraging two libraries:

* `gputools` by Martin Weigert (https://github.com/maweigert/gputools) for affine transformations. Note that you may need to install the develop branch of `gputools` as the code in this repo relies on this fixes https://github.com/maweigert/gputools/issues/12 that may not have made it into the main branch (at the time of this writing) and the conda channels.
* `flowdec` by Eric Czech (https://github.com/hammerlab/flowdec) for deconvolution.


## Documentation and explanation of the algorithm 

The following notebooks illustrate the basic algorithms used and provide examples for batch processing.

* **Deskewing** [Jupyter notebook that illustrates how to deskew and rotate with affine transforms](./Python/00_Lattice_Light_Sheet_Deskew.ipynb)
* **Deconvolution** [Demonstrates deskewing both on raw data with skewed PSF (less voxels, therefore faster) and on deskewed data with unskewed PSF](./Python/01_Lattice_Light_Sheet_Deconvolution.ipynb)
* **Batch Processing** [Batch process experiment folders](./Python/02_Batch_Process.ipynb) 

## Sample image and PSF file

**DROPBOX links fixed. (The dropbox links in a previous version were no longer valid)**

Sample images are too large for Github:
* sample image file (courtesy of Felix Kraus / Monash University)  
https://www.dropbox.com/s/34ei5jj0qgylf8q/drp1_dendra2_test_1_CamA_ch0_stack0000_488nm_0000000msec_0018218290msecAbs.tif?dl=0
* corresponding PSF
https://www.dropbox.com/s/39ljascy4vkp0tk/488_PSF_galvo_CamA_ch0_stack0000_488nm_0000000msec_0016836088msecAbs.tif?dl=0

### How is this different from `LLSpy` and `cudaDeconv`?

LLSpy (by Talley Lambert https://github.com/tlambert03/LLSpy) is a batch processing GUI front-end for processing lattice-lightsheet data.
The deconvolution and deskew part of the processing in LLSpy is performed by the `cudaDeconv` library from Janelia Farm.
LLSpy also includes corrections for residual pixel intensities and provides registration between channels - none of this
is currently implemented in this project.
This project was started as an effort to provide an open-source, GPU accelerated implementation of deskew/deconvolve
 because `cudaDeconv` was only available as a binary distribution (after signing a non-disclosure-agreement). Meanwhile,
 Dan Milkie has released `cudaDeconv` as open source: https://github.com/dmilkie/cudaDecon .  
  

## License

The code in this repository falls under a BSD-3 license, with the exceptions of parts from other projects (which will
have their respective licenses reproduced in the source files.)