# Lattice Lightsheet Deskew and Deconvolution with GPU acceleration in Python

## CAVEAT: 

work in progress ! 

## About - Lattice Lightsheet Deskew/Deconv in Python

This repository will provide python based-code for deskewing and deconvolution of lattice light sheet data.
The aim is that the code can run **both** on GPU and on CPU (although deconvolution on GPU will be painfully slow). 

Currently this is mainly leveraging two libraries:

* `gputools` by Martin Weigert (https://github.com/maweigert/gputools) for affine transformations. Note that you may need to install the develop branch of `gputools` as the code in this repo relies on this fixes https://github.com/maweigert/gputools/issues/12 that may not have made it into the main branch (at the time of this writing) and the conda channels.
* `flowdec` by Eric Czech (https://github.com/hammerlab/flowdec) for deconvolution.


## Documentation and explanation of the algorithm 

The following notebooks illustrate the basic algorithms used and provide examples for batch processing.

* **Deskewing** [Jupyter notebook that illustrates how to deskew and rotate with affine transforms](./Python/00_Lattice_Light_Sheet_Deskew.ipynb)
* **Deconvolution** [Demonstrates deskewing both on raw data with skewed PSF (less voxels, therefore faster) and on deskewed data with unskewed PSF](./Python/01_Lattice_Light_Sheet_Deconvolution.ipynb)
* **Batch Processing** [Batch process experiment folders](./Python/03_Batch_Process.ipynb) 

## Sample image and PSF file

**DROPBOX links fixed. (The dropbox links in a previous version were no longer valid)**

Sample images are too large for Github:
* sample image file (courtesy of Felix Kraus / Monash University)  
https://www.dropbox.com/s/34ei5jj0qgylf8q/drp1_dendra2_test_1_CamA_ch0_stack0000_488nm_0000000msec_0018218290msecAbs.tif?dl=0
* corresponding PSF
https://www.dropbox.com/s/39ljascy4vkp0tk/488_PSF_galvo_CamA_ch0_stack0000_488nm_0000000msec_0016836088msecAbs.tif?dl=0

### How is this different from LLspy ?

LLSpy (by Talley Lambert https://github.com/tlambert03/LLSpy) is a front-end for batch processing lattice-lightsheet data.
The actual processing is performed by the `cudaDeconv` library from Janelia, which unfortunately is not developed in the
open and only distributed after signing a NDA. 

The code in this repository is intended to develop into an open-source, GPU accelerated library 
for deskewing and deconvolving lattice light sheet data. The open source license still needs to be determined 
after discussions with @jni and other contributors. I assume it will be the same license that scikit-image uses.

## Todo (these will be added to the issue tracker) 

* add more example datasets
* add batch sumission for HPC clusters using `dask-jobqueue` https://github.com/dask/dask-jobqueue
* Flowdec currently requires CUDA for GPU-acceleration. An `opencl`-based deconvolution would open this up to more graphics accelerators. Alternatively check whether we can get the `ROCm` version of tensorflow running with flowdec to at least support AMD workstation cards https://github.com/ROCmSoftwarePlatform/tensorflow-upstream
* develop and add PSF processing utilities, similar to PSF distiller
