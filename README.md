# Lattice Lightsheet Deskew and Deconvolution with GPU acceleration in Python

## CAVEAT: 

work in progress ! 
Will start adding code from a private repo in the coming weeks and add documentation and sample datasets.

## About - Lattice Lightsheet Deskew/Deconv in Python

This repository will provide python based-code for deskewing and deconvolution of lattice light sheet data.
The aim is that the code can run **both** on GPU and on CPU (although deconvolution on GPU will be painfully slow). 

Currently this is mainly leveraging two libraries:

* `gputools` by Martin Weigert (https://github.com/maweigert/gputools) for affine transformations. Note that you may need to install the develop branch of `gputools` as the code in this repo relies on this fixes https://github.com/maweigert/gputools/issues/12 that may not have made it into the main branch (at the time of this writing) and the conda channels.
* `flowdec` by Eric Czech (https://github.com/hammerlab/flowdec) for deconvolution.

## Documentation and explanation of the algorithm 

There are two explanatory notebooks that detail the steps:

* **Deskewing**
https://github.com/VolkerH/Lattice_Lightsheet_Deskew_Deconv/blob/master/Python/00_Lattice_Light_Sheet_Deskew.ipynb
* **Deconvolution**, both on raw data with skewed PSF (less voxels, much faster) or on deskewed data with PSF
https://github.com/VolkerH/Lattice_Lightsheet_Deskew_Deconv/blob/master/Python/01_Lattice_Light_Sheet_Deconvolution.ipynb

### How is this different from LLspy ?

LLSpy (by Talley Lambert https://github.com/tlambert03/LLSpy) is a front-end for batch processing lattice-lightsheet data.
The actual processing is performed by the `cudaDeconv` library from Janelia, which unfortunately is not developed in the
open and only distributed after signing a NDA. 

The code in this repository is intended to develop into an open-source, GPU accelerated library 
for deskewing and deconvolving lattice light sheet data. The open source license still needs to be determined 
after discussions with @jni and other contributors. I assume it will be the same license that scikit-image uses.



## Todo (this will be added to the issue tracker) 

* add example datasets
* add batch scripts
* add batch sumission for HPC clusters using `dask-jobqueue` https://github.com/dask/dask-jobqueue
* Flowdec currently requires CUDA for GPU-acceleration. An `opencl`-based deconvolution would open this up to more graphics accelerators.
* develop and add PSF processing utilities, similar to PSF distiller
