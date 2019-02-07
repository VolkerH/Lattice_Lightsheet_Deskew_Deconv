# Lattice  Lightsheet Deskew and Deconvolutoin with GPU acceleration

## CAVEAT: 

work in progress ! 
Will start adding code from a private repo in the coming weeks and add documentation and sample datasets.

## About - Lattice Lightsheet Deskew/Deconv in Python

This repository will provide python based-code for deskewing and deconvolution of lattice light sheet data.
The aim is that the code can run **both** on GPU and on CPU (although deconvolution on GPU will be painfully slow). 
Currently this is mainly leveraging two libraries:

* `gputools` by Martin Weigert (https://github.com/maweigert/gputools) for affine transformations
* `flowdec` by Eric Czech (https://github.com/hammerlab/flowdec) for deconvolution

## Documentation and explanation of the algorithm 

There are two explanatory notebooks that explain the steps:

Deskewing
https://github.com/VolkerH/Lattice_Lightsheet_Deskew_Deconv/blob/master/Python/00_Lattice_Light_Sheet_Deskew.ipynb
Deconvolution, both on raw data with skewed PSF (less voxels, much faster) or on deskewed data with PSF
https://github.com/VolkerH/Lattice_Lightsheet_Deskew_Deconv/blob/master/Python/01_Lattice_Light_Sheet_Deconvolution.ipynb

### How is this different from LLspy ?

LLSpy (by Talley Lambert https://github.com/tlambert03/LLSpy) is a front-end for batch processing lattice-lightsheet data.
The actual processing is performed by the `cudaDeconv` library from Janelia, which unfortunately is not developed in the
open and only distributed after signing a NDA. The code in this repository 

Open-source, GPU accelerated code for deskewing and deconvolving lattice light sheet data.

## Todo (this will be added to the issue tracker) 

* add example datasets
* add batch scripts
* add batch sumission for HPC clusters using `dask-jobqueue` https://github.com/dask/dask-jobqueue
* Flowdec currently requires CUDA for GPU-acceleration. An `opencl`-based deconvolution would open this up to more graphics accelerators.
* add PSF processing utilities, similar to PSF distiller
