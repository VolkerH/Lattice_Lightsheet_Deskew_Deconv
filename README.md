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
