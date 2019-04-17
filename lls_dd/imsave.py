import tifffile
import warnings
import numpy as np
import logging
from typing import Union

# The code in this file is derived from code in Talley Lambert's LLSpy project
# https://github.com/tlambert03/LLSpy/blob/develop/llspy/util.py
#
# The license associated with LLSpy is reproduced below

# also see https://pypi.org/project/tifffile/

"""
Copyright (c) 2017 - President and Fellows of Harvard College.
All rights reserved.

Developed by:  Talley Lambert, PhD
Cell Biology Microscopy Facility, Harvard Medical School
http://cbmf.hms.harvard.edu/
Harvard University case number HU 7053 - Lattice Light Sheet Software (LLSpy)

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal with the
Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions, and the following disclaimers.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimers in the documentation
  and/or other materials provided with the distribution.

* Neither the names of the Cell Biology Microscopy Facility, Harvard Medical School,
  Harvard University, the Harvard shield or logo, nor the names of its contributors
  may be used to endorse or promote products derived from this Software without
  specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
"""


logger = logging.getLogger("lls_dd")


def reorderstack(arr: np.ndarray, inorder: str = "zyx", outorder: str = "tzcyx"):
    """rearrange order of array, used when resaving a file for Fiji."""
    inorder = inorder.lower()
    assert arr.ndim == len(inorder), "The array dimensions must match the inorder dimensions"
    for _ in range(len(outorder) - arr.ndim):
        arr = np.expand_dims(arr, 0)
    for i in outorder:
        if i not in inorder:
            inorder = i + inorder
    arr = np.transpose(arr, [inorder.find(n) for n in outorder])
    return arr


# TODO: I copied this in from Talley's LLSPy with the intention of writing the scale,
# however, I don't like the stack reordering introducing additional length 1 dimensions
# as I can't open a whole folder of 5 dim .tif files in spimage.
def imsave(
    outpath: str,
    arr: np.array,
    compress: Union[int, str] = 0,
    dx: float = 1,
    dz: float = 1,
    dt: float = 1,
    unit: str = "micron",
):
    """sample wrapper for tifffile.imsave imagej=True.
    see documentation in tiffile
    """
    # TODO: actually the array should be in TZCYXS order according to tifffile docu. S == Series ??
    # array must be in TZCYX order
    md = {
        "unit": unit,
        "spacing": dz,
        "finterval": dt,
        "hyperstack": "true",
        "mode": "composite",
        "loop": "true",
    }

    big_t = True if arr.nbytes > 3_758_096_384 else False  # > 3.5GB make a bigTiff
    if arr.ndim == 3:
        arr = reorderstack(arr)  # assume that 3 dimension array is ZYX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logger.debug(f"saving {outpath}")
        tifffile.imsave(
            outpath,
            arr,
            compress=compress,
            bigtiff=big_t,
            imagej=False,
            resolution=(1 / dx, 1 / dx),
            metadata=md,
        )
