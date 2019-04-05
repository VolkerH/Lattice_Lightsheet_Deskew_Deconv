#import lls_dd.deconv_gputools_rewrite as ld
import lls_dd.deconvolution as ld
import tifffile
import numpy as np
import pathlib
from prefetch_generator import BackgroundGenerator

def volreadgen(files):
    for f in files:
        vol = np.squeeze(tifffile.imread(str(f)))
        yield f,vol
        
def run():
    psf = np.squeeze(tifffile.imread("PSF_642.tif"))
    dec = ld.init_rl_deconvolver()

    psf = psf[12:-10,: ,: ]
    files = list(pathlib.Path('.').glob("*ch0_*.tif"))
    for f,vol in BackgroundGenerator(volreadgen(files)):
        print(vol.shape)
        res = ld.deconv_volume(vol[12:-10, :, :], psf, dec, 10)
        #print(f"min {np.min(res)}")
        #print(f"max {np.max(res)}")
        outpath = f.parent / "dec" / f.name
        print(f"saving as {str(outpath)}")
        tifffile.imsave(str(outpath), res)


run()
