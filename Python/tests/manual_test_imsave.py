
import sys 
sys.path.append("..")
import imsave
import numpy.random

def create_image_with_scale():
    """a quick test to create an image with scaling information.
    Confirmed with Fiji that the pixel scaling is correct.
    """
    test_im = numpy.random.randint(0,4096, (100,512,512), dtype=numpy.uint16)
    imsave.imsave(test_im, "scaled_random.tif", dx = 0.333, dz = 10 )

create_image_with_scale()