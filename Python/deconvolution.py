from flowdec import restoration as tfd_restoration
from flowdec import data as fd_data
from functools import partial

def init_rl_deconvolver():
    """ initializes the tensorflow-based Richardson Lucy Deconvolver """
    return tfd_restoration.RichardsonLucyDeconvolver(n_dims=3, start_mode='input').initialize()

def deconv_volume(vol, psf, deconvolver, n_iter):
    """ perform RL deconvolution using deconvolver 
    input:
    vol : input volume
    psf : psf (numpy array) 
    deconvolver : see init_rl_deconvolver
    n_iter: number of iterations

    TODO: add observer callback so that progress updates for each iteration
    can be displayed. Also, add option to save intermediate results within
    a certain range of iterations.
    """ 
    print("PSF", psf)
    print("deconlver", deconvolver)
    aq = fd_data.Acquisition(data=vol, kernel=psf)
    return deconvolver.run(aq, niter=n_iter).data

# TODO move get_deconv into a separate file with deconvolution stuff
def get_deconv_function(psf, deconvolver, n_iter):
    deconv_func = partial(deconv_volume, psf=psf, deconvolver=deconvolver, n_iter=n_iter)
    return deconv_func
    # where should we create the deconvolve objects ?
    #def deconv_func(input):
    #    print(f"dummy stub for created deconv function with shape {input_shape} with PSF of shape {psf.shape} and {n_iter} iterations")
    #    print(f"processing input with shape {input.shape}")
    #    return(input)
    #return deconv_func