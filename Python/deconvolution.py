from flowdec import restoration as tfd_restoration
from flowdec import data as fd_data


def init_rl_deconvolver():
    """ initializes the tensorflow-based Richardson Lucy Deconvolver """
    return tfd_restoration.RichardsonLucyDeconvolver(ndims=3, start_mode='input').initialize()

def deconv_volume(vol, kernel, deconvolver, n_iter=10):
    """ perform RL deconvolution using deconvolver 
    input:
    vol : input volumen
    kernel : psf to 
    deconvolver : see init_rl_deconvolver
    n_iter: number of iterations

    TODO: add observer callback so that progress updates for each iteration
    can be displayed. Also, add option to save intermediate results within
    a certain range of iterations.
    """ 
    aq = fd_data.Acquisition(data=vol, kernel=kernel)
    return deconvolver.run(aq, niter=n_iter).data
