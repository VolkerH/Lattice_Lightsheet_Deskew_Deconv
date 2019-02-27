from experiment_folder import Experimentfolder
from process_llsm_experiment import  *

input_folder = "~/Data/20181219_Felix_Dendra2_Drp1_test/"
fixed_settings_file = "fixed_settings.json"
ef = Experimentfolder(input_folder, fixed_settings_file)

ep = ExperimentProcessor(ef, exp_outfolder="/tmp/volker_batch_deskew/Experiment_20181219/")
ep.do_MIP = True # create MIPs.

ep.do_deskew = True  # deskew the raw data
ep.do_rotate = False  # don't rotate the raw data, we'll just do this for the deconvolved data

ep.do_deconv = True  # deconvolve the raw data
ep.do_deconv_rotate = False  # don't deskew the deconvolved data
ep.do_deconv_deskew = True  # deskew+rotate the coverslip of the deconvolved data

ep.skip_existing = True
ep.deconv_n_iter = 10
ep.bg_subtract_value = 95
ep.MIP_method = "montage"

ep.process_stack_subfolder(ef.stacks[0])