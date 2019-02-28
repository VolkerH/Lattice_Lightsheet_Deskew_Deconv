from experiment_folder import Experimentfolder
from process_llsm_experiment import  *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
input_folder = "/home/vhil0002/Data/20181219_Felix_Dendra2_Drp1_test/"
fixed_settings_file = "fixed_settings.json"
ef = Experimentfolder(input_folder, fixed_settings_file)


ep = ExperimentProcessor(ef, exp_outfolder="/tmp/volker_batch_deskew3/Experiment_20181219/")
ep.do_MIP = True # create MIPs.

ep.do_deskew = False  # deskew the raw data
ep.do_rotate = False  # don't rotate the raw data, we'll just do this for the deconvolved data

ep.do_deconv = True  # deconvolve the raw data
ep.do_deconv_rotate = True  # don't deskew the deconvolved data
ep.do_deconv_deskew = False  # deskew+rotate the coverslip of the deconvolved data

ep.skip_existing = False
ep.deconv_n_iter = 10
ep.bg_subtract_value = 95
ep.MIP_method = "montage"


subfolder = ef.stacks[3]
print(subfolder)
ep.process_stack_subfolder(subfolder)
#ep.process_all()
