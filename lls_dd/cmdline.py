from lls_dd import Experimentfolder, ExperimentProcessor
import os
import pathlib
import click
from .settings import create_fixed_settings
import pdb

class ProcessCmd(object):
    def __init__(self, exp_folder, fixed_settings=None, home=None, debug=False):
        if not fixed_settings:
            # check whether default settings are present ?
            defaultpath = pathlib.Path(home) / "fixed_settings.json"
            if not defaultpath.exists():
                print("No fixed settings file found.")
                print(f"Generating a default one in {str(defaultpath)}")  # TODO prompt users whether they want one
                create_fixed_settings(str(defaultpath))
            fixed_settings = str(defaultpath)

        self.home = os.path.abspath(home or '.')
        self.debug = debug
        self.exp_folder = pathlib.Path(exp_folder)
        self.ef = Experimentfolder(self.exp_folder, fixed_settings)

    def hello(self):
        print("hello. my home is", self.home)
        print("the experiment folder is", str(self.exp_folder))


pass_process_cmd = click.make_pass_decorator(ProcessCmd)


# For commandline interface see http://click.palletsprojects.com/en/7.x/complex/
@click.group()
@click.option('--home', envvar='LLS_DD_HOME', default='.lls_dd')
@click.option('--debug/--no-debug', default=False,
              envvar='LLS_DD_DEBUG')
@click.option('-f', '--fixed_settings', default=None, help='.json file with fixed settings')
@click.argument('exp_folder')
@click.pass_context
def cli(ctx, exp_folder, home, debug, fixed_settings):
    """lls_dd: lattice lightsheet deskew and deconvolution utility"""
    ctx.obj = ProcessCmd(exp_folder, fixed_settings, home, debug)


@cli.command(short_help='Processes an experiment folder or individual stacks therein.')
@click.option('-M', '--MIP', is_flag=True, default=True, help='calculate maximum intensity projections')
@click.option('--deskew_rot', is_flag=True, default=False,
              help='rotate deskewed data to coverslip coordinates and save')
@click.option('--deskew', is_flag=True, default=False, help='save deskewed data')
@click.option('-i', '--iterations', default=0, help="if >0, perform deconvolution this number of Richardson-Lucy "
                                                    "iterations")
@click.option('-r', '--decon_rot', is_flag=True, default=True, help="if  deconvolution was chosen, rotate deconvolved "
                                                                    "and deskewed data to coverslip coordinates"
                                                                    " and save.")
@click.option('-s', '--decon_deskew', is_flag=True, default=False,
              help="if  deconvolution was chosen, rotate deconvolved "
                   "and deskewed data to coverslip coordinates"
                   " and save.")
@click.option('-n', '--number', default=None, help="stack number to process. if not provided, all stacks are processed")
@click.option('--mstyle', default='montage', type=click.Choice(['montage', 'multi']), help="MIP output style")
@click.option('--skip_existing', default='True', help="if this opting is given, files for which the output "
                                                      "already exists will not be processed")
@click.option('--lzw', default=0, help="lossless compression level for tiff (0-9). 0 is no compression")
@click.argument('out_folder', required=False)
@pass_process_cmd
def process(processcmd, out_folder, mip, deskew_rot, deskew, iterations, number,
            decon_rot, decon_deskew, mstyle, skip_existing, lzw):
    """experiment folder to process (required) output folder (optional) Otherwise same as input"""

    ep = ExperimentProcessor(processcmd.ef, out_folder)
    ep.do_MIP = mip
    ep.do_deskew = deskew
    ep.do_rotate = deskew_rot
    ep.do_deconv = iterations > 0
    ep.do_deconv_deskew = decon_deskew
    ep.do_deconv_rotate = decon_rot
    ep.skip_existing = skip_existing
    ep.deconv_n_iter = iterations
    ep.lzw = lzw
    ep.MIP_method = mstyle
    #pdb.set_trace()
    print(processcmd.exp_folder)
    if out_folder:
        print(processcmd.hello())
        print(out_folder)
    else:
        print("no output folder, will use input folder")
    if number:
        print(f"processing stack nunmber {int(number)}")
        ep.process_stack_subfolder(processcmd.ef.stacks[0])
    else:
        print(f"proessing all stacks")

@cli.command()
@pass_process_cmd
def stacks(processcmd):
    """ list stacks in experiment folder """
    print(processcmd.ef.stacks)


@cli.command()
@pass_process_cmd
def psfs(processcmd):
    """ list psfs in experiment folder """
    print("listing psfs in ", processcmd.exp_folder)
    print(processcmd.ef.PSFs)

# def hello():
#    print("hello world")

# def list():
#    ef = Experimentfolder(".", "fixed_settings.json")

# def process():
#    ef = Experimentfolder(".", "fixed_settings.json")
#    ep = ExperimentProcessor(ef)
#    ep.process_all()