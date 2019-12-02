from lls_dd import Experimentfolder, ExperimentProcessor
import os
import pathlib
import click
from lls_dd.settings import create_fixed_settings


class ProcessCmd(object):
    def __init__(self, exp_folder, fixed_settings=None, home=None, debug=False):
        if not fixed_settings:
            # check whether default settings are present ?
            defaultpath = pathlib.Path(home) / "fixed_settings.json"
            if not defaultpath.exists():
                print("No fixed settings file found.")
                print(
                    f"Generating a default one in {str(defaultpath)}"
                )  # TODO prompt users whether they want one
                pathlib.Path(home).mkdir(parents=True, exist_ok=True)
                create_fixed_settings(str(defaultpath))
            fixed_settings = str(defaultpath)

        self.home = os.path.abspath(home or ".")
        self.debug = debug
        self.exp_folder = pathlib.Path(exp_folder)
        self.ef = Experimentfolder(self.exp_folder, fixed_settings)


pass_process_cmd = click.make_pass_decorator(ProcessCmd)


# For commandline interface see http://click.palletsprojects.com/en/7.x/complex/
# somehow the order of commands is bugging me, I think this is what is discussed
# in this issue https://github.com/pallets/click/issues/108 with some mentioned
# workarounds.
@click.group()
@click.option("--home", envvar="LLS_DD_HOME", default="~/.lls_dd")
@click.option("--debug/--no-debug", default=False, envvar="LLS_DD_DEBUG")
@click.option("-f", "--fixed_settings", default=None, help=".json file with fixed settings")
@click.argument("exp_folder")
@click.pass_context
def cli(ctx, exp_folder, home, debug, fixed_settings):
    """lls_dd: lattice lightsheet deskew and deconvolution utility"""
    ctx.obj = ProcessCmd(exp_folder, fixed_settings, os.path.expanduser(home), debug)


@cli.command(short_help="Processes an experiment folder or individual stacks therein.")
@click.option("-M", "--MIP", is_flag=True, default=False, help="calculate maximum intensity projections")
@click.option(
    "--rot", is_flag=True, default=False, help="rotate deskewed data to coverslip coordinates and save"
)
@click.option("--deskew", is_flag=True, default=False, help="save deskewed data")
@click.option(
    "-b", "--backend", default="flowdec", help='deconvolution backend, either "flowdec" or "gputools"'
)
@click.option(
    "-i",
    "--iterations",
    default=0,
    help="if >0, perform deconvolution this number of Richardson-Lucy " "iterations",
)
@click.option(
    "-c",
    "--camera-subtract",
    default = 100,
    help="value to be substracted from greyvalues (camera offset)",
)
@click.option(
    "-r",
    "--decon-rot",
    is_flag=True,
    default=False,
    help="if  deconvolution was chosen, rotate deconvolved "
    "and deskewed data to coverslip coordinates"
    " and save.",
)
@click.option(
    "-s",
    "--decon-deskew",
    is_flag=True,
    default=False,
    help="if  deconvolution was chosen, deskew the deconvolved "
    " data and save.",
)
@click.option(
    "-n",
    "--number",
    default=None,
    help="stack number to process. if not provided, all stacks are processed",
)
@click.option(
    "--mstyle", default="montage", type=click.Choice(["montage", "multi"]), help="MIP output style"
)
@click.option(
    "--skip_existing",
    is_flag=True,
    default="False",
    help="if this opting is given, files for which the output " "already exists will not be processed",
)
@click.option("--lzw", default=0, help="lossless compression level for tiff (0-9). 0 is no compression")
@click.argument("out_folder", required=False)
@pass_process_cmd
def process(
    processcmd,
    out_folder,
    rot,
    mip,
    deskew,
    backend,
    iterations,
    camera_subtract,
    number,
    decon_rot,
    decon_deskew,
    mstyle,
    skip_existing,
    lzw,
):
    """experiment folder to process (required) output folder (optional) Otherwise same as input"""

    ep = ExperimentProcessor(processcmd.ef, exp_outfolder=out_folder)
    ep.do_MIP = mip
    ep.do_deskew = deskew
    ep.do_rotate = rot
    ep.do_deconv = iterations > 0
    ep.do_deconv_deskew = decon_deskew
    ep.do_deconv_rotate = decon_rot
    ep.skip_existing = skip_existing
    ep.deconv_n_iter = iterations
    # ep.lzw = lzw
    ep.MIP_method = mstyle
    ep.deconv_backend = backend
    ep.bg_subtract_value = camera_subtract,
    print(processcmd.ef)
    print(ep)

    if number:
        print(f"processing stack number {int(number)}")
        ep.process_stack_subfolder(processcmd.ef.stacks[int(number)])
    else:
        print(f"processing all stacks")
        ep.process_all()


@cli.command()
@pass_process_cmd
def stacks(processcmd):
    """ list stacks in experiment folder """
    print(processcmd.ef._str_stacks())


@cli.command()
@pass_process_cmd
def psfs(processcmd):
    """ list psfs in experiment folder """
    print(processcmd.ef._str_PSFs())
