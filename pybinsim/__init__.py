import argparse
import logging
from importlib.metadata import version

from pybinsim.application import BinSim


def parse_cmd_args():
    class _LicenseAction(argparse.Action):
        def __call__(self, _parser, namespace, values, option_string=None):
            print(open("../LICENSE", mode="r", encoding="utf-8").read())
            _parser.exit()

    class _VersionAction(argparse.Action):
        def __call__(self, _parser, namespace, values, option_string=None):
            print(version(__package__))
            _parser.exit()

    def _print_arg(arg):
        arg_value = getattr(_args, arg)
        arg_default = parser.get_default(arg)
        if arg_value != arg_default:
            print(f'"{arg}" using "{arg_value}" (default "{arg_default}")')

    parser = argparse.ArgumentParser(
        prog=__package__,
        description="Real-time dynamic binaural synthesis with head tracking.",
    )
    parser.add_argument(
        "-l",
        "--license",
        action=_LicenseAction,
        nargs=0,
        help="show LICENSE information and exit",
    )
    parser.add_argument(
        "-v",
        "--version",
        action=_VersionAction,
        nargs=0,
        help="show version information and exit",
    )
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="pyBinSimSettings.txt",
        help="file containing rendering configuration (see example)",
    )

    # parse arguments
    _args = parser.parse_args()
    _print_arg("config_file")

    print("All unnamed arguments use default values.\n")
    return _args


def init_logging():
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)16s:%(lineno)-3d - %(levelname)7s - "
        "%(name)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    log = logging.getLogger(__package__)
    log.addHandler(console_handler)
    return log


args = parse_cmd_args()

logger = init_logging()
logger.info(f"Starting {__package__} v{version(__package__)}")
