import logging
from importlib.metadata import version

from pybinsim.application import BinSim


def init_logging():
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(formatter)
    log = logging.getLogger(__package__)
    log.addHandler(console_handler)
    return log


logger = init_logging()
logger.info(f"Starting {__package__} v{version(__package__)}")
