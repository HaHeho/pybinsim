import logging
from importlib.metadata import version

from pybinsim.application import BinSim


def init_logging():
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("pybinsim")
    logger.addHandler(console_handler)

    return logger


logger = init_logging()
logger.info(f"Starting pybinsim v{version('pybinsim')}")
