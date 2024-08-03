import logging

import pybinsim

pybinsim.logger.setLevel(logging.INFO)  # defaults to INFO
# Use logging.WARNING for printing warnings only

with pybinsim.BinSim("settings.cfg") as binsim:
    binsim.stream_start()
