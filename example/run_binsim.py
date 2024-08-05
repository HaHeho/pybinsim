import logging

import pybinsim

pybinsim.logger.setLevel(logging.DEBUG)  # defaults to INFO
# Use logging.WARNING for printing warnings only

# # utilize command line arguments instead of a hard coded file
# with pybinsim.BinSim(config_file=pybinsim.args.config_file) as binsim:
#     binsim.stream_start()

with pybinsim.BinSim("pyBinSimSettings_isoperare.txt") as binsim:
    binsim.stream_start()
