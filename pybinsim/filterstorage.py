# This file is part of the pyBinSim project.
#
# Copyright (c) 2017 A. Neidhardt, F. Klein, N. Knoop, T. Köllmer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import time
from enum import Enum
from pathlib import Path

import numpy as np
import scipy.io as sio
import soundfile as sf
import torch
from scipy.spatial import cKDTree

from pybinsim.pose import Pose, SourcePose
from pybinsim.utility import get_nearest_neighbour_key, sph2cart


class Filter(object):
    def __init__(
        self, inputfilter, irBlocks, block_size, torch_settings, filename=None
    ):
        self.log = logging.getLogger(f"{__package__}.{self.__class__.__name__}")

        # Torch options
        self.torch_device = torch.device(torch_settings)

        self.ir_blocks = irBlocks
        self.block_size = block_size

        self.TF_blocks = irBlocks
        self.TF_block_size = block_size + 1

        # input shape: (ir_length, 2)
        ir_blocked = np.empty((2, irBlocks, block_size))

        # if filter is mono - for whatever reason - use mono channel on both
        # ir blocks
        if inputfilter.shape[1] != 2:
            ir_blocked[0,] = np.reshape(
                inputfilter[:, 0], (irBlocks, block_size)
            )
            ir_blocked[1,] = np.reshape(
                inputfilter[:, 0], (irBlocks, block_size)
            )
        else:
            ir_blocked[0,] = np.reshape(
                inputfilter[:, 0], (irBlocks, block_size)
            )
            ir_blocked[1,] = np.reshape(
                inputfilter[:, 1], (irBlocks, block_size)
            )

        self.IR_blocked = torch.as_tensor(
            ir_blocked, dtype=torch.float32, device=self.torch_device
        )

        # not used
        self.filename = filename

        self.fd_available = False
        self.TF_blocked = None

    def getFilter(self):
        return self.IR_blocked

    def getFilterTD(self):
        if self.fd_available:
            self.log.warning("No time domain filter available!")
            return torch.zeros((2, self.ir_blocks, self.block_size))
        else:
            return self.IR_blocked

    def storeInFDomain(self):
        self.TF_blocked = torch.fft.rfft(
            self.IR_blocked, dim=2, n=self.block_size * 2
        )

        self.fd_available = True

        # Discard time domain data
        self.IR_blocked = None

    def getFilterFD(self):
        if not self.fd_available:
            self.log.warning("No frequency domain filter available!")
            return torch.zeros(
                (2, self.ir_blocks, self.block_size + 1), dtype=torch.complex64
            )
        else:
            return self.TF_blocked


class FilterType(Enum):
    Undefined = "Unidentified filter"
    ds_Filter = "Direct sound filter"
    early_Filter = "Early filter"
    late_Filter = "Late filter"
    sd_Filter = "Source directivity filter"
    headphone_Filter = "Headphone filter"


class FilterStorage(object):
    """Class for storing all filters mentioned in the filter list"""

    def __init__(
        self,
        block_size,
        filter_source,
        filter_list_name,
        filter_database,
        torch_settings,
        useHeadphoneFilter=False,
        headphoneFilterSize=0,
        ds_filterSize=0,
        early_filterSize=0,
        late_filterSize=0,
        sd_filterSize=0,
        useNearestNeighbour=False,
    ):
        self.log = logging.getLogger(f"{__package__}.{self.__class__.__name__}")
        self.log.info("Init")
        self.block_size = block_size

        self.ds_size = ds_filterSize
        self.ds_blocks = self.ds_size // self.block_size

        self.early_size = early_filterSize
        self.early_blocks = self.early_size // self.block_size

        self.late_size = late_filterSize
        self.late_blocks = self.late_size // self.block_size

        self.sd_size = sd_filterSize
        self.sd_blocks = self.sd_size // self.block_size

        self.useHeadphoneFilter = useHeadphoneFilter
        if self.useHeadphoneFilter:
            self.headPhoneFilterSize = headphoneFilterSize
            self.headphone_blocks = self.headPhoneFilterSize // self.block_size

        self.torch_settings = torch_settings

        self.default_ds_filter = Filter(
            inputfilter=np.zeros((self.ds_size, 2), dtype="float32"),
            irBlocks=self.ds_blocks,
            block_size=self.block_size,
            torch_settings=self.torch_settings,
        )
        self.default_early_filter = Filter(
            inputfilter=np.zeros((self.early_size, 2), dtype="float32"),
            irBlocks=self.early_blocks,
            block_size=self.block_size,
            torch_settings=self.torch_settings,
        )
        self.default_late_filter = Filter(
            inputfilter=np.zeros((self.late_size, 2), dtype="float32"),
            irBlocks=self.late_blocks,
            block_size=self.block_size,
            torch_settings=self.torch_settings,
        )
        self.default_sd_filter = Filter(
            inputfilter=np.zeros((self.sd_size, 2), dtype="float32"),
            irBlocks=self.sd_blocks,
            block_size=self.block_size,
            torch_settings=self.torch_settings,
        )

        self.default_ds_filter.storeInFDomain()
        self.default_early_filter.storeInFDomain()
        self.default_late_filter.storeInFDomain()
        self.default_sd_filter.storeInFDomain()

        self.filter_source = filter_source
        self.filter_list_path = filter_list_name
        self.filter_database = filter_database
        self.useNearestNeighbour = useNearestNeighbour

        self.headphone_filter = None
        self.filter_list = None

        # format: [key, {filter}]
        self.ds_filter_dict = {}
        self.early_filter_dict = {}
        self.late_filter_dict = {}
        self.sd_filter_dict = {}

        if self.filter_source == "wav":
            self.filter_list = open(self.filter_list_path, "r")
            self.log.info("Loading wav format filters according to filter list")
            self.load_wav_filters()
        elif self.filter_source == "mat":
            self.log.info("Loading mat format filters")
            self.mat_file = sio.loadmat(filter_database)
            self.mat_vars = sio.whosmat(filter_database)
            self.parse_and_load_matfile()

        def _add_grid_summary_to_dict(filter_dict):
            if not filter_dict:  # is empty
                return
            # get all unique string identifiers from custom field in keys
            for custom_id in list(
                dict.fromkeys(key[4] for key in filter_dict.keys())
            ):
                # match target source via custom field
                _filter_dict = dict(
                    (key, value)
                    for key, value in filter_dict.items()
                    if key[4] == custom_id
                )
                # gather all orientations in Cartesian coordinates
                orientations_sph = np.asarray(
                    list(key[0] for key in _filter_dict.keys())
                )
                orientations_cart = np.asarray(
                    sph2cart(
                        azimuth=orientations_sph[:, 0],
                        elevation=orientations_sph[:, 1],
                        radius=1.0,
                    )
                ).T
                # make and store KDTree
                filter_dict[f"{custom_id}_orientations_kdtree"] = cKDTree(
                    orientations_cart
                )
                # store also spherical orientations for faster access
                filter_dict[f"{custom_id}_orientations_sph"] = orientations_sph

        if self.useNearestNeighbour:
            self.log.info(
                "Adding filter entries for nearest-neighbour selection"
            )
            _add_grid_summary_to_dict(self.ds_filter_dict)
            _add_grid_summary_to_dict(self.early_filter_dict)
            _add_grid_summary_to_dict(self.late_filter_dict)
            _add_grid_summary_to_dict(self.sd_filter_dict)

    def parse_and_load_matfile(self):
        for var in self.mat_vars:
            var_name = var[0]
            self.log.info(f"Loading mat variable: {var_name}")

            mat_data = self.mat_file[var_name]
            for row in range(mat_data.shape[1]):
                filter_value_list = np.concatenate(
                    [
                        mat_data["listenerOrientation"][0, row],
                        mat_data["listenerPosition"][0, row],
                        mat_data["sourceOrientation"][0, row],
                        mat_data["sourcePosition"][0, row],
                        mat_data["custom"][0, row],
                    ],
                    axis=1,
                )
                filter_pose = Pose.from_filterValueList(filter_value_list)
                key = filter_pose.create_key()

                current_filter = mat_data["filter"][0, row]
                row_type = str.upper(mat_data["type"][0, row][0])
                if row_type == "HP":
                    if not self.useHeadphoneFilter:
                        self.log.info("Skipping headphone filter")
                        continue
                    current_filter = Filter(
                        inputfilter=self.check_filter(
                            filter_type=FilterType.headphone_Filter,
                            current_filter=current_filter,
                        ),
                        irBlocks=self.headphone_blocks,
                        block_size=self.block_size,
                        torch_settings=self.torch_settings,
                    )
                    current_filter.storeInFDomain()
                    self.headphone_filter = current_filter

                elif row_type == "SD":
                    # skip 'listenerOrientation' and 'listenerPosition'
                    filter_value_list = filter_value_list[:, 6:]
                    filter_pose = SourcePose.from_filterValueList(
                        filter_value_list
                    )
                    key = filter_pose.create_key()

                    current_filter = Filter(
                        inputfilter=self.check_filter(
                            filter_type=FilterType.sd_Filter,
                            current_filter=current_filter,
                        ),
                        irBlocks=self.sd_blocks,
                        block_size=self.block_size,
                        torch_settings=self.torch_settings,
                    )
                    current_filter.storeInFDomain()
                    self.sd_filter_dict[key] = current_filter

                elif row_type == "DS":
                    current_filter = Filter(
                        inputfilter=self.check_filter(
                            filter_type=FilterType.ds_Filter,
                            current_filter=current_filter,
                        ),
                        irBlocks=self.ds_blocks,
                        block_size=self.block_size,
                        torch_settings=self.torch_settings,
                    )
                    current_filter.storeInFDomain()
                    self.ds_filter_dict[key] = current_filter

                elif row_type == "ER":
                    current_filter = Filter(
                        inputfilter=self.check_filter(
                            filter_type=FilterType.early_Filter,
                            current_filter=current_filter,
                        ),
                        irBlocks=self.early_blocks,
                        block_size=self.block_size,
                        torch_settings=self.torch_settings,
                    )
                    current_filter.storeInFDomain()
                    self.early_filter_dict[key] = current_filter

                elif row_type == "LR":
                    current_filter = Filter(
                        inputfilter=self.check_filter(
                            filter_type=FilterType.late_Filter,
                            current_filter=current_filter,
                        ),
                        irBlocks=self.late_blocks,
                        block_size=self.block_size,
                        torch_settings=self.torch_settings,
                    )
                    current_filter.storeInFDomain()
                    self.late_filter_dict[key] = current_filter

                else:
                    raise RuntimeError("Filter identifier wrong or missing")

            # Delete parsed variable
            self.mat_file.pop(var_name)

        # clear whole mat_file after parsing
        self.mat_file = []

    def parse_filter_list(self):
        """
        Generator for filter list lines

        Lines are assumed to have a format like
        0 0 40 1 1 0 brirWav_APA/Ref_A01_1_040.wav

        The headphone filter starts with HPFILTER instead of the positions.

        Lines can be commented with a '#' as first character.

        :return: Iterator of (Pose, filter-path) tuples
        """
        for line in self.filter_list:
            # comment out lines in the list with a "#"
            if line.startswith("#") or line == "\n":
                continue

            line_content = line.split()
            filter_path = line_content[-1]

            # if line.startswith("HPFILTER"):
            # handle headphone filter
            if line.startswith("HP"):
                if self.useHeadphoneFilter:
                    self.log.info(f"Loading headphone filter: {filter_path}")
                    self.headphone_filter = Filter(
                        self.load_wav_filter(
                            filter_path, FilterType.headphone_Filter
                        ),
                        self.headphone_blocks,
                        self.block_size,
                        self.torch_settings,
                    )
                    self.headphone_filter.storeInFDomain()
                    continue
                else:
                    # self.headphone_filter = Filter(
                    #     self.load_wav_filter(filter_path),
                    #     self.ir_blocks,
                    #     self.block_size,
                    # )
                    self.log.info(f"Skipping headphone filter: {filter_path}")
                    continue

            if line.startswith("SD"):
                # # TODO: Should SD filters also be skipped if not used
                # #  (analogue to HP filters)?
                # filter_type = FilterType.sd_Filter
                # filter_value_list = np.concatenate(
                #     [
                #         self.matfile[self.matvarname]["sourceOrientation"][0][
                #             row
                #         ],
                #         self.matfile[self.matvarname]["sourcePosition"][0][row],
                #         self.matfile[self.matvarname]["custom"][0][row],
                #     ],
                #     axis=1,
                # )
                # filter_pose = SourcePose.from_filterValueList(filter_value_list)
                # key = filter_pose.create_key()
                # current_filter = Filter(
                #     self.check_filter(
                #         filter_type,
                #         self.matfile[self.matvarname]["filter"][0][row],
                #     ),
                #     self.sd_blocks,
                #     self.block_size,
                #     self.torch_settings,
                # )
                # current_filter.storeInFDomain()
                # self.sd_filter_dict.update({key: current_filter})
                # continue
                raise NotImplementedError("This code needs to be fixed")

            if line.startswith("DS"):
                filter_type = FilterType.ds_Filter
                filter_value_list = tuple(line_content[1:-1])
                filter_pose = Pose.from_filterValueList(filter_value_list)
            elif line.startswith("ER"):
                filter_type = FilterType.early_Filter
                filter_value_list = tuple(line_content[1:-1])
                filter_pose = Pose.from_filterValueList(filter_value_list)
            elif line.startswith("LR"):
                filter_type = FilterType.late_Filter
                filter_value_list = tuple(line_content[1:-1])
                filter_pose = Pose.from_filterValueList(filter_value_list)
            else:
                raise RuntimeError("Filter identifier wrong or missing")

            yield filter_pose, filter_path, filter_type

    def load_wav_filters(self):
        """
        Load filters from files

        :return: None
        """
        self.log.info("Start loading filters...")
        start = time.time()

        parsed_filter_list = list(self.parse_filter_list())

        # # check if all files are available
        # are_files_missing = False
        # for pose, filter_path in parsed_filter_list:
        #     fn_filter = Path(filter_path)
        #     if not fn_filter.exists():
        #         self.log.warn(f"Wavefile not found: {fn_filter}")
        #         are_files_missing = True
        # if are_files_missing:
        #     raise FileNotFoundError("Some files are missing")
        #
        # for pose, filter_path in parsed_filter_list:
        #     self.log.debug(f"Loading {filter_path}")
        #     key = pose.create_key()
        #     loaded_filter = self.load_filter(filter_path)
        #     current_filter = Filter(
        #         inputfilter=loaded_filter,
        #         irBlocks=self.ir_blocks,
        #         block_size=self.block_size,
        #         filename=filter_path,
        #     )
        #     self.filter_dict.update({key: current_filter})

        for filter_pose, filter_path, filter_type in parsed_filter_list:
            # Skip undefined types (e.g. old format)
            if filter_type == FilterType.Undefined:
                continue
            fn_filter = Path(filter_path)

            # check for missing filters and throw exception if not found
            if not Path(filter_path).exists():
                self.log.warning(f"Wavefile not found: {fn_filter}")
                raise FileNotFoundError(f"File {fn_filter} is missing.")

            self.log.debug(f"Loading {filter_path}")
            key = filter_pose.create_key()
            if filter_type == FilterType.ds_Filter:
                current_filter = Filter(
                    inputfilter=self.load_wav_filter(filter_path, filter_type),
                    irBlocks=self.ds_blocks,
                    block_size=self.block_size,
                    torch_settings=self.torch_settings,
                )
                current_filter.storeInFDomain()
                self.ds_filter_dict[key] = current_filter

            if filter_type == FilterType.early_Filter:
                current_filter = Filter(
                    inputfilter=self.load_wav_filter(filter_path, filter_type),
                    irBlocks=self.early_blocks,
                    block_size=self.block_size,
                    torch_settings=self.torch_settings,
                )
                current_filter.storeInFDomain()
                self.early_filter_dict[key] = current_filter

            if filter_type == FilterType.late_Filter:
                current_filter = Filter(
                    inputfilter=self.load_wav_filter(filter_path, filter_type),
                    irBlocks=self.late_blocks,
                    block_size=self.block_size,
                    torch_settings=self.torch_settings,
                )
                current_filter.storeInFDomain()
                self.late_filter_dict[key] = current_filter

        end = time.time()
        self.log.info(f"Finished loading filters in {str(end - start)} sec.")
        # self.log.info(
        #     f"filter_dict size: {total_size(self.filter_dict) // 1024 // 1024} MiB"
        # )

    def _get_filter(self, pose, filter_type, filter_dict, default_filter):
        f_str = filter_type.value
        key = pose.create_key()
        try:
            if self.useNearestNeighbour:
                key = get_nearest_neighbour_key(filter_dict, key)
            result_filter = filter_dict[key]
            if result_filter.filename:
                self.log.debug(f"{f_str} use file={result_filter.filename}")
            else:
                self.log.debug(f"{f_str} use {key=}")
        except KeyError:
            self.log.warning(f"{f_str} not found for {key=}")
            result_filter = default_filter
        except ValueError as e:
            self.log.warning(f"{f_str} not found for {e}")
            result_filter = default_filter
        return result_filter

    def get_sd_filter(self, source_pose):
        """
        Searches in the dict if key is available and return corresponding filter
        When no filter is found, defaultFilter is returned which results in silence

        :param source_pose:
        :return: corresponding filter for pose
        """
        return self._get_filter(
            pose=source_pose,
            filter_type=FilterType.sd_Filter,
            filter_dict=self.sd_filter_dict,
            default_filter=self.default_sd_filter,
        )

    def get_ds_filter(self, pose):
        """
        Searches in the dict if key is available and return corresponding filter
        When no filter is found, defaultFilter is returned which results in silence

        :param pose:
        :return: corresponding filter for pose
        """
        return self._get_filter(
            pose=pose,
            filter_type=FilterType.ds_Filter,
            filter_dict=self.ds_filter_dict,
            default_filter=self.default_ds_filter,
        )

    def get_early_filter(self, pose):
        """
        Searches in the dict if key is available and return corresponding filter
        When no filter is found, defaultFilter is returned which results in silence

        :param pose:
        :return: corresponding filter for pose
        """
        return self._get_filter(
            pose=pose,
            filter_type=FilterType.early_Filter,
            filter_dict=self.early_filter_dict,
            default_filter=self.default_early_filter,
        )

    def get_late_filter(self, pose):
        """
        Searches in the dict if key is available and return corresponding filter
        When no filter is found, defaultFilter is returned which results in silence

        :param pose:
        :return: corresponding filter for pose
        """
        return self._get_filter(
            pose=pose,
            filter_type=FilterType.late_Filter,
            filter_dict=self.late_filter_dict,
            default_filter=self.default_late_filter,
        )

    def get_headphone_filter(self):
        if self.headphone_filter is None:
            raise RuntimeError("Headphone filter not loaded")

        return self.headphone_filter

    def load_wav_filter(self, filter_path, filter_type):
        current_filter, _ = sf.read(filter_path, dtype="float32")
        return self.check_filter(filter_type, current_filter)

    def check_filter(self, filter_type, current_filter):
        # TODO: Check sampling rate (fs)
        filter_size = np.size(current_filter, axis=0)
        match filter_type:
            case FilterType.ds_Filter:
                target_size = self.ds_size
            case FilterType.early_Filter:
                target_size = self.early_size
            case FilterType.late_Filter:
                target_size = self.late_size
            case FilterType.sd_Filter:
                target_size = self.sd_size
            case FilterType.headphone_Filter:
                target_size = self.headPhoneFilterSize
            case _:
                raise ValueError(f"Unknown filter type {filter_type}")

        if filter_size < target_size:
            self.log.debug(
                f"{filter_type.value} too short: Zero-padding from "
                f"{filter_size} samples to {target_size} samples"
            )
            current_filter = np.pad(
                current_filter, ((0, target_size - filter_size), (0, 0))
            )
        elif filter_size > target_size:
            self.log.debug(
                f"{filter_type.value} too long: Truncating from "
                f"{filter_size} samples to {target_size} samples"
            )
            current_filter = current_filter[target_size]

        return current_filter

    def close(self):
        self.log.info("Close")
