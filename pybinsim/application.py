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

""" Module contains main loop and configuration of pyBinSim """
import logging
import sys
import time

import numpy as np
import sounddevice as sd
import torch

from pybinsim.convolver import ConvolverTorch
from pybinsim.filterstorage import FilterStorage, FilterType
from pybinsim.input_buffer import InputBufferMulti
from pybinsim.osc_receiver import OscReceiver
from pybinsim.parsing import parse_boolean, parse_soundfile_list
from pybinsim.pkg_receiver import CONFIG_SOUNDFILE_PLAYER_NAME, PkgReceiver
from pybinsim.pose import Pose, SourcePose
from pybinsim.soundhandler import LoopState, SoundHandler
from pybinsim.zmq_receiver import ZmqReceiver


class BinSimConfig(object):
    def __init__(self):
        self.log = logging.getLogger(f"{__package__}.{self.__class__.__name__}")

        # Default Configuration
        self.configurationDict = {
            "soundfile": "",
            "blockSize": 256,
            "ds_filterSize": 512,
            "early_filterSize": 4096,
            "late_filterSize": 16384,
            "directivity_filterSize": 512,
            "filterSource[mat/wav]": "mat",
            "filterList": "brirs/filter_list_kemar5.txt",
            "filterDatabase": "brirs/database.mat",
            "enableCrossfading": False,
            "useHeadphoneFilter": False,
            "headphone_filterSize": 1024,
            "useNearestNeighbour": False,
            "loudnessFactor": 1.0,  # float
            "maxChannels": 8,
            "samplingRate": 48000,
            "loopSound": True,
            "pauseConvolution": False,
            "pauseAudioPlayback": False,
            "torchConvolution[cpu/cuda]": "cuda",
            "torchStorage[cpu/cuda]": "cuda",
            "ds_convolverActive": True,
            "early_convolverActive": True,
            "late_convolverActive": True,
            "sd_convolverActive": False,
            # only set for bench_audio_callback.py!
            "audio_callback_benchmark": False,
            "recv_type": "osc",
            "recv_protocol": "tcp",
            "recv_ip": "127.0.0.1",
            "recv_port": 10000,  # starting port in case of OSC
        }

    def read_from_file(self, filepath):
        with open(filepath, mode="r") as config:
            for line in config:
                line = str.strip(line)  # remove surrounding whitespaces
                try:
                    # split on whitespace
                    [key, value] = str.split(line, maxsplit=1)
                except ValueError:
                    if len(line):
                        self.log.debug(f"Line '{line}' skipped")
                    continue  # skip empty lines

                if key in self.configurationDict:
                    config_value_type = type(self.configurationDict[key])

                    if config_value_type is bool:
                        # evaluate 'False' to False
                        boolean_config = parse_boolean(value)
                        if boolean_config is None:
                            self.log.warning(
                                f"Cannot convert '{value}' to bool. (key: '{key}')"
                            )
                        else:
                            self.configurationDict[key] = boolean_config
                    else:
                        # use type(str) - ctors of int, float, ...
                        self.configurationDict[key] = config_value_type(value)

                elif key.startswith("#"):
                    self.log.debug(f"Entry '{key}' ignored")
                else:
                    self.log.warning(f"Entry '{key}' is unknown")

    def get(self, setting):
        return self.configurationDict[setting]

    def set(self, setting, value):
        if isinstance(self.configurationDict[setting], bool) and not isinstance(
            value, bool
        ):
            value_bool = parse_boolean(value)
            if value_bool is None:
                self.log.warning(f"Cannot convert '{value}' to bool")
            else:
                value = value_bool

        if isinstance(self.configurationDict[setting], type(value)):
            self.configurationDict[setting] = value
        else:
            self.log.warning(
                f"New value for entry '{setting}' has wrong type "
                f"{type(value)} instead of {type(self.configurationDict[setting])}"
            )


class BinSim(object):
    """
    Main pyBinSim program logic
    """

    def __init__(self, config_file):
        self.log = logging.getLogger(f"{__package__}.{self.__class__.__name__}")
        self.log.info("Init")

        self.cpu_usage_update_rate = 100
        self.time_usage = np.zeros(
            self.cpu_usage_update_rate - 1, dtype="float32"
        )
        self.time_usage_index = 0

        # Read Configuration File
        self.config = BinSimConfig()
        self.config.read_from_file(config_file)

        self.nChannels = self.config.get("maxChannels")
        self.sampleRate = self.config.get("samplingRate")
        self.blockSize = self.config.get("blockSize")

        self.result = None
        self.block = None
        self.stream = None

        self.convolverHP = None
        self.ds_convolver = None
        self.early_convolver = None
        self.late_convolver = None
        self.sd_convolver = None
        self.input_Buffer = None
        self.input_BufferHP = None
        self.input_BufferSD = None
        self.filterStorage = None
        self.pkgReceiver = None
        self.soundHandler = None
        self.initialize_pybinsim()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__cleanup()

    def stream_start(self):
        self.log.info("Stream start")
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sampleRate,
                dtype="float32",
                channels=2,
                latency="low",
                blocksize=self.blockSize,
                callback=audio_callback(self),
            )
            # pydevd.settrace(suspend=False, trace_only_current_thread=True)
            with self.stream as s:
                self.log.info(f"latency: {s.latency * 1e3:.2f} ms")
                while True:
                    sd.sleep(1000)
        except KeyboardInterrupt:
            print("KEYBOARD")
        except Exception as e:
            print(e)

    def initialize_pybinsim(self):
        self.result = torch.zeros(2, self.blockSize, dtype=torch.float32)
        self.block = torch.zeros(
            [self.nChannels, self.blockSize], dtype=torch.float32
        )

        ds_size = self.config.get("ds_filterSize")
        early_size = self.config.get("early_filterSize")
        late_size = self.config.get("late_filterSize")
        sd_size = self.config.get("directivity_filterSize")
        hp_size = self.config.get("headphone_filterSize")

        def _get_minimum_size(filter_size, filter_type):
            if filter_size >= self.blockSize:
                return filter_size
            self.log.info(
                f"{filter_type.value} shorter than block size: Zero-padding "
                f"from {filter_size} samples to {self.blockSize} samples"
            )
            return self.blockSize

        ds_size = _get_minimum_size(ds_size, FilterType.ds_Filter)
        early_size = _get_minimum_size(early_size, FilterType.early_Filter)
        late_size = _get_minimum_size(late_size, FilterType.late_Filter)
        sd_size = _get_minimum_size(sd_size, FilterType.sd_Filter)
        hp_size = _get_minimum_size(hp_size, FilterType.headphone_Filter)

        def _get_multiple_size(filter_size, filter_type):
            if not np.mod(filter_size, self.blockSize):
                return filter_size
            new_size = (
                int(np.ceil(filter_size / self.blockSize)) * self.blockSize
            )
            self.log.info(
                f"{filter_type.value} not multiple of block size: Zero-padding "
                f"from {filter_size} samples to {new_size} samples"
            )
            return new_size

        ds_size = _get_multiple_size(ds_size, FilterType.ds_Filter)
        early_size = _get_multiple_size(early_size, FilterType.early_Filter)
        late_size = _get_multiple_size(late_size, FilterType.late_Filter)
        sd_size = _get_multiple_size(sd_size, FilterType.sd_Filter)
        hp_size = _get_multiple_size(hp_size, FilterType.headphone_Filter)

        # Create FilterStorage
        self.filterStorage = FilterStorage(
            block_size=self.blockSize,
            filter_source=self.config.get("filterSource[mat/wav]"),
            filter_list_name=self.config.get("filterList"),
            filter_database=self.config.get("filterDatabase"),
            torch_settings=self.config.get("torchStorage[cpu/cuda]"),
            useHeadphoneFilter=self.config.get("useHeadphoneFilter"),
            headphoneFilterSize=hp_size,
            ds_filterSize=ds_size,
            early_filterSize=early_size,
            late_filterSize=late_size,
            sd_filterSize=sd_size,
            useNearestNeighbour=self.config.get("useNearestNeighbour"),
        )

        # Create SoundHandler
        self.soundHandler = SoundHandler(
            block_size=self.blockSize,
            n_channels=self.nChannels,
            fs=self.sampleRate,
        )

        soundfiles = parse_soundfile_list(self.config.get("soundfile"))
        loop_config = (
            LoopState.LOOP if self.config.get("loopSound") else LoopState.SINGLE
        )
        self.soundHandler.create_player(
            soundfiles, CONFIG_SOUNDFILE_PLAYER_NAME, loop_state=loop_config
        )

        # Start a PkgReceiver
        recv_type = self.config.get("recv_type")
        recv_select = {"zmq": ZmqReceiver, "osc": OscReceiver}
        self.pkgReceiver = recv_select.get(recv_type, PkgReceiver)(
            self.config, self.soundHandler
        )
        self.pkgReceiver.start_listening()
        time.sleep(1)

        # Create input buffers
        torch_settings = self.config.get("torchConvolution[cpu/cuda]")
        self.input_Buffer = InputBufferMulti(
            block_size=self.blockSize,
            inputs=self.nChannels,
            torch_settings=torch_settings,
        )
        self.input_BufferHP = InputBufferMulti(
            block_size=self.blockSize,
            inputs=2,
            torch_settings=torch_settings,
        )
        self.input_BufferSD = InputBufferMulti(
            block_size=self.blockSize,
            inputs=2,
            torch_settings=torch_settings,
        )

        # Create N convolvers depending on the number of wav channels
        self.log.info("Number of Channels: " + str(self.nChannels))

        enableCrossfading = self.config.get("enableCrossfading")
        self.ds_convolver = ConvolverTorch(
            ir_size=ds_size,
            block_size=self.blockSize,
            stereoInput=False,
            sources=self.nChannels,
            interpolate=enableCrossfading,
            torch_settings=torch_settings,
        )
        self.early_convolver = ConvolverTorch(
            ir_size=early_size,
            block_size=self.blockSize,
            stereoInput=False,
            sources=self.nChannels,
            interpolate=enableCrossfading,
            torch_settings=torch_settings,
        )
        self.late_convolver = ConvolverTorch(
            ir_size=late_size,
            block_size=self.blockSize,
            stereoInput=False,
            sources=self.nChannels,
            interpolate=enableCrossfading,
            torch_settings=torch_settings,
        )
        self.sd_convolver = ConvolverTorch(
            ir_size=sd_size,
            block_size=self.blockSize,
            stereoInput=True,
            sources=self.nChannels,
            interpolate=enableCrossfading,
            torch_settings=torch_settings,
        )

        self.ds_convolver.active = self.config.get("ds_convolverActive")
        self.early_convolver.active = self.config.get("early_convolverActive")
        self.late_convolver.active = self.config.get("late_convolverActive")
        self.sd_convolver.active = self.config.get("sd_convolverActive")

        # HP Equalization convolver
        self.convolverHP = None
        if self.config.get("useHeadphoneFilter"):
            self.convolverHP = ConvolverTorch(
                ir_size=self.config.get("headphone_filterSize"),
                block_size=self.blockSize,
                stereoInput=True,
                sources=1,
                interpolate=False,
                torch_settings=torch_settings,
            )
            hpfilter = self.filterStorage.get_headphone_filter()
            self.convolverHP.setAllFilters([hpfilter])

    def __cleanup(self):
        # Close everything when BinSim is finished
        # self.oscReceiver.close()
        self.pkgReceiver.close()
        self.stream.close()
        self.filterStorage.close()
        self.input_Buffer.close()
        self.input_BufferHP.close()
        self.ds_convolver.close()
        self.early_convolver.close()
        self.late_convolver.close()

        if self.config.get("useHeadphoneFilter") and self.convolverHP:
            self.convolverHP.close()


def audio_callback(binsim):
    """Wrapper for callback to hand over custom data"""
    assert isinstance(binsim, BinSim)

    # The python-sounddevice Callback
    def callback(outdata, _frame_count, _time_info, status):
        # print("python-sounddevice callback")
        debug = "pydevd" in sys.modules
        if debug:
            import pydevd

            pydevd.settrace(suspend=False, trace_only_current_thread=True)

        # Update config
        # binsim.current_config = binsim.oscReceiver.get_current_config()
        binsim.current_config = binsim.pkgReceiver.get_current_config()

        amount_channels = binsim.current_config.get("maxChannels")
        if amount_channels == 0:
            return

        if binsim.current_config.get("pauseAudioPlayback"):
            binsim.block = torch.as_tensor(
                binsim.soundHandler.get_zeros(), dtype=torch.float32
            )
        else:
            loudness = callback.config.get("loudnessFactor")
            binsim.block = torch.as_tensor(
                binsim.soundHandler.get_block(loudness), dtype=torch.float32
            )

        if binsim.current_config.get("pauseConvolution"):
            if amount_channels == 2:
                binsim.result = binsim.block
            else:
                mix = torch.mean(binsim.block[:amount_channels, :], dim=0)
                binsim.result[0, :] = mix
                binsim.result[1, :] = mix
        else:
            input_buffers = binsim.input_Buffer.process(binsim.block)

            for n in range(amount_channels):
                if binsim.pkgReceiver.is_ds_filter_update_necessary(n):
                    filterList = list()
                    for sourceId in range(amount_channels):
                        filterValueList = (
                            binsim.pkgReceiver.get_current_ds_filter_values(
                                sourceId
                            )
                        )
                        _filter = binsim.filterStorage.get_ds_filter(
                            Pose.from_filterValueList(filterValueList)
                        )
                        filterList.append(_filter)
                    binsim.ds_convolver.setAllFilters(filterList)
                    break

            for n in range(amount_channels):
                if binsim.pkgReceiver.is_early_filter_update_necessary(n):
                    filterList = list()
                    for sourceId in range(amount_channels):
                        filterValueList = (
                            binsim.pkgReceiver.get_current_early_filter_values(
                                sourceId
                            )
                        )
                        _filter = binsim.filterStorage.get_early_filter(
                            Pose.from_filterValueList(filterValueList)
                        )
                        filterList.append(_filter)
                    binsim.early_convolver.setAllFilters(filterList)
                    break

            for n in range(amount_channels):
                if binsim.pkgReceiver.is_late_filter_update_necessary(n):
                    filterList = list()
                    for sourceId in range(amount_channels):
                        filterValueList = (
                            binsim.pkgReceiver.get_current_late_filter_values(
                                sourceId
                            )
                        )
                        _filter = binsim.filterStorage.get_late_filter(
                            Pose.from_filterValueList(filterValueList)
                        )
                        filterList.append(_filter)
                    binsim.late_convolver.setAllFilters(filterList)
                    break

            ds = binsim.ds_convolver.process(input_buffers)

            for n in range(amount_channels):
                if binsim.pkgReceiver.is_sd_filter_update_necessary(n):
                    sd_filterList = list()
                    for sourceId in range(amount_channels):
                        filterValueList = (
                            binsim.pkgReceiver.get_current_sd_filter_values(
                                sourceId
                            )
                        )
                        sd_filter = binsim.filterStorage.get_sd_filter(
                            SourcePose.from_filterValueList(filterValueList)
                        )
                        sd_filterList.append(sd_filter)
                    binsim.sd_convolver.setAllFilters(sd_filterList)
                    break

            # apply source directivity to ds block if needed
            if callback.config.get("sd_convolverActive"):
                sd_buffer = binsim.input_BufferSD.process(ds[:, 0, :])
                ds = binsim.sd_convolver.process(
                    sd_buffer
                )  # let's keep the name "ds" for now
            early = binsim.early_convolver.process(input_buffers)
            late = binsim.late_convolver.process(input_buffers)

            # combine early and late into ds
            ds.add_(early).add_(late)
            binsim.result = ds[:, 0, :]

            # Finally apply Headphone Filter
            if callback.config.get("useHeadphoneFilter"):
                result_buffer = binsim.input_BufferHP.process(binsim.result)
                binsim.result = binsim.convolverHP.process(result_buffer)[
                    :, 0, :
                ]

        outdata[:] = np.transpose(binsim.result.detach().cpu().numpy())

        # Report buffer underrun - Still working with sounddevice package?
        if status == 4:
            binsim.log.warn("Output buffer underrun occurred")

        # Report clipping
        if np.max(np.abs(outdata)) > 1:
            binsim.log.warn("Clipping occurred: Adjust loudnessFactor!")

        binsim.time_usage[binsim.time_usage_index] = binsim.stream.cpu_load
        binsim.time_usage_index = (binsim.time_usage_index + 1) % len(
            binsim.time_usage
        )

        if (
            binsim.ds_convolver.get_counter() % binsim.cpu_usage_update_rate
            == 0
        ):
            percentiles = np.percentile(binsim.time_usage, (0, 50, 100))
            mean = np.mean(binsim.time_usage)
            binsim.log.info(
                f"audio callback utilization: mean {mean:>6.2%} "
                f"| min {percentiles[0]:>6.2%} "
                f"| median {percentiles[1]:>6.2%} "
                f"| max {percentiles[2]:>6.2%}"
            )

    callback.config = binsim.config

    return callback
