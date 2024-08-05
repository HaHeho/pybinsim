import logging

import numpy as np

from pybinsim.filterstorage import FilterType
from pybinsim.parsing import parse_soundfile_list
from pybinsim.soundhandler import LoopState, PlayState, SoundHandler

CONFIG_SOUNDFILE_PLAYER_NAME = "config_soundfile"


class PkgReceiver(object):
    def __init__(self, current_config, soundhandler: SoundHandler):
        self.log = logging.getLogger(f"{__package__}.{self.__class__.__name__}")
        self.log.info("Init")

        self.soundhandler = soundhandler

        # Basic settings
        self.ip = current_config.get("recv_ip")
        self.port = current_config.get("recv_port")
        self.proto = current_config.get("recv_protocol")
        self.maxChannels = 100

        self.currentConfig = current_config

        # Default values; Stores filter keys for all channels/convolvers
        self.ds_filters_updated = [True] * self.maxChannels
        self.early_filters_updated = [True] * self.maxChannels
        self.late_filters_updated = [True] * self.maxChannels
        self.sd_filters_updated = [True] * self.maxChannels

        self.default_filter_value = np.zeros((1, 15))
        self.default_sd_filter_value = np.zeros((1, 9))

        self.valueList_ds_filter = np.tile(
            self.default_filter_value, [self.maxChannels, 1]
        )
        self.valueList_early_filter = np.tile(
            self.default_filter_value, [self.maxChannels, 1]
        )
        self.valueList_late_filter = np.tile(
            self.default_filter_value, [self.maxChannels, 1]
        )
        self.valueList_sd_filter = np.tile(
            self.default_sd_filter_value, [self.maxChannels, 1]
        )

        self.record_audio_callback_benchmark_data = current_config.get(
            "audio_callback_benchmark"
        )
        if self.record_audio_callback_benchmark_data:
            self.times_azimuth_received = list()

    def start_listening(self):
        """Start PkgReceiver thread"""
        pass

    @staticmethod
    def select_slice(i):
        switcher = {
            "/pyBinSim_ds_Filter": slice(0, 15),
            "/pyBinSim_ds_Filter_Short": slice(0, 9),
            "/pyBinSim_ds_Filter_Orientation": slice(0, 3),
            "/pyBinSim_ds_Filter_Position": slice(3, 6),
            "/pyBinSim_ds_Filter_sourceOrientation": slice(6, 9),
            "/pyBinSim_ds_Filter_sourcePosition": slice(9, 12),
            "/pyBinSim_ds_Filter_Custom": slice(12, 15),
            "/pyBinSim_early_Filter": slice(0, 15),
            "/pyBinSim_early_Filter_Short": slice(0, 9),
            "/pyBinSim_early_Filter_Orientation": slice(0, 3),
            "/pyBinSim_early_Filter_Position": slice(3, 6),
            "/pyBinSim_early_Filter_sourceOrientation": slice(6, 9),
            "/pyBinSim_early_Filter_sourcePosition": slice(9, 12),
            "/pyBinSim_early_Filter_Custom": slice(12, 15),
            "/pyBinSim_late_Filter": slice(0, 15),
            "/pyBinSim_late_Filter_Short": slice(0, 9),
            "/pyBinSim_late_Filter_Orientation": slice(0, 3),
            "/pyBinSim_late_Filter_Position": slice(3, 6),
            "/pyBinSim_late_Filter_sourceOrientation": slice(6, 9),
            "/pyBinSim_late_Filter_sourcePosition": slice(9, 12),
            "/pyBinSim_late_Filter_Custom": slice(12, 15),
            "/pyBinSim_sd_Filter": slice(0, 9),
        }
        return switcher.get(i, [])

    def handle_ds_filter_input(self, identifier, channel, *args):
        """
        Handler for tracking information

        :param identifier:
        :param channel:
        :param args:
        :return:
        """
        self.log.debug(f"{identifier=}, {channel=}, {args=}")
        key_slice = self.select_slice(identifier)
        if len(args) == len(self.valueList_ds_filter[channel, key_slice]):
            if np.all(args == self.valueList_ds_filter[channel, key_slice]):
                self.log.debug(f"Same {FilterType.ds_Filter.value} as before")
            else:
                self.ds_filters_updated[channel] = True
                self.valueList_ds_filter[channel, key_slice] = args
        else:
            self.log.warning(
                f"OSC identifier/key mismatch for {key_slice=}, {len(args)=}"
            )

    def handle_early_filter_input(self, identifier, channel, *args):
        """
        Handler for tracking information

        :param identifier:
        :param channel:
        :param args:
        :return:
        """
        self.log.debug(f"{identifier=}, {channel=}, {args=}")
        key_slice = self.select_slice(identifier)

        if len(args) == len(self.valueList_early_filter[channel, key_slice]):
            if np.all(args == self.valueList_early_filter[channel, key_slice]):
                self.log.debug(
                    f"Same {FilterType.early_Filter.value} as before"
                )
            else:
                self.early_filters_updated[channel] = True
                self.valueList_early_filter[channel, key_slice] = args
        else:
            self.log.warning(
                f"OSC identifier/key mismatch for {key_slice=}, {len(args)=}"
            )

    def handle_late_filter_input(self, identifier, channel, *args):
        """
        Handler for tracking information

        :param identifier:
        :param channel:
        :param args:
        :return:
        """
        self.log.debug(f"{identifier=}, {channel=}, {args=}")
        key_slice = self.select_slice(identifier)

        if len(args) == len(self.valueList_late_filter[channel, key_slice]):
            if np.all(args == self.valueList_late_filter[channel, key_slice]):
                self.log.debug(f"Same {FilterType.late_Filter.value} as before")
            else:
                self.late_filters_updated[channel] = True
                self.valueList_late_filter[channel, key_slice] = args
        else:
            self.log.warning(
                f"OSC identifier/key mismatch for {key_slice=}, {len(args)=}"
            )

    def handle_sd_filter_input(self, identifier, channel, *args):
        """
        Handler for tracking information

        :param identifier:
        :param channel:
        :param args:
        :return:
        """
        self.log.debug(f"{identifier=}, {channel=}, {args=}")
        key_slice = self.select_slice(identifier)
        if len(args) == len(self.valueList_sd_filter[channel, key_slice]):
            if np.all(args == self.valueList_sd_filter[channel, key_slice]):
                self.log.debug(f"Same {FilterType.sd_Filter.value} as before")
            else:
                self.sd_filters_updated[channel] = True
                self.valueList_sd_filter[channel, key_slice] = args
        else:
            self.log.warning(
                f"OSC identifier/key mismatch for {key_slice=}, {len(args)=}"
            )

    def handle_file_input(self, identifier, soundpath):
        """Handler for playlist control"""
        assert identifier == "/pyBinSimFile"
        assert isinstance(soundpath, str)
        self.soundhandler.stop_all_players()
        self.soundhandler.create_player(
            parse_soundfile_list(soundpath),
            CONFIG_SOUNDFILE_PLAYER_NAME,
            loop_state=(
                LoopState.LOOP
                if self.currentConfig.get("loopSound")
                else LoopState.SINGLE
            ),
        )
        self.log.info(f"Setting player sound path to '{soundpath}'")

    def handle_play(
        self,
        identifier,
        soundfile_list,
        start_channel=0,
        loop="single",
        player_name=None,
        volume=1.0,
        play="play",
    ):
        assert identifier == "/pyBinSimPlay"
        assert isinstance(soundfile_list, str)
        assert isinstance(start_channel, int)
        assert isinstance(loop, str)
        loop = loop.lower()
        if player_name is None:
            player_name = soundfile_list
        volume = float(volume)
        assert isinstance(play, str)
        play = play.lower()

        if loop == "loop":
            loop_state = LoopState.LOOP
        elif loop == "single":
            loop_state = LoopState.SINGLE
        else:
            raise ValueError("Loop argument must be 'loop' or 'single'")
        if play == "play":
            play_state = PlayState.PLAYING
        elif play == "pause":
            play_state = PlayState.PAUSED
        else:
            raise ValueError("Play argument must be 'play' or 'pause'")

        filepaths = parse_soundfile_list(soundfile_list)
        self.soundhandler.create_player(
            filepaths,
            player_name,
            start_channel,
            loop_state,
            play_state,
            volume,
        )
        self.log.info(
            f"Starting player '{player_name}' at channel {start_channel}, "
            f"{loop_state}, {play_state}, volume {volume:.3f}"
        )

    def handle_player_control(self, identifier, player_name, play):
        assert identifier == "/pyBinSimPlayerControl"

        if play == "play":
            play_state = PlayState.PLAYING
        elif play == "pause":
            play_state = PlayState.PAUSED
        elif play == "stop":
            play_state = PlayState.STOPPED
        else:
            raise ValueError("Play argument must be 'play', 'pause' or 'stop'")

        try:
            self.soundhandler.get_player(player_name).play_state = play_state
            self.log.info(f"Setting player '{player_name}' to {play_state}")
        except AttributeError:
            self.log.warning(
                f"Ignored setting player '{player_name}' to {play_state}"
            )

    def handle_player_channel(self, identifier, player_name, channel):
        assert identifier == "/pyBinSimPlayerChannel"
        assert isinstance(channel, int)

        self.soundhandler.set_player_start_channel(player_name, channel)
        self.log.info(f"Setting player '{player_name}' to channel {channel}")

    def handle_player_volume(self, identifier, player_name, volume):
        assert identifier == "/pyBinSimPlayerVolume"

        volume = float(volume)
        self.soundhandler.set_player_volume(player_name, volume)
        self.log.info(
            f"Setting player '{player_name}' to volume {volume:.3f} "
            f"({20.0 * np.log10(volume):+.1f} dB)"
        )

    def handle_stop_all_players(self, identifier):
        assert identifier == "/pyBinSimStopAllPlayers"

        self.soundhandler.stop_all_players()
        self.log.info("stopping all players")

    def handle_audio_pause(self, identifier, value):
        """Handler for playback control"""
        assert identifier == "/pyBinSimPauseAudioPlayback"

        self.currentConfig.set("pauseAudioPlayback", value)
        self.log.info(f"{'Pausing' if value else 'Unpausing'} audio")

    def handle_convolution_pause(self, identifier, value):
        """Handler for playback control"""
        assert identifier == "/pyBinSimPauseConvolution"

        self.currentConfig.set("pauseConvolution", value)
        self.log.info(f"{'Pausing' if value else 'Unpausing'} convolution")

    def handle_loudness(self, identifier, value):
        """Handler for loudness control"""
        assert identifier == "/pyBinSimLoudness"

        self.currentConfig.set("loudnessFactor", float(value))
        self.log.info(
            f"Changing loudness to {value:.3f} "
            f"({20.0 * np.log10(value):+.1f} dB)"
        )

    def is_ds_filter_update_necessary(self, channel):
        """Check if there is a new direct filter for channel"""
        return self.ds_filters_updated[channel]

    def is_early_filter_update_necessary(self, channel):
        """Check if there is a new early reverb filter for channel"""
        return self.early_filters_updated[channel]

    def is_late_filter_update_necessary(self, channel):
        """Check if there is a new late reverb filter for channel"""
        return self.late_filters_updated[channel]

    def is_sd_filter_update_necessary(self, channel):
        """Check if there is a new source directivity filter for channel"""
        return self.sd_filters_updated[channel]

    def get_current_ds_filter_values(self, channel):
        """Return key for filter"""
        self.ds_filters_updated[channel] = False
        return self.valueList_ds_filter[channel, :]

    def get_current_early_filter_values(self, channel):
        """Return key for late reverb filters"""
        self.early_filters_updated[channel] = False
        return self.valueList_early_filter[channel, :]

    def get_current_late_filter_values(self, channel):
        """Return key for late reverb filters"""
        self.late_filters_updated[channel] = False
        return self.valueList_late_filter[channel, :]

    def get_current_sd_filter_values(self, channel):
        """Return key for source directivity filters"""
        self.sd_filters_updated[channel] = False
        return self.valueList_sd_filter[channel, :]

    def get_current_config(self):
        return self.currentConfig

    def get_times_azimuth_received_and_reset(self):
        result = self.times_azimuth_received
        self.times_azimuth_received = list()
        return result

    def close(self):
        pass
