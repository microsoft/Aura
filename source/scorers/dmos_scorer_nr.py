import os
import numpy as np
import math
import subprocess
import librosa
import time
import shutil
import noisereduce as nr

import soundfile as sf
import tensorflow as tf

from ..utils.utils import WrapInferenceSession
from ..utils.audiolib import load_audio_file, infer_mos


class DMOSScorerFromNoiseReduce(object):

    def __init__(self, onnx_dns_model, onnx_dnsmos_models, input_length, score_criteria={'MOS_sig': 1.0}, sampling_rate=16000, no_stft=True):

        self.session_sig = WrapInferenceSession(onnx_dnsmos_models['sig_model_path'])
        self.session_bak_ovr = WrapInferenceSession(onnx_dnsmos_models['bak_ovr_model_path'])

        self.input_length = input_length
        self.no_stft = no_stft
        self.frame_size = int(0.02 * sampling_rate)  # 20ms
        self.shift = self.frame_size // 2

        if self.no_stft:
            tf.compat.v1.enable_eager_execution()
            win_len = self.frame_size
            hop_len = int(self.shift)
            self.stft_length = tf.cast(win_len, dtype=tf.int32)
            self.stft_shift = tf.cast(hop_len, dtype=tf.int32)
            self.n_fft = tf.cast(win_len, dtype=tf.int32)
            self.win_fn = tf.signal.hann_window
            self.hop_len = hop_len
            # self.inverse_win_fn = tf.signal.inverse_stft_window_fn(tf.cast(hop_len, dtype=tf.int32))

        #self.feed_dict_names = [inp.name for inp in self.input_tensors]

        assert all(key in ['MOS_sig', 'MOS_bak', 'MOS_ovr'] for key in score_criteria.keys())
        self.score_criteria = score_criteria


    @classmethod
    def from_dict(cls, config_dict):
        """
        Instantiate a DNMOSScorer object using parameters from config_dict
        :param config_dict:
        :return: DNMOSFilter object
        """
        onnx_dnsmos_models = config_dict['dnsmos_path']
        input_length = config_dict['input_length']

        no_stft = True
        if 'no_stft' in config_dict:
            no_stft = config_dict['no_stft']

        return cls(onnx_dnsmos_models, input_length, no_stft=no_stft)

    def denoise(self, mig_filename):
        """
        Apply model in self.model_path to denoise audio file in mig_filename and compute mos_sig,
        mos_bak, mos_ovr using dnmos_p835 model
        :param mig_filename: string, audio filename
        :return: float, float, float mos_sig, mos_bak, mos_ovr
        """

        audio, fs = load_audio_file(mig_filename, input_length=self.input_length)
        if audio is None:
            return -1, -1, -1, audio, fs

        enhanced_sig = self.run(audio, fs)

        dns_mos_sig, dns_mos_bak, dns_mos_ovr = self.infer_mos(enhanced_sig, fs)

        return dns_mos_sig, dns_mos_bak, dns_mos_ovr, audio, fs

    def score(self, mig_data):
        """
        Compute dmos for criteria in self.criteria.key() and compute an average score using weight from
        self.criteria.values()
        :param mig_data:
        :return:
        """
        mig_data = mig_data.to_dict()
        mig_filename = mig_data['clip_url']

        # compute mos after denoising
        dns_mos_sig, dns_mos_bak, dns_mos_ovr, audio, fs = self.denoise(mig_filename)
        if dns_mos_sig < 0:
            return -1, -1, -1, -1, -1, -1, -1, mig_filename

        dns_mos_dict = {'MOS_sig': dns_mos_sig, 'MOS_bak': dns_mos_bak, 'MOS_ovr': dns_mos_ovr}

        if 'MOS_ovr' in mig_data:
            mos_sig, mos_bak, mos_ovr = mig_data['MOS_sig'], mig_data['MOS_bak'], mig_data['MOS_ovr']
        else:
            mos_sig, mos_bak, mos_ovr = self.infer_mos(audio, fs)
        mos_dict = {'MOS_sig': mos_sig, 'MOS_bak': mos_bak, 'MOS_ovr': mos_ovr}

        score = 0
        for score_criteria, score_weight in self.score_criteria.items():
            score += score_weight * (dns_mos_dict[score_criteria] - mos_dict[score_criteria])

        return dns_mos_sig, dns_mos_bak, dns_mos_ovr, score, mos_sig, mos_bak, mos_ovr, mig_filename

    def infer_mos(self, audio, fs):
        """
        Compute mos_sig, mos_bak, mos_ovr predicted by models in self.session_sig, self.session_bak_ovr
        :param audio: np.array 1 D array
        :param fs: int, sample rate
        :return: mos_sig, mos_bak, mos_ovr
        """

        input_length = self.input_length
        sig, bak, ovr = infer_mos(audio, fs, input_length, self.session_sig, self.session_bak_ovr)

        return sig, bak, ovr

    def run(self, audio, fs):
        return nr.reduce_noise(y=audio, sr=fs)