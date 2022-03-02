import numpy as np

from ..utils.audiolib import standardize_audio_size, load_audio_file, infer_mos
from ..utils.utils import WrapInferenceSession


class DNSMOSFilter(object):
    """
    Filter clean speech by looking at audio with a mos_criteria < threshold
    """

    def __init__(self, onnx_models, mos_criteria, input_length):

        self.session_sig = WrapInferenceSession(onnx_models['sig_model_path'])
        self.session_bak_ovr = WrapInferenceSession(onnx_models['bak_ovr_model_path'])

        self.input_length = input_length

        assert all(key in ['MOS_sig', 'MOS_bak', 'MOS_ovr'] for key in mos_criteria.keys())
        self.mos_criteria = mos_criteria

    @classmethod
    def from_dict(cls, config_dict):
        """
        Instantiate a DNMOSFilter object using parameters from config_dict
        :param config_dict:
        :return: DNMOSFilter object
        """
        onnx_models = config_dict['dnsmos_path']
        input_length = config_dict['input_length']

        mos_criteria = config_dict['mos_criteria']

        return cls(onnx_models, mos_criteria, input_length)

    def classify_as_clean(self, audio_path):
        """

        :param fs:
        :param audio:
        :return:
        """
        audio, fs = load_audio_file(audio_path, input_length=self.input_length)

        if audio is None:
            return True, -1, -1, -1, audio_path

        if len(audio.shape) > 1:
            audio = audio.T
            audio = audio.sum(axis=0) / audio.shape[0]

        mos_sig, mos_bak, mos_ovr = self.infer_mos(audio, fs)
        mos_dict = {'MOS_sig': mos_sig, 'MOS_bak': mos_bak,'MOS_ovr': mos_ovr}

        clean_speech = False

        for mos_type, mos_value in self.mos_criteria.items():
            clean_speech = (mos_dict[mos_type] > mos_value)

            if clean_speech:
                break

        return clean_speech, mos_sig, mos_bak, mos_ovr, audio_path

    def infer_mos(self, audio, fs):
        """
        Compute mos_sig, mos_bak, mos_ovr predicted by models in self.session_sig, self.session_bak_ovr
        :param audio: np.array 1 D array
        :param fs: int, sample rate
        :return: mos_sig, mos_bak, mos_ovr
        """

        input_length = self.input_length
        audio = standardize_audio_size(audio, fs, input_length)

        sig, bak, ovr = infer_mos(audio, fs, input_length, self.session_sig, self.session_bak_ovr)

        return sig, bak, ovr