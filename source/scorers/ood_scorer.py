import numpy as np

from ..utils.audiolib import standardize_audio_size, load_audio_file, audio_melspec
from ..utils.utils import WrapInferenceSession


class OODScorer(object):
    """
    Classify audio as in or out of distribution relative DNS current test set
    """

    def __init__(self, onnx_model, input_length):

        self.session = WrapInferenceSession(onnx_model)

        self.input_length = input_length

    @classmethod
    def from_dict(cls, config_dict):
        """
        Instantiate a OODScorer object using parameters from config_dict
        :param config_dict:
        :return: OODScorer object
        """
        onnx_models = config_dict['model_path']
        input_length = config_dict['input_length']


        return cls(onnx_models, input_length)

    def score(self, mig_data):
        """

        :param fs:
        :param audio:
        :return:
        """
        mig_data = mig_data.to_dict()
        mig_filename = mig_data['clip_url']
        audio, fs = load_audio_file(mig_filename, input_length=self.input_length)

        if audio is None:
            return -1, mig_filename

        prob = self.predict_prob_clean(audio, fs)
        pred = prob[0][0]
        pred = np.exp(pred)
        pred = pred /pred.sum()
        score_is_testset = pred[3] / (pred[2] + pred[3])

        return 1 - score_is_testset, mig_filename

    def predict_prob_clean(self, audio, fs):
        """
        Compute probability to be a clean speech
        :param audio: np.array 1 D array
        :param fs: int, sample rate
        :return: mos_sig, mos_bak, mos_ovr
        """

        input_length = self.input_length
        audio = standardize_audio_size(audio, fs, input_length)
        input_features = audio_melspec(audio, sr=fs)[None, ...]

        onnx_inputs = {inp.name: input_features for inp in self.session.get_inputs()}
        prob = self.session.run(None, onnx_inputs)

        return prob
