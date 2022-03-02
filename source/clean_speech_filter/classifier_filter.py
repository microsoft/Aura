from ..utils.audiolib import standardize_audio_size, load_audio_file, audio_melspec
from ..utils.utils import WrapInferenceSession


class ClassFilter(object):
    """
    Filter clean speech by looking at audio with a mos_criteria < threshold
    """

    def __init__(self, onnx_model, input_length):

        self.session = WrapInferenceSession(onnx_model)

        self.input_length = input_length

    @classmethod
    def from_dict(cls, config_dict):
        """
        Instantiate a ClassFilter object using parameters from config_dict
        :param config_dict:
        :return: DNMOSFilter object
        """
        onnx_models = config_dict['model_path']
        input_length = config_dict['input_length']


        return cls(onnx_models, input_length)

    def classify_as_clean(self, audio_path):
        """

        :param fs:
        :param audio:
        :return:
        """
        audio, fs = load_audio_file(audio_path, input_length=self.input_length)

        if audio is None:
            return True, audio_path
        try: # if we are unable to compute the probability, we consider it as clean, so that it is discarded.
            prob = self.predict_prob_clean(audio, fs)
            pred = prob[0][0].argmax()
        except:
            pred = 0

        clean_speech = False
        if pred == 0:
            clean_speech = True

        return clean_speech, audio_path

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
