import numpy as np
import tensorflow as tf

from ..utils.audiolib import standardize_audio_size, load_audio_file, audio_melspec
from ..utils.utils import WrapInferenceSession


class OODClusterScorer(object):
    """
    Classify audio as in or out of distribution relative DNS current test set
    """

    def __init__(self, onnx_model, input_length, centroids):

        self.session = WrapInferenceSession(onnx_model)
        self.input_length = input_length

        clustering = np.load(centroids)
        self.centers = clustering['centroids']
        self.covariance = np.linalg.inv(clustering['covariance'])

    @classmethod
    def from_dict(cls, config_dict):
        """
        Instantiate a OODScorer object using parameters from config_dict
        :param config_dict:
        :return: OODScorer object
        """
        onnx_models = config_dict['model_path']
        input_length = config_dict['input_length']
        centroids = config_dict['centroids']

        return cls(onnx_models, input_length, centroids)

    def score(self, mig_data):
        """

        :param fs:
        :param audio:
        :return:
        """
        mig_data = mig_data.to_dict()
        mig_filename = mig_data['clip_url']
        audio, fs = load_audio_file(mig_filename, input_length=self.input_length, standardize=False)

        if audio is None:
            return -1, mig_filename

        logprob = self.predict(audio, fs)
        score = 1 - np.exp(-logprob)

        return score, mig_filename

    def predict(self, audio, fs):
        """
        Compute probability to be in distribution using cluster assignment and distribution over assignment
        :param audio: np.array 1 D array
        :param fs: int, sample rate
        :return: mos_sig, mos_bak, mos_ovr
        """

        input_length = int(np.ceil(self.input_length)) * fs
        n = int(audio.shape[0] / input_length)
        if n <= 1:
            audio_splits = [audio]
        else:

            audio = audio[:int(n * input_length)]
            audio_splits = np.split(audio, int(audio.shape[0] / input_length))

        score_list = []

        for audio in audio_splits:
            audio = standardize_audio_size(audio, fs, self.input_length)
            input_features = audio_melspec(audio, sr=fs)[None, ...]

            onnx_inputs = {inp.name: input_features for inp in self.session.get_inputs()}
            outputs = self.session.get_outputs()[1]
            out = self.session.run([outputs.name], onnx_inputs)

            latent = out[0]

            x_exp = np.mean(latent, axis=(1))
            c_exp = self.centers #tf.expand_dims(self.centers, 0)
            covariance = self.covariance

            deviation = x_exp - c_exp
            deviation = deviation[..., None]

            dist = np.matmul(covariance, deviation)[..., 0]  # (nlcuster, dim, 1)
            dist = np.sum((x_exp - c_exp) * dist, 1)#(ncluster)

            assignment = dist.argmin()
            score = dist[assignment]

            score_list.append(score)

        return np.concatenate([score_list]).mean()


