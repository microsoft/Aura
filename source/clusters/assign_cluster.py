import numpy as np
import pandas as pd

from ..utils.audiolib import standardize_audio_size, load_audio_file, audio_melspec
from ..utils.utils import WrapInferenceSession


class Cluster(object):
    """
    Assign audio file to cluster_id to each audio file
    """

    def __init__(self, onnx_model, input_length, centroids, tags):

        self.session = WrapInferenceSession(onnx_model)
        self.input_length = input_length

        clustering = np.load(centroids)
        self.centers = clustering['centroids']
        self.covariance = clustering['covariance']
        self.labels = clustering['centroid_labels']
        self.tags = tags

    @classmethod
    def from_dict(cls, config_dict):
        """
        Generate Cluster object using parameters from config_dict
        :param config_dict:
        :return: Cluster object
        """
        onnx_models = config_dict['model_path']
        input_length = config_dict['input_length']
        centroids = config_dict['centroids']
        tags = pd.read_csv(config_dict['tags_file'])
        tags.set_index(np.arange(len(tags)), inplace=True)

        return cls(onnx_models, input_length, centroids, tags)

    def assign(self, mig_data):
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

        label_list, cluster_list = self.predict(audio, fs)

        label = ','.join([self.tags.loc[lab, 'name'] for lab in label_list])
        cluster_id = ','.join([str(lab) for lab in cluster_list])

        return cluster_id, label, mig_filename

    def predict(self, audio, fs):
        """
        Assign a cluster_id to audio
        :param audio: np.array 1 D array
        :param fs: int, sample rate
        :return: cluster_id, cluster_label, label
        """

        input_length = int(np.ceil(self.input_length)) * fs
        n = int(audio.shape[0] / input_length)
        if n <= 1:
            audio_splits = [audio]
        else:
            audio = audio[:int(n * input_length)]
            audio_splits = np.split(audio, int(audio.shape[0] / input_length))

        label_list = []
        cluster_list = []

        for audio in audio_splits:
            audio = standardize_audio_size(audio, fs, self.input_length)
            input_features = audio_melspec(audio, sr=fs)[None, ...]

            onnx_inputs = {inp.name: input_features for inp in self.session.get_inputs()}
            out_latent = self.session.get_outputs()[1]
            #out_label = self.session.get_outputs()[0]
            out = self.session.run([out_latent.name], onnx_inputs)

            latent = out[0]

            x_exp = np.mean(latent, axis=(1))
            c_exp = self.centers
            deviation = x_exp - c_exp
            deviation = deviation[..., None]
            covariance = self.covariance

            dist = np.matmul(covariance, deviation)[..., 0]  # (nlcuster, dim, 1)
            dist = np.sum((x_exp - c_exp) * dist, 1)

            assignment = dist.argmin()
            labels = self.labels[assignment]
            labels = labels.argsort()[-3:][::-1].tolist()

            for label in labels:
                if label not in label_list:
                    label_list.append(label)

            if assignment not in cluster_list:
                cluster_list.append(assignment)

        return label_list, cluster_list


