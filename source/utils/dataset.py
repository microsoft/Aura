import os
import glob
import urllib.request
from azureml import data

import numpy as np
import soundfile as sf
import pandas as pd
import tensorflow as tf

from ..utils.audiolib import get_one_zero_label, audio_melspec


class ReadDataFromCSV(object):
    """
    From a directory collect all csv files in a dataframe and read each audio_url
    """

    def __init__(self, data_csv, column_url='clip_url', frac_samples=None):

        self.audio_path = pd.read_csv(data_csv)

        if frac_samples is not None:
            self.audio_path = self.audio_path.sample(frac=frac_samples)

        n = len(self.audio_path)
        self.audio_path.set_index(np.arange(n), inplace=True)

        self.column_url = column_url
        self.columns = self.audio_path.columns

    def __call__(self):
        """
        Create an iterator that iterates over the data
        :return:
        """
        return [audio_row[self.column_url] for _, audio_row in self.audio_path.iterrows()]

    def __len__(self):
        return len(self.audio_path)

    def filter(self, criteria):
        """
        Adjust dataset according to criteria
        :param criteria: dict
        :return:
        """
        for criteria_key, criteria_value in criteria.items():
            self.audio_path = self.audio_path[self.audio_path[criteria_key] == criteria_value]

        n = len(self.audio_path)
        self.audio_path.set_index(np.arange(n), inplace=True)


class ReadDataFromDataFrame(ReadDataFromCSV):

    def __init__(self, dataframe, column_url='file_url', frac_samples=None):
        self.audio_path = dataframe

        if frac_samples is not None:
            self.audio_path = self.audio_path.sample(frac=frac_samples)

        n = len(self.audio_path)
        self.audio_path.set_index(np.arange(n), inplace=True)

        self.column_url = column_url
        self.columns = self.audio_path.columns


class SpectralFeatures(object):
    """
    Collect audio-clip dataset into an iterator of log-mel spectograms
    """

    def __init__(self, dir, random=True, input_len_second=4.95, frac_samples=None, range=None, filename='clip-url'):

        self.input_len_second = input_len_second
        self.dir = dir
        self.filename = filename

        self.clip_files = glob.glob(os.path.join(self.dir, '*.csv'))

        audio_path_list = []
        for csv_clip_file in self.clip_files:
            df = pd.read_csv(csv_clip_file)
            audio_path_list.append(df)

        self.audio_path = pd.concat(audio_path_list)

        if random:
            self.audio_path = self.audio_path.sample(frac=1)

        if frac_samples is not None:
            self.audio_path = self.audio_path.sample(frac=frac_samples)

        if range is not None:
            self.audio_path = self.audio_path.loc[range[0]:range[1], :]

        n = len(self.audio_path)
        self.audio_path.set_index(np.arange(n), inplace=True)

    def standardize_audio_size(self, audio, fs):
        """
        Adjust audio size to be of size fs * self.input_len_second
        If len(audio) > fs * self.input_len_second, sample a sub audio clip of size fs * self.input_len_second
        If len(audio) < fs * self.input_len_second, pad the audio clip with itself
        :param audio: np.array
        :param fs: int, sampling rate
        :return:
        """
        audio = np.tile(audio, np.ceil(fs * self.input_len_second / audio.shape[0]).astype('int32'))

        if len(audio) > self.input_len_second * fs:
            start_idx = np.random.randint(0, len(audio) - self.input_len_second * fs)
            end_idx = start_idx + int(self.input_len_second * fs)

            audio = audio[start_idx:end_idx]

        return audio

    def __getitem__(self, item):

        clip_url = self.audio_path.loc[item, self.filename]
        local_url = os.path.basename(clip_url)

        local_name, _ = urllib.request.urlretrieve(clip_url, local_url)
        audio, fs = sf.read(local_name)

        # standardize the size of all clips to fs * self.input_len
        audio = self.standardize_audio_size(audio, fs)

        return audio_melspec(audio)

    def __call__(self):
        """
        Create a generator that iterates over the data
        :return:
        """

        for audio_index, audio_row in self.audio_path.iterrows():
            clip_url = audio_row[self.filename]
            local_url = os.path.basename(clip_url)

            try:
                local_name = tf.keras.utils.get_file(origin=clip_url, fname=local_url)
            except:
                print(f'Error when reading file {clip_url}')
                continue

            audio, fs = sf.read(local_name)

            # standardize the size of all clips to fs * self.input_len
            audio = self.standardize_audio_size(audio, fs)

            yield audio_melspec(audio)

    def __len__(self):
        return len(self.audio_path)


class SpectralFeaturesWithLabels(SpectralFeatures):

    """
    Add noise type label to iterator from SpectralFeatures
    """

    def __init__(self, dir, tags_file, random=True, input_len_second=4.95, frac_samples=None, range=None, filename='clip-url'):

        super().__init__(dir, random=random, input_len_second=input_len_second, frac_samples=frac_samples, range=range, filename=filename)

        tags_mapping = pd.read_csv(tags_file)
        tags_mapping['tag_numerical'] = np.arange(len(tags_mapping))
        tags_mapping.set_index('id', inplace=True)

        self.tags_mapping = tags_mapping.to_dict('index')
        self.num_labels = len(tags_mapping)

    def __getitem__(self, item):

        clip_url = self.audio_path.loc[item, self.filename]
        local_url = os.path.basename(clip_url)
        tag_code = self.audio_path.loc[item, 'tag_code']

        local_name, _ = urllib.request.urlretrieve(clip_url, local_url)
        audio, fs = sf.read(local_name)

        # standardize the size of all clips to fs * self.input_len
        audio = self.standardize_audio_size(audio, fs)

        # labels
        label_one_hot = get_one_zero_label(tag_code, self.tags_mapping, num_labels=self.num_labels)

        return audio_melspec(audio), label_one_hot


    def __call__(self):
        """
        Create a generator that iterates over the data
        :return:
        """

        for audio_index, audio_row in self.audio_path.iterrows():
            clip_url = audio_row[self.filename]
            local_url = os.path.basename(clip_url)

            try:
                local_name, _ = urllib.request.urlretrieve(clip_url, local_url)
                audio, fs = sf.read(local_name)
            except:
                print(f'Error when reading file {clip_url}')
                continue

            if audio.shape[0] == 0:
                continue

            # standardize the size of all clips to fs * self.input_len
            audio = self.standardize_audio_size(audio, fs)

            # labels
            label_one_hot = get_one_zero_label(audio_row['tag_code'], self.tags_mapping, num_labels=self.num_labels)

            yield audio_melspec(audio), label_one_hot


class SpectralFeaturesWithLabelsAndSNR(SpectralFeatures):

    """
    Add noise type label and snr to iterator from SpectralFeatures
    """

    def __init__(self, dir, tags_file, random=True, input_len_second=4.95, frac_samples=None, range=None, filename='clip-url'):

        super().__init__(dir, random=random, input_len_second=input_len_second, frac_samples=frac_samples, range=range, filename=filename)

        tags_mapping = pd.read_csv(tags_file)
        tags_mapping['tag_numerical'] = np.arange(len(tags_mapping))
        tags_mapping.set_index('id', inplace=True)

        self.tags_mapping = tags_mapping.to_dict('index')
        self.num_labels = len(tags_mapping)

    def __getitem__(self, item):

        clip_url = self.audio_path.loc[item, self.filename]
        local_url = os.path.basename(clip_url)
        tag_code = self.audio_path.loc[item, 'tag_code']
        snr = self.audio_path.loc[item, 'snr']

        local_name, _ = urllib.request.urlretrieve(clip_url, local_url)
        audio, fs = sf.read(local_name)

        # standardize the size of all clips to fs * self.input_len
        audio = self.standardize_audio_size(audio, fs)

        # labels
        label_one_hot = get_one_zero_label(tag_code, self.tags_mapping, num_labels=self.num_labels)

        return audio_melspec(audio), label_one_hot, snr


    def __call__(self):
        """
        Create a generator that iterates over the data
        :return:
        """

        for audio_index, audio_row in self.audio_path.iterrows():
            clip_url = audio_row[self.filename]
            local_url = os.path.basename(clip_url)

            try:
                local_name, _ = urllib.request.urlretrieve(clip_url, local_url)
                audio, fs = sf.read(local_name)
            except:
                print(f'Error when reading file {clip_url}')
                continue

            if audio.shape[0] == 0:
                continue

            # standardize the size of all clips to fs * self.input_len
            audio = self.standardize_audio_size(audio, fs)

            # labels
            label_one_hot = get_one_zero_label(audio_row['tag_code'], self.tags_mapping, num_labels=self.num_labels)

            yield audio_melspec(audio), label_one_hot, audio_row['snr']

class SpectralFeaturesWithLabelsSmoothing(SpectralFeaturesWithLabels):
    """
    Overwrite the __getitem__ method to generate audio file and label that are weighted average of two
    audio files and labels.
    """

    def __getitem__(self, item):

        clip_url = self.audio_path.loc[item, self.filename]
        local_url = os.path.basename(clip_url)
        tag_code = self.audio_path.loc[item, 'tag_code']

        local_name, _ = urllib.request.urlretrieve(clip_url, local_url)
        audio, fs = sf.read(local_name)

        # pair audio
        id = np.random.randint(0, len(self.audio_path))
        clip_url2 = self.audio_path.loc[id, self.filename]
        local_url2 = os.path.basename(clip_url2)
        tag_code2 = self.audio_path.loc[id, 'tag_code']

        local_name2, _ = urllib.request.urlretrieve(clip_url2, local_url2)
        audio2, fs2 = sf.read(local_name2)

        # standardize the size of all clips to fs * self.input_len
        audio = self.standardize_audio_size(audio, fs)
        audio2 = self.standardize_audio_size(audio2, fs2)

        alpha = np.random.rand()
        audio = alpha * audio + (1 - alpha) * audio2

        # labels
        label_one_hot = get_one_zero_label(tag_code, self.tags_mapping, num_labels=self.num_labels)
        label_one_hot2 = get_one_zero_label(tag_code2, self.tags_mapping, num_labels=self.num_labels)
        label_one_hot = alpha * label_one_hot + (1 - alpha) * label_one_hot2

        return audio_melspec(audio), label_one_hot