"""
@author: chkarada + xgitiaux
"""

# Note: This single process audio synthesizer will attempt to use each noisy
# speech sourcefile once, as it does not randomly sample from these files
import os
import glob
import argparse
import urllib.request
from pathlib import Path
import configparser as cp
import random

import librosa
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from scipy import signal

from ..utils.audiolib import audioread, audiowrite, segmental_snr_mixer, activitydetector, is_clipped, \
    standardize_audio_size

from ..utils.log import get_logger


MAXTRIES = 50
MAXFILELEN = 100

np.random.seed(4)

logger = get_logger(__name__)

def balance_label(df, label_column, n=20, controlled_categories=['/m/04rlf', '/m/0jbk', '/m/04szw', '/m/07yv9']):
    """
    Generate a balance dataset according to label in label_column with n samples per class

    :param df: panda df
    :param label_column: string
    :param n: int
    """
    cols_list = list(df.columns)
    labels = list(set(df[label_column]))
    labels = [w for lab in labels for w in lab.split(',')]
    labels = list(set(labels))

    df_list = []
    for label in labels:
        if label not in list(df.columns):
            df[label] = df[label_column].apply(lambda x: label in x.split(',')).astype('int32')

    for label in labels:
        d = df[(df[label] == 1)]
        if controlled_categories is not None:

            if label in controlled_categories:
                d = d[d[controlled_categories].sum(1) <= 1]
            else:
                d = d[d[controlled_categories].sum(1) < 1]

            replace = False
            if n > len(d):
                replace = True

        if len(d) > 0:
            d = d.sample(n=n, replace=replace)
        df_list.append(d[cols_list])

    data = pd.concat(df_list)
    return data


def add_pyreverb(clean_speech, rir):
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")

    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[0: clean_speech.shape[0]]

    return reverb_speech


def pick_clean_speech(fs_output, clean_files_list, size=10, seed=None, tmp=None, clean_activity_threshold=None):
    """
    Randomly choose a clean speech to be convolved with a background noise. The
    chosen clip needs to be not clipped.

    The clean speech size is adjusted to size: if len(clean_speech) > size, takes a random subset;
    else tile clean_speech to match size

    :param fs_output: int sample rate of noise clip to be convolved with clean speech
    :param clean_files: list of clean speech file names
    :param size: float size of the clean speech to return
    :param seed:
    :return:
    """

    notfound = True
    tries = 0

    while (notfound) & (tries < MAXTRIES):

        tries += 1
        clip_url = random.choice(clean_files_list)

        local_url = os.path.basename(clip_url)
        if tmp is not None:
            local_url = os.path.join(tmp, local_url)

        try:
            audio_url, _ = urllib.request.urlretrieve(clip_url, local_url)
            audio, fs = audioread(audio_url)
        except:
            continue

        if ~is_clipped(audio):
            notfound = False

        if audio.shape[0] == 0:
            notfound = True

        if clean_activity_threshold is not None:
            try:
                dect = activitydetector(audio)
                if dect < clean_activity_threshold:
                    notfound = True
            except:
                notfound = True

    if tries > MAXTRIES:
        return [], None

    # resample
    if fs != fs_output:
        audio = librosa.resample(audio, fs, fs_output)

    # adjust audio size to noise size
    if audio.shape[0] != size * fs_output:
        audio = standardize_audio_size(audio, fs_output, size)

    return audio, clip_url


def build_audio(noise_file, clean_speech_files_list, save_dir, snr_lower=-5, snr_upper=20, target_lower=-35,
                target_upper=-15, seed=None, dest_name=None, tmp=None, noisy_pair=False, clean_activity_threshold=None):
    """

    :param noisy_pair:
    :param dest_name:
    :param seed:
    :param activity_threshold:
    :param target_upper:
    :param target_lower:
    :param snr_upper:
    :param snr_lower:
    :param save_dir:
    :param noise_files:
    :param index:
    :param clean_speech_files_array:
    :param save: directory to write noisy audio file
    :return: list of labels of the noise convolved with audio_files[index], path_clean, path_noisy
    """
    clip_url = noise_file['clip-url']
    local_url = os.path.basename(clip_url)

    logger.info(f'Add noise to {clip_url}')

    if Path(local_url).is_file():
        noise_url = local_url
    else:
        try:
            noise_url, _ = urllib.request.urlretrieve(clip_url, local_url)
        except:
            logger.info(f"Issue loading {clip_url}")
            return [], None, None

    try:
        noise, fs = audioread(noise_url)
    except:
        logger.info(f"Issue loading {clip_url}")
        return [], None, None

    labels = noise_file['tag_name'].split(',')
    labels_code = noise_file['label'].split(',')

    has_speech = noise_file['has_speech']

    # pick clean speech: one or two clean
    num_noisyspeech = 1 + 1 * (noisy_pair)
    snr = random.randint(snr_lower, snr_upper)

    clip_list = []
    clean_url_list = []
    for i in range(num_noisyspeech):
        clean_speech_audio, clean_speech_url = pick_clean_speech(fs,
                                                                 clean_speech_files_list,
                                                                 seed=seed,
                                                                 size=noise.shape[0] / fs,
                                                                 tmp=tmp,
                                                                 clean_activity_threshold=clean_activity_threshold
                                                                 )
        if len(clean_speech_audio) == 0:
            if noisy_pair:
                return [], None, None, None
            else:
                return [], None, None

        # mix clean speech and noise
        clean_snr, noise_snr, noisy_snr, target_level = segmental_snr_mixer(clean_speech_audio,
                                                                            noise,
                                                                            snr,
                                                                            target_upper=target_upper,
                                                                            target_lower=target_lower)

        if is_clipped(clean_snr) or is_clipped(noise_snr) or is_clipped(noisy_snr):
            logger.info(f"Warning: File {clip_url} has unexpected clipping returning without writing audio to disk")
            if noisy_pair:
                return [], None, None, None
            else:
                return [], None, None

        if has_speech:
            clip_list.append(noise)
        else:
            clip_list.append(noisy_snr)

        clean_url_list.append(clean_speech_url)

    labels_name_path = '_'.join(lab.replace(' ', '') for lab in labels[0:3])

    path_noisy_list = []

    for i, clean_speech_url in enumerate(clean_url_list):
        local_url = os.path.basename(clean_speech_url).split('.wav')[0]
        path_noisy = os.path.join(save_dir, 'noisy', f'{local_url}_{labels_name_path}.wav')

        noisy_snr = clip_list[i]
        audiowrite(path_noisy, noisy_snr, fs)

        if dest_name is not None:
            path_noisy = os.path.join(dest_name, 'noisy', f'{local_url}_{labels_name_path}.wav')
            path_noisy_list.append(path_noisy)

    if has_speech:
        labels = [lab for lab in labels if lab not in ['Male speech', ' man speaking', 'Female speech', ' woman speaking', 'Speech', 'Narration',' monologue']]
        labels_code = [lab for lab in labels_code if lab not in ['/m/02zsn', '/m/05zppz', '/m/09x0r', '/m/02qldy']]

    return labels, labels_code, path_noisy_list, snr, clip_url, has_speech


def create_noisy_audio(cfg, data_dir, dest_dir, dest_name=None, num_workers=2):
    """

    :param cfg:
    :param data_dir:
    :param dest_dir:
    :param dest_name:
    :param num_workers:
    :return:
    """

    # get configuration values
    config = cp.ConfigParser()
    config._interpolation = cp.ExtendedInterpolation()
    config.read(cfg)

    config['path']['data_dir'] = data_dir
    config['path']['dest_dir'] = dest_dir

    destination_name = config['path']['dest_dir']
    if dest_name is not None:
        config['path']['dest_name'] = dest_name
        destination_name = config['audio_parameters']['files_names']

    save_dir = config['audio_parameters']['files_destination']

    clean = list(config['speech_directories'].values())

    snr_lower = config.getint('audio_parameters', 'snr_lower')
    snr_upper = config.getint('audio_parameters', 'snr_upper')
    target_level_lower = config.getint('audio_parameters', 'target_level_lower')
    target_level_upper = config.getint('audio_parameters', 'target_level_upper')
    exclude_speech = config.getboolean('audio_parameters', 'exclude_speech')
    include_only = config.getboolean('audio_parameters', 'include_only')
    noisy_pair = config.getboolean('audio_parameters', 'noisy_pair')
    balance = config.getboolean('audio_parameters', 'balance')
    balance_speech = config.getboolean('audio_parameters', 'balance_speech')
    clean_activity_threshold = config.getfloat('audio_parameters', 'clean_activity_threshold')

    frac_sample = config.getboolean('audio_parameters', 'frac_sample')
    if frac_sample:
        noisy = [(data, float(config['frac_sample'][name])) for name, data in config['noise_directories'].items()]
    else:
        noisy = [(data, 1.0) for data in config['noise_directories'].values()]

    clean_files_list = []
    for clean_directory in clean:
        clean_files = glob.glob(os.path.join(clean_directory, '*.csv'))
        for clean_file in clean_files:
            df = pd.read_csv(clean_file)
            df['clean_type'] = clean_directory
            clean_files_list.append(df[['clip_url', 'clean_type']])

    clean_files_index = pd.concat(clean_files_list)
    if balance_speech:
        n = config.getint('balance_parameters', 'n_speech')
        clean_files_index = clean_files_index.groupby('clean_type').sample(n=n)

    logger.info(f'Clean speech composition: {clean_files_index.groupby("clean_type").size()}')
    clean_files_list = clean_files_index['clip_url'].tolist()

    noise_files_list = []
    for noise_directory in noisy:
        noise_files = glob.glob(os.path.join(noise_directory[0], '*.csv'))
        for noise_file_path in noise_files:
            df = pd.read_csv(noise_file_path)
            df = df.sample(frac=noise_directory[1])
            noise_files_list.append(df)

    noise_files_index = pd.concat(noise_files_list)

    if exclude_speech:
        noise_files_index['is_speech'] = noise_files_index['label'].apply(lambda x: len(set(x.split(',')) & {'/m/02zsn', '/m/05zppz', '/m/09x0r', '/m/02qldy'}) == len(x.split(',')))
        noise_files_index = noise_files_index[noise_files_index['is_speech'] == False]

    noise_files_index['has_speech'] = noise_files_index['label'].apply(lambda x: len(set(x.split(',')) & {'/m/02zsn', '/m/05zppz', '/m/09x0r', '/m/02qldy'}) > 0).astype('int32')
    noise_files_index = noise_files_index[noise_files_index['has_speech'] == 0]

    if include_only:
        labels = config['labels']
        labels_list = []
        for _, string_id in labels.items():
            labels_list += string_id.split(',')

        noise_files_index['out-of-distribution'] = noise_files_index['label'].apply(lambda x: x in labels_list)
        noise_files_index = noise_files_index[noise_files_index['out-of-distribution'] == True]

    # split before balancing
    train, validate = np.split(noise_files_index.sample(frac=1, random_state=42), [int(.9 * len(noise_files_index))])

    if balance:
        n = config.getint('balance_parameters', 'n')
        controlled_categories = config['balance_parameters']['controlled_categories'].split(',')
        train = balance_label(train, 'label', n=n, controlled_categories=controlled_categories)
        validate = balance_label(validate, 'label', n=int(n/10), controlled_categories=controlled_categories)

    logger.info(f'Number of noise files in train to proceed: {len(train)}')
    logger.info(f'Number of noise files in train to proceed: {len(validate)}')

    local_folder = '../temp'
    os.makedirs(local_folder, exist_ok=True)

    data = {'train': train, 'validate': validate}

    data_processed = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:

        for dataname, data in data.items():
            futures = []

            for i, noise_audio in data.iterrows():
                futures.append(executor.submit(build_audio,
                                           noise_audio,
                                           clean_files_list,
                                           save_dir,
                                           dest_name=destination_name,
                                           snr_lower=snr_lower,
                                           snr_upper=snr_upper,
                                           target_lower=target_level_lower,
                                           target_upper=target_level_upper,
                                           seed=i,
                                           tmp=local_folder,
                                           noisy_pair=noisy_pair,
                                           clean_activity_threshold=clean_activity_threshold
                                           )
                           )

            labels = [f.result() for f in futures]

            labels_csv = pd.DataFrame(labels, columns=['tag_name', 'tag_code', 'noisy_file', 'snr', 'noise_url', 'has_speech'])

            labels_csv['num_tags'] = labels_csv['tag_name'].apply(lambda x: len(x))
            labels_csv = labels_csv[labels_csv['num_tags'] > 0]

            columns_list = ['noisy_file']
            if noisy_pair:
                columns_list = ['noisy_file1', 'noisy_file2']

            noisy_path_df = pd.DataFrame(labels_csv.noisy_file.tolist(), columns=columns_list, index=labels_csv.index)
            labels_csv.drop('noisy_file', axis=1, inplace=True)
            labels_csv = labels_csv.join(noisy_path_df)

            labels_csv['tag_name'] = labels_csv['tag_name'].apply(lambda lab_list: ','.join(lab for lab in lab_list))
            labels_csv['tag_code'] = labels_csv['tag_code'].apply(lambda lab_list: ','.join(lab for lab in lab_list))

            labels_csv.to_csv(f'{save_dir}/noisyspeech_{dataname}_files.csv')

            data_processed.append(labels_csv)

    labels_all = pd.concat(data_processed)
    labels_all.to_csv(f'{save_dir}/noisyspeech_all_files.csv')


if __name__ == '__main__':
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../noise_synthetizer.cfg')
    parser.add_argument('--data_dir')
    parser.add_argument('--dest_dir')
    parser.add_argument('--dest_name', default=None)
    parser.add_argument('--num_workers', type=int, default=min(32, os.cpu_count() + 4))
    args = parser.parse_args()

    s = time.time()
    create_noisy_audio(args.config, args.data_dir, args.dest_dir,
                       num_workers=args.num_workers,
                       dest_name=args.dest_name
                       )
    e = time.time()

    print(f'running time is {e - s}')
