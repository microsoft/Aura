import argparse
import os

import pandas as pd
import numpy as np
import yaml
import time
import datetime

from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

from source.utils.dataset import ReadDataFromCSV, ReadDataFromDataFrame
from source.utils.log import get_logger

from source.clean_speech_filter import *
from source.scorers import *
from source.clusters import *

logger = get_logger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/aml_config.yml')
    parser.add_argument('--clean_speech_classifier', help='model path to onnx file')
    parser.add_argument('--sig_dnsmos_path', help='sig dnmos model path to onnx file')
    parser.add_argument('--bak_ovr_dnsmos_path', help='bak/ovr dnmos model path to onnx file')
    parser.add_argument('--noise_type_classifier_path', default=None, help='Path to onnx file for noise type classifier '
                                                                           'whose embedding layer is used for ood '
                                                                           'detection and clustering')
    parser.add_argument('--ood_centroids', default=None, help='path to centroids file that contains the cluster centroid'
                                                             'of DNS train set')
    parser.add_argument('--centroids', default=None,
                        help='path to centroids file that contains the cluster centroid'
                             'of learned from a balanced data of noise types')
    parser.add_argument('--save_dir')

    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    data_csv = config['data_csv']
    # Change frac_samples when dealing with large data
    dataset = ReadDataFromCSV(data_csv, frac_samples=None, column_url=config['column_name'])

    tstamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    experience_name = config['experience_name']
    outfolder = args.save_dir
    outfolder = f'{outfolder}/{tstamp}_{experience_name}'
    os.makedirs(outfolder, exist_ok=True)

    with open(f'{outfolder}/configuration.yml', 'w') as file:
        yaml.dump(config, file)

    s = time.time()

    logger.info(f'Total number of audio files is: {len(dataset)}')

    # add dnmos models
    config['dnsmos_models']['dnsmos_path'] = {}
    config['dnsmos_models']['dnsmos_path']['sig_model_path'] = args.sig_dnsmos_path
    config['dnsmos_models']['dnsmos_path']['bak_ovr_model_path'] = args.bak_ovr_dnsmos_path

    # classify audio as clean or noisy speech
    clean_speech_filter_name = config['clean_speech_filter'].pop('name')
    config['clean_speech_filter']['model_path'] = args.clean_speech_classifier

    if clean_speech_filter_name == 'DNSMOSFilter':
        config['clean_speech_filter'].update(config['dnsmos_models'])

    clean_speech_filter = globals()[clean_speech_filter_name].from_dict(config['clean_speech_filter'])

    with ProcessPoolExecutor(max_workers=24) as executor:
        futures = []
        for data in dataset():
            futures.append(executor.submit(clean_speech_filter.classify_as_clean, data))

        results = [f.result() for f in futures]

    if clean_speech_filter_name == 'DNSMOSFilter':
        column_urls = ['clean_speech', 'MOS_sig', 'MOS_bak', 'MOS_ovr', 'clip_url']
    else:
        column_urls = ['clean_speech', 'clip_url']

    data_clean_speech = pd.DataFrame(results, columns=column_urls)
    data_clean_speech.to_csv(f'{outfolder}/clean_noisy_{clean_speech_filter_name}_{tstamp}.csv')

    dataset_noisyspeech = ReadDataFromDataFrame(data_clean_speech, column_url=column_urls)
    dataset_noisyspeech.filter({'clean_speech': False})

    logger.info(f'Total number of noisyspeech is: {len(dataset_noisyspeech)}')

    # detect challenging conditions among noisyspeech
    scorer_dict = config['challenging_detector']
    scorer_name = scorer_dict.pop('name')
    scorer_dict.update(config['dnsmos_models'])

    scorer = globals()[scorer_name].from_dict(scorer_dict)

    with Pool() as p:
        denoised_mos = p.map(scorer.score, dataset_noisyspeech())

        score_data = pd.DataFrame(denoised_mos,
                                  columns=['dns_mos_sig', 'dns_mos_bak', 'dns_mos_ovr',
                                           'score', 'mos_sig', 'mos_bak', 'mos_ovr', 'clip_url'])

        score_data = score_data[score_data.dns_mos_sig > 0]

    # add out-of-distribution detection
    if 'ood_scorer' in config:
        ood_scorer_dict = config['ood_scorer']
        ood_scorer_dict['model_path'] = args.noise_type_classifier_path
        ood_scorer_dict['centroids'] = args.ood_centroids
        name = ood_scorer_dict.pop('name')

        ood_scorer = globals()[name].from_dict(ood_scorer_dict)
        with Pool() as p:
            ood_scores = p.map(ood_scorer.score, dataset_noisyspeech())

        ood_scores_data = pd.DataFrame(ood_scores, columns=['ood_score', 'clip_url'])

    score_data['score'] = np.exp(-score_data['score'])
    if 'ood_scorer' in config:
        score_data = pd.merge(score_data, ood_scores_data, on='clip_url', how='inner')
        score_data['score'] = score_data['ood_score'] * score_data['score']

    if 'clustering' in config:
        cluster_dict = config['clustering']
        name = cluster_dict.pop('name')
        cluster_dict['model_path'] = args.noise_type_classifier_path
        cluster_dict['centroids'] = args.centroids

        cluster_model = globals()[name].from_dict(cluster_dict)
        with Pool() as p:
            cluster_pool = p.map(cluster_model.assign, dataset_noisyspeech())

        cluster_data = pd.DataFrame(cluster_pool, columns=['cluster_id', 'labels', 'clip_url'])
        score_data = pd.merge(score_data, cluster_data, on='clip_url', how='inner')

    score_data.to_csv(f'{outfolder}/full_{scorer_name}_{tstamp}.csv')

    # sampling challenging conditions
    if config['sampling_method'] == 'rank':
        score_data.sort_values(by='score', ascending=False, inplace=True)
        score_data_sample = score_data.head(config['n'])

    elif config['sampling_method'] == 'diversity' and 'clustering' in config:
        remain_to_allocate = config['n']
        rank_allocated = 0
        score_data['rank'] = score_data.groupby('cluster_id')['score'].rank()

        df_list = []
        while remain_to_allocate > 0:
            df = score_data[score_data['rank'] == rank_allocated]
            remain_to_allocate -= len(df)
            rank_allocated += 1
            df_list.append(df)

        score_data_sample = pd.concat(df_list).sort_values(by='score').head(config['n'])


        score_data_sample.to_csv(f'{outfolder}/sample_{scorer_name}_{tstamp}.csv')

    e = time.time()
    logger.info(f'Total processing time for {len(dataset)} files: {e - s}')

