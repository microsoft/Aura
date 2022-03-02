import argparse
import time
from concurrent.futures import ProcessPoolExecutor

from source.utils.tfrecords import create_tfrecords
from source.utils.log import get_logger
from source.utils.dataset import *

logger = get_logger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--num_tfrecords', type=int, default=1000)
    parser.add_argument('--datasetname', default='noisyspeech_unbalanced_snr')
    parser.add_argument('--nworkers', type=int, default=min(32, os.cpu_count() + 4))
    parser.add_argument('--tags_file', default='../../../../data/synthetic/tags/noisyspeech_full_tags.csv')
    parser.add_argument('--filename', default='clip_url')
    parser.add_argument('--labels', action='store_true')
    parser.add_argument('--snr', action='store_true')
    parser.add_argument('--smoothing', action='store_true')
    parser.add_argument('--input_length', type=float, default=9.99)

    args = parser.parse_args()

    tag_files = args.tags_file
    if args.labels and args.snr:
        dataset_all = SpectralFeaturesWithLabelsAndSNR(args.data_dir,
                                                       tag_files,
                                                       random=True,
                                                       filename=args.filename,
                                                       frac_samples=1.0,
                                                       input_len_second=args.input_length)
    elif args.labels:
        dataset_all = SpectralFeaturesWithLabels(args.data_dir,
                                                 tag_files,
                                                 random=True,
                                                 filename=args.filename,
                                                 frac_samples=1.0,
                                                 input_len_second=args.input_length)
    else:
        dataset_all = SpectralFeatures(args.data_dir,
                                       random=True,
                                       filename=args.filename,
                                       frac_samples=1.0,
                                       input_len_second=args.input_length)

    n = len(dataset_all)

    dset_list = []
    train, validate = np.split(dataset_all.audio_path.sample(frac=1, random_state=42),
                               [int(.9 * len(dataset_all.audio_path))])

    logger.info(f'Number of records to proceed: {len(dataset_all)}')
    num_tfrecords = args.num_tfrecords

    train_list = list(train[dataset_all.filename].index)
    train_list = [train_list[i:i + num_tfrecords] for i in range(0, len(train_list), num_tfrecords)]

    validate_list = list(validate[dataset_all.filename].index)
    validate_list = [validate_list[i:i + num_tfrecords] for i in range(0, len(validate_list), num_tfrecords)]

    if args.smoothing:
        smooth = train.sample(frac=0.3)
        smooth_dset = SpectralFeaturesWithLabelsSmoothing(args.data_dir,
                                                          tag_files,
                                                          random=True,
                                                          filename=args.filename,
                                                          frac_samples=1.0,
                                                          input_len_second=9.99)

        smooth_dset.audio_path = smooth_dset.audio_path[
            smooth_dset.audio_path[smooth_dset.filename].isin(smooth[smooth_dset.filename])]
        smooth = smooth_dset.audio_path
        smooth_list = list(smooth[smooth_dset.filename].index)
        smooth_list = [smooth_list[i:i + num_tfrecords] for i in range(0, len(smooth_list), num_tfrecords)]

    s = time.time()

    nworkers = args.nworkers

    logger.info(f'Number of tfrecords to write: {len(train_list)}')
    logger.info(f'Number of tfrecords to write: {len(validate_list)}')

    if args.smoothing:
        logger.info(f'Number of tfrecords to write: {len(smooth_list)}')

    with ProcessPoolExecutor(max_workers=args.nworkers) as executor:

        futures = []
        for i, data in enumerate(train_list):
            futures.append(
                executor.submit(create_tfrecords, data, f'{args.save_dir}/train', f'{args.datasetname}_{i}',
                                dataset_all, snr=args.snr, labels=args.labels))

        if args.smoothing:
            for i, data in enumerate(smooth_list):
                futures.append(
                    executor.submit(create_tfrecords, data, f'{args.save_dir}/train', f'{args.datasetname}_smooth_{i}',
                                    smooth_dset, snr=args.snr, labels=args.labels))

        for i, data in enumerate(validate_list):
            futures.append(
                executor.submit(create_tfrecords, data, f'{args.save_dir}/validate', f'{args.datasetname}_{i}',
                                dataset_all, snr=args.snr, labels=args.labels))

        results = [f.result() for f in futures]

    e = time.time()
    logger.info(f'Time to proceed {n} files: {e - s}')
