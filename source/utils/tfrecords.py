import os

import tensorflow as tf

from ..utils.log import get_logger

logger = get_logger(__name__)


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_example(feature, labels=None, snr=None):
    """

    :param feature: log-mel features
    :param labels: labels
    :return: tf.train.Examples
    """
    feature_label = {'feature': _float_feature(feature.reshape(-1).tolist()),
                     }
    if labels is not None:
        feature_label['labels'] = _float_feature(labels.tolist())
    if snr is not None:
        feature_label['snr'] = _float_feature([snr])

    return tf.train.Example(features=tf.train.Features(feature=feature_label))


def create_tfrecords(audio_files, save_path, datasetname, dataset, snr=False, labels=False):
    """
    Create a tfrecords with all the files in audio_files
    :param audio_files: list of fike url
    :param dataset: a SpectralFeature object
    :param save_path: string, where to save the tfrecord
    :param datasetname: string, name to add to tfrecord filename

    """
    os.makedirs(save_path, exist_ok=True)

    with tf.io.TFRecordWriter(os.path.join(save_path, f'{datasetname}.tfrec')) as writer:
        for i, idx in enumerate(audio_files):
            clip_url = dataset.audio_path.loc[idx, dataset.filename]
            print(f'Adding {clip_url} to tfrecords')

            snr_value = None
            labels_value = None

            try:
                if snr and labels:
                    features, labels_value, snr_value = dataset[idx]
                elif labels:
                    features, labels_value = dataset[idx]
                else:
                    features = dataset[idx]

                tf_example = create_example(features, labels=labels_value, snr=snr_value)
                writer.write(tf_example.SerializeToString())

            except:
                print(f'Issue with {clip_url}')
                continue

        writer.close()

        print(f'Process {len(audio_files)} records')


def parse_tfrecord(tf_example, num_mel=64, num_frames=496, num_labels=None, snr=False, labels=False, label_float=False):
    label_type = tf.float32 if label_float else tf.int64

    if labels:
        assert num_labels is not None

    if snr and labels:
        feature_description = {'feature': tf.io.FixedLenFeature([num_mel * num_frames], tf.float32),
                               'labels': tf.io.FixedLenFeature([num_labels], label_type),
                               'snr': tf.io.FixedLenFeature([1], tf.float32)
                               }
    elif labels:
        feature_description = {'feature': tf.io.FixedLenFeature([num_mel * num_frames], tf.float32),
                           'labels': tf.io.FixedLenFeature([num_labels], label_type)
                           }
    else:
        feature_description ={'feature': tf.io.FixedLenFeature([num_mel * num_frames], tf.float32),
         }

    parsed = tf.io.parse_single_example(tf_example, feature_description)
    return parsed


def resize(data, num_mel, num_frames, num_labels=None, labels=False, snr=False):
    if labels:
        assert num_labels is not None

    if labels and snr:
        return tf.reshape(data['feature'], (num_frames, num_mel)), data['labels'], data['snr']
    elif labels:
        return tf.reshape(data['feature'], (num_frames, num_mel)), data['labels']
    else:
        return tf.reshape(data['feature'], (num_frames, num_mel)), tf.zeros(num_labels, dtype=tf.float32)

