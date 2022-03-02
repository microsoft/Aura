import os
import datetime

import tensorflow as tf
import pandas as pd
import numpy as np

from azureml.core.run import Run

from ..utils.tfrecords import parse_tfrecord, resize
from ..utils.log import get_logger
from ..models import *

# run - azure
run = Run.get_context()
logger = get_logger(__name__)


class FocalMultiLabelLoss(tf.keras.losses.Loss):
    """Focal loss with class level weights"""

    def __init__(self, *args, gamma=2.0, alpha=0.1, weights=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.gamma = gamma
        self.eps = 10**-8
        self.weights = weights
        self.alpha = alpha
        self.weights_0 = tf.clip_by_value(tf.convert_to_tensor(weights[0], dtype=tf.float32), 0.0001, 2)
        self.weights_1 = tf.clip_by_value(tf.convert_to_tensor(weights[1], dtype=tf.float32), 0.0001, 2)

    def call(self, y_true, prob):

        p = tf.math.maximum(prob, self.eps)
        q = tf.math.maximum(1 - prob, self.eps)

        y_true = tf.cast(y_true, dtype=tf.float32)

        pos_term = - (q ** self.gamma) * tf.math.log(p)
        if self.weights is not None:
            pos_term = pos_term * self.weights_1

        neg_term = - (p ** self.gamma) * tf.math.log(q)
        if self.weights is not None:
            neg_term = neg_term * self.weights_0

        labels = y_true * (1 - self.alpha) + self.alpha / y_true.shape[1]

        loss = labels * pos_term + (1 - labels) * neg_term

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return loss
        elif self.reduction == tf.keras.losses.Reduction.SUM:
            return tf.reduce_sum(loss)
        else:
            return tf.reduce_mean(loss)

class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    """
    A callback class for logging metrics using Azure Machine Learning Python SDK
    """

    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss_noise_type')
        mse = logs.get('mse')
        accuracy = logs.get('accuracy')
        auc = logs.get('auc')
        map = logs.get('map')

        val_map = logs.get('val_map')
        val_auc = logs.get('val_auc')
        val_accuracy = logs.get('val_accuracy')
        val_mse = logs.get('val_mse')

        if loss:
            run.log('loss', float(loss))

        if mse:
            run.log('mse', float(mse))

        if accuracy:
            run.log('accuracy', float(accuracy))

        if auc:
            run.log('AUC', float(auc))

        if map:
            run.log('mAP', float(map))

        if val_accuracy:
            run.log('val_accuracy', float(val_accuracy))

        if val_auc:
            run.log('val_AUC', float(val_auc))

        if val_map:
            run.log('val_mAP', float(val_map))

        if mse:
            run.log('val_mse', float(val_mse))

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        if batch_loss:
            run.log('batch_loss', float(batch_loss))

class MultiLossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    """
    A callback class for logging metrics using Azure Machine Learning Python SDK
    """

    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('output__1_loss')
        mse = logs.get('output_2_mse')

        auc = logs.get('output_1_auc')
        map = logs.get('output_1_map')

        val_map = logs.get('val_output_1_map')
        val_auc = logs.get('val_output_1_auc')
        val_mse = logs.get('val_output_2_mse')

        if loss:
            run.log('loss', float(loss))

        if mse:
            run.log('mse', float(mse))

        if auc:
            run.log('AUC', float(auc))

        if map:
            run.log('mAP', float(map))

        if val_auc:
            run.log('val_AUC', float(val_auc))

        if val_map:
            run.log('val_mAP', float(val_map))

        if mse:
            run.log('val_mse', float(val_mse))

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        if batch_loss:
            run.log('batch_loss', float(batch_loss))


def get_model(args, num_replicas):
    """
    Construct tensorflow model from checkpoint in args.path_model_tf
    """

    if args.balancing or args.output_bias:
        assert args.tags is not None
        tags_mapping = pd.read_csv(args.tags)
        tags_mapping['tag_numerical'] = np.arange(len(tags_mapping))
        tags_mapping.set_index('tag_numerical', inplace=True)

        weight_for_1 = 1 / (tags_mapping['count']) * 0.5
        weight_for_0 = (1 / (1 - tags_mapping['count'])) * 0.5

        class_weight = {0: weight_for_0, 1: weight_for_1}
        bias = np.log(tags_mapping['count'] / (1 - tags_mapping['count']))

    else:
        class_weight = {0: 1.0, 1:1.0}
        bias = None

    loss_noise_type = FocalMultiLabelLoss(gamma=args.gamma, alpha=args.alpha, weights=class_weight)
    loss_snr = tf.keras.losses.MeanSquaredError()

    output_bias = None
    if args.output_bias:
        output_bias = bias

    # load model tensorflow model
    model_pretrained_extractor_checkpoint = args.pretrained_extractor

    model = globals()[args.model_name](extract_init=model_pretrained_extractor_checkpoint, nclass=args.num_labels, output_bias=output_bias)

    if args.path_model_tf is not None:
        model.load_weights(tf.train.latest_checkpoint(args.path_model_tf)).expect_partial()

    # build model
    num_frames = args.num_frames
    num_mel = args.num_mel

    batch_size = args.batch_size * num_replicas
    inputs = tf.zeros((batch_size, num_frames, num_mel))
    model(inputs)

    model.build(input_shape=(batch_size, num_frames, num_mel))
    print('Building spectral model done')

    metrics = [[tf.keras.metrics.BinaryAccuracy(),
               tf.keras.metrics.AUC(curve='ROC', multi_label=True, name='auc'),
               tf.keras.metrics.AUC(curve='PR', multi_label=True, name='map'),
               ],
               [tf.keras.metrics.MeanSquaredError(name='mse')]
               ]

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer, loss=[loss_noise_type, loss_snr],
                  metrics=metrics, loss_weights=[1.0, 0.01])
    print('Compiling model done')

    return model


def get_data_from_tfrecords(args, num_replicas):
    """
    Create a tf.data from tf records in args.train_dir/args.validation_dir
    :param args:
    :param num_replicas:
    :return:
    """

    num_frames = args.num_frames
    num_mel = args.num_mel
    num_labels = args.num_labels

    batch_size = args.batch_size * num_replicas

    autotune = tf.data.AUTOTUNE

    train_filenames = tf.io.gfile.glob(f'{args.train_dir}/*.tfrec')
    train_filenames = np.random.choice(train_filenames, len(train_filenames), replace=False)
    train_dataset = tf.data.TFRecordDataset(train_filenames, num_parallel_reads=autotune) \
        .map(lambda example: parse_tfrecord(example, num_mel=num_mel, num_frames=num_frames, num_labels=num_labels, snr=True, labels=True),
             num_parallel_calls=autotune) \
        .map(lambda example: resize(example, num_frames=num_frames, num_mel=num_mel, snr=True, labels=True), num_parallel_calls=autotune) \
        .shuffle(10 * batch_size) \
        .batch(batch_size) \

    val_filenames = tf.io.gfile.glob(f'{args.validation_dir}/*.tfrec')
    val_dataset = tf.data.TFRecordDataset(val_filenames, num_parallel_reads=autotune) \
        .map(lambda example: parse_tfrecord(example, num_mel=num_mel, num_frames=num_frames, num_labels=num_labels, snr=True, labels=True),
             num_parallel_calls=autotune) \
        .map(lambda example: resize(example, num_frames=num_frames, num_mel=num_mel), num_parallel_calls=autotune, snr=True, labels=True) \
        .batch(batch_size) \
        .prefetch(autotune) \


    return train_dataset, val_dataset


def train(args):
    strategy = tf.distribute.MirroredStrategy()
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    if run._run_id.startswith("OfflineRun"):
        run.number = 0

    experiment_name = args.experiment_name
    tstamp = datetime.datetime.now().strftime("%Y%m%d%H%M")

    checkpoint_path = os.path.join(save_dir, "checkpoints", experiment_name, f'run_{tstamp}_{run.number}')
    checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        save_best_only=True,
        monitor='val_map',
        mode='max',
        filepath=checkpoint_prefix,
        save_weights_only=True)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_map',
                                                               min_delta=0,
                                                               patience=10,
                                                               verbose=0,
                                                               mode='max',
                                                               baseline=None,
                                                               restore_best_weights=False)

    with strategy.scope():
        model = get_model(args, strategy.num_replicas_in_sync)
        train_loader, validation_loader = get_data_from_tfrecords(args, strategy.num_replicas_in_sync)

    model.fit(train_loader,
              epochs=args.num_epochs,
              callbacks=[MultiLossAndErrorPrintingCallback(), checkpoint_callback, early_stopping_callback],
              validation_data=validation_loader
              )

