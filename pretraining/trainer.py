import argparse
import os
from pathlib import Path

import sklearn.model_selection
import tensorflow as tf

from pretraining import datasets
from pretraining.utils import task_solver
from transplant.datasets import icentia11k
from transplant.evaluation import CustomCheckpoint, f1
from transplant.modules.utils import build_input_tensor_from_shape
from transplant.utils import (matches_spec, load_pkl, save_pkl)

from clr.learningratefinder import LearningRateFinder
from clr.clr_callback import CyclicLR
from clr import config
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")


def _create_dataset_from_generator(patient_ids, samples_per_patient=None):
    print('[INFO] Create dataset from generator')

    samples_per_patient = samples_per_patient or args.samples_per_patient
    if args.task == 'rhythm':
        dataset = datasets.rhythm_dataset(
            db_dir=str(args.train), patient_ids=patient_ids, frame_size=args.frame_size,
            unzipped=args.unzipped, samples_per_patient=samples_per_patient)
    elif args.task == 'beat':
        dataset = datasets.beat_dataset(
            db_dir=str(args.train), patient_ids=patient_ids, frame_size=args.frame_size,
            unzipped=args.unzipped, samples_per_patient=samples_per_patient)
    elif args.task == 'hr':
        dataset = datasets.heart_rate_dataset(
            db_dir=str(args.train), patient_ids=patient_ids, frame_size=args.frame_size,
            unzipped=args.unzipped, samples_per_patient=samples_per_patient)
    else:
        raise ValueError('unknown task: {}'.format(args.task))
    return dataset


def _create_dataset_from_data(data):
    print('[INFO] Create dataset from data')

    x, y = data['x'], data['y']
    if args.task in ['rhythm', 'beat', 'hr']:
        spec = (tf.TensorSpec((None, args.frame_size, 1), tf.float32),
                tf.TensorSpec((None,), tf.int32))
    elif args.task == 'cpc':
        spec = ({'context': tf.TensorSpec((None, args.context_size, args.frame_size, 1), tf.float32),
                 'samples': tf.TensorSpec((None, args.ns + 1, args.frame_size, 1), tf.float32)},
                tf.TensorSpec((None,), tf.int32))
    else:
        raise ValueError('unknown task: {}'.format(args.task))
    if not matches_spec((x, y), spec, ignore_batch_dim=True):
        raise ValueError('data does not match the required spec: {}'.format(spec))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', type=Path, required=True, help='Job output directory.')
    parser.add_argument('--task', required=True, help='Training task: `rhythm`, `beat`, `hr`.')
    parser.add_argument('--train', type=Path, required=True, help='Path to the train directory or a pickled file.')
    parser.add_argument('--val-file', type=Path, help='Path to the pickled validation file.\nOverrides --val-size.')
    parser.add_argument('--cache-val', type=Path, help='Path to where the newly created validation set will be cached.')
    parser.add_argument('--weights-file', type=Path, help='Path to a checkpoint to load the weights from.')
    parser.add_argument('--unzipped', action='store_true', help='Whether files in the train directory are unzipped.')
    parser.add_argument('--val-patients', type=float, default=None,
                        help='Number of patients or proportion of patients '
                             'that will be moved from train to validation.')
    parser.add_argument('--val-size', type=int, default=None,
                        help='Size of the validation set when collecting data from train directory.')
    parser.add_argument('--arch', default='resnet18', help='Architecture of the ECG feature extractor: '
                                                           '`resnet18`, `resnet34` or `resnet50`.')
    parser.add_argument('--stages', type=int, default=None, help='Stages of the densenet network '
                                                                 'that will be pretrained.')
    parser.add_argument('--frame-size', type=int, default=2048, help='Frame size.')
    # parser.add_argument('--context-size', type=int, default=8, help='Context size measured in frames.')
    # parser.add_argument('--ns', type=int, default=1, help='Number of negative samples for the CPC.')
    # parser.add_argument('--positive-offset', type=int, default=0,
    #                     help='Offset of the positive sample from the context.')
    # parser.add_argument('--context-overlap', type=int, default=0, help='CPC Context overlap.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('--samples-per-patient', type=int, default=1000,
                        help='Number of data points that are sampled from a patient file once it is read.')
    parser.add_argument('--val-samples-per-patient', type=int, default=None,
                        help='Number of data points that are sampled from a validation patient file once it is read.\n'
                             'By default equal to --samples-per-patient.')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='Number of steps per epoch.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs.')
    parser.add_argument('--val-metric', default='loss', help='Performance metric: either `loss`, `acc` or `f1`.')
    parser.add_argument('--data-parallelism', type=int, default=1, help='Number of data loaders running in parallel.')
    parser.add_argument('--seed', type=int, default=None, help='Random state.')
    parser.add_argument("-f", "--lr-find", type=int, default=0, help="whether or not to find optimal learning rate")

    args, _ = parser.parse_known_args()

    if args.val_metric not in ['loss', 'acc', 'f1']:
        raise ValueError('Unknown metric: {}'.format(args.val_metric))

    os.makedirs(str(args.job_dir), exist_ok=True)
    print('[INFO] Creating working directory in {}'.format(args.job_dir))

    seed = args.seed or np.random.randint(2 ** 16)
    print('[INFO] Setting random state {}'.format(seed))
    np.random.seed(seed)

    # Number of data points that are sampled from a validation patient file once it is read
    # By default equal to 'samples_per_patient'.
    if args.val_samples_per_patient is None:
        args.val_samples_per_patient = args.samples_per_patient

    if not args.val_file and args.val_patients:     # Using val_patients, not using val_file
        # Number of patients or proportion of patients
        # that will be moved from train to validation.
        if args.val_patients >= 1:  # using number of patients
            args.val_patients = int(args.val_patients)

    if args.val_file:   # Using val_file
        print('[INFO] Loading validation data from file {} ...'.format(args.val_file))
        val = load_pkl(str(args.val_file))  # read data from val_file
        validation_data = _create_dataset_from_data(val)    # create validation dataset
    else:
        val = None
        validation_data = None

    if args.train.is_file():    # used to check 'train' if the entry is a file or not.
        print('[INFO] Loading train data from file {} ...'.format(args.train))
        train = load_pkl(str(args.train))   # read 'train' data file
        if val:     # if validation dataset is also data file
            # remove training examples of patients who belong to the validation set
            train_mask = np.isin(element=train['patient_ids'], test_elements=val['patient_ids'], invert=True)
            train = {key: array[train_mask] for key, array in train.items()}    # create dictionaries
        elif args.val_patients:     # if validation dataset is number of patients
            # if args.task == 'cpc':
            #     print('--val-patients is ignored when train is a pickled file because the negative samples '
            #           'in the validation set cannot be guaranteed to come from only the validation patients.')
            # else:
            print('[INFO] Splitting data into train and validation')
            _, val_patients_ids = sklearn.model_selection.train_test_split(np.unique(train['patient_ids']),
                                                                           test_size=args.val_patients)
            # remove training examples of patients who belong to the validation set
            val_mask = np.isin(train['patient_ids'], val_patients_ids)
            val = {key: array[val_mask] for key, array in train.items()}    # create dictionaries
            validation_data = _create_dataset_from_data(val)    # create validation dataset
            train_mask = ~val_mask
            train = {key: array[train_mask] for key, array in train.items()}
        train_size = len(train['y'])
        steps_per_epoch = None
        train_data = _create_dataset_from_data(train).shuffle(train_size)   # create train dataset
    else:   # not file
        print('[INFO] Building train data generators')
        train_patient_ids = icentia11k.ds_patient_ids   # return patient's id
        if val:     # if validation set is a file
            # remove patients who belong to the validation set from train data
            train_patient_ids = np.setdiff1d(ar1=train_patient_ids, ar2=val['patient_ids'])
        elif args.val_patients:     # using number of patient
            print('[INFO] Splitting patients into train and validation')
            train_patient_ids, val_patient_ids = sklearn.model_selection.train_test_split(train_patient_ids,
                                                                                          test_size=args.val_patients)
            # validation size is one validation epoch by default
            val_size = args.val_size or (len(val_patient_ids) * args.val_samples_per_patient)
            print('[INFO] Collecting {} validation samples ...'.format(val_size))
            validation_data = _create_dataset_from_generator(val_patient_ids, args.val_samples_per_patient)
            val_x, val_y = next(validation_data.batch(val_size).as_numpy_iterator())
            val = {'x': val_x, 'y': val_y, 'patient_ids': val_patient_ids}
            if args.cache_val:
                print('[INFO] Caching the validation set in {} ...'.format(args.cache_val))
                save_pkl(str(args.cache_val), x=val_x, y=val_y, patient_ids=val_patient_ids)
            validation_data = _create_dataset_from_data(val)
        steps_per_epoch = args.steps_per_epoch
        if args.data_parallelism > 1:
            split = len(train_patient_ids) // args.data_parallelism
            train_patient_ids = tf.convert_to_tensor(train_patient_ids)
            train_data = tf.data.Dataset.range(args.data_parallelism).interleave(
                lambda i: _create_dataset_from_generator(train_patient_ids[i * split:(i + 1) * split],
                                                         args.samples_per_patient),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            train_data = _create_dataset_from_generator(train_patient_ids, args.samples_per_patient)
        buffer_size = 16 * args.samples_per_patient  # data from 16 patients
        train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE).shuffle(buffer_size)

    train_data = train_data.batch(args.batch_size)  # train dataset

    if val:
        validation_data = validation_data.batch(args.batch_size)    # validation dataset

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        print('[INFO] Building model ...')

        # Adding local features learning part
        model = task_solver(task=args.task, arch=args.arch, stages=args.stages)

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=config.MIN_LR),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')])

        # initialize the weights of the model
        # Returns the output shapes for elements of the input dataset / iterator.
        input_shape, _ = tf.compat.v1.data.get_output_shapes(train_data)

        print('[INFO] Output shape of train data ...{}'.format(input_shape))

        # Returns the output shapes for elements of the input dataset / iterator.
        input_dtype, _ = tf.compat.v1.data.get_output_types(train_data)

        inputs = build_input_tensor_from_shape(shape=input_shape, dtype=input_dtype, ignore_batch_dim=True)
        model(inputs)

        print('[INFO] Model parameters: {:,d}'.format(model.count_params()))

        if args.weights_file:
            print('[INFO] Loading weights from file {} ...'.format(args.weights_file))
            model.load_weights(filepath=str(args.weights_file))

        if args.val_metric in ['loss', 'acc']:
            monitor = ('val_' + args.val_metric) if val else args.val_metric
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=str(args.job_dir / 'epoch_{epoch:02d}' / 'model.weights'),
                monitor=monitor,
                save_best_only=False,
                save_weights_only=True,
                mode='auto',
                verbose=1)
        elif args.val_metric == 'f1':
            if val:
                checkpoint = CustomCheckpoint(
                    filepath=str(args.job_dir / 'epoch_{epoch:02d}' / 'model.weights'),
                    data=(validation_data, val['y']),
                    score_fn=f1,
                    save_best_only=False,
                    verbose=1)
            else:
                raise ValueError('f1 metric may only be used in combination with the validation set.')
        else:
            raise ValueError('Unknown metric: {}'.format(args.val_metric))

        logger = tf.keras.callbacks.CSVLogger(filename=str(args.job_dir / 'history.csv'))

        # check to see if we are attempting to find an optimal learning rate
        # before training for the full number of epochs
        if args.lr_find > 0:
            # initialize the learning rate finder and then train with learning
            # rates ranging from 1e-10 to 1e+1
            print("[INFO] Finding learning rate...")
            lrf = LearningRateFinder(model)
            lrf.find(trainData=train_data, startLR=1e-10, endLR=1e+1, stepsPerEpoch=steps_per_epoch, epochs=3)

            # plot the loss for the various learning rates and save the
            # resulting plot to disk
            lrf.plot_loss()
            plt.savefig(config.LRFIND_PLOT_PATH)

            # gracefully exit the script so we can adjust our learning rates
            # in the config and then train the network for our full set of
            # epochs
            print("[INFO] learning rate finder complete")
            print("[INFO] examine plot and adjust learning rates before training")
            sys.exit(0)

        #
        #
        # model.fit(x=train_data,
        #           steps_per_epoch=steps_per_epoch,
        #           epochs=args.epochs,
        #           validation_data=validation_data,
        #           callbacks=[checkpoint, logger],
        #           verbose=2)
