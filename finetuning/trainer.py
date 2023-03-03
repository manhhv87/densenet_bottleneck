import argparse
import os
from pathlib import Path

import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold

from finetuning.utils import (ecg_feature_extractor, train_test_split)
from transplant.evaluation import (auc, f1, multi_f1, CustomCheckpoint, f_max,
                                   f_beta_metric, g_beta_metric, f1_2018)
from transplant.utils import (create_predictions_frame, load_pkl, is_multiclass)

from clr.clr_callback import CyclicLR
from clr import config

import gc
from warnings import warn


def _create_dataset_from_data(data):
    """
    input:  data = {'x': x,
                    'y': labels.to_numpy(),
                    'record_ids': labels.index.to_numpy(),
                    'classes': labels.columns.to_numpy()}
    return: data and label
    """
    return tf.data.Dataset.from_tensor_slices((data['x'], data['y']))


def _create_model(n_classes, act, dat_x, weights_file):
    # include fc layer
    old_model = ecg_feature_extractor()
    old_model.add(tf.keras.layers.Dense(units=n_classes, activation=act))

    # initialize the weights of the model
    inputs_func = tf.keras.layers.Input(shape=dat_x.shape[1:], dtype=dat_x.dtype)
    old_model(inputs_func)

    print('[INFO] Model parameters: {:,d}'.format(old_model.count_params()))

    if weights_file:  # pre-trained weigh
        print('[INFO] Loading weights from file {} ...'.format(weights_file))
        old_model.load_weights(str(weights_file))

    old_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.MIN_LR, beta_1=0.9,
                                                         beta_2=0.98, epsilon=1e-9),
                      loss=loss,
                      metrics=[accuracy])

    return old_model, old_model.get_weights()


# we initialize it multiple times
def _init_weight(same_old_model, first_weights):
    same_old_model.set_weights(first_weights)


def parse_args():
    """Parse all the arguments provided from the CLI.
       Returns:
           A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--job-dir', type=Path, required=True,
                        help='Job output directory.')
    parser.add_argument('--train', type=Path, required=True,
                        help='Path to the train file.')
    parser.add_argument('--val', type=Path,
                        help='Path to the validation file. Overrides --val-size.')
    parser.add_argument('--val-size', type=float, default=None,
                        help='Size of the validation set or proportion of the train set.')
    parser.add_argument('--subset', type=float, default=None,
                        help='Size of a subset of the train set or proportion of the train set.')
    parser.add_argument('--weights-file', type=Path,
                        help='Path to pretrained weights or a checkpoint of the model.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--val-metric', default='loss',
                        help='Validation metric used to find the best model at each epoch. Supported metrics are:'
                             '`loss`, `acc`, `f1`, `auc`, `fmax`, `fmetric`, `gmetric`, `f2018`.')
    parser.add_argument('--channel', type=int, default=None,
                        help='Use only the selected channel. By default use all available channels.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random state.')
    parser.add_argument('--verbose', action='store_true',
                        help='Show debug messages.')
    parser.add_argument('--k-fold', type=int, default=None,
                        help='k-fold cross validation')

    return parser.parse_known_args()


if __name__ == '__main__':
    args, unk = parse_args()

    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    if args.val_metric not in ['loss', 'acc', 'f1', 'auc', 'fmax', 'fmetric', 'gmetric', 'f2018']:
        raise ValueError('Unknown metric: {}'.format(args.val_metric))

    os.makedirs(name=str(args.job_dir), exist_ok=True)
    print('[INFO] Creating working directory in {}'.format(args.job_dir))

    seed = args.seed or np.random.randint(2 ** 16)
    print('[INFO] Setting random state {}'.format(seed))
    np.random.seed(seed)

    if not args.k_fold:  # Không sử dụng k-fold
        # Không sử dụng val file riêng biệt mà chia val từ train set
        if not args.val and args.val_size:
            if args.val_size >= 1:  # Lấy theo số lượng của patients, nếu <= 1 thì đó là theo tỷ lệ của train set
                args.val_size = int(args.val_size)

        # Tiếp tục chia train set sau khi đã chia thành train và val set
        # Lớn hơn hoặc bằng 1 thì sẽ là lấy theo số lượng patient, nếu <= 1 thì đó là theo tỷ lệ của train set
        if args.subset and args.subset >= 1:
            args.subset = int(args.subset)

        print('[INFO] Loading train data from {} ...'.format(args.train))
        train = load_pkl(file=str(args.train))

        if args.val:  # Loading val set từ val file
            print('[INFO] Loading validation data from {} ...'.format(args.val))
            val = load_pkl(file=str(args.val))
        elif args.val_size:  # Chia tỷ lệ val set từ train set mà không phải là load từ file
            original_train_size = len(train['x'])  # Kích thước của toàn bộ dữ liệu dataset
            train, val = train_test_split(train, test_size=args.val_size,
                                          stratify=train['y'])  # Chia thành train và val set
            new_train_size = len(train['x'])  # Trả về kích thước của train set mới
            new_val_size = len(val['x'])  # trả về kích thước của val set mới
            print('[INFO] Split data into train {:.2%} and validation {:.2%}'.format(
                new_train_size / original_train_size, new_val_size / original_train_size))
        else:  # Không sử dụng val set
            val = None

        if args.test:  # Sử dụng test set file riêng biệt
            print('[INFO] Loading test data from {} ...'.format(args.test))
            test = load_pkl(str(args.test))
        else:  # Không sử dụng test set
            test = None

        if args.subset:  # Tiếp tục chia train set sau khi đã chia train dataset ban đầu thành new train set và val set
            original_train_size = len(train['x'])  # Trả về kích thước của train set
            train, _ = train_test_split(train, train_size=args.subset, stratify=train['y'])  # Trả về new train set
            new_train_size = len(train['x'])  # Trả về kích thước của new train set
            print('[INFO] Using only {:.2%} of train data'.format(new_train_size / original_train_size))

        if args.channel is not None:
            train['x'] = train['x'][:, :, args.channel:args.channel + 1]
            if val:
                val['x'] = val['x'][:, :, args.channel:args.channel + 1]
            if test:
                test['x'] = test['x'][:, :, args.channel:args.channel + 1]

        print('[INFO] Train data shape:', train['x'].shape)

        # Disable AutoShard.
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_data = _create_dataset_from_data(train).with_options(options).shuffle(len(train['x'])).batch(
            args.batch_size)
        val_data = _create_dataset_from_data(val).with_options(options).batch(args.batch_size) if val else None
        test_data = _create_dataset_from_data(test).with_options(options).batch(args.batch_size) if test else None

        if train:
            train_size = len(train['x'])
            print('[INFO] Train size {} ...'.format(train_size))
        if val:
            val_size = len(val['x'])
            print('[INFO] Validation size {} ...'.format(val_size))
        if test:
            test_size = len(test['x'])
            print('[INFO] Test size {} ...'.format(test_size))

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            print('[INFO] Building model ...')
            num_classes = len(train['classes'])

            if is_multiclass(train['y']):
                activation = 'sigmoid'
                loss = tf.keras.losses.BinaryCrossentropy()
                accuracy = tf.keras.metrics.BinaryAccuracy(name='acc')
            else:
                activation = 'softmax'
                loss = tf.keras.losses.CategoricalCrossentropy()
                accuracy = tf.keras.metrics.CategoricalAccuracy(name='acc')

            # Creating model
            inputs = tf.keras.layers.Input(shape=train['x'].shape[1:], dtype=train['x'].dtype)
            backbone_model = ecg_feature_extractor(input_layer=inputs)
            # x = tf.keras.layers.GlobalMaxPooling1D()(backbone_model.output)
            # x = tf.keras.layers.GlobalAveragePooling1D()(backbone_model.output)
            # x = tf.keras.layers.Flatten()(backbone_model.output)

            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True))(
                backbone_model.output)
            # x = tf.keras.layers.LSTM(units=64, return_sequences=True)(backbone_model.output)
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
            # x = tf.keras.layers.GlobalAveragePooling1D()(x)

            x = tf.keras.layers.Dense(units=num_classes, activation=activation)(x)
            model = tf.keras.models.Model(inputs=backbone_model.input, outputs=x)
            model.summary()

            print('[INFO] Model parameters: {:,d}'.format(model.count_params()))

            if args.weights_file:  # Sử dụng trọng số đã được pre-trained
                # initialize weights (excluding the optimizer state) to load the pretrained resnet
                # the optimizer state is randomly initialized in the `model.compile` function
                print('[INFO] Loading weights from file {} ...'.format(args.weights_file))
                model.load_weights(str(args.weights_file))

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.MIN_LR),
                          loss=loss, metrics=[accuracy])

            callbacks = [tf.keras.callbacks.CSVLogger(filename=str(args.job_dir / 'history.csv'))]

            if args.val_metric in ['loss', 'acc']:
                monitor = ('val_' + args.val_metric) if val else args.val_metric
                checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=str(args.job_dir / 'best_model.weights'),
                                                                monitor=monitor,
                                                                save_best_only=True,
                                                                save_weights_only=True,
                                                                mode='auto',
                                                                verbose=1)
            elif args.val_metric == 'f1':
                if is_multiclass(train['y']):
                    score_fn = multi_f1
                else:
                    score_fn = f1

                checkpoint = CustomCheckpoint(filepath=str(args.job_dir / 'best_model.weights'),
                                              data=(val_data, val['y']) if val else (train_data, train['y']),
                                              score_fn=score_fn,
                                              save_best_only=True,
                                              verbose=1)

            elif args.val_metric == 'auc':
                checkpoint = CustomCheckpoint(filepath=str(args.job_dir / 'best_model.weights'),
                                              data=(val_data, val['y']) if val else (train_data, train['y']),
                                              score_fn=auc,
                                              save_best_only=True,
                                              verbose=1)

            elif args.val_metric == 'fmax':
                checkpoint = CustomCheckpoint(filepath=str(args.job_dir / 'best_model.weights'),
                                              data=(val_data, val['y']),  # if val else (train_data, train['y']),
                                              score_fn=f_max,
                                              save_best_only=True,
                                              verbose=1)

            elif args.val_metric == 'fmetric':
                checkpoint = CustomCheckpoint(filepath=str(args.job_dir / 'best_model.weights'),
                                              data=(val_data, val['y']),  # if val else (train_data, train['y']),
                                              score_fn=f_beta_metric,
                                              save_best_only=True,
                                              verbose=1)

            elif args.val_metric == 'gmetric':
                checkpoint = CustomCheckpoint(filepath=str(args.job_dir / 'best_model.weights'),
                                              data=(val_data, val['y']),  # if val else (train_data, train['y']),
                                              score_fn=g_beta_metric,
                                              save_best_only=True,
                                              verbose=1)

            elif args.val_metric == 'f2018':
                checkpoint = CustomCheckpoint(filepath=str(args.job_dir / 'best_model.weights'),
                                              data=(val_data, val['y']),  # if val else (train_data, train['y']),
                                              score_fn=f1_2018,
                                              save_best_only=True,
                                              verbose=1)

            else:
                raise ValueError('Unknown metric: {}'.format(args.val_metric))

            callbacks.append(checkpoint)

            if val:
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,
                                                                  min_delta=0.001, verbose=1)
                callbacks.append(early_stopping)

            # otherwise, we have already defined a learning rate space to train over,
            # so compute the step size and initialize the cyclic learning rate method
            clr = CyclicLR(mode=config.CLR_METHOD,
                           base_lr=config.MIN_LR,
                           max_lr=config.MAX_LR,
                           step_size=config.STEP_SIZE * train_size // args.batch_size)

            callbacks.append(clr)

            model.fit(train_data,
                      validation_data=val_data,
                      epochs=args.epochs,
                      callbacks=callbacks,
                      verbose=1)

            # load best model for inference
            print('[INFO] Loading the best weights from file {} ...'.format(str(args.job_dir / 'best_model.weights')))
            model.load_weights(filepath=str(args.job_dir / 'best_model.weights'))

            # Save the entire model as a SavedModel.
            print('[INFO] Saving model ...')
            model.save(str(args.job_dir / 'best_model'))

            if val:
                print('[INFO] Predicting validation data ...')
                val_y_prob = model.predict(x=val['x'], batch_size=args.batch_size)
                val_predictions = create_predictions_frame(y_prob=val_y_prob, y_true=val['y'],
                                                           class_names=train['classes'],
                                                           record_ids=val['record_ids'])
                val_predictions.to_csv(path_or_buf=args.job_dir / 'val_predictions.csv', index=False)


    else:  # Đánh giá theo k-fold cross validation
        print('[INFO] Loading train data from {} ...'.format(args.train))
        data_set = load_pkl(file=str(args.train))
        x, y, record_ids, classes = data_set['x'], data_set['y'], data_set['record_ids'], data_set['classes']

        if args.channel is not None:
            data_set['x'] = data_set['x'][:, :, args.channel:args.channel + 1]

        print('[INFO] Train data shape:', data_set['x'].shape)
        print('[INFO] Building model ...')
        num_classes = len(data_set['classes'])

        if is_multiclass(data_set['y']):
            activation = 'sigmoid'
            loss = tf.keras.losses.BinaryCrossentropy()
            accuracy = tf.keras.metrics.BinaryAccuracy(name='acc')
        else:
            activation = 'softmax'
            loss = tf.keras.losses.CategoricalCrossentropy()
            accuracy = tf.keras.metrics.CategoricalAccuracy(name='acc')

        model_old, weights = _create_model(num_classes, activation, data_set['x'], args.weights_file)

        kf = KFold(n_splits=args.k_fold, shuffle=True)
        foldNum = 0
        all_scores_mse = []
        all_scores_f1_each_class = []
        all_scores_macro_f1 = []
        all_scores_macro_auc = []

        idx_data = np.arange(len(x))

        for train_idx, val_idx in kf.split(idx_data):
            foldNum += 1

            train = {'x': x[train_idx],
                     'y': y[train_idx],
                     'record_ids': record_ids[train_idx],
                     'classes': classes}
            val = {'x': x[val_idx],
                   'y': y[val_idx],
                   'record_ids': record_ids[val_idx],
                   'classes': classes}

            # Disable AutoShard.
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            train_data = _create_dataset_from_data(train).with_options(options).shuffle(len(train['x'])).batch(
                args.batch_size)
            val_data = _create_dataset_from_data(val).with_options(options).batch(args.batch_size)

            train_size = len(train['x'])
            print('[INFO] Train size {} ...'.format(train_size))
            val_size = len(val['x'])
            print('[INFO] Validation size {} ...'.format(val_size))

            # instead of creating a new model, we just reset its weights
            model = model_old
            _init_weight(model, weights)

            callbacks = []

            logger = tf.keras.callbacks.CSVLogger(filename=str(args.job_dir / 'history.csv'))
            callbacks.append(logger)

            if args.val_metric in ['loss', 'acc']:
                monitor = ('val_' + args.val_metric)  # if val else args.val_metric
                checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=str(args.job_dir / 'best_model.weights'),
                                                                monitor=monitor,
                                                                save_best_only=True,
                                                                save_weights_only=True,
                                                                mode='auto',
                                                                verbose=1)
            elif args.val_metric == 'f1':
                if is_multiclass(train['y']):
                    score_fn = multi_f1
                else:
                    score_fn = f1

                checkpoint = CustomCheckpoint(filepath=str(args.job_dir / 'best_model.weights'),
                                              data=(val_data, val['y']),
                                              # if val else (train_data, partial_train_data['y']),
                                              score_fn=score_fn,
                                              save_best_only=True,
                                              verbose=1)

            elif args.val_metric == 'auc':
                checkpoint = CustomCheckpoint(filepath=str(args.job_dir / 'best_model.weights'),
                                              data=(val_data, val['y']),
                                              # if val else (train_data, partial_train_data['y']),
                                              score_fn=auc,
                                              save_best_only=True,
                                              verbose=1)

            elif args.val_metric == 'fmax':
                checkpoint = CustomCheckpoint(filepath=str(args.job_dir / 'best_model.weights'),
                                              data=(val_data, val['y']),  # if val else (train_data, train['y']),
                                              score_fn=f_max,
                                              save_best_only=True,
                                              verbose=1)

            elif args.val_metric == 'fmetric':
                checkpoint = CustomCheckpoint(filepath=str(args.job_dir / 'best_model.weights'),
                                              data=(val_data, val['y']),  # if val else (train_data, train['y']),
                                              score_fn=f_beta_metric,
                                              save_best_only=True,
                                              verbose=1)

            elif args.val_metric == 'gmetric':
                checkpoint = CustomCheckpoint(filepath=str(args.job_dir / 'best_model.weights'),
                                              data=(val_data, val['y']),  # if val else (train_data, train['y']),
                                              score_fn=g_beta_metric,
                                              save_best_only=True,
                                              verbose=1)

            elif args.val_metric == 'f2018':
                checkpoint = CustomCheckpoint(filepath=str(args.job_dir / 'best_model.weights'),
                                              data=(val_data, val['y']),  # if val else (train_data, train['y']),
                                              score_fn=f1_2018,
                                              save_best_only=True,
                                              verbose=1)

            else:
                raise ValueError('Unknown metric: {}'.format(args.val_metric))

            callbacks.append(checkpoint)

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,
                                                              min_delta=0.001, verbose=1)
            callbacks.append(early_stopping)

            # otherwise, we have already defined a learning rate space to train over,
            # so compute the step size and initialize the cyclic learning rate method
            stepSize = config.STEP_SIZE * train_size // args.batch_size
            clr = CyclicLR(mode=config.CLR_METHOD,
                           base_lr=config.MIN_LR,
                           max_lr=config.MAX_LR,
                           step_size=stepSize)

            callbacks.append(clr)

            print('[INFO] Training fold {}/{} ...'.format(foldNum, args.k_fold))
            model.fit(train_data,
                      validation_data=val_data,
                      epochs=args.epochs,
                      callbacks=callbacks,
                      verbose=1)
            print("============================================================================")

            tf.compat.v1.reset_default_graph()
            del model
            gc.collect()
            tf.keras.backend.clear_session()
