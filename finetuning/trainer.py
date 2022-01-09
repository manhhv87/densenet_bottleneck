import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from transplant.utils import read_predictions

from finetuning.utils import (ecg_feature_extractor, train_test_split)
from transplant.evaluation import (auc, f1, f1_classes, multi_f1, CustomCheckpoint, f_max, challenge2020_metrics)
from transplant.utils import (create_predictions_frame, load_pkl, is_multiclass)

from clr.learningratefinder import LearningRateFinder
from clr.clr_callback import CyclicLR
from clr import config


def _create_dataset_from_data(data):
    """
    input:  data = {'x': x,
                    'y': labels.to_numpy(),
                    'record_ids': labels.index.to_numpy(),
                    'classes': labels.columns.to_numpy()}
    return: data and label
    """
    return tf.data.Dataset.from_tensor_slices((data['x'], data['y']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', type=Path, required=True, help='Job output directory.')
    parser.add_argument('--train', type=Path, required=True, help='Path to the train file.')
    parser.add_argument('--val', type=Path, help='Path to the validation file.\n'
                                                 'Overrides --val-size.')
    parser.add_argument('--test', type=Path, help='Path to the test file.')
    parser.add_argument('--weights-file', type=Path, help='Path to pretrained weights or a checkpoint of the model.')
    parser.add_argument('--val-size', type=float, default=None,
                        help='Size of the validation set or proportion of the train set.')
    parser.add_argument('--arch', default='resnet18', help='Network architecture: '
                                                           '`resnet18`, `resnet34` or `resnet50`.')
    parser.add_argument('--subset', type=float, default=None, help='Size of a subset of the train set '
                                                                   'or proportion of the train set.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('--val-metric', default='loss',
                        help='Validation metric used to find the best model at each epoch. Supported metrics are:'
                             '`loss`, `acc`, `f1`, `auc`, `fmax`, `fg`.')
    parser.add_argument('--channel', type=int, default=None, help='Use only the selected channel. '
                                                                  'By default use all available channels.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs.')
    parser.add_argument('--seed', type=int, default=None, help='Random state.')
    parser.add_argument('--verbose', action='store_true', help='Show debug messages.')
    parser.add_argument('--k-fold', type=int, default=None, help='k-fold cross validation')
    args, _ = parser.parse_known_args()

    if args.val_metric not in ['loss', 'acc', 'f1', 'auc', 'fmax', 'fg']:
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

        train_data = _create_dataset_from_data(train).shuffle(len(train['x'])).batch(args.batch_size)
        val_data = _create_dataset_from_data(val).batch(args.batch_size) if val else None
        test_data = _create_dataset_from_data(test).batch(args.batch_size) if test else None

        train_size = len(train['x'])
        val_size = len(val['x'])
        test_size = len(test['x'])
        print('[INFO] Train, Validation and Test size {}, {}, {} ...'.format(train_size, val_size, test_size))

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

            # not include fc layer
            model = ecg_feature_extractor(arch=args.arch)
            model.add(tf.keras.layers.Dense(units=num_classes, activation=activation))

            # initialize the weights of the model
            inputs = tf.keras.layers.Input(shape=train['x'].shape[1:], dtype=train['x'].dtype)
            model(inputs)  # complete model

            print('[INFO] Model parameters: {:,d}'.format(model.count_params()))

            if args.weights_file:  # Sử dụng trọng số đã được pre-trained
                # initialize weights (excluding the optimizer state) to load the pretrained resnet
                # the optimizer state is randomly initialized in the `model.compile` function
                print('[INFO] Loading weights from file {} ...'.format(args.weights_file))
                model.load_weights(str(args.weights_file))

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.MIN_LR, beta_1=0.9,
                                                             beta_2=0.98, epsilon=1e-9),
                          loss=loss,
                          metrics=[accuracy])

            callbacks = []

            logger = tf.keras.callbacks.CSVLogger(filename=str(args.job_dir / 'history.csv'))
            callbacks.append(logger)

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
                                              data=(val_data, val['y']),     # if val else (train_data, train['y']),
                                              score_fn=f_max,
                                              save_best_only=True,
                                              verbose=1)

            elif args.val_metric == 'fg':
                checkpoint = CustomCheckpoint(filepath=str(args.job_dir / 'best_model.weights'),
                                              data=(val_data, val['y']),    # if val else (train_data, train['y']),
                                              score_fn=challenge2020_metrics,
                                              save_best_only=True,
                                              verbose=1)

            else:
                raise ValueError('Unknown metric: {}'.format(args.val_metric))

            callbacks.append(checkpoint)

            if val:
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,
                                                                  min_delta=0.001, verbose=1)
                callbacks.append(early_stopping)

            # otherwise, we have already defined a learning rate space to train
            # over, so compute the step size and initialize the cyclic learning
            # rate method

            # stepSize = config.STEP_SIZE * train_size // args.batch_size
            stepSize = config.STEP_SIZE * train_size // args.batch_size
            clr = CyclicLR(mode=config.CLR_METHOD,
                           base_lr=config.MIN_LR,
                           max_lr=config.MAX_LR,
                           step_size=stepSize)

            callbacks.append(clr)

            # Disable AutoShard.
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            train_data = train_data.with_options(options)
            val_data = val_data.with_options(options)

            model.fit(train_data, epochs=args.epochs, verbose=1, validation_data=val_data, callbacks=callbacks)

            # load best model for inference
            print('[INFO] Loading the best weights from file {} ...'.format(str(args.job_dir / 'best_model.weights')))
            model.load_weights(filepath=str(args.job_dir / 'best_model.weights'))

            print('[INFO] Predicting training data ...')
            train_y_prob = model.predict(x=train['x'], batch_size=args.batch_size)
            train_predictions = create_predictions_frame(y_prob=train_y_prob,
                                                         y_true=train['y'],
                                                         class_names=train['classes'],
                                                         record_ids=train['record_ids'])
            train_predictions.to_csv(path_or_buf=args.job_dir / 'train_predictions.csv', index=False)

            if val:
                print('[INFO] Predicting validation data ...')
                val_y_prob = model.predict(x=val['x'], batch_size=args.batch_size)
                val_predictions = create_predictions_frame(y_prob=val_y_prob, y_true=val['y'],
                                                           class_names=train['classes'],
                                                           record_ids=val['record_ids'])
                val_predictions.to_csv(path_or_buf=args.job_dir / 'val_predictions.csv', index=False)

            if test:
                print('[INFO] Predicting test data ...')
                test_y_prob = model.predict(x=test['x'], batch_size=args.batch_size)
                test_predictions = create_predictions_frame(y_prob=test_y_prob, y_true=test['y'],
                                                            class_names=train['classes'],
                                                            record_ids=test['record_ids'])
                test_predictions.to_csv(path_or_buf=args.job_dir / 'test_predictions.csv', index=False)

    else:  # Đánh giá theo k-fold cross validation
        print('[INFO] Loading train data from {} ...'.format(args.train))
        data_set = load_pkl(file=str(args.train))
        x, y, record_ids, classes = data_set['x'], data_set['y'], data_set['record_ids'], data_set['classes']

        if args.channel is not None:
            data_set['x'] = data_set['x'][:, :, args.channel:args.channel + 1]

        print('[INFO] Train data shape:', data_set['x'].shape)

        idx_data = np.arange(len(x))

        kf = KFold(n_splits=args.k_fold, shuffle=True)
        foldNum = 0
        all_scores_mse = []
        all_scores_f1_each_class = []
        all_scores_macro_f1 = []

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

                # not include fc layer
                model = ecg_feature_extractor(arch=args.arch)
                model.add(tf.keras.layers.Dense(units=num_classes, activation=activation))

                # initialize the weights of the model
                inputs = tf.keras.layers.Input(shape=train['x'].shape[1:], dtype=train['x'].dtype)
                model(inputs)  # complete model

                print('[INFO] Model parameters: {:,d}'.format(model.count_params()))

                if args.weights_file:  # Sử dụng trọng số đã được pre-trained
                    # initialize weights (excluding the optimizer state) to load the pretrained resnet
                    # the optimizer state is randomly initialized in the `model.compile` function
                    print('[INFO] Loading weights from file {} ...'.format(args.weights_file))
                    model.load_weights(str(args.weights_file))

                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.MIN_LR,
                                                                 beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                              loss=loss,
                              metrics=[accuracy])

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
                                                  data=(val_data, val['y']),    # if val else (train_data, train['y']),
                                                  score_fn=f_max,
                                                  save_best_only=True,
                                                  verbose=1)

                elif args.val_metric == 'fg':
                    checkpoint = CustomCheckpoint(filepath=str(args.job_dir / 'best_model.weights'),
                                                  data=(val_data, val['y']),    # if val else (train_data, train['y']),
                                                  score_fn=challenge2020_metrics,
                                                  save_best_only=True,
                                                  verbose=1)

                else:
                    raise ValueError('Unknown metric: {}'.format(args.val_metric))

                callbacks.append(checkpoint)

                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,
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

                # rlo = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7,
                #                                            verbose=1, min_lr=config.MIN_LR)
                #
                # callbacks.append(rlo)

                print('[INFO] Training fold {}/{} ...'.format(foldNum, args.k_fold))
                model.fit(train_data,
                          epochs=args.epochs,
                          verbose=1,
                          validation_data=val_data,
                          callbacks=callbacks)

                # load best model for inference
                print(
                    '[INFO] Loading the best weights from file {} ...'.format(str(args.job_dir / 'best_model.weights')))
                model.load_weights(filepath=str(args.job_dir / 'best_model.weights'))

                print('[INFO] Predicting training data for fold {} ...'.format(foldNum))
                train_y_prob = model.predict(x=train['x'], batch_size=args.batch_size)
                train_predictions = create_predictions_frame(y_prob=train_y_prob,
                                                             y_true=train['y'],
                                                             class_names=train['classes'],
                                                             record_ids=train['record_ids'])
                train_predictions.to_csv(path_or_buf=str(args.job_dir) + '/train_predictions_' + str(foldNum) + '.csv',
                                         index=False)

                print('[INFO] Predicting validation data for fold {} ...'.format(foldNum))
                val_y_prob = model.predict(x=val['x'], batch_size=args.batch_size)
                val_predictions = create_predictions_frame(y_prob=val_y_prob,
                                                           y_true=val['y'],
                                                           class_names=train['classes'],
                                                           record_ids=val['record_ids'])
                val_predictions.to_csv(path_or_buf=str(args.job_dir) + '/val_predictions_' + str(foldNum) + '.csv',
                                       index=False)

                val_pre = read_predictions(str(args.job_dir) + '/val_predictions_' + str(foldNum) + '.csv')
                y_true = val_pre['y_true']
                y_prob = val_pre['y_prob']
                macro_f1 = f1(y_true, y_prob)
                all_scores_macro_f1.append(macro_f1)
                print('[INFO] macro f1 for fold {} is {}'.format(foldNum, macro_f1))
                f1_each_class = f1_classes(y_true, y_prob)
                all_scores_f1_each_class.append(f1_each_class)
                print('[INFO] f1 for each class for fold {} is {}'.format(foldNum, f1_each_class))

                print('[INFO] Evaluates the model on the validation data ...')
                val_mse, val_mae = model.evaluate(val_data, verbose=1)
                all_scores_mse.append(val_mse)
                print('[INFO] Validation MSE for fold {} is {}'.format(foldNum, val_mse))
                print("============================================================================")

        scores_f1_each_class_array = np.concatenate(all_scores_f1_each_class, axis=0).reshape((5, 4))

        print("Results ...")
        print("============================================================================")

        print('[INFO] macro f1, mean and standard deviation values for all of folds is {}, {} and {}'.format(
            all_scores_macro_f1, np.mean(all_scores_macro_f1), np.std(all_scores_macro_f1)))
        print('[INFO] f1, mean and standard deviation values for each class of folds is {}, {} and {}'.format(
            all_scores_f1_each_class, scores_f1_each_class_array.mean(axis=0), scores_f1_each_class_array.std(axis=0)))
        print('[INFO] mse, mean and standard deviation values for folds is {} and {}'.format(
            all_scores_mse, np.mean(all_scores_mse), np.std(all_scores_mse)))
