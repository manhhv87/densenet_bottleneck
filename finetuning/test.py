import argparse
from pathlib import Path

import tensorflow as tf
import numpy as np

from transplant.utils import read_predictions
from transplant.evaluation import (auc, f1, f1_classes, f_max, f_beta_metric, g_beta_metric, f1_2018)
from transplant.utils import (create_predictions_frame, load_pkl)

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


def parse_args():
    """Parse all the arguments provided from the CLI.
       Returns:
           A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--job-dir', type=Path, required=True,
                        help='Job output directory.')
    parser.add_argument('--path_to_model',  # or model_date_order.hdf5
                        help='file containing training model.')
    parser.add_argument('--test', type=Path,
                        help='Path to the test file.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--val-metric', default='loss',
                        help='Validation metric used to find the best model at each epoch. Supported metrics are:'
                             '`loss`, `acc`, `f1`, `auc`, `fmax`, `fmetric`, `gmetric`, `f2018`.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random state.')
    parser.add_argument('--verbose', action='store_true',
                        help='Show debug messages.')

    return parser.parse_known_args()


if __name__ == '__main__':
    """Parse all the arguments provided from the CLI.
       Returns:
           A list of parsed arguments.
    """
    args, unk = parse_args()

    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    if args.val_metric not in ['loss', 'acc', 'f1', 'auc', 'fmax', 'fmetric',
                               'gmetric', 'f2018', 'faf', 'fblock', 'fpc', 'fst']:
        raise ValueError('Unknown metric: {}'.format(args.val_metric))

    seed = args.seed or np.random.randint(2 ** 16)
    print('[INFO] Setting random state {}'.format(seed))
    np.random.seed(seed)

    print('[INFO] Loading test data from {} ...'.format(args.test))
    test = load_pkl(str(args.test))
    print('[INFO] Test size {} ...'.format(len(test['x'])))

    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    test_data = _create_dataset_from_data(test).with_options(options).batch(args.batch_size) if test else None

    # Import model
    model = tf.keras.models.load_model(args.path_to_model, compile=False)

    print('[INFO] Predicting test data ...')
    test_y_prob = model.predict(x=test['x'], batch_size=args.batch_size)
    test_predictions = create_predictions_frame(y_prob=test_y_prob, y_true=test['y'],
                                                class_names=test['classes'],
                                                record_ids=test['record_ids'])
    test_predictions.to_csv(path_or_buf=args.job_dir / 'test_predictions.csv', index=False)

    test_pre = read_predictions(str(args.job_dir) + '/test_predictions.csv')
    y_true = test_pre['y_true']
    y_prob = test_pre['y_prob']

    # Evaluation on F1
    if args.val_metric == 'f1':
        macro_f1 = f1(y_true, y_prob)
        print('[INFO] macro f1 is {}'.format(macro_f1))

        f1_each_class = f1_classes(y_true, y_prob)
        print('[INFO] f1 for each class is {}'.format(f1_each_class))

    # Evaluation on AUC
    if args.val_metric == 'auc':
        macro_auc = auc(y_true, y_prob)
        print('[INFO] macro AUC is {}'.format(macro_auc))

    # Evaluation on Fmax
    if args.val_metric == 'fmax':
        f_max = f_max(y_true, y_prob)
        print('[INFO] f_max is {}'.format(f_max))

    # Evaluation on Fbeta=2
    if args.val_metric == 'fmetric':
        f_beta = f_beta_metric(y_true, y_prob)
        print('[INFO] f_beta is {}'.format(f_beta))

    # Evaluation on Gbeta=2
    if args.val_metric == 'gmetric':
        g_beta = g_beta_metric(y_true, y_prob)
        print('[INFO] g_beta is {}'.format(g_beta))

    # Evaluation on f2018
    if args.val_metric == 'f2018':
        f2018 = f1_2018(y_true, y_prob)
        print('[INFO] f2018 is {}'.format(f2018))
