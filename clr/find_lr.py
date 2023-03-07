import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np

from finetuning.utils import ecg_feature_extractor
from transplant.utils import load_pkl, is_multiclass

from clr.learningratefinder import LearningRateFinder
from clr import config
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

    parser.add_argument('--train', type=Path, required=True,
                        help='Path to the train file.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random state.')

    return parser.parse_known_args()


if __name__ == '__main__':
    args, unk = parse_args()

    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    seed = args.seed or np.random.randint(2 ** 16)
    print('[INFO] Setting random state {}'.format(seed))
    np.random.seed(seed)

    print('[INFO] Loading train data from {} ...'.format(args.train))
    train = load_pkl(file=str(args.train))

    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_data = _create_dataset_from_data(train).with_options(options).shuffle(len(train['x'])).batch(
        args.batch_size, drop_remainder=True)

    print('[INFO] Train size {} ...'.format(len(train['x'])))

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

        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True))(
            backbone_model.output)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)

        x = tf.keras.layers.Dense(units=num_classes, activation=activation)(x)
        model = tf.keras.models.Model(inputs=backbone_model.input, outputs=x)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.MIN_LR),
                      loss=loss, metrics=[accuracy])

        # use the learning rate finder to find a suitable range to train our network
        lrf = LearningRateFinder(model)
        lrf.find(trainData=train_data,
                 startLR=1e-10, endLR=1e+1,
                 stepsPerEpoch=config.STEP_SIZE * (len(train['x']) // args.batch_size),
                 batchSize=args.batch_size)
