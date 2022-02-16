import numpy as np
import sklearn.model_selection
import tensorflow as tf

from transplant.modules.densenet1d import _DenseNet


def ecg_feature_extractor(arch=None, input_layer=None, stages=None):
    # Furthermore, we use larger filter sizes (7, 5, 5, 3) at each stage respectively,
    # which we have observed to outperform the suggested smaller 3 × 3 filters.
    # See Table 1 in Deep Residual Learning for Image Recognition
    if arch is None or arch == 'resnet18':
        resnet = _DenseNet(input_layer=input_layer,
                           num_outputs=None,
                           blocks=(6, 4, 6, 0)[:stages],
                           first_num_channels=16,
                           growth_rate=8,
                           kernel_size=(8, 6, 8, 4),
                           bottleneck=True,
                           dropout_rate=None,
                           include_top=False).dense_model()
    else:
        raise ValueError('unknown architecture: {}'.format(arch))

    feature_extractor = tf.keras.Sequential([input_layer,
                                             resnet,
                                             tf.keras.layers.GlobalAveragePooling1D()])  # not fc layer
    return feature_extractor


def train_test_split(data_set, **options):
    x, y, record_ids, classes = data_set['x'], data_set['y'], data_set['record_ids'], data_set['classes']
    assert len(x) == len(y) == len(record_ids)
    idx = np.arange(len(x))
    train_idx, test_idx = sklearn.model_selection.train_test_split(idx, **options)
    train = {'x': x[train_idx],
             'y': y[train_idx],
             'record_ids': record_ids[train_idx],
             'classes': classes}
    test = {'x': x[test_idx],
            'y': y[test_idx],
            'record_ids': record_ids[test_idx],
            'classes': classes}
    return train, test