import numpy as np
import sklearn.model_selection
import tensorflow as tf

from transplant.modules.densenet1d import _DenseNet


def ecg_feature_extractor(input_layer=None, stages=None):
    backbone_model = _DenseNet(input_layer=input_layer,
                               num_outputs=None,
                               # blocks=(6, 4, 6, 0)[:stages],
                               # blocks=(6, 12, 24, 16)[:stages],   # DenseNet-121
                               blocks=(6, 12, 32, 32)[:stages],     # DenseNet-169
                               #blocks=(6, 12, 48, 32)[:stages],  # DenseNet-201
                               # blocks=(6, 12, 64, 48)[:stages],  # DenseNet-264
                               # first_num_channels=16,
                               first_num_channels=64,
                               # growth_rate=8,
                               growth_rate=32,
                               # kernel_size=(8, 6, 8, 4),
                               kernel_size=(3, 3, 3, 3),
                               bottleneck=True,
                               dropout_rate=None,
                               include_top=False).model()

    return backbone_model

    # feature_extractor = tf.keras.Sequential([base_model,
    #                                          tf.keras.layers.GlobalAveragePooling1D()])  # not fc layer
    # return feature_extractor


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

