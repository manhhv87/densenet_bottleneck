import numpy as np
import sklearn.model_selection
import tensorflow as tf
from sklearn.metrics import (confusion_matrix,
                             precision_score, recall_score, f1_score,
                             precision_recall_curve, average_precision_score)

from transplant.modules.densenet1d import _DenseNet


def ecg_feature_extractor(input_layer=None, stages=None):
    backbone_model = _DenseNet(input_layer=input_layer,
                               num_outputs=None,
                               blocks=(6, 4, 6, 0)[:stages],  # Own model
                               # blocks=(6, 12, 24, 16)[:stages],   # DenseNet-121
                               # blocks=(6, 12, 32, 32)[:stages],     # DenseNet-169
                               # blocks=(6, 12, 48, 32)[:stages],  # DenseNet-201
                               # blocks=(6, 12, 64, 48)[:stages],  # DenseNet-264
                               first_num_channels=16,
                               # first_num_channels=64,
                               growth_rate=8,
                               # growth_rate=32,
                               kernel_size=(8, 6, 8, 4),
                               # kernel_size=(3, 3, 3, 3),
                               bottleneck=True,
                               dropout_rate=None,
                               include_top=False).model()

    return backbone_model


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


def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""
    n = np.shape(y_true)[1]
    opt_precision = []
    opt_recall = []
    opt_threshold = []
    for k in range(n):
        # Get precision-recall curve
        precision, recall, threshold = precision_recall_curve(y_true[:, k], y_score[:, k])

        # Compute f1 score for each point (use nan_to_num to avoid nans messing up the results)
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))

        # Select threshold that maximize f1 score
        index = np.argmax(f1_score)
        opt_precision.append(precision[index])
        opt_recall.append(recall[index])
        # t = threshold[index-1] if index != 0 else threshold[0]-1e-10
        opt_threshold.append(threshold[index])

    return np.array(opt_precision), np.array(opt_recall), np.array(opt_threshold)
