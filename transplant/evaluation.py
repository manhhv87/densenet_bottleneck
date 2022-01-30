import warnings

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score


def auc(y_true, y_prob):
    if y_prob.ndim != 2:
        raise ValueError('y_prob must be a 2d matrix with class probabilities for each sample')
    if y_true.shape != y_prob.shape:
        raise ValueError('shapes do not match')
    return roc_auc_score(y_true, y_prob, average='macro')


def f1(y_true, y_prob, multiclass=False, threshold=None):
    # threshold may also be a 1d array of thresholds for each class
    if y_prob.ndim != 2:
        raise ValueError('y_prob must be a 2d matrix with class probabilities for each sample')

    if y_true.ndim == 1:  # we assume that y_true is sparse (consequently, multiclass=False)
        if multiclass:
            raise ValueError('if y_true cannot be sparse and multiclass at the same time')
        depth = y_prob.shape[1]
        y_true = _one_hot(y_true, depth)

    if multiclass:
        if threshold is None:
            threshold = 0.5
        y_pred = y_prob >= threshold
    else:
        y_pred = y_prob >= np.max(y_prob, axis=1)[:, None]

    return f1_score(y_true, y_pred, average='macro')


def f1_classes(y_true, y_prob, multiclass=False, threshold=None):
    # threshold may also be a 1d array of thresholds for each class
    if y_prob.ndim != 2:
        raise ValueError('y_prob must be a 2d matrix with class probabilities for each sample')

    if y_true.ndim == 1:  # we assume that y_true is sparse (consequently, multiclass=False)
        if multiclass:
            raise ValueError('if y_true cannot be sparse and multiclass at the same time')
        depth = y_prob.shape[1]
        y_true = _one_hot(y_true, depth)

    if multiclass:
        if threshold is None:
            threshold = 0.5
        y_pred = y_prob >= threshold
    else:
        y_pred = y_prob >= np.max(y_prob, axis=1)[:, None]

    return f1_score(y_true, y_pred, average=None)


def f_max(y_true, y_prob, thresholds=None):
    """ source: https://github.com/helme/ecg_ptbxl_benchmarking """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)

    pr, rc = macro_precision_recall(y_true, y_prob, thresholds)
    f1s = (2 * pr * rc) / (pr + rc)
    i = np.nanargmax(f1s)

    # return f1s[i], thresholds[i]
    return f1s[i]


def macro_precision_recall(y_true, y_prob, thresholds):  # multi-class multi-output
    """ source: https://github.com/helme/ecg_ptbxl_benchmarking """
    # expand analysis to the number of thresholds
    y_true = np.repeat(y_true[None, :, :], len(thresholds), axis=0)
    y_prob = np.repeat(y_prob[None, :, :], len(thresholds), axis=0)
    y_pred = y_prob >= thresholds[:, None, None]

    # compute true positives
    tp = np.sum(np.logical_and(y_true, y_pred), axis=2)

    # compute macro average precision handling all warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        den = np.sum(y_pred, axis=2)
        precision = tp / den
        precision[den == 0] = np.nan
        with warnings.catch_warnings():  # for nan slices
            warnings.simplefilter("ignore", category=RuntimeWarning)
            av_precision = np.nanmean(precision, axis=1)

    # compute macro average recall
    recall = tp / np.sum(y_true, axis=2)
    av_recall = np.mean(recall, axis=1)

    return av_precision, av_recall


def apply_thresholds(pred, thresholds):
    """
    Apply class-wise thresholds to prediction score in order to get binary format.
    BUT: if no score is above threshold, pick maximum. This is needed due to metric issues.
    """

    tmp = []
    for p in pred:
        tmp_p = (p > thresholds).astype(int)
        if np.sum(tmp_p) == 0:
            tmp_p[np.argmax(p)] = 1
        tmp.append(tmp_p)
    tmp = np.array(tmp)
    return tmp


def challenge2020_metrics(y_true, y_prob, beta_f=2, beta_g=2, class_weights=None, single=False):
    """ source: https://github.com/helme/ecg_ptbxl_benchmarking/blob/516740dd2964d67c213ab6df9eba5d50b2245d00/code/utils/utils.py#L100 """
    f_beta = 0
    g_beta = 0

    y_pred = apply_thresholds(y_prob, 0.5)
    num_samples, num_classes = y_true.shape

    if single:  # if evaluating single class in case of threshold-optimization
        sample_weights = np.ones(num_samples)
    else:
        sample_weights = y_true.sum(axis=1)

    if class_weights is None:
        class_weights = np.ones(num_classes)

    for k, w_k in enumerate(class_weights):
        tp, fp, tn, fn = 0., 0., 0., 0.
        for i in range(num_samples):
            if y_true[i, k] == y_pred[i, k] == 1:
                tp += 1. / sample_weights[i]
            if y_pred[i, k] == 1 and y_true[i, k] != y_pred[i, k]:
                fp += 1. / sample_weights[i]
            if y_true[i, k] == y_pred[i, k] == 0:
                tn += 1. / sample_weights[i]
            if y_pred[i, k] == 0 and y_true[i, k] != y_pred[i, k]:
                fn += 1. / sample_weights[i]
        f_beta += w_k * ((1 + beta_f ** 2) * tp) / ((1 + beta_f ** 2) * tp + fp + beta_f ** 2 * fn)
        g_beta += w_k * tp / (tp + fp + beta_g * fn)
    f_beta /= class_weights.sum()
    g_beta /= class_weights.sum()

    return f_beta, g_beta


def f_beta_metric(y_true, y_prob):
    f_beta, _ = challenge2020_metrics(y_true=y_true, y_prob=y_prob)
    return f_beta


def g_beta_metric(y_true, y_prob):
    _, g_beta = challenge2020_metrics(y_true=y_true, y_prob=y_prob)
    return g_beta


def challenge2020_scores(y_true, y_prob):
    A = np.zeros((9, 9), dtype=np.int)
    y_pred = apply_thresholds(y_prob, 0.5)
    num_samples, num_classes = y_true.shape

    for class_i in range(num_classes):
        for i in range(num_samples):
            if y_true[i, class_i] == y_pred[i, class_i] == 1:
                A[class_i][class_i] += 1
            else:
                A[np.argmax(y_pred[i])][class_i] += 1
    return A


def f1_2018(y_true, y_prob):
    # y_pred = apply_thresholds(y_prob, 0.5)
    # num_samples, num_classes = y_true.shape
    #
    # print('[INFO] num_samples')
    # print(num_samples)
    #
    # print('[INFO] num_classes')
    # print(num_classes)
    #
    # print('[INFO] y_true')
    # print(y_true)
    # print('[INFO] y_true_shape')
    # print(y_true.shape)
    # print('[INFO] y_true_1')
    # print(np.argmax(y_true[1]))
    #
    # print('[INFO] y_pred')
    # print(y_pred)
    # print('[INFO] y_pred_shape')
    # print(y_pred.shape)
    # print('[INFO] y_pred_1')
    # print(np.argmax(y_pred[1]))

    A = challenge2020_scores(y_true=y_true, y_prob=y_prob)
    F11 = 2 * A[0][0] / (np.sum(A[0, :]) + np.sum(A[:, 0]))
    F12 = 2 * A[1][1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
    F13 = 2 * A[2][2] / (np.sum(A[2, :]) + np.sum(A[:, 2]))
    F14 = 2 * A[3][3] / (np.sum(A[3, :]) + np.sum(A[:, 3]))
    F15 = 2 * A[4][4] / (np.sum(A[4, :]) + np.sum(A[:, 4]))
    F16 = 2 * A[5][5] / (np.sum(A[5, :]) + np.sum(A[:, 5]))
    F17 = 2 * A[6][6] / (np.sum(A[6, :]) + np.sum(A[:, 6]))
    F18 = 2 * A[7][7] / (np.sum(A[7, :]) + np.sum(A[:, 7]))
    F19 = 2 * A[8][8] / (np.sum(A[8, :]) + np.sum(A[:, 8]))
    F1 = (F11 + F12 + F13 + F14 + F15 + F16 + F17 + F18 + F19) / 9
    return F1


def f_af(y_true, y_prob):
    A = challenge2020_scores(y_true=y_true, y_prob=y_prob)
    Faf = 2 * A[1][1] / (np.sum(A[1, :]) + np.sum(A[:, 1]))
    return Faf


def f_block(y_true, y_prob):
    _, _, A = challenge2020_scores(y_true=y_true, y_prob=y_prob)
    Fblock = 2 * (A[2][2] + A[3][3] + A[4][4]) / (np.sum(A[2:5, :]) + np.sum(A[:, 2:5]))
    return Fblock


def f_pc(y_true, y_prob):
    _, _, A = challenge2020_scores(y_true=y_true, y_prob=y_prob)
    Fpc = 2 * (A[5][5] + A[6][6]) / (np.sum(A[5:7, :]) + np.sum(A[:, 5:7]))
    return Fpc


def f_st(y_true, y_prob):
    _, _, A = challenge2020_scores(y_true=y_true, y_prob=y_prob)
    Fst = 2 * (A[7][7] + A[8][8]) / (np.sum(A[7:9, :]) + np.sum(A[:, 7:9]))
    return Fst


def _one_hot(x, depth):
    x_one_hot = np.zeros((x.size, depth))
    x_one_hot[np.arange(x.size), x] = 1
    return x_one_hot


def multi_f1(y_true, y_prob):
    return f1(y_true, y_prob, multiclass=True, threshold=0.5)


class CustomCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, data, score_fn, best=-np.Inf, save_best_only=False, batch_size=None, verbose=0):
        super().__init__()
        self.filepath = filepath
        self.data = data
        self.score_fn = score_fn
        self.save_best_only = save_best_only
        self.batch_size = batch_size
        self.verbose = verbose
        self.best = best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        x, y_true = self.data
        y_prob = self.model.predict(x, batch_size=self.batch_size)
        score = self.score_fn(y_true, y_prob)
        logs.update({self.metric_name: score})
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        if score > self.best:
            if self.verbose:
                print('\nEpoch %05d: %s improved from %0.5f to %0.5f, saving model to %s'
                      % (epoch + 1, self.metric_name, self.best, score, filepath))
            self.model.save_weights(filepath, overwrite=True)
            self.best = score
        elif not self.save_best_only:
            if self.verbose:
                print('\nEpoch %05d: %s (%.05f) did not improve from %0.5f, saving model to %s'
                      % (epoch + 1, self.metric_name, score, self.best, filepath))
            self.model.save_weights(filepath, overwrite=True)
        else:
            if self.verbose:
                print('\nEpoch %05d: %s (%.05f) did not improve from %0.5f'
                      % (epoch + 1, self.metric_name, score, self.best))

    @property
    def metric_name(self):
        return self.score_fn.__name__
