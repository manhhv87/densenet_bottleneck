import numpy as np
import pandas as pd

import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2
from itertools import combinations

import tensorflow as tf
import sklearn.model_selection
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


def get_scores(y_true, y_pred, score_fun, threshold):
    # nclasses = np.shape(y_true)[1]
    scores = []

    # for name, fun in score_fun.items():
    #     scores += [[fun(y_true[:, k], y_pred[:, k]) for k in range(nclasses)]]

    for name, fun in score_fun.items():
        if name == 'AUC':
            scores += [fun(y_true, y_pred)]
        else:
            scores += [fun(y_true, y_pred, threshold)]

    return np.array(scores).T


def specificity_score(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred, labels=[0, 1])
    spc = m[0, 0] * 1.0 / (m[0, 0] + m[0, 1])

    return spc


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


def affer_results(y_true, y_pred):
    """Return true positives, false positives, true negatives, false negatives.
    Args:
        y_true : ndarray
            True value
        y_pred : ndarray
            Predicted value
    Returns:
        tn, tp, fn, fp: ndarray
            Boolean matrices containing true negatives, true positives, false negatives and false positives.
        cm : ndarray
            Matrix containing: 0 - true negative, 1 - true positive,
                               2 - false negative, and 3 - false positive.
    """

    tn = (y_true == y_pred) & (y_pred == 0)  # True negative
    tp = (y_true == y_pred) & (y_pred == 1)  # True positive
    fp = (y_true != y_pred) & (y_pred == 1)  # False positive
    fn = (y_true != y_pred) & (y_pred == 0)  # False negative

    # Generate matrix of "tp, fp, tn, fn"
    m, n = np.shape(y_true)
    cm = np.zeros((m, n), dtype=int)
    cm[tn] = 0
    cm[tp] = 1
    cm[fn] = 2
    cm[fp] = 3

    return tn, tp, fn, fp, cm


# %% Generate table with scores for the average model (Table 2)
def generate_table(y_true, y_prob, score_fun, threshold, diagnosis):
    # Compute scores
    scores = get_scores(y_true, y_prob, score_fun, threshold)

    # Put them into a data frame
    scores_df = pd.DataFrame(scores, index=score_fun.keys(), columns=diagnosis)

    # Save results
    scores_df.to_excel("./output/tables/scores.xlsx", float_format='%.3f')
    scores_df.to_csv("./output/tables/scores.csv", float_format='%.3f')

    return scores_df


# %% Confusion matrices (Supplementary Table 1)
def plot_confusion_matrix(y_true, y_prod, nclasses, predictor_names, diagnosis, threshold):
    mask = y_prod > threshold

    # Get neural network prediction
    # This data was also saved in './data/annotations/dnn.csv'
    y_pred = np.zeros_like(y_prod)  # return an array of zeros with the same shape and type as a given array.
    y_pred[mask] = 1  # return an array with 1 value if each of mask's element is true

    M = [confusion_matrix(y_true[:, k], y_pred[:, k], labels=[0, 1])
         for k in range(nclasses)]

    M_xarray = xr.DataArray(np.expand_dims(np.array(M), axis=0),
                            dims=['predictor', 'diagnosis', 'true label', 'predicted label'],
                            coords={'predictor': predictor_names,
                                    'diagnosis': diagnosis,
                                    'true label': ['not present', 'present'],
                                    'predicted label': ['not present', 'present']})

    confusion_matrices = M_xarray.to_dataframe('n')
    confusion_matrices = confusion_matrices.reorder_levels([1, 2, 3, 0], axis=0)
    confusion_matrices = confusion_matrices.unstack()
    confusion_matrices = confusion_matrices.unstack()
    confusion_matrices = confusion_matrices['n']

    confusion_matrices.to_excel("./output/tables/confusion_matrices.xlsx", float_format='%.3f')
    confusion_matrices.to_csv("./output/tables/confusion_matrices.csv", float_format='%.3f')


# %% Compute scores and bootstraped version of these scores
def compute_score_bootstraped(y_true, y_prob, nclasses, score_fun, bootstrap_nsamples, threshold):
    # Compute bootstraped samples
    np.random.seed(123)  # NEVER change this =P
    n, _ = np.shape(y_true)
    samples = np.random.randint(n, size=n * bootstrap_nsamples)

    # Get samples
    y_true_resampled = np.reshape(y_true[samples, :], (bootstrap_nsamples, n, nclasses))
    y_prob_resampled = np.reshape(y_prob[samples, :], (bootstrap_nsamples, n, nclasses))

    # Apply functions
    scores_resampled = np.array([get_scores(y_true_resampled[i, :, :], y_prob_resampled[i, :, :], score_fun, threshold)
                                 for i in range(bootstrap_nsamples)])

    # Sort scores
    scores_resampled.sort(axis=0)

    return scores_resampled


# %% Print box plot (Supplementary Figure 1)
def plot_box(scores_resampled_list, predictor_names, bootstrap_nsamples, score_fun):
    # Convert to xarray
    scores_resampled_xr = xr.DataArray(np.array(scores_resampled_list),
                                       dims=['n', 'score_fun'],
                                       coords={'n': range(bootstrap_nsamples),
                                               'score_fun': list(score_fun.keys())})

    scores_resampled_list_df = []

    for sf in score_fun:
        f1_score_resampled_xr = scores_resampled_xr.sel(score_fun=sf)

        # Convert to dataframe
        scores_resampled_list_df.append(f1_score_resampled_xr.to_dataframe(name=sf).reset_index(level=[0]))

    scores_resampled_concat_df = pd.concat(scores_resampled_list_df, axis=1)
    scores_resampled_concat_df = scores_resampled_concat_df.drop(columns='n')
    df_melted = pd.melt(scores_resampled_concat_df)
    df_melted = df_melted.drop(df_melted[df_melted['variable'] == 'score_fun'].index)

    # Plot seaborn
    ax = sns.boxplot(x='variable', y='value', data=df_melted)

    # Save results
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("")
    plt.ylabel("", fontsize=16)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('./output/figures/boxplot_bootstrap.pdf')

    scores_resampled_xr.to_dataframe(name='score').to_csv('./output/figures/boxplot_bootstrap_data.txt')


# %% Compute scores and bootstraped version of these scores on alternative splits
def compute_score_bootstraped_splits(y_true, y_score_best, score_fun, bootstrap_nsamples, percentiles, diagnosis):
    scores_resampled_list = []
    scores_percentiles_list = []

    for name in ['normal_order', 'date_order', 'individual_patients', 'base_model']:
        print(name)

        # Get data
        yn_true = y_true
        yn_score = np.load(
            './dnn_predicts/other_splits/model_' + name + '.npy') if not name == 'base_model' else y_score_best

        # Compute threshold
        nclasses = np.shape(yn_true)[1]
        opt_precision, opt_recall, threshold = get_optimal_precision_recall(yn_true, yn_score)
        mask_n = yn_score > threshold
        yn_pred = np.zeros_like(yn_score)
        yn_pred[mask_n] = 1

        # Compute bootstraped samples
        np.random.seed(123)  # NEVER change this =P
        n, _ = np.shape(yn_true)
        samples = np.random.randint(n, size=n * bootstrap_nsamples)

        # Get samples
        y_true_resampled = np.reshape(yn_true[samples, :], (bootstrap_nsamples, n, nclasses))
        y_doctors_resampled = np.reshape(yn_pred[samples, :], (bootstrap_nsamples, n, nclasses))

        # Apply functions
        scores_resampled = np.array([get_scores(y_true_resampled[i, :, :], y_doctors_resampled[i, :, :], score_fun)
                                     for i in range(bootstrap_nsamples)])
        # Sort scores
        scores_resampled.sort(axis=0)

        # Append
        scores_resampled_list.append(scores_resampled)

        # Compute percentiles index
        i = [int(p / 100.0 * bootstrap_nsamples) for p in percentiles]

        # Get percentiles
        scores_percentiles = scores_resampled[i, :, :]

        # Convert percentiles to a dataframe
        scores_percentiles_df = pd.concat([pd.DataFrame(x, index=diagnosis, columns=score_fun.keys())
                                           for x in scores_percentiles], keys=['p1', 'p2'], axis=1)

        # Change multiindex levels
        scores_percentiles_df = scores_percentiles_df.swaplevel(0, 1, axis=1)
        scores_percentiles_df = scores_percentiles_df.reindex(level=0, columns=score_fun.keys())

        # Append
        scores_percentiles_list.append(scores_percentiles_df)

    return scores_resampled_list


# %% Print box plot on alternative splits (Supplementary Figure 2 (a))
def plot_box_splits(scores_resampled_list, bootstrap_nsamples, score_fun):
    scores_resampled_xr = xr.DataArray(np.array(scores_resampled_list),
                                       dims=['predictor', 'n', 'diagnosis', 'score_fun'],
                                       coords={
                                           'predictor': ['random', 'by date', 'by patient', 'original DNN'],
                                           'n': range(bootstrap_nsamples),
                                           'diagnosis': ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST'],
                                           'score_fun': list(score_fun.keys())})

    # Remove everything except f1_score
    sf = 'F1 score'
    fig, ax = plt.subplots()
    f1_score_resampled_xr = scores_resampled_xr.sel(score_fun=sf)

    # Convert to dataframe
    f1_score_resampled_df = f1_score_resampled_xr.to_dataframe(name=sf).reset_index(level=[0, 1, 2])

    # Plot seaborn
    ax = sns.boxplot(x="diagnosis", y=sf, hue="predictor", data=f1_score_resampled_df,
                     order=['1dAVb', 'SB', 'AF', 'ST', 'RBBB', 'LBBB'],
                     palette=sns.color_palette("Set1", n_colors=8))
    plt.axvline(3.5, color='black', ls='--')
    plt.axvline(5.5, color='black', ls='--')
    plt.axvspan(3.5, 5.5, alpha=0.1, color='gray')

    # Save results
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("")
    plt.ylabel("F1 score", fontsize=16)
    plt.legend(fontsize=17)
    plt.ylim([0.4, 1.05])
    plt.xlim([-0.5, 5.5])
    plt.tight_layout()
    plt.savefig('./outputs/figures/boxplot_bootstrap_other_splits_{0}.pdf'.format(sf))
    f1_score_resampled_df.to_csv('./outputs/figures/boxplot_bootstrap_other_splits_data.txt', index=False)
