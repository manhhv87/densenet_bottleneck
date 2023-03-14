import pandas as pd
import numpy as np

from sklearn.metrics import (confusion_matrix,
                             precision_score, recall_score, f1_score,
                             precision_recall_curve, average_precision_score)

from transplant.evaluation import (auc, f_max, f_beta_metric, g_beta_metric, f1_2018)
from finetuning.utils import (specificity_score, generate_table, plot_pre_rec_curve, plot_confusion_matrix,
                              compute_score_bootstraped, plot_box, McNemar_score, kappa_score_dnn_cardio_emerg_stud,
                              kappa_score_cert_card, compute_score_bootstraped_splits, plot_box_splits)

# %% Constants
score_fun = {'AUC': auc, 'Fmax': f_max, 'Fbeta': f_beta_metric,
             'Gbeta': g_beta_metric, 'F2018': f1_2018}
diagnosis = ['Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
nclasses = len(diagnosis)
predictor_names = ['DNN']

bootstrap_nsamples = 1000
percentiles = [2.5, 97.5]

# Get threshold that yield the best precision recall using "get_optimal_precision_recall" on validation set
# (we rounded it up to three decimal cases to make it easier to read...)
threshold = np.array([0.510, 0.856, 0.570, 0.484, 0.556, 0.200, 0.633, 0.278,
                      0.611])  # corresponding to 'Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE'

# %% Read datasets
# Get true values
y_true = pd.read_csv('./data/annotations/gold_standard.csv').values

# get y_score for different models
y_score_list = [pd.read_csv('./dnn_predicts/model_' + str(i + 1) + '.csv').values for i in range(10)]
y_score_list = [y_score[:, 10:].astype(np.float64) for y_score in y_score_list]

# %% Get average model
# Get micro average precision (return micro average precision (mAP) between 0.946 and 0.961; we choose the one
# with mAP immediately above the median value of all executions (the one with mAP = 0.951))
micro_avg_precision = [average_precision_score(y_true, y_score, average='micro')
                       for y_score in y_score_list]

# get ordered index
# These realizations have micro average precision (mAP) between 0.946 and 0.961
index = np.argsort(micro_avg_precision)  # sorting the data in ascending order and return data's index in array
print('Micro average precision')
print(np.array(micro_avg_precision)[index])

# get 6th best model (immediately above median) out 10 different models
# we choose the one with mAP immediately above the median value of all executions (the one with mAP = 0.951)
# (We could not choose the model with mAP equal to the median value because 10 is an even number;
# hence, there is no single middle value.)
k_dnn_best = index[5]
y_score_best = y_score_list[k_dnn_best]  # score of the best model (6th model)

# We consider our model to have predicted the abnormality when its output—a number between 0 and 1—is above a threshold.
# Note: changing on own dataset
mask = y_score_best > threshold

# Get neural network prediction
# This data was also saved in './data/annotations/dnn.csv'
y_neuralnet = np.zeros_like(y_score_best)  # return an array of zeros with the same shape and type as a given array.
y_neuralnet[mask] = 1  # return an array with 1 value if each of mask's element is true

# %% Generate table with scores for the average model (Table 2)
scores_list = generate_table(y_true=y_true, score_fun=score_fun, diagnosis=diagnosis, y_neuralnet=y_neuralnet)

# %% Confusion matrices (Supplementary Table 1)
plot_confusion_matrix(y_true=y_true, nclasses=nclasses, diagnosis=diagnosis, y_neuralnet=y_neuralnet)

# %% Compute scores and bootstraped version of these scores
scores_percentiles_list, scores_resampled_list = compute_score_bootstraped(y_true=y_true,
                                                                           nclasses=nclasses,
                                                                           score_fun=score_fun,
                                                                           percentiles=percentiles,
                                                                           bootstrap_nsamples=bootstrap_nsamples,
                                                                           y_neuralnet=y_neuralnet,
                                                                           diagnosis=diagnosis,
                                                                           predictor_names=predictor_names)

# %% Print box plot (Supplementary Figure 1)
plot_box(scores_resampled_list=scores_resampled_list, predictor_names=predictor_names,
         bootstrap_nsamples=bootstrap_nsamples, score_fun=score_fun)

# %% McNemar test (Supplementary Table 3)
# McNemar_score(y_true=y_true, y_neuralnet=y_neuralnet, diagnosis=diagnosis)

# # %% Kappa score classifiers (Supplementary Table 2(a))
# kappa_score_dnn_cardio_emerg_stud(names=["DNN", "cardio.", "emerg.", "stud."],
#                                   predictors=[y_neuralnet, y_cardio, y_emerg, y_student],
#                                   diagnosis=diagnosis)

# # %% Kappa score dataset generation (Supplementary Table 2(b))
# kappa_score_cert_card(y_neuralnet=y_neuralnet, y_cardiologist1=y_cardiologist1,
#                       y_cardiologist2=y_cardiologist2, diagnosis=diagnosis)

# # %% Compute scores and bootstraped version of these scores on alternative splits
# scores_resampled_list = compute_score_bootstraped_splits(y_true=y_true, y_score_best=y_score_best,
#                                                          score_fun=score_fun, bootstrap_nsamples=bootstrap_nsamples,
#                                                          percentiles=percentiles, diagnosis=diagnosis)

# # %% Print box plot on alternative splits (Supplementary Figure 2 (a))
# plot_box_splits(scores_resampled_list=scores_resampled_list,
#                 bootstrap_nsamples=bootstrap_nsamples,
#                 score_fun=score_fun)
