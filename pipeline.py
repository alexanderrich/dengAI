from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from statsmodels.tools import eval_measures

import preprocess
import model


sj_train, iq_train = preprocess.load_data('data/dengue_features_train.csv',
                                    labels_path="data/dengue_labels_train.csv")
sj_test, iq_test = preprocess.load_data('data/dengue_features_test.csv')

predictions = {}

regs = [ .5, 1, 2, 5, 10, 20]

for data in [{'name': 'sj', 'train': sj_train, 'test': sj_test, 'subtrain_length': 800},
                 {'name': 'iq', 'train': iq_train, 'test': iq_test, 'subtrain_length': 400}]:

    train_subtrain = data['train'].head(data['subtrain_length']).copy()
    train_subtest = data['train'].tail(data['train'].shape[0] - data['subtrain_length']).copy()
    train_subtrain, tests = preprocess.feature_normalization(train_subtrain, [train_subtest, data['test']])
    train_subtest = tests[0]
    test = tests[1]

    lag_cols = [column for column in train_subtrain.columns.values if column not in ['total_cases', 'constant']]

    train_subtrain = preprocess.add_lags(train_subtrain, lag_cols, [1,2,3])
    train_subtest = preprocess.add_lags(train_subtest, lag_cols, [1,2,3])
    test = preprocess.add_lags(test, lag_cols, [1,2,3])

    x_cols = [column for column in train_subtrain.columns.values if column != 'total_cases']

    best_fit = 100000
    for reg in regs:
        fitted_model = model.regularized_absloss_poisson(train_subtrain, x_cols, 'total_cases', alpha=.01, lambda_reg=reg)
        fit_val = fitted_model.get_abs_error(train_subtest)
        if fit_val < best_fit:
            best_fit = fit_val
            best_reg = reg
            best_fitted_model = fitted_model
    print("best reg for", data['name'], ":", best_reg)
    print("test loss for", data['name'], ":", best_fit)

    # train_full = preprocess.feature_normalization(data['train'])
    # fitted_model = model.regularized_absloss_poisson(train_subtrain, x_cols, 'total_cases', alpha=.01, lambda_reg=best_reg)

    predictions[data['name']] = best_fitted_model.predict(test)

submission = pd.read_csv("data/submission_format.csv",
                         index_col=[0, 1, 2])
submission.total_cases = np.concatenate([predictions['sj'], predictions['iq']])
submission.to_csv("data/submission.csv")
