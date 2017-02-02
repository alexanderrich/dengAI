import pandas as pd
import numpy as np

def load_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    df.drop('week_start_date', axis=1, inplace=True)
    df['constant'] = 1.0
    # fill missing values
    df.fillna(method='ffill', inplace=True)
    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)
    # separate san juan and iquitos
    sj = df.loc['sj'].copy()
    iq = df.loc['iq'].copy()
    return sj, iq

def feature_normalization(train, test_sets=[], exclude=['total_cases', 'constant']):
    train = train.copy()
    if len(test_sets):
        test_sets = [x.copy() for x in test_sets]
    for column in train:
        if column not in exclude:
            train_col = train.loc[:, column]
            train_mean = train_col.mean()
            train_std = train_col.std()
            train.loc[:, column] = (train_col - train_mean) / train_std
            for t in test_sets:
                t.loc[:, column] = (t.loc[:, column] - train_mean) / train_std
    if len(test_sets):
        return train, test_sets
    else:
        return train

def add_lags(data, columns, lags):
    data = data.copy()
    for column in columns:
        for lag in lags:
            data[column + '_lag' + str(lag)] = data[column].shift(lag)
    data = data.fillna(0)
    return data
