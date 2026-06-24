import os
import sys

sys.path.append('../../')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools as it
import argparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
import math
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from classes.datagen import tDNMSGenerator, genFactory
from classes.models import LeakyRNN
from classes.models import LeakyRNNCell

import models_database as mdb

from analysis_tools.mpl_helpers import PdfPlotter
import analysis_tools.jPlots as jP

def load_model(model, n_time_bins=20, task='just_short_match'):
    model_info = mdb.get_model(model)
    model_path = model_info['path']
    model_attrs = mdb.get_model_attributes(model)

    input_noise = model_attrs['input_noise']
    rnn_layer = 0
    model = tf.keras.models.load_model(model_path)

    if task is None:
        task = model_info['task_name']

    trials_gen = genFactory.create(
        task, input_noise=input_noise, batch_size=1, n_blocks=1)

    X2 = trials_gen.generate_trials(26)
    formatted_X = tDNMSGenerator.format_validation(X2)[0]
    y = pd.DataFrame(
        model.layers[rnn_layer](
            np.expand_dims(X2[['light', 'odor']], 0))[0])
    y.index = X2.index

    #drop the first block to remove any errors from initiation
    y = y.drop(np.arange(4), level='trial', axis=0)

    time_bins = (
        y.index.get_level_values('idx').to_series() * n_time_bins /
            len(y.index.unique('idx'))
        ).apply(np.floor).astype(int)

    idx = y.index.to_frame()
    idx['time_bin'] = time_bins.values
    y.index = pd.MultiIndex.from_frame(idx)
    return y


def test_model(model_id, y):
    test_trials = np.array_split(y.index.unique('trial'), 10)
    train_trials = [y.index.unique('trial').difference(t) for t in test_trials]
    def train_slice(train, test, y):
        y_train = y.reindex(train, level='trial')
        y_test = y.reindex(test, level='trial')

        clf_time = LinearDiscriminantAnalysis(n_components=2)
        clf_time.fit(
            y_train.values, y=y_train.index.get_level_values('time_bin'))

        predictions = pd.DataFrame([], index=y_test.index)
        predictions['all'] = clf_time.predict(y_test)

        for trial_type in y_train.index.unique('type'):
            clf_time = LinearDiscriminantAnalysis(n_components=2)
            clf_time.fit(
                y_train.xs(trial_type, level='type').values,
                y=y_train.xs(trial_type, level='type'
                ).index.get_level_values('time_bin'))
            predictions[trial_type] = clf_time.predict(y_test)
        return predictions

    predictions = pd.concat(
        list(map(train_slice, train_trials, test_trials, it.repeat(y))), keys=np.arange(len(train_trials)))
    predictions.index.names = ['split'] + predictions.index.names[1:]
    return predictions


def calc_error(predictions):
    circ_error = lambda a, b, n: np.min(
        np.array(list(map(
            lambda x, y: np.abs(x-y),
            it.repeat(a), (b, b+n, b-n)))))
    err = predictions.groupby('time_bin').apply(
        lambda x, circ_error=circ_error: x.map(
            lambda a, circ_error=circ_error: circ_error(
                a, x.index.get_level_values('time_bin'), 20)))
    return err.droplevel(0)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_ids', nargs='+',
        help='model ids from database to run time decoding analysis on (one or more)')
    args = parser.parse_args(argv)
    print(args.model_ids)

    for model_id in args.model_ids:
        print(f'[training on {model_id}]')
        y = load_model(model_id)
        predictions = test_model(model_id, y)
        error = calc_error(predictions)
        result = pd.concat(
            (predictions, error), axis=1,
            keys=('predictions', 'circ_error'))

        save_path = Path(mdb.get_model(model_id)['path']).parent / f'model.{model_id}_time_decoding.h5'
        print(f'saving: {save_path}')
        result.to_hdf(save_path, key='decoding')


if __name__ == '__main__':
    main(sys.argv[1:])
