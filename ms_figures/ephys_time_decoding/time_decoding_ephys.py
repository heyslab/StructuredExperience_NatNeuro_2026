import os
import sys

sys.path.append('../')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

from pathlib import Path
import numpy as np
import pandas as pd
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

#from classes.datagen import tDNMSGenerator, genFactory
#from classes.models import LeakyRNN
#from classes.models import LeakyRNNCell

import models_database as mdb
from analysis_tools.progressbar import ProgressBar

from analysis_tools.mpl_helpers import PdfPlotter
import analysis_tools.jPlots as jP

TRIAL_COLORS = {
    'LS': '#eb0d8c',
    'SL': '#2bace2',
    'SS': '#f89521',
    'LL': '#2b958c',
    'S' : '#f89521'}
SHAPING_COLORS = {'no_shaping': 'tab:blue', 'shaping': 'tab:orange'}

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  
    "axes.facecolor":    (1.0, 1.0, 1.0, 0.0),  
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),
    })



def smoothed_spikes(cluster, spikes_file, p=type('',(object,),{"increment": lambda x: None})()):
    p.increment()
    spikes = pd.read_hdf(spikes_file, key=cluster)
    spikes = spikes.dropna(axis=0).set_index(
        ['trial', 'trial_type', 'result'])
    spikes = spikes.reset_index().where(
        spikes.reset_index()['trial_type'] != 'None'
        ).dropna(how='all').set_index(spikes.index.names)

    def create_trial_tc(spike_times, nbins=40, n_trials=1):
        event_count = pd.cut(
            spike_times, np.linspace(0, 20, nbins + 1)
            ).value_counts().sort_index().values / (nbins / 20)

        x_vals = np.linspace(0, 20-20/nbins, nbins) + 20/nbins/2
        spike_rate = pd.Series(
            event_count / n_trials
            ).rolling(
                16, win_type='gaussian', center=True, min_periods=1
                ).mean(std=4)
        return spike_rate

    rate_maps = spikes['trial_time'].groupby(['trial', 'trial_type', 'result']).apply(create_trial_tc).unstack(-1)
    norm_map = rate_maps.subtract(rate_maps.mean(1), axis=0).div(rate_maps.std(1), axis=0).fillna(0)

    return norm_map.stack()



def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()

    data_path = Path(args.directory)
    save_path = data_path / 'analysis'
    jP.make_folder(save_path)
    spikes_file = data_path / 'spikes.h5'
    cache_file = save_path / 'time_decoding.h5'

    print(spikes_file)

    jP.set_rcParams(plt)
    dpi = 300

    with pd.HDFStore(spikes_file, 'r') as f:
        clusters = f.keys()
    p = ProgressBar(len(clusters))
    firing_rates = pd.Series(clusters, index=clusters).apply(smoothed_spikes, spikes_file=spikes_file, p=p)

    Y = firing_rates.T.fillna(0)
    names = list(Y.index.names)
    names[-1] = 'time_bin'
    names[names.index('trial_type')] = 'type'
    Y.index.names = names
 
    def test_model(y, cache_file):
        model_id = 0
        print(f'[Decoding Model {model_id}]')
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
                if 'U' in trial_type:
                    continue
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
        predictions.to_hdf(cache_file, key=f'model_{model_id}')

        #circ_error = lambda a, b, n: np.min(
        #    np.array(list(map(
        #        lambda x, y: (x-y),
        #        it.repeat(a), (b, b+n, b-n)))) ** 2)
        #err = predictions.groupby('time_bin').apply(
        #    lambda x, circ_error=circ_error: x.map(
        #        lambda a, circ_error=circ_error: circ_error(
        #            a, x.index.get_level_values('time_bin'), 20)))
        #squared_error = err.droplevel(0).sort_index()
        #squared_error.columns.name = 'decode_type'

        #percent_correct = pd.DataFrame(
        #    predictions.values == np.expand_dims(
        #        predictions.index.get_level_values('time_bin').to_series(
        #            index=predictions.index).values, -1),
        #    index=predictions.index, columns=predictions.columns
        #    ).groupby(['split', 'time_bin']).mean().groupby('split').mean()
        #return squared_error

    test_model(Y, cache_file=cache_file)

if __name__ == '__main__':
    main(sys.argv[1:])
