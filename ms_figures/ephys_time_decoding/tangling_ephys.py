import os
import sys

sys.path.append('../')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

import random
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

    def create_trial_tc(spike_times, nbins=200, n_trials=1):
        event_count = pd.cut(
            spike_times, np.linspace(0, 20, nbins + 1)
            ).value_counts().sort_index().values / (nbins / 20)

        x_vals = np.linspace(0, 20-20/nbins, nbins) + 20/nbins/2
        spike_rate = pd.Series(
            event_count / n_trials
            ).rolling(
                400, win_type='gaussian', center=True, min_periods=1
                ).mean(std=20)
        return spike_rate

    rate_maps = spikes['trial_time'].groupby(['trial', 'trial_type', 'result']).apply(create_trial_tc).unstack(-1)
    norm_map = rate_maps.subtract(rate_maps.mean(1), axis=0).div(rate_maps.std(1), axis=0).fillna(0)
    return norm_map.stack()


def model_tangling(y, delta_t=0.1):
    e = (y**2).sum(axis=1).mean() * 0.1

    def calc_t(t, y, e, p):
        p.increment()
        denom = ((y.iloc[t] - y)**2).sum(axis=1)
        diff_y = y.diff() / delta_t
        num = ((diff_y.iloc[t] - diff_y)**2).sum(axis=1)
        return (num/(denom + e)).max()

    p = ProgressBar(len(y))
    t = pd.Series(np.arange(len(y)), index=y.index)[1:].apply(calc_t, y=y, e=e, p=p)
    return t


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()

    data_path = Path(args.directory)
    save_path = data_path / 'analysis'
    jP.make_folder(save_path)
    spikes_file = data_path / 'spikes.h5'
    cache_file = save_path / 'tangling.h5'
    spikes_cache = 'spikes.h5'

    print(spikes_file)

    jP.set_rcParams(plt)
    dpi = 300

    with pd.HDFStore(spikes_file, 'r') as f:
        clusters = f.keys()
    random.seed(0)
    p = ProgressBar(len(clusters))
    try:
        firing_rates = pd.read_hdf(spikes_cache, key=data_path.parts[-1])
    except:
        firing_rates = pd.Series(clusters, index=clusters).apply(smoothed_spikes, spikes_file=spikes_file, p=p)
        firing_rates.to_hdf(spikes_cache, key=data_path.parts[-1])
    Y = firing_rates.T.fillna(0)
    Y = Y.drop('FA', level='result')
    Y = Y.loc[Y.index.get_level_values('trial') < Y.index.unique('trial')[25]] 

    names = list(Y.index.names)
    names[-1] = 'time_bin'
    names[names.index('trial_type')] = 'type'
    Y.index.names = names
    pca = PCA(n_components=8)
    pcs = pca.fit_transform(Y)
    tangling = model_tangling(pd.DataFrame(pcs, index=Y.index)) 
    print()
    print(tangling.mean())
    tangling.to_hdf(cache_file, key='tangling')  
 
if __name__ == '__main__':
    main(sys.argv[1:])
