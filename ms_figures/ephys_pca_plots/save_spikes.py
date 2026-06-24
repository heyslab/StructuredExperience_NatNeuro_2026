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

import models_database as mdb
from analysis_tools.progressbar import ProgressBar

from analysis_tools.mpl_helpers import PdfPlotter
import analysis_tools.jPlots as jP


def smoothed_spikes(cluster, spikes_file, p=type('',(object,),{"increment": lambda x: None})()):
    p.increment()
    spikes = pd.read_hdf(spikes_file, key=cluster)
    spikes = spikes.dropna(axis=0).set_index(
        ['trial', 'trial_type', 'result'])
    spikes = spikes.reset_index().where(
        spikes.reset_index()['trial_type'] != 'None'
        ).dropna(how='all').set_index(spikes.index.names)

    def create_trial_tc(spike_times, nbins=80, n_trials=1):
        tmax = 20
        event_count = pd.cut(
            spike_times, np.linspace(0, tmax, nbins + 1)
            ).value_counts().sort_index().values / (nbins / tmax)

        x_vals = np.linspace(0, tmax-tmax/nbins, nbins) + tmax/nbins/2
        sigma = 4
        spike_rate = pd.Series(
            event_count / n_trials
            ).rolling(
                sigma**2, win_type='gaussian', center=True, min_periods=1
                ).mean(std=sigma)
        idx_frame = pd.Series(x_vals, name='trial_time',
                              index=pd.Index(np.arange(nbins), name='idx')).reset_index()
        spike_rate.index = pd.MultiIndex.from_frame(idx_frame)
        return spike_rate

    rate_maps = spikes['trial_time'].groupby(['trial', 'trial_type', 'result'])\
                                    .apply(create_trial_tc).unstack(['trial_time', 'idx'])
    norm_map = ((rate_maps - np.mean(rate_maps)) / np.std(rate_maps.values)).fillna(0)
    return norm_map.stack(['trial_time', 'idx'], future_stack=True)



def main(argv):
    sl_shaping_paths = list(map(
        Path,
         (
          '/data1/dua/sl_spike_data/M2_2202025_g0',
          '/data1/dua/sl_spike_data/M2_2212025_g0',
          '/data1/dua/sl_spike_data/M1_2182025_g0',
          '/data1/dua/sl_spike_data/M1_2202025_g0',
          '/data1/dua/sl_spike_data/M4_2252025_g0',
          '/data1/dua/sl_spike_data/M4_2272025_g1',
          '/data1/dua/sl_spike_data/M5_2272025_g0',
          '/data1/dua/sl_spike_data/M5_3012025_g0'
          )))

    shaping_paths = list(map(
        Path,
        ('/data1/jack/tDNMS_EC_project/AC01_ephys/AC01_01292025_g0',
         '/heys-nas-LabData/jack/neuropix_data/j2_05092024/j2_05092024_g0',
         '/heys-nas-LabData/tDNMS_EC_project/data/AB02_ephys/AB02_08272024_g0',
         '/data1/jack/tDNMS_EC_project/AB02_ephys/AB02_08282024_g0',
         '/data1/jack/tDNMS_EC_project/AB04_ephys/AB04_09092024_g0'
         )))
    model_types = pd.Series(
        ['shaping']*len(shaping_paths) + ['sl_shaping']*len(sl_shaping_paths),
        name='model_type').reset_index()
    model_types.columns= ['model_id'] + list(model_types.columns[1:])
    all_paths = pd.Series(shaping_paths + sl_shaping_paths,
                          index=pd.MultiIndex.from_frame(model_types))
 

    for directory in all_paths:
        data_path = Path(directory)
        spikes_file = data_path / 'spikes.h5'
        spikes_cache = 'spikes.h5'

        print(spikes_file)

        jP.set_rcParams(plt)
        dpi = 300

        with pd.HDFStore(spikes_file, 'r') as f:
            clusters = f.keys()
        p = ProgressBar(len(clusters))
        firing_rates = pd.Series(clusters, index=clusters).apply(smoothed_spikes, spikes_file=spikes_file, p=p)
        firing_rates.T.to_hdf(spikes_cache, key=data_path.parts[-1])

if __name__ == '__main__':
    main(sys.argv[1:])
