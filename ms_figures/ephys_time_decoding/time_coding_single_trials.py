# [Figures 6i]

import os
import sys

sys.path.append('../')

from sklearn import metrics
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

from pathlib import Path
import numpy as np
import pandas as pd
import itertools as it
import argparse

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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

def main(argv):

    jP.set_rcParams(plt)
    dpi = 300

    sl_shaping_paths = list(map(
        Path,
         ('/data1/dua/sl_spike_data/M2_2202025_g0',
          '/data1/dua/sl_spike_data/M2_2212025_g0',
          '/data1/dua/sl_spike_data/M1_2182025_g0',
          '/data1/dua/sl_spike_data/M1_2202025_g0',
          '/data1/dua/sl_spike_data/M4_2252025_g0',
          '/data1/dua/sl_spike_data/M4_2272025_g1',
          '/data1/dua/sl_spike_data/M5_2272025_g0',
          '/data1/dua/sl_spike_data/M5_3012025_g0')))

    shaping_paths = list(map(
        Path, 
        ('/data1/jack/tDNMS_EC_project/AC01_ephys/AC01_01292025_g0',
         '/heys-nas-LabData/jack/neuropix_data/j2_05092024/j2_05092024_g0',
         '/heys-nas-LabData/tDNMS_EC_project/data/AB02_ephys/AB02_08272024_g0',
         '/data1/jack/tDNMS_EC_project/AB02_ephys/AB02_08282024_g0',
         '/data1/jack/tDNMS_EC_project/AB04_ephys/AB04_09092024_g0')))
    model_types = pd.Series(
        ['shaping']*len(shaping_paths) + ['sl_shaping']*len(sl_shaping_paths),
        name='model_type').reset_index()
    model_types.columns= ['model_id'] + list(model_types.columns[1:])
    all_paths = pd.Series(shaping_paths + sl_shaping_paths,
                          index=pd.MultiIndex.from_frame(model_types))
    predictions = (all_paths / 'analysis/time_decoding.h5')\
                      .apply(pd.read_hdf, key='model_0')
    predictions = pd.concat(
        predictions.values, keys=pd.MultiIndex.from_tuples(predictions.keys(),
        names=predictions.index.names))

    predictions = predictions * 0.5
    idx = predictions.index.to_frame()
    idx['time_bin'] *= 0.5
    predictions.index = pd.MultiIndex.from_frame(idx)

    def calc_error(predictions):
        circ_error = lambda a, b, n: np.min(
            np.array(list(map(
                lambda x, y: (x-y),
                it.repeat(a), (b, b+n, b-n)))) ** 2)
        err = predictions.groupby('time_bin').apply(
            lambda x, circ_error=circ_error: x.map(
                lambda a, circ_error=circ_error: circ_error(
                    a, x.index.get_level_values('time_bin'), 20)))
        squared_error = err.droplevel(0).sort_index()
        squared_error.columns.name = 'decode_type'

        return squared_error
    sq_err = calc_error(predictions)

    x = sq_err.groupby(['model_type', 'type']).mean().stack().xs('SL', level='type')
    print(x.unstack().T.div(x.unstack().T.xs('SL')))
    x = sq_err.groupby(['model_type', 'type']).mean().stack().xs('LS', level='type')
    print(x.unstack().T.div(x.unstack().T.xs('LS')))
    x = sq_err.groupby(['model_type', 'type']).mean().stack().xs('SS', level='type')
    print(x.unstack().T.div(x.unstack().T.xs('SS')))

    path = Path('/analysis/ms_figures/ephys_time_singleTrials')
    path.mkdir(exist_ok=True, parents=True)

    PdfPlotter(path / 'bytrial_decoding.pdf', fixed_margins=jP.default_margins())
    plt.figure(figsize=(1.5, 2), dpi=300)
    ax = plt.gca()

    errs_ls = sq_err['SL'].xs('LS', level='type').groupby(['model_type', 'result','trial', 'model_id']).mean()
    errs_sl = sq_err['LS'].xs('SL', level='type').groupby(['model_type', 'result','trial', 'model_id']).mean()
    errs  = pd.concat((errs_ls, errs_sl))


    def box_plotter(X, i, colors, ax):
        flierprops={'marker': '.', 'markersize': 2, 'mfc': 'k'}
        boxprops={'facecolor': next(colors), 'edgecolor': 'k'}
        medianprops={'color': 'k'}
        ax.boxplot(
            [X], positions=[next(i)], widths=0.5, flierprops=flierprops,
            boxprops=boxprops, patch_artist=True, medianprops=medianprops)

    positions = iter([0, 1, 3, 4])

    box_plotter(errs['shaping']['H'].values, positions, iter(['tab:orange']), ax)
    box_plotter(errs['shaping']['M'].values, positions, iter(['tab:orange']), ax)
    box_plotter(errs['sl_shaping']['H'].values, positions, iter(['tab:green']), ax)
    box_plotter(errs['sl_shaping']['M'].values, positions, iter(['tab:green']), ax)
    jP.configure_spines(ax)
    ax.set_ylabel('Time Decoding MSE')
    ax.set_ylim(0, 85)
    ax.set_xticklabels(['Correct', 'Incorrect', 'Correct', 'Incorrect'], rotation=45, rotation_mode='anchor', ha='right', fontsize=5)

    errs.name = 'values'
    X = errs.reset_index()
    model = ols("values ~ C(model_type)*C(result)", X).fit()
    aov_table = anova_lm(model, typ=2)
    print(aov_table)
    tukey = pairwise_tukeyhsd(
        endog=X['values'], groups=list(map(str, zip(X['model_type'],
        X['result']))), alpha=0.05)
    print(tukey)
    print(tukey.pvalues)

    symbols = jP.significance_symbols(tukey.pvalues)
    lbl_y = errs.max() + jP.annotation_padding(ax, 0.08, axis='y')
    ax.text(0.5, lbl_y, symbols.iloc[0], ha='center', va='center', fontsize=5)
    ax.text(3.5, lbl_y, symbols.iloc[5], ha='center', va='center', fontsize=5)
    lbl_y = lbl_y + jP.annotation_padding(ax, 0.1, axis='y')
    jP.annotation(ax, (1, 3), lbl_y, symbols.iloc[3], va='center')
    lbl_y = lbl_y + jP.annotation_padding(ax, 0.1, axis='y')
    jP.annotation(ax, (0, 3), lbl_y, symbols.iloc[1], va='center')
    lbl_y = lbl_y + jP.annotation_padding(ax, 0.1, axis='y')
    jP.annotation(ax, (1, 4), lbl_y, symbols.iloc[4], va='bottom')
    lbl_y = lbl_y + jP.annotation_padding(ax, 0.1, axis='y')
    jP.annotation(ax, (0, 4), lbl_y, symbols.iloc[2], va='center')

    ax.text(
        0.5, 1, 'S/FT', color='tab:orange', transform=ax.get_xaxis_transform(),
        ha='center', va='bottom', clip_on=False)
    ax.text(
        3.5, 1, 'SL/FT', color='tab:green', transform=ax.get_xaxis_transform(),
        ha='center', va='bottom', clip_on=False)
    plt.show()

   
if __name__ == '__main__':
    main(sys.argv[1:])


