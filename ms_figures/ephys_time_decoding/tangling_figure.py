#[Figure S8e]
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
    path = Path('/analysis/ms_figures/ephys_tangling')
    path.mkdir(exist_ok=True, parents=True)
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
    tangling = (all_paths / 'analysis/tangling.h5')\
                      .apply(pd.read_hdf, key='tangling')
    print(tangling.mean(1).groupby('model_type').mean())
    print(scipy.stats.ttest_ind(*tangling.mean(1).groupby('model_type').apply(list).values))
    trial_tangling = tangling.T.stack().stack().groupby(['model_id', 'model_type', 'trial'])\
                             .mean().groupby('model_type').apply(list)
    pvalue = scipy.stats.ttest_ind(*trial_tangling.values)

    PdfPlotter(path / 'tangling_violin.pdf', fixed_margins=jP.default_margins())
    plt.figure(figsize=(1.5, 2.5), dpi=300)
    ax = plt.gca()

    color='tab:orange'
    parts = ax.violinplot([trial_tangling['shaping']], positions=[0], widths=0.25)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
    parts['cmaxes'].set_edgecolor(color)
    parts['cmins'].set_edgecolor(color)
    parts['cbars'].set_edgecolor(color)

    color='tab:green'
    parts = ax.violinplot([trial_tangling['sl_shaping']], positions=[1], widths=0.25)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
    parts['cmaxes'].set_edgecolor(color)
    parts['cmins'].set_edgecolor(color)
    parts['cbars'].set_edgecolor(color)
    jP.configure_spines(ax)
    ax.ticklabel_format(axis='y', scilimits=(0,0))
    ax.set_xticks([0, 1])
    ax.set_yticks([1e3, 2e3])
    ax.set_ylim([0, 2.2e3])
    ax.set_xticklabels(['SL+LS', 'SL Only'], rotation=40, rotation_mode='anchor', ha='right')
    ylbl = trial_tangling.apply(np.max).max() + jP.annotation_padding(ax, 0.08)
    ax.set_ylabel('Mean Trial Tangling')
    sigs = jP.significance_symbols([pvalue.pvalue])
    jP.annotation(ax, [0, 1], ylbl, sigs.iloc[0], va='bottom')
    print(pvalue)
    plt.show()

    PdfPlotter(path / 'tangling_box.pdf', fixed_margins=jP.default_margins())
    plt.figure(figsize=(2, 3), dpi=300)
    ax = plt.gca()

    color='tab:orange'
    flierprops={'marker': '.', 'markersize': 2, 'mfc': 'k'}
    boxprops={'facecolor': color, 'edgecolor': 'k'}
    medianprops={'color': 'k'}
    ax.boxplot(
        [trial_tangling['shaping']], positions=[0], widths=0.5, flierprops=flierprops,
        boxprops=boxprops, patch_artist=True, medianprops=medianprops)

    color='tab:green'
    flierprops={'marker': '.', 'markersize': 2, 'mfc': 'k'}
    boxprops={'facecolor': color, 'edgecolor': 'k'}
    medianprops={'color': 'k'}
    ax.boxplot(
        [trial_tangling['sl_shaping']], positions=[1], widths=0.5, flierprops=flierprops,
        boxprops=boxprops, patch_artist=True, medianprops=medianprops)

    jP.configure_spines(ax)
    ax.ticklabel_format(axis='y', scilimits=(0,0))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['SL+LS', 'SL Only'], rotation=45, rotation_mode='anchor', ha='right')
    ylbl = trial_tangling.apply(np.max).max() + jP.annotation_padding(ax, 0.08)
    sigs = jP.significance_symbols([pvalue.pvalue])
    jP.annotation(ax, [0, 1], ylbl, sigs.iloc[0])

    plt.show()



if __name__ == '__main__':
    main(sys.argv[1:])


