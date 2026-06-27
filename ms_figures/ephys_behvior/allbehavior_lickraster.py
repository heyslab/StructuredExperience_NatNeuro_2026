# [Figure S9c,d]
import sys
sys.path.append('../')

import os
import glob
import pandas as pd
import numpy as np
import itertools as it
import argparse
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import analysis_tools.jPlots as jP
from analysis_tools.mpl_helpers import PdfPlotter
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d

TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'LL': '#253D5B', 'S': 'k', 'L': 'r',
                'MM': '#DA627D', 'XL': '#96ACB7'}

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  
    "axes.facecolor":    (1.0, 1.0, 1.0, 0.0),  
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),
    })

def resp_eventplot(ax, bd):
    trial_type = bd.index.unique('trial_type')[0]

    odors = bd.groupby('trial_time')['odor'].mean()
    odors = odors > 0.5
    odor_diff = odors.diff()
    cue_times = odor_diff.where(odor_diff != 0).dropna().index.to_list()
    ax.axvspan(cue_times[0], cue_times[1], color=TRIAL_COLORS[trial_type], alpha=0.5)
    if len(cue_times) > 2:
        ax.axvspan(cue_times[2], cue_times[3], color=TRIAL_COLORS[trial_type], alpha=0.5)

    trial_data = bd.xs(trial_type, level='trial_type')
    licks = trial_data['Lick_Start'].where(trial_data['Lick_Start'] == 1).dropna()
    idx = trial_data.droplevel('trial_time').groupby(trial_data.droplevel('trial_time').index).head(1).index
    lick_times = licks.index.to_frame()['trial_time']\
        .groupby(['model_id', 'trial']).apply(list).reindex(idx)
    filled_licks = lick_times.apply(lambda x: x if type(x) == type([]) else []).values.tolist()
    ax.eventplot(filled_licks, color='k', lw=0.5)
    ax.set_xlim(0, 20)
    jP.configure_spines(ax)
    ax.set_ylim(-1, len(lick_times))
    ax.set_xticks(np.arange(5, 21, 5))


def load_data(path):
    bd = pd.read_hdf(path)
    if 'lick' in bd.columns:
        bd['Lick_Start'] = bd['lick'].astype(int).diff() == 1
    elif 'licking' in bd.columns:
        bd['Lick_Start'] = bd['licking'].astype(int).diff() == 1
    return bd

def main(argv):
    jP.set_rcParams(plt)
    margins = jP.default_margins()
    margins['top'] = 50
    dpi = 300

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
        ('/data3/jack/tDNMS_EC_project/AC01_ephys/AC01_01292025_g0',
         '/heys-nas-LabData/jack/neuropix_data/j2_05092024/j2_05092024_g0',
         '/data3/jack/tDNMS_EC_project/AB02_ephys/AB02_08272024_g0',
         '/data3/jack/tDNMS_EC_project/AB02_ephys/AB02_08282024_g0',
         '/data3/jack/tDNMS_EC_project/AB04_ephys/AB04_09092024_g0')))
    model_types = pd.Series(
        ['shaping']*len(shaping_paths) + ['sl_shaping']*len(sl_shaping_paths),
        name='model_type').reset_index()
    model_types.columns= ['model_id'] + list(model_types.columns[1:])
    all_paths = pd.Series(shaping_paths + sl_shaping_paths,
                          index=pd.MultiIndex.from_frame(model_types))
    bd = (all_paths / 'behavior.h5').apply(load_data)
    bd = pd.concat(bd.values, keys=bd.index)

    bd = bd.droplevel(-1).reset_index()\
           .set_index(['model_id', 'model_type', 'trial_time', 'trial', 'trial_type'])
    bd = bd[bd.index.get_level_values('trial').notna()]
    all_bd = bd.copy()


    path = Path('/analysis/ms_figures/ephys')
    path.mkdir(exist_ok=True, parents=True)

    for model_type in ('shaping', 'sl_shaping'):
        bd = all_bd.xs(model_type, level='model_type')
        PdfPlotter(path / f'alllick_raster.{model_type}.pdf', fixed_margins=margins)
        plt.figure(figsize=(2.5, 2.5), dpi=300)
        gs = gridspec.GridSpec(1, 3, wspace=0.45)

        axs = list(map(plt.subplot, gs))
        resp_eventplot(axs[0], bd.xs('SL', level='trial_type', drop_level=False))
        resp_eventplot(axs[1], bd.xs('LS', level='trial_type', drop_level=False))
        resp_eventplot(axs[2], bd.xs('SS', level='trial_type', drop_level=False))

        axs[0].set_ylabel('Trials')
        axs[0].set_title('SL', color=TRIAL_COLORS['SL'], pad=2)
        axs[1].set_title('LS', color=TRIAL_COLORS['LS'], pad=2)
        axs[2].set_title('SS', color=TRIAL_COLORS['SS'], pad=2)

        for ax in axs:
            ax.set_xlabel('Time (s)')
            ax.set_xticks([10, 20])
            ax.set_yticks(np.arange(50, ax.get_ylim()[-1], 50))
        plt.show()

        PdfPlotter(path / f'alllick_raster.{model_type}_2.75.pdf', fixed_margins=margins)
        plt.figure(figsize=(2.75, 2.5), dpi=300)
        gs = gridspec.GridSpec(1, 3, wspace=0.45)

        axs = list(map(plt.subplot, gs))
        resp_eventplot(axs[0], bd.xs('SL', level='trial_type', drop_level=False))
        resp_eventplot(axs[1], bd.xs('LS', level='trial_type', drop_level=False))
        resp_eventplot(axs[2], bd.xs('SS', level='trial_type', drop_level=False))

        axs[0].set_ylabel('Trials')
        axs[0].set_title('SL', color=TRIAL_COLORS['SL'], pad=2)
        axs[1].set_title('LS', color=TRIAL_COLORS['LS'], pad=2)
        axs[2].set_title('SS', color=TRIAL_COLORS['SS'], pad=2)

        for ax in axs:
            ax.set_xlabel('Time (s)')
            ax.set_xticks([10, 20])
            ax.set_yticks(np.arange(50, ax.get_ylim()[-1], 50))
        plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
