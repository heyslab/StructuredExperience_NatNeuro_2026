# [Figure 7d,e,f,g]
import sys
sys.path.append('../')
sys.path.append('/home/jack/code/ephys_tools/scripts/tdnms')

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
        .groupby(['mouse_name', 'session', 'trial', 'probe_type']).apply(list).reindex(idx)
    filled_licks = lick_times.apply(lambda x: x if type(x) == type([]) else []).values.tolist()
    ax.eventplot(filled_licks, color='k', lw=0.5)
    ax.set_xlim(0, 20)
    jP.configure_spines(ax)
    ax.set_ylim(-1, len(lick_times))
    ax.set_xticks(np.arange(5, 21, 5))


def load_data(path):
    bd = pd.read_hdf(path)
    bd['Lick_Start'] = bd['lick'].astype(int).diff() == 1
    return bd

def main(argv):
    jP.set_rcParams(plt)
    margins = jP.default_margins()
    margins['top'] = 50
    dpi = 300

    data_path = Path('/heys-nas-LabData/Cambria/Jack Project - tDNMS to probe (MM, LL)/BaselineAndProbes')
    mice = ['J3', 'J6', 'J7']
    mice = pd.Series(mice, index=pd.Index(mice, name='mouse_name'))
    data_paths = pd.concat((
            mice.apply(lambda x, p=data_path: list((p / x).glob('*LL/behavior.h5'))),
            mice.apply(lambda x, p=data_path: list((p / x).glob('*MM/behavior.h5'))),
            mice.apply(lambda x, p=data_path: list((p / x).glob('*XL/behavior.h5')))),
        keys=pd.Index(['LL', 'MM', 'XL'], name='probe_type'))
    data_paths = data_paths.apply(lambda x: pd.Series(x, index=pd.Index(np.arange(len(x)), name='session'))).stack()
    data_paths = data_paths.apply(pd.Series).stack()
    colors = {'LL': '#253D5B', 'MM': '#DA627D', 'XL': '#96ACB7'}

    bd = data_paths.apply(load_data)
    bd = pd.concat(bd.values, keys=bd.index)

    bd = bd.droplevel('time').reset_index()\
           .set_index(['mouse_name', 'session', 'trial_time', 'trial', 'trial_type', 'probe_type'])
    bd = bd[bd.index.get_level_values('trial').notna()]

    data_path = Path('/analysis/ms_figures/probe_mice')
    data_path.mkdir(exist_ok=True, parents=True)

    PdfPlotter(data_path / 'nonprobe_lick_raster.pdf', fixed_margins=margins)
    plt.figure(figsize=(4, 3), dpi=300)
    gs = gridspec.GridSpec(1, 3, wspace=0.45)

    axs = list(map(plt.subplot, gs))
    resp_eventplot(axs[0], bd.xs('SL', level='trial_type', drop_level=False))
    resp_eventplot(axs[1], bd.xs('LS', level='trial_type', drop_level=False))
    resp_eventplot(axs[2], bd.xs('SS', level='trial_type', drop_level=False))

    axs[0].set_ylabel('tDNMS Trials')
    axs[0].set_title('SL', color=TRIAL_COLORS['SL'], pad=2)
    axs[1].set_title('LS', color=TRIAL_COLORS['LS'], pad=2)
    axs[2].set_title('SS', color=TRIAL_COLORS['SS'], pad=2)

    for ax in axs:
        ax.set_xlabel('Time (s)')
    plt.show()

    PdfPlotter(data_path / 'nonprobe_lick_raster_4.5.pdf', fixed_margins=margins)
    plt.figure(figsize=(4.5, 3), dpi=300)
    gs = gridspec.GridSpec(1, 3, wspace=0.45)

    axs = list(map(plt.subplot, gs))
    resp_eventplot(axs[0], bd.xs('SL', level='trial_type', drop_level=False))
    resp_eventplot(axs[1], bd.xs('LS', level='trial_type', drop_level=False))
    resp_eventplot(axs[2], bd.xs('SS', level='trial_type', drop_level=False))

    axs[0].set_ylabel('tDNMS Trials')
    axs[0].set_title('SL', color=TRIAL_COLORS['SL'], pad=2)
    axs[1].set_title('LS', color=TRIAL_COLORS['LS'], pad=2)
    axs[2].set_title('SS', color=TRIAL_COLORS['SS'], pad=2)

    for ax in axs:
        ax.set_xlabel('Time (s)')
    plt.show()

    PdfPlotter(data_path / 'allprobe_lick_raster.pdf', fixed_margins=margins)
    plt.figure(figsize=(2.5, 3), dpi=300)
    gs = gridspec.GridSpec(3, 1)

    axs = list(map(plt.subplot, gs))
    resp_eventplot(axs[0], bd.xs('LL', level='trial_type', drop_level=False))
    resp_eventplot(axs[1], bd.xs('MM', level='trial_type', drop_level=False))
    resp_eventplot(axs[2], bd.xs('XL', level='trial_type', drop_level=False))

    axs[1].set_yticks([])
    axs[2].set_yticks([])
    axs[0].set_ylabel('Trials')

    axs[0].set_title('LL', color=TRIAL_COLORS['LL'], pad=2)
    axs[1].set_title('MM', color=TRIAL_COLORS['MM'], pad=2)
    axs[2].set_title('XL', color=TRIAL_COLORS['XL'], pad=2)

    for ax in axs:
        ax.set_xlabel('Time (s)')
    axs[1].set_yticks([])
    axs[2].set_yticks([])
    plt.show()
 
    for trial_type in ('LL', 'MM', 'XL'):
        PdfPlotter(data_path / f'allprobe_lick_raster.{trial_type}.pdf', fixed_margins=margins)
        plt.figure(figsize=(2.5, 1), dpi=300)
        ax = plt.gca()

        resp_eventplot(ax, bd.xs(trial_type, level='trial_type', drop_level=False))
        ax.set_ylabel(f'{trial_type} Trials', color=TRIAL_COLORS[trial_type])
        ax.set_xlabel('Time (s)')
        plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
