# [Figure 6e]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import itertools as it

import spikeinterface.full as si

from analysis_tools.mpl_helpers import PdfPlotter
from analysis_tools.progressbar import ProgressBar
import analysis_tools.jPlots as jP
#import expt_classes

plt.rcParams.update({
    "figure.facecolor":  (1.0, 0.0, 0.0, 0.0),  # red   with alpha = 30%
    "axes.facecolor":    (0.0, 1.0, 0.0, 0.0),  # green with alpha = 50%
    "savefig.facecolor": (0.0, 0.0, 1.0, 0.0),  # blue  with alpha = 20%
    })

TRIAL_COLORS = {
    'LS': '#eb0d8c',
    'SL': '#2bace2',
    'SS': '#f89521',
    'LL': '#2b958c',
    'S' : '#f89521'}


def find_spike_paths(path):
    if (path / 'spike_times.npy').exists():
        return [path]

    subdirs = [p for p in path.iterdir() if p.is_dir()]
    if len(subdirs):
        return list(it.chain.from_iterable(map(find_spike_paths, subdirs)))

    return []


def plot_gen(counts, dpi=300):
    plt.figure(figsize=(2.5, 1), dpi=dpi)
    gs = iter(gridspec.GridSpec(1, 3, wspace=0.3, hspace=0.2))
    while True:
        gss = next(gs).subgridspec(len(counts) + 1, 1, height_ratios=[25] + counts.values.tolist())

        yield list(map(plt.subplot, gss))


def plot_cluster(axs, cluster, spikes_file, odors, trials):
    spikes = pd.read_hdf(spikes_file, key=cluster)
    spikes = spikes.dropna(axis=0).set_index(
        ['trial', 'trial_type', 'result'])
    spikes = spikes.reset_index().where(
        spikes.reset_index()['trial_type'] != 'None'
        ).dropna(how='all').set_index(spikes.index.names)

    def plot_avg_line(ax, spike_times, color, n_trials):
        nbins = 60
        event_count = pd.cut(
            spike_times, np.linspace(0, 20, nbins + 1)
            ).value_counts().sort_index().values / (20 / nbins)

        x_vals = np.linspace(0, 20-20/nbins, nbins) + 20/nbins/2
        spike_rate = pd.Series(
            event_count / n_trials
            ).rolling(
                16, win_type='gaussian', center=True, min_periods=1
                ).mean(std=2)
        ax.plot(x_vals, spike_rate.values,
                 c=color, clip_on=False)
        ax.set_xlim(x_vals[0], x_vals[-1])
        ax.set_axis_off()

    def plot_trial_type(spikes, axs_dict, trials, odors):
        trial_type = spikes.index.unique('trial_type')[0]
        ax = axs_dict[trial_type]
        valid_trial_spikes = spikes.droplevel(
            ['trial_type', 'result'])[
                spikes.index.unique('trial').intersection(trials[trial_type])]

        events = valid_trial_spikes.groupby(
            'trial').apply(list).apply(
                lambda x: [a for a in x if ~np.isnan(a)]).apply(
                    lambda x: x if type(x) is type([]) else [])

        ax.eventplot(events, color='k', alpha=0.8, lw=0.25)
        ax.set_xticks([])
        ax.axvspan(
            odors[0][trial_type, 'start'], odors[0][trial_type, 'stop'], alpha=0.2,
            color='tab:blue')
        ax.axvspan(
            odors[1][trial_type, 'start'], odors[1][trial_type, 'stop'], alpha=0.2,
            color='tab:blue')
        ax.set_xlim(0, 20)
        ax.set_ylim(-0.5, len(trials[trial_type] + 0.5))
        plot_avg_line(
            axs_dict['tuning_curves'], valid_trial_spikes,
            TRIAL_COLORS[trial_type], len(trials[trial_type]))
        jP.configure_spines(ax)


    def plot_trial_type_hm(spikes, axs_dict, trials, odors):
        trial_type = spikes.index.unique('trial_type')[0]
        ax = axs_dict[trial_type]
        valid_trial_spikes = spikes.droplevel(
            ['trial_type', 'result'])[
                spikes.index.unique('trial').intersection(trials[trial_type])]
        plot_avg_line(
            axs_dict['tuning_curves'], valid_trial_spikes,
            TRIAL_COLORS[trial_type], len(trials[trial_type]))

        def create_trial_tc(spike_times, nbins=40, n_trials=1):
            event_count = pd.cut(
                spike_times, np.linspace(0, 20, nbins + 1)
                ).value_counts().sort_index().values / (nbins / 20)

            x_vals = np.linspace(0, 20-nbins/20, nbins) + nbins/20/2
            spike_rate = pd.Series(
                event_count / n_trials
                ).rolling(
                    16, win_type='gaussian', center=True, min_periods=1
                    ).mean(std=4)
            return spike_rate

        rate_maps = valid_trial_spikes.groupby('trial').apply(create_trial_tc)
        rate_maps = rate_maps.unstack(1).reindex(trials[trial_type], level='trial')
        norm_map = rate_maps.subtract(rate_maps.mean(1), axis=0).div(rate_maps.std(1), axis=0).fillna(0)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap', ('#ffffff', TRIAL_COLORS[trial_type]))
        axs_dict[trial_type].imshow(norm_map, vmin=0, aspect='auto', cmap=cmap, extent=[0, 20, 0, len(trials[trial_type])])

    axs_dict = {k: v for (k, v) in
                zip(['tuning_curves'] + list(trials.index.get_level_values('trial_type')), axs)}
    spikes['trial_time'].groupby('trial_type').apply(
        plot_trial_type, axs_dict=axs_dict, trials=trials, odors=odors)
    list(map(lambda x: x.spines['bottom'].set_visible(False), axs[:-1]))
    list(map(lambda x: x.spines['top'].set_visible(False), axs[2:]))
    list(map(lambda x: x.set_xticks([]), axs[:-1]))
    jP.configure_spines(axs[0])



def main(argv):
    data_paths = ['/data3/jack/tDNMS_EC_project/AC01_ephys/AC01_01292025_g0'] * 3
    data_paths[1] = '/data3/jack/tDNMS_EC_project/AB02_ephys/AB02_08282024_g0'
    clusters = [
        'group_1/cluster_294',
        #'group_1/cluster_311',
        'group_0/cluster_168',
        'group_1/cluster_361']
    model_types = ['shaping'] * 3

    data_paths = data_paths + ['/data1/dua/sl_spike_data/M2_2212025_g0'] * 3
    clusters = clusters + [
            'group_0/cluster_104',
            'group_0/cluster_259',
            'group_0/cluster_93']
    model_types = model_types + ['ls_shaping'] * 3
    ylims = [(0, 10)] * 6
    ylims[0] = (0, 10 * 0.6)
    ylims[1] = (0, 10 * 0.6)
    ylims[3] = (0, 10 * 0.6)
    units = pd.DataFrame({'path':data_paths, 'unit': clusters, 'ylims': ylims}, index=model_types)

    hit_only = False
    save_path = Path('/analysis/ms_figures/ephys_updated')
    save_path.mkdir(exist_ok=True, parents=True)

    jP.set_rcParams(plt)
    dpi = 300


    plot_order = ['SL', 'LS', 'SS']

    #margins={'left': 80, 'right': 80, 'top': 80, 'bottom': 80}
    margins = jP.default_margins()
    PdfPlotter(save_path / 'example_units_tdnms.pdf', fixed_margins=margins)

    plt.figure(figsize=(2.7, 2.75), dpi=dpi)
    gs0 = gridspec.GridSpec(2, 1)
    p = ProgressBar(len(clusters))
    def plot_mapper(path, unit, specs, counter, title, ylims, p=ProgressBar.nocount(), labelx=False):
        i = next(counter)
        spikes_file = Path(path) / 'spikes.h5'
        behavior_file = Path(path) / 'behavior.h5'

        bd = pd.read_hdf(behavior_file)
        if hit_only:
            bd = bd.where(bd['result'].apply(lambda x: x in ('H', 'CR'))).dropna(how='all')

        trials = bd.set_index(['trial_type'])['trial'].groupby('trial_type').unique()
        trials.reindex(plot_order)
        counts = trials.apply(len)
        odor_cues = bd.reset_index().set_index(
            ['trial_time', 'trial', 'trial_type']
            )['odor'].groupby(['trial_type']).apply(
                lambda x: x.xs(x[0].index.unique('trial')[0], level='trial')
                ).droplevel(-1).unstack('trial_type').fillna(False)

        odor_transitions = pd.concat(
            (odor_cues.apply(lambda x: x[x.fillna(False).astype(int).diff() > 0].index),
             odor_cues.apply(lambda x: x[x.fillna(False).astype(int).diff() < 0].index)),
            keys=('start', 'stop')).unstack(0).T

        gs = specs[i].subgridspec(len(counts) + 1, 1, height_ratios=[25] + counts.values.tolist()) 
        ax = list(map(plt.subplot, gs))
        #ax[0].set_title(f'Neuron {i + 1}', pad=-1)
        p.increment()
        plot_cluster(
            ax, unit, spikes_file, odor_transitions, trials)
        
        list(map(lambda x: x.set_yticks([]), ax))
        if i == 0:
            ax[2].set_ylabel(title)
            jP.set_ylabel_position(ax[2], nlines=2.5)

        ax[-1].set_xlabel('Time (s)')

        ax[0].set_axis_on()
        ax[0].spines['left'].set_bounds(1, 6)
        ax[0].set_yticks([3])
        ax[0].tick_params(axis='y', length=0)
        if i == 0:
            ax[0].set_yticklabels(['5\nsp/s'], fontsize=5, va='center')
        else:
            #ax[0].spines['left'].set_visible(False)
            ax[0].set_yticks([])
        ax[0].set_ylim(ylims)
        if labelx:
            ax[-1].set_xticks([10, 20])

    counter = it.count()
    gs = gs0[0].subgridspec(1, 3, wspace=0.3, hspace=0.2)
    units.xs('shaping').apply(
        lambda x, specs=gs, counter=counter:
            plot_mapper(**x, title='LS+SL\nSpikes/Trial', specs=gs, counter=counter), axis=1)

    counter = it.count()
    gs = gs0[1].subgridspec(1, 3, wspace=0.3, hspace=0.2)
    units.xs('ls_shaping').apply(
        lambda x, specs=gs, counter=counter:
            plot_mapper(**x, title='SL Only\nSpikes/Trial', specs=gs, counter=counter, labelx=True), axis=1)
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
