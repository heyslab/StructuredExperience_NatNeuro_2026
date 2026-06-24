# [Figure S1b,c,d,e]
import os
import datetime
import sys

sys.path.append('../')

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import functools
import argparse
import itertools as it

import matplotlib.gridspec as gridspec

from analysis_tools.mpl_helpers import PdfPlotter
import analysis_tools.jPlots as jP
from analysis_tools.progressbar import ProgressBar

from classes.models import LeakyRNN, LeakyRNNCell
from classes.datagen import genFactory
import models_database as mdb

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  
    "axes.facecolor":    (1.0, 1.0, 1.0, 0.0),  
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0)})


TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'S': '#f89521'}
SHAPING_COLORS = {'no_shaping': 'tab:blue', 'shaping': 'tab:orange'}


def error_compare_plot(predictions, ax):
    def calc_trial_max(X):
        trial_type = X.index.unique('type')[0]
        if trial_type == 'SS':
            return pd.DataFrame([X.max(), None], index=('out', 'in'))
        elif trial_type == 'LS':
            max_out = max(X.loc['0s':'11s'].max(), X.loc['16s':'20s'].max())
            max_in = X.loc['13s':'15s'].max()
            return  pd.Series([max_out, max_in], index=('out', 'in'))
        elif trial_type == 'SL':
            max_out = max(X.loc['0s':'8s'].max(), X.loc['16s':'20s'].max())
            max_in = X.loc['13s':'15s'].max()
            return pd.Series([max_out, max_in], index=('out', 'in'))

    trial_maxs = predictions.stack()\
                            .groupby(['model_type', 'trial'])\
                            .apply(calc_trial_max)[0].unstack()
    maxes_list = trial_maxs.groupby(['model_type'])\
                           .apply(lambda x: pd.DataFrame(
                                x.values.T, index=x.keys())).apply(list, axis=1) 
    fixed_trial_maxes = maxes_list.apply(
        lambda x: [a for a in x if np.isfinite(a)])

    step_size = 0.8
    def violin_plot(X, ctr, ax, color, p=ProgressBar.nocount()):
        p.increment()
        offset = 0.5
        i = next(ctr)
        parts = ax.violinplot([X['out'], X['in']], positions=[i+0, i+offset], widths=0.25)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
        parts['cmaxes'].set_edgecolor(color)
        parts['cmins'].set_edgecolor(color)
        parts['cbars'].set_edgecolor(color)

        ax.set_xticks(list(ax.get_xticks()) + [i, i+offset])
    
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlim(-0.25, len(fixed_trial_maxes)/2 * step_size + 1.25)
    ctr = it.count(step=step_size)
    fixed_trial_maxes.xs('no_shaping', level='model_type', drop_level=False)\
                     .unstack().apply(violin_plot, ctr=ctr, ax=ax, axis=1,
                                      color='tab:blue')
    next(ctr)
    fixed_trial_maxes.xs('shaping', level='model_type', drop_level=False)\
                     .unstack().apply(violin_plot, ctr=ctr, ax=ax, axis=1,
                                      color='tab:orange')
    ax.set_ylim(0, 1)
    ax.set_xticklabels(
        [['Error', 'Response'][i%2] for i, _ in enumerate(ax.get_xticks())],
        rotation=35, rotation_mode='anchor', ha='right', fontsize=5)
    ax.set_ylabel('Trial Peak', labelpad=0)
    jP.configure_spines(ax, fix_ylabel=False)

    bounds = ax.get_xlim()
    xticks = ax.get_xticks()
    ax.text(bounds[0]+0.05, 1, 'No\nShaping', color='tab:blue', ma='center',
            transform=ax.get_xaxis_transform(), ha='left', va='bottom', fontsize=5)

    ax.text(bounds[1]+0.25, 1, 'Shaping\n+ Full Task',
            color='tab:orange', transform=ax.get_xaxis_transform(), ha='right',
            va='bottom', ma='center', fontsize=5)

    sorted_maxes = fixed_trial_maxes.apply(sorted)
    def calculate_cutoff(maxs):
        scores = []
        for i in range(1, len(maxs['out'])):
            out_corr = len(maxs['out']) - i + 1
            scores.append(
                np.sum(np.array(maxs['in']) >
                       maxs['out'][-i]) + out_corr)

        threshold = maxs['out'][-np.argmax(scores) - 1]
        score = np.max(scores)/(len(maxs['in']) + len(maxs['out']))
        if score == 1:
            threshold = (max(maxs['out']) + min(maxs['in'])) / 2
        return threshold, score

    def draw_threshold_error(ax, score, threshold, maxs, color, xticks):
        if score < 1:
            x_display = (ax.transData.transform((xticks[0]-0.2, 0))[0],
                         ax.transData.transform((xticks[0]+0.2, 0))[0])
            x_vals = (ax.transAxes.inverted().transform((x_display[0], 0))[0],
                      ax.transAxes.inverted().transform((x_display[1], 0))[0])
            ax.axhspan(threshold, np.max(maxs['out']), xmin=x_vals[0],
                       xmax=x_vals[1], color='red', alpha=0.15, zorder=-1e3)
            x_display = (ax.transData.transform((xticks[1]-0.2, 0))[0],
                         ax.transData.transform((xticks[1]+0.2, 0))[0])
            x_vals = (ax.transAxes.inverted().transform((x_display[0], 0))[0],
                      ax.transAxes.inverted().transform((x_display[1], 0))[0])
            ax.axhspan(np.min(maxs['in']), threshold, xmin=x_vals[0],
                       xmax=x_vals[1], color='red', alpha=0.15, zorder=-1e3)

    threshold_no, score = calculate_cutoff(sorted_maxes['no_shaping'])
    draw_threshold_error(ax, score, threshold_no, sorted_maxes['no_shaping'],
                         SHAPING_COLORS['no_shaping'], xticks[:2])
    ax.axhline(threshold_no, xmin=0, xmax=0.475, color=SHAPING_COLORS['no_shaping'])
    ax.text(0.525, threshold_no, f'{int(np.round(score*100))}%',
            transform=ax.get_yaxis_transform(), ha='right', va='bottom',
            fontsize=5, color=SHAPING_COLORS['no_shaping'])

    threshold, score = calculate_cutoff(sorted_maxes['shaping'])
    draw_threshold_error(ax, score, threshold, sorted_maxes['shaping'],
                         SHAPING_COLORS['shaping'], xticks[2:])
    ax.axhline(threshold, xmin=0.525, xmax=1, color=SHAPING_COLORS['shaping'])
    ax.text(1.025, threshold, f'{int(np.round(score*100))}%',
            transform=ax.get_yaxis_transform(), ha='right', va='bottom',
            fontsize=5, color=SHAPING_COLORS['shaping'])

    ax.set_yticks([0.2, 0.6, 1.0])
    ax.set_yticklabels([0.2, 0.6, 1.0], fontsize=5)
    return threshold_no, threshold


def load_X2(noise_level, cache_path, n_trials=15):
    cache_key = f'/X_{int(noise_level*100)}'
    if cache_path is not None and cache_path.is_file():
        with pd.HDFStore(cache_path, 'r') as f:
            if cache_key in f.keys():
                return pd.read_hdf(cache_path, key=cache_key)

    trial_gen = genFactory.create(
        'just_short_match', input_noise=noise_level, batch_size=1,
        n_blocks=1)

    X2 = trial_gen.generate_trials(n_trials)
    if cache_path is not None:
        X2.to_hdf(cache_path, key=cache_key)
    return X2


def load_predictions(X2, models, cache_path, overwrite=False):
    noise_level = X2.index.unique('noise')[0]
    cache_key = f'/predictions_{int(noise_level*100)}'
    if cache_path is not None and cache_path.is_file() and not overwrite:
        with pd.HDFStore(cache_path, 'r') as f:
            if cache_key in f.keys():
                return pd.read_hdf(cache_path, key=cache_key)

    formatted_X = np.expand_dims(X2[['light', 'odor']], 0)
    predictions = models.apply(
        lambda m, X=formatted_X, i=X2.index:
            pd.Series(m.predict(X).squeeze(), index=i)).T

    time_idx = pd.to_timedelta(predictions.index.get_level_values('idx')/10,
                               unit='s')
    idx = predictions.index.to_frame()
    idx['time'] = time_idx
    predictions.index = pd.MultiIndex.from_frame(
        idx[['time'] + predictions.index.names])

    if cache_path is not None:
        predictions.to_hdf(cache_path, key=cache_key)

    return predictions


def main(argv):
    models = pd.Series(
        [9, 132],  index=pd.Index(['no_shaping', 'shaping'],
        name='model_type'))

    margins = jP.default_margins()

    model_infos = models.apply(mdb.get_model).apply(pd.Series)
    model_infos = pd.concat(
        (model_infos, models.apply(mdb.get_model_attributes)), axis=1)

    path = Path('/analysis/ms_figures/noisy_error')
    cache_path = Path('extra_short_data.hdf')
    jP.make_folder(path)

    dpi = 300
    jP.set_rcParams(plt)

    trained_noise = model_infos['input_noise'].head(1)[0]
    noise_levels = [np.round(trained_noise * x, 2) for x in [1, 4, 7, 10]]

    X2 = pd.concat(list(map(load_X2, noise_levels, it.repeat(cache_path))),
                   keys=pd.Index(noise_levels, name='noise'))
    models = model_infos['path'].apply(tf.keras.models.load_model)
    predictions = X2.groupby('noise').apply(
        load_predictions, models=models, cache_path=cache_path,
        overwrite=False).droplevel(0)

    adj_margins = margins.copy()
    adj_margins['bottom'] = 110

    for noise_level in noise_levels:
        PdfPlotter(path / f'noisy_errors.{noise_level}.pdf', fixed_margins=adj_margins)
        plot_predictions = predictions.xs(noise_level, level='noise', drop_level=False)\
                                      .reindex(np.arange(0, 12), level='trial')

        cue_transitions = X2.reindex(plot_predictions.droplevel('time').index)['cues'].diff()
        cue_start = cue_transitions[cue_transitions == 1]\
                                   .index.to_series().apply(cue_transitions.index.get_loc)
        cue_stop = cue_transitions[cue_transitions == -1]\
                                   .index.to_series().apply(cue_transitions.index.get_loc)
        cue_times = pd.concat(
            (cue_start.droplevel('idx'), cue_stop.droplevel('idx')),
            keys=('start', 'stop'), axis=1)/10

        target = X2['response'].reindex(
            plot_predictions.index.unique('trial'), level='trial').xs(noise_level, level='noise')

        plt.figure(figsize=(3.25, 1.5), dpi=dpi)
        gs = gridspec.GridSpec(2, 2, hspace=0.05, wspace=0.3, width_ratios=(3, 1.1))
        ax_dict = {'no_shaping': plt.subplot(gs[0, 0]), 'shaping': plt.subplot(gs[1, 0])}

        def plot_cues(cue_times, ax):
            plot_args = {
                'color': TRIAL_COLORS[cue_times.index.unique('type')[0]],
                'alpha': 0.15,
                'zorder': -100}
            cue_times.apply(
                lambda x, ax=ax, plot_args=plot_args:
                    ax.axvspan(*x, **plot_args), axis=1)

        def plot_model_behavior(predictions, ax_dict, cue_times, target, plot_cues=plot_cues):
            model_type = predictions.name
            ax = ax_dict[model_type]
            cue_times.groupby('type').apply(plot_cues, ax=ax)

            ax.plot(np.arange(target.shape[0])/10, target.values, c='k',
                ls='-', label='target')
            ax.plot(np.arange(predictions.shape[0])/10, predictions.values, c='r',
                lw=1, label='response')
            jP.configure_spines(ax, fix_xlabel=False)
            ax.set_yticks([])
            ax.set_xlim(0, len(predictions)/10)

        plot_predictions[['no_shaping', 'shaping']].apply(
            plot_model_behavior, ax_dict=ax_dict, cue_times=cue_times, target=target)
        ax_dict['no_shaping'].spines['bottom'].set_visible(False)
        ax_dict['no_shaping'].set_xticks([])
        ax_dict['shaping'].tick_params(axis='x', labelsize=5)
        ax_dict['shaping'].set_xlabel('Time (s)')

        ax_dict['no_shaping'].text(
            -0.01, 0.5, 'No\nShaping', color=SHAPING_COLORS['no_shaping'],
            fontsize=5, va='center', ha='right', transform=ax_dict['no_shaping'].transAxes, rotation=90, ma='center')
        ax_dict['shaping'].text(
            -0.01, 0.5, 'With\nShaping', color=SHAPING_COLORS['shaping'],
            fontsize=5, va='center', ha='right', transform=ax_dict['shaping'].transAxes, rotation=90, ma='center')
        
        ax_dict['shaping'].text(
            0.03, 0.55, r'$\xi_{in} \sim \cal{N}(0, ' + str(noise_level) + r')$',
            transform=ax_dict['shaping'].figure.transFigure, rotation=90,
            ha='center', va='center')

        legend_lines = [matplotlib.lines.Line2D([0], [0], color='k'),
                        matplotlib.lines.Line2D([0], [0], color='r')]
        ax_dict['no_shaping'].legend(
            legend_lines, ['target', 'response'], loc='lower right',
            bbox_to_anchor=(1.01, 0.9), ncols=2)

        thresholds = error_compare_plot(predictions.xs(noise_level, level='noise'), plt.subplot(gs[:, 1]))
        ax_dict['no_shaping'].axhline(thresholds[0], ls='--', color='k', lw=0.5)
        ax_dict['shaping'].axhline(thresholds[1], ls='--', color='k', lw=0.5)
        plt.show()


    
if __name__ == '__main__':
    main(sys.argv[1:])
