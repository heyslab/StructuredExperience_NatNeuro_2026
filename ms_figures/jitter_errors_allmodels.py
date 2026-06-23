# [Figure 1l,m]
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
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy

import matplotlib.gridspec as gridspec

from analysis_tools.mpl_helpers import PdfPlotter
import analysis_tools.jPlots as jP
from analysis_tools.progressbar import ProgressBar

from classes.models import LeakyRNN, LeakyRNNCell
from classes.datagen import genFactory, tDNMSTemporalJitter
import models_database as mdb

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  
    "axes.facecolor":    (1.0, 1.0, 1.0, 0.0),  
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0)})


TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'S': '#f89521'}
SHAPING_COLORS = {'no_shaping': 'tab:blue', 'shaping': 'tab:orange'}


def error_compare(predictions):
    def calc_trial_max(X):
        trial_type = X.index.unique('type')[0]
        if trial_type == 'SS':
            return pd.DataFrame([X.max(), None], index=('out', 'in'))

        cue_transitions = X.index.to_frame()['cues'].diff()
        cue_start = X[cue_transitions == 1]\
                                   .index.to_frame()['time']\
                                   .droplevel(['time', 'idx', 'cues'])
        cue_stop = X[cue_transitions == -1]\
                                  .index.to_frame()['time']\
                                  .droplevel(['time', 'idx', 'cues'])
        cue_times = pd.concat((cue_start, cue_stop), axis=1, keys=('start', 'stop'))
        cue2_start = cue_times['start'].iloc[-1]
        cue2_stop = cue_times['stop'].iloc[-1]
        max_out = max(X.loc['0s':cue2_start].max(),
                      X.loc[cue2_stop + pd.Timedelta(seconds=3):'20s'].max())
        max_in = X.loc[cue2_stop:cue2_stop + pd.Timedelta(seconds=2)].max()
        return pd.Series([max_out, max_in], index=('out', 'in'))

    trial_maxs = predictions.groupby(['jitter', 'trial']).apply(calc_trial_max)[0].unstack()
    maxes_list = trial_maxs.groupby(['jitter'])\
                           .apply(lambda x: pd.DataFrame(
                                x.values.T, index=x.keys())).apply(list, axis=1) 
    fixed_trial_maxes = maxes_list.apply(
        lambda x: [a for a in x if np.isfinite(a)])

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
        return score

    return sorted_maxes.unstack().apply(calculate_cutoff, axis=1)


def load_X2(jitter_level, n_trials=25, noise_level=0.15):
    trial_gen = tDNMSTemporalJitter(
        jitter=jitter_level, input_noise=noise_level, batch_size=1, n_blocks=1)

    X2 = trial_gen.generate_trials(n_trials)
    return X2


def load_predictions(X2, models, overwrite=False):
    jitter_level = X2.index.unique('jitter')[0]
    formatted_X = np.expand_dims(X2[['light', 'odor']], 0)
    predictions = models.apply(
        lambda m, X=formatted_X, i=X2.index:
            pd.Series(m.predict(X).squeeze(), index=i)).T

    time_idx = pd.to_timedelta(predictions.index.get_level_values('idx')/10,
                               unit='s')
    time_idx.name = 'time'
    predictions = predictions.set_index(time_idx, append=True)\
                             .reorder_levels(
        np.roll(np.arange(len(predictions.index.names) + 1), 1))
    predictions = predictions.set_index(X2['cues'], append=True)
    return predictions


def main(argv):
    no_model_ids = [9, 18, 19, 24, 28, 29, 31, 32, 93, 109, 116, 119]
    with_model_ids = [20, 22, 25, 21, 23, 26, 27, 30, 76, 79, 82, 84]
    cache_file = Path('jitter_errors.hdf')
    idx = pd.MultiIndex.from_tuples(
        list(zip(no_model_ids, it.repeat('no_shaping'))) +
        list(zip(with_model_ids, it.repeat('shaping'))),
        names=('model_id', 'model_type'))
    model_ids = pd.Series(no_model_ids + with_model_ids,  index=idx)

    margins = jP.default_margins()
    adj_margins = margins.copy()
    adj_margins['bottom'] = 110

    model_infos = model_ids.apply(mdb.get_model).apply(pd.Series)
    model_infos = pd.concat(
        (model_infos, model_ids.apply(mdb.get_model_attributes)), axis=1)

    path = Path('/analysis/ms_figures/jittery_error')
    jP.set_rcParams(plt)
    jP.make_folder(path)
    dpi = 300

    trained_noise = model_infos['input_noise'].head(1).values[0]
    jitter_levels = [0, 0.25, 0.5, 0.75,  1.0, 1.25, 1.5, 1.75, 2]

    if not cache_file.exists():
        X2 = pd.concat(list(map(load_X2, jitter_levels)),
                       keys=pd.Index(jitter_levels, name='jitter'))
        models = model_infos['path'].apply(tf.keras.models.load_model)
        print('Computing Predictions')
        predictions = X2.groupby('jitter').apply(
            load_predictions, models=models, overwrite=False).droplevel(0)

        scores = predictions.apply(error_compare)
        scores.to_hdf(cache_file, key='scores')
    else:
        scores = pd.read_hdf(cache_file)

    stacked_scores = scores.T.stack()
    stacked_scores.name = 'values'
    X = stacked_scores.reset_index()
    model = ols("values ~ C(model_type)*C(jitter)", X).fit()
    aov_table = anova_lm(model, typ=2)
    print(aov_table)

    ttest = scores.T.stack().groupby(['model_type', 'jitter']).apply(list)\
                  .groupby('jitter').apply(lambda x: scipy.stats.ttest_ind(*x, equal_var=False))\
                  .apply(pd.Series, index=['ttest', 'pval'])
    ttest_corr =  (ttest['pval'] * len(jitter_levels)).apply(np.round, decimals=5)

    PdfPlotter(path / 'all_models.pdf', fixed_margins=adj_margins)
    plt.figure(figsize=(1.75, 2), dpi=dpi)
    ax = plt.gca()

    def plot_scores(X, ax, labels_dict=dict(no_shaping='NS', shaping='S/FT')):
        model_type = X.index.unique('model_type')[0]
        print(model_type)
        means = X.mean()
        stds = X.std()
        noise_levels = X.columns.get_level_values('jitter')

        ax.fill_between(noise_levels, means + stds, means - stds,
                        color=SHAPING_COLORS[model_type], alpha=0.15)
        ax.plot(noise_levels, means, color=SHAPING_COLORS[model_type],
                label=labels_dict[model_type], clip_on=False)
        jP.configure_spines(ax, fix_xlabel=False)

    (scores * 100).T.groupby('model_type').apply(plot_scores, ax=ax)
    ax.set_ylabel('Correctly Seperated')
    ax.set_xlabel(r'Cue Temporal Jitter $\sim \cal{N}(0, x)$')
    jP.percent_y(ax)
    ax.set_ylim(75, 100)
    ax.set_yticks([80, 90, 100])
    ax.set_xlim(0, 2.0)
    ax.legend(loc='lower left')
    jP.annotation(ax, (0.25-0.1, 0.25+0.1), 100, '*', va='bottom')
    jP.annotation(ax, (0.5-0.1, 0.75+0.1), 100, '***', va='bottom')
    jP.annotation(ax, (1.0-0.1, 1.25+0.1), 100, '**', va='bottom')
    jP.annotation(ax, (1.5-0.1, 2.0), 100, '***', va='bottom')
    plt.show()


    PdfPlotter(path / 'all_models_larger.pdf', fixed_margins=adj_margins)
    plt.figure(figsize=(2, 1.75), dpi=dpi)
    ax = plt.gca()

    (scores * 100).T.groupby('model_type').apply(plot_scores, ax=ax)
    ax.set_ylabel('Correct Trials')
    ax.set_xlabel(r'Cue Temporal Jitter $\sim \cal{N}(0, x)$')
    jP.percent_y(ax)
    ax.set_ylim(75, 100)
    ax.set_yticks([80, 90, 100])
    ax.set_xlim(0, 2.0)
    ax.axvline(jitter_levels[2], c='r', ls='--', zorder=-100)
    ax.axvline(jitter_levels[-1], c='r', ls='--', clip_on=False, zorder=-100)
    ax.text(
        0.4, 0.9, '1.', transform=ax.get_xaxis_transform(), fontsize=5,
        color='r', ha='center')
    ax.text(
        1.9, 0.9, '2.', transform=ax.get_xaxis_transform(), fontsize=5,
        color='r', ha='center')
    ax.legend(loc='lower left')
    jP.annotation(ax, (0.25-0.1, 0.25+0.1), 100, '*', va='bottom')
    jP.annotation(ax, (0.5-0.1, 0.75+0.1), 100, '***', va='bottom')
    jP.annotation(ax, (1.0-0.1, 1.25+0.1), 100, '**', va='bottom')
    jP.annotation(ax, (1.5-0.1, 2.0), 100, '***', va='bottom')
    plt.show()


    PdfPlotter(path / 'all_models_2.25.pdf', fixed_margins=adj_margins)
    plt.figure(figsize=(2.25, 1.75), dpi=dpi)
    ax = plt.gca()

    (scores * 100).T.groupby('model_type').apply(plot_scores, ax=ax)
    ax.set_ylabel('Correct Trials')
    ax.set_xlabel(r'Cue Temporal Jitter $\sim \cal{N}(0, x)$')
    jP.percent_y(ax)
    ax.set_ylim(75, 100)
    ax.set_yticks([80, 90, 100])
    ax.set_xlim(0, 2.0)
    ax.axvline(jitter_levels[2], c='r', ls='--', zorder=-100)
    ax.axvline(jitter_levels[-1], c='r', ls='--', clip_on=False, zorder=-100)
    ax.text(
        0.4, 0.9, '1.', transform=ax.get_xaxis_transform(), fontsize=5,
        color='r', ha='center')
    ax.text(
        1.9, 0.9, '2.', transform=ax.get_xaxis_transform(), fontsize=5,
        color='r', ha='center')
    ax.legend(loc='lower left')
    jP.annotation(ax, (0.25-0.1, 0.25+0.1), 100, '*', va='bottom')
    jP.annotation(ax, (0.5-0.1, 0.75+0.1), 100, '***', va='bottom')
    jP.annotation(ax, (1.0-0.1, 1.25+0.1), 100, '**', va='bottom')
    jP.annotation(ax, (1.5-0.1, 2.0), 100, '***', va='bottom')
    plt.show()


    info = ['counts: ', str(model_ids.groupby('model_type').count()), '\n']
    info = info + ['pvalues:\n', str(ttest), '\n']
    info = info + ['pvalues (corrected):\n', str(ttest_corr)]
    info = info + ['\n\n\n']
    info = info + [str(aov_table), '\n']
    info_file = path / 'info_file.txt'
    with open(info_file, 'w') as f:
        f.writelines(info)

    
    margins = jP.default_margins()
    margins['left'] = margins['left'] / 2
    margins['bottom'] = 110
    margins['right'] = 45
    PdfPlotter(path / 'low_example.pdf', fixed_margins=margins)

    plt.figure(figsize=(1.25, 1.75), dpi=300)
    gs = gridspec.GridSpec(2, 1)
    axs = list(map(plt.subplot, gs))
    for ax in axs:
        ax.set_xlim(0, 20)
        ax.axvspan(3, 8, ymax=0.8, color='gray')
        ax.axvspan(11, 13, ymax=0.8, color='gray')
        ax.set_yticks([])
        ax.set_ylim(0, 2)

    ax = axs[0]
    ax.set_ylabel('Low Jitter')
    ax.set_xticks([])
    import random
    start_jitter = [0.244, -0.009]
    stop_jitter = [-0.104, 0.137]
    cue_times = pd.DataFrame(
        {'start': [3+start_jitter[0], 11+start_jitter[1]],
         'stop': [8+stop_jitter[0]+start_jitter[0], 13+stop_jitter[1]+start_jitter[1]]})
    ax.axvspan(
        cue_times['start'].iloc[0], cue_times['stop'].iloc[0], ymax=0.5,
        color=TRIAL_COLORS['LS'])
    ax.axvspan(
        cue_times['start'].iloc[1], cue_times['stop'].iloc[1], ymax=0.5,
        color=TRIAL_COLORS['LS'])
    ax.text(0.05, 0.9, '1.', transform=ax.transAxes, c='r')
    jP.configure_spines(ax)

    arrowprops = dict(arrowstyle='-|>', color='k', lw=0.3, shrinkA=0, shrinkB=0, mutation_scale=4)
    ax.annotate('', (11+start_jitter[1], 0.7),
                xytext=(11, 0.7), textcoords=ax.transData, arrowprops=arrowprops)
    ax.annotate('', (11+start_jitter[1], 0.3),
                xytext=(11, 0.3), textcoords=ax.transData, arrowprops=arrowprops)
    ax.annotate('', (3+start_jitter[0], 0.3),
                xytext=(3, 0.3), textcoords=ax.transData, arrowprops=arrowprops)
    ax.annotate('', (3+start_jitter[0], 0.7),
                xytext=(3, 0.7), textcoords=ax.transData, arrowprops=arrowprops)


    ax = axs[1]
    start_jitter = [-0.597, 2.17]
    stop_jitter = [-1.342, 1.6]
    cue_times = pd.DataFrame(
        {'start': [3+start_jitter[0], 11+start_jitter[1]],
         'stop': [8+stop_jitter[0]+start_jitter[0], 13+stop_jitter[1]+start_jitter[1]]})
 
    ax.axvspan(
        cue_times['start'].iloc[0], cue_times['stop'].iloc[0], ymax=0.5,
        color=TRIAL_COLORS['LS'])
    ax.axvspan(
        cue_times['start'].iloc[1], cue_times['stop'].iloc[1], ymax=0.5,
        color=TRIAL_COLORS['LS'])
    jP.configure_spines(ax, fix_xlabel=False)
    ax.text(0.05, 0.9, '2.', transform=ax.transAxes, c='r')
    arrowprops = dict(arrowstyle='-|>', color='k', lw=1, shrinkA=0, shrinkB=0, mutation_scale=4)
    ax.annotate('', (11+start_jitter[1], 0.7),
                xytext=(11, 0.7), textcoords=ax.transData, arrowprops=arrowprops)
    ax.annotate('', (11+start_jitter[1], 0.3),
                xytext=(11, 0.3), textcoords=ax.transData, arrowprops=arrowprops)
    ax.annotate('', (3+start_jitter[0], 0.3),
                xytext=(3, 0.3), textcoords=ax.transData, arrowprops=arrowprops)
    ax.annotate('', (3+start_jitter[0], 0.7),
                xytext=(3, 0.7), textcoords=ax.transData, arrowprops=arrowprops)

    ax.set_ylabel('High Jitter')
    ax.set_xlabel('Time (s)')
    plt.show()
    
if __name__ == '__main__':
    main(sys.argv[1:])
