# [Figure 1h,i]
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

from classes.models import LeakyRNN, LeakyRNNCell
from classes.datagen import genFactory
import models_database as mdb

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  
    "axes.facecolor":    (1.0, 1.0, 1.0, 0.0),  
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0)})


TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'S': '#f89521'}
SHAPING_COLORS = {'no_shaping': 'tab:blue', 'shaping': 'tab:orange'}

def main(argv):
    models = pd.Series(
        [9, 130, 132, 128],  index=pd.Index(['no_shaping', 'only_shaping', 'shaping', 'include_long'],
        name='model_type'))

    margins={'left': 80, 'right': 45, 'top': 45, 'bottom': 80}

    model_infos = models.apply(mdb.get_model).apply(pd.Series)
    model_infos = pd.concat(
        (model_infos, models.apply(mdb.get_model_attributes)), axis=1)

    path = Path('/analysis/ms_figures/behavior')
    cache_path = Path('testing_data.hdf')
    jP.make_folder(path)

    dpi = 300
    jP.set_rcParams(plt)

    models = model_infos['path'].apply(tf.keras.models.load_model)

    
    trial_gen = genFactory.create(
        'just_short_match', input_noise=model_infos['input_noise'].head(1), batch_size=1,
        n_blocks=1)
    if cache_path.is_file():
        X2 = pd.read_hdf(cache_path, key='X')
        X_no_match = pd.read_hdf(cache_path, key='X_nomatch')
    else:
        X2 = trial_gen.generate_trials(25)
        X2.to_hdf(cache_path, key='X')
        trial_gen = genFactory.create(
            'no_match', input_noise=model_infos['input_noise'].head(1), batch_size=1,
            n_blocks=1)
        X_no_match = trial_gen.generate_trials(25)
        X_no_match.to_hdf(cache_path, key='X_nomatch')

    long_gen = genFactory.create(
        'include_long_match', input_noise=model_infos['input_noise'].head(1), batch_size=1,
        n_blocks=1)
    X_long = long_gen.generate_trials(25)
    X_long.to_hdf(cache_path, key='X_long')

    predictions_full = None
    if cache_path.is_file():
        try:
            predictions_full = pd.read_hdf(cache_path, key='predictions')
            predictions_nomatch = pd.read_hdf(cache_path, key='predictions_nomatch')
        except:
            pass

    if predictions_full is None:
        formatted_X = trial_gen.format_validation(X2)[0]
        cols = X2.index.names
        X2 = X2.reset_index().set_index(cols + ['cues'])
        index = X2.index

        predictions_full = models.apply(
            lambda m, X=formatted_X, i=X2.index:
                pd.Series(m.predict(X).squeeze(), index=i)).T
        predictions_full.to_hdf(cache_path, key='predictions')

        formatted_nomatch = trial_gen.format_validation(X_no_match)[0]
        cols = X_no_match.index.names
        X_no_match = X_no_match.reset_index().set_index(cols + ['cues'])
        index_nomatch = X_no_match.index

        predictions_nomatch = models.apply(
            lambda m, X=formatted_nomatch, i=X_no_match.index:
                pd.Series(m.predict(X).squeeze(), index=i)).T
        predictions_nomatch.to_hdf(cache_path, key='predictions_nomatch')


    predictions = predictions_full.reindex(np.arange(0, 8), level='trial')
    pred_nomatch = predictions_nomatch.reindex(np.arange(1, 9), level='trial')[['only_shaping']]

    cue_transitions = predictions.index.to_frame()['cues'].droplevel('cues').diff()
    cue_transitions = cue_transitions.reset_index().reset_index().set_index(
        ['index', 'trial', 'type', 'idx'])

    cue_times = pd.concat(
        list(map(cue_transitions[cue_transitions != 0
                                 ].dropna().reset_index().set_index(
                                     ['cues', 'trial', 'type'])['index'].xs,
                 (1, -1))), axis=1, keys=('start', 'stop'))/10

    cue_transitions = pred_nomatch.index.to_frame()['cues'].droplevel('cues').diff()
    cue_transitions = cue_transitions.reset_index().reset_index().set_index(
        ['index', 'trial', 'type', 'idx'])

    cue_nomatch = pd.concat(
        list(map(cue_transitions[cue_transitions != 0
                                 ].dropna().reset_index().set_index(
                                     ['cues', 'trial', 'type'])['index'].xs,
                 (1, -1))), axis=1, keys=('start', 'stop'))/10

    target = X2['response'].reindex(predictions.index.unique('trial'), level='trial')
    target_nomatch = X_no_match['response'].reindex(pred_nomatch.index.unique('trial'), level='trial')

    PdfPlotter(path / f'multiple_trials.pdf', fixed_margins=margins)
    plt.figure(figsize=(2, 1.5), dpi=dpi)
    gs = gridspec.GridSpec(2, 1, hspace=0.05)
    axs = map(plt.subplot, gs)
    ax_dict = {'no_shaping': next(axs), 'shaping': next(axs)}

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
        jP.configure_spines(ax)
        ax.set_yticks([])
        ax.set_xlim(0, len(predictions)/10)

    predictions[['no_shaping', 'shaping']].apply(
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
        0.03, 0.55, 'Model Response',
        transform=ax_dict['shaping'].figure.transFigure, rotation=90,
        ha='center', va='center')

    legend_lines = [matplotlib.lines.Line2D([0], [0], color='k'),
                    matplotlib.lines.Line2D([0], [0], color='r')]
    ax_dict['no_shaping'].legend(
        legend_lines, ['target', 'response'], loc='lower right',
        bbox_to_anchor=(1.1, 0.65))

    plt.show()


    PdfPlotter(path / f'multiple_trials_3plots.pdf', fixed_margins=margins)
    plt.figure(figsize=(2.5, 2.2), dpi=dpi)
    gs = gridspec.GridSpec(3, 1, hspace=0.05)
    axs = map(plt.subplot, gs)
    ax_dict = {'no_shaping': next(axs), 'only_shaping': next(axs), 'shaping': next(axs)}

    predictions.apply(plot_model_behavior, ax_dict=ax_dict, cue_times=cue_times, target=target)
    ax_dict['no_shaping'].spines['bottom'].set_visible(False)
    ax_dict['no_shaping'].set_xticks([])
    ax_dict['only_shaping'].spines['bottom'].set_visible(False)
    ax_dict['only_shaping'].set_xticks([])

    ax_dict['shaping'].tick_params(axis='x', labelsize=5)
    ax_dict['shaping'].set_xlabel('Time (s)')

    ax_dict['no_shaping'].text(
        -0.01, 0.5, 'No\nShaping', color=SHAPING_COLORS['no_shaping'],
        fontsize=5, va='center', ha='right', transform=ax_dict['no_shaping'].transAxes, rotation=90, ma='center')

    ax_dict['only_shaping'].text(
        -0.01, 0.5, 'After\nShaping', color=SHAPING_COLORS['shaping'],
        fontsize=5, va='center', ha='right', transform=ax_dict['only_shaping'].transAxes, rotation=90, ma='center')
    ax_dict['shaping'].text(
        -0.01, 0.5, 'Shaping +\nFull Task', color=SHAPING_COLORS['shaping'],
        fontsize=5, va='center', ha='right', transform=ax_dict['shaping'].transAxes, rotation=90, ma='center')
    
    ax_dict['shaping'].text(
        0.03, 0.52, 'Model Response',
        transform=ax_dict['shaping'].figure.transFigure, rotation=90,
        ha='center', va='center')

    legend_lines = [matplotlib.lines.Line2D([0], [0], color='k'),
                    matplotlib.lines.Line2D([0], [0], color='r')]
    ax_dict['no_shaping'].legend(
        legend_lines, ['target', 'response'], loc='lower right',
        bbox_to_anchor=(1, 0.65))

    plt.show()

    PdfPlotter(path / f'multiple_trials_3plots_shaping.pdf', fixed_margins=margins)
    plt.figure(figsize=(2.5, 2.2), dpi=dpi)
    gs = gridspec.GridSpec(3, 1, hspace=0.05)
    axs = list(map(plt.subplot, gs))
    ax_dict = {'no_shaping': axs[0], 'nomatch': axs[1], 'shaping': axs[2]}

    predictions.drop('only_shaping', axis=1)\
        .apply(plot_model_behavior, ax_dict=ax_dict, cue_times=cue_times,
               target=target)
    pred_nomatch.name = 'nomatch'
    plot_model_behavior(pred_nomatch, ax_dict={'nomatch': axs[1]},
                        cue_times=cue_nomatch, target=target_nomatch)
    ax_dict['no_shaping'].spines['bottom'].set_visible(False)
    ax_dict['no_shaping'].set_xticks([])

    ax_dict['nomatch'].spines['bottom'].set_visible(False)
    ax_dict['nomatch'].set_xticks([])

    ax_dict['shaping'].tick_params(axis='x', labelsize=5)
    ax_dict['shaping'].set_xlabel('Time (s)')

    ax_dict['no_shaping'].text(
        -0.01, 0.5, 'No\nShaping', color=SHAPING_COLORS['no_shaping'],
        fontsize=5, va='center', ha='right', transform=ax_dict['no_shaping'].transAxes, rotation=90, ma='center')

    ax_dict['nomatch'].text(
        -0.01, 0.5, 'After\nShaping', color=SHAPING_COLORS['shaping'],
        fontsize=5, va='center', ha='right', transform=ax_dict['nomatch'].transAxes, rotation=90, ma='center')
    ax_dict['shaping'].text(
        -0.01, 0.5, 'Shaping +\nFull Task', color=SHAPING_COLORS['shaping'],
        fontsize=5, va='center', ha='right', transform=ax_dict['shaping'].transAxes, rotation=90, ma='center')
    
    ax_dict['shaping'].text(
        0.03, 0.52, 'Model Response',
        transform=ax_dict['shaping'].figure.transFigure, rotation=90,
        ha='center', va='center')

    legend_lines = [matplotlib.lines.Line2D([0], [0], color='k'),
                    matplotlib.lines.Line2D([0], [0], color='r')]
    ax_dict['no_shaping'].legend(
        legend_lines, ['target', 'response'], loc='lower right',
        bbox_to_anchor=(1, 0.65))

    plt.show()


    margins_adj = margins.copy()
    margins_adj['top'] = 90
    PdfPlotter(path / f'no_shaping.pdf', fixed_margins=margins_adj)
    plt.figure(figsize=(2, 1), dpi=dpi)
    ax = plt.gca()
    ax_dict = {'shaping': ax}

    plot_model_behavior(predictions['no_shaping'], ax_dict={'no_shaping': ax}, cue_times=cue_times, target=target)

    ax.tick_params(axis='x', labelsize=5)
    ax.set_xlabel('Time (s)')

    ax.text(
        -0.01, 0.5, 'No Shaping', color=SHAPING_COLORS['no_shaping'],
        fontsize=5, va='center', ha='right', transform=ax.transAxes, rotation=90, ma='center')
    
    transform = matplotlib.transforms.blended_transform_factory(
        ax.figure.transFigure, ax.transAxes)
    ax.text(
        0.03, 0.5, 'Model Response',
        transform=transform, rotation=90,
        ha='center', va='center')

    legend_lines = [matplotlib.lines.Line2D([0], [0], color='k'),
                    matplotlib.lines.Line2D([0], [0], color='r')]

    plt.show()
 

    PdfPlotter(path / f'shaping_2plots.pdf', fixed_margins=margins)
    plt.figure(figsize=(2, 1.5), dpi=dpi)
    gs = gridspec.GridSpec(2, 1, hspace=0.05)
    axs = list(map(plt.subplot, gs))
    ax_dict = {'nomatch': axs[0], 'shaping': axs[1]}

    plot_model_behavior(predictions['shaping'], ax_dict={'shaping': axs[1]}, cue_times=cue_times, target=target)
    pred_nomatch.name = 'nomatch'
    plot_model_behavior(pred_nomatch, ax_dict={'nomatch': axs[0]},
                        cue_times=cue_nomatch, target=target_nomatch)

    ax_dict['nomatch'].spines['bottom'].set_visible(False)
    ax_dict['nomatch'].set_xticks([])

    ax_dict['shaping'].tick_params(axis='x', labelsize=5)
    ax_dict['shaping'].set_xlabel('Time (s)')

    ax_dict['nomatch'].text(
        -0.01, 0.5, 'After\nShaping', color=SHAPING_COLORS['shaping'],
        fontsize=5, va='center', ha='right', transform=ax_dict['nomatch'].transAxes, rotation=90, ma='center')
    ax_dict['shaping'].text(
        -0.01, 0.5, 'Shaping +\nFull Task', color=SHAPING_COLORS['shaping'],
        fontsize=5, va='center', ha='right', transform=ax_dict['shaping'].transAxes, rotation=90, ma='center')
    
    ax_dict['shaping'].text(
        0.03, 0.52, 'Model Response',
        transform=ax_dict['shaping'].figure.transFigure, rotation=90,
        ha='center', va='center')

    legend_lines = [matplotlib.lines.Line2D([0], [0], color='k'),
                    matplotlib.lines.Line2D([0], [0], color='r')]

    plt.show()

    margins_adj = jP.default_margins().copy()
    margins_adj['top'] = 75
    margins_adj['right'] = 45
    PdfPlotter(path / f'no_shaping_2.5.pdf', fixed_margins=margins_adj)
    plt.figure(figsize=(2.5, 1), dpi=dpi)
    ax = plt.gca()
    plot_model_behavior(
        predictions['no_shaping'], ax_dict={'no_shaping': ax},
        cue_times=cue_times, target=target)

    ax.tick_params(axis='x', labelsize=5)
    ax.set_xlabel('Time (s)')

    ax.text(
        -0.01, 0.5, 'No Shaping', color=SHAPING_COLORS['no_shaping'],
        fontsize=5, va='center', ha='right', transform=ax.transAxes,
        rotation=90, ma='center')
    
    transform = matplotlib.transforms.blended_transform_factory(
        ax.figure.transFigure, ax.transAxes)

    ax.set_ylabel('Model\nResponse')
    jP.set_ylabel_position(ax, nlines=2.5)

    legend_lines = [matplotlib.lines.Line2D([0], [0], color='k'),
                    matplotlib.lines.Line2D([0], [0], color='r')]

    plt.show()
 

    PdfPlotter(path / f'shaping_2plots_2.5.pdf', fixed_margins=margins_adj)
    plt.figure(figsize=(2.5, 1.5), dpi=dpi)
    gs = gridspec.GridSpec(2, 1, hspace=0.05)
    axs = list(map(plt.subplot, gs))
    ax_dict = {'nomatch': axs[0], 'shaping': axs[1]}

    plot_model_behavior(predictions['shaping'], ax_dict={'shaping': axs[1]}, cue_times=cue_times, target=target)
    pred_nomatch.name = 'nomatch'
    plot_model_behavior(pred_nomatch, ax_dict={'nomatch': axs[0]},
                        cue_times=cue_nomatch, target=target_nomatch)

    ax_dict['nomatch'].spines['bottom'].set_visible(False)
    ax_dict['nomatch'].set_xticks([])

    ax_dict['shaping'].tick_params(axis='x', labelsize=5)
    ax_dict['shaping'].set_xlabel('Time (s)')

    ax_dict['nomatch'].text(
        -0.01, 0.5, 'After\nShaping', color=SHAPING_COLORS['shaping'],
        fontsize=5, va='center', ha='right', transform=ax_dict['nomatch'].transAxes, rotation=90, ma='center')
    ax_dict['shaping'].text(
        -0.01, 0.5, 'Shaping +\nFull Task', color=SHAPING_COLORS['shaping'],
        fontsize=5, va='center', ha='right', transform=ax_dict['shaping'].transAxes, rotation=90, ma='center')
    
    ax_dict['shaping'].text(
        0.03, 0.52, 'Model Response',
        transform=ax_dict['shaping'].figure.transFigure, rotation=90,
        ha='center', va='center', fontsize=7)

    legend_lines = [matplotlib.lines.Line2D([0], [0], color='k'),
                    matplotlib.lines.Line2D([0], [0], color='r')]

    plt.show()
 

if __name__ == '__main__':
    main(sys.argv[1:])
