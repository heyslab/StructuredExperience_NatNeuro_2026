# [Figure 5c,g]
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

TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'S': '#f89521'}
SHAPING_COLORS = {'no_shaping': 'tab:blue', 'shaping': 'tab:orange', 'sl_only': 'tab:green', 'ls_only': 'tab:red'}

def main(argv):
    models = pd.Series(
        [47, 68, 65],  index=pd.Index(['sl_shaping', 'ls_shaping_1', 'ls_shaping_2'],
        name='model_type'))

    margins={'left': 80, 'right': 90, 'top': 45, 'bottom': 80}

    model_infos = models.apply(mdb.get_model).apply(pd.Series)
    model_infos = pd.concat(
        (model_infos, models.apply(mdb.get_model_attributes)), axis=1)

    path = Path('/analysis/ms_figures/shaping_tasks_behavior')
    cache_path = Path('shaping_tasks_testing_data.hdf')
    jP.make_folder(path)

    dpi = 300
    jP.set_rcParams(plt)

    models = model_infos['path'].apply(tf.keras.models.load_model)

    
    resave = True
    if cache_path.is_file():
        try:
            X2 = pd.read_hdf(cache_path, key='X')
            X_ls = pd.read_hdf(cache_path, key='X_ls')
            X_sl = pd.read_hdf(cache_path, key='X_sl')
            resave = False
        except:
            pass

    if resave:
        trial_gen_ls = genFactory.create(
            'ls_only', input_noise=model_infos['input_noise'].head(1), batch_size=1,
            n_blocks=1)
        trial_gen_sl = genFactory.create(
            'sl_only', input_noise=model_infos['input_noise'].head(1), batch_size=1,
            n_blocks=1)
        trial_gen = genFactory.create(
            'just_short_match', input_noise=model_infos['input_noise'].head(1), batch_size=1,
            n_blocks=1)

        X2 = trial_gen.generate_trials(25)
        X2.to_hdf(cache_path, key='X')
        X_ls = trial_gen_ls.generate_trials(25)
        X_ls.to_hdf(cache_path, key='X_ls')
        X_sl = trial_gen_sl.generate_trials(25)
        X_sl.to_hdf(cache_path, key='X_sl')

    if cache_path.is_file():
        try:
            predictions_full = pd.read_hdf(cache_path, key='predictions')
            predictions_ls = pd.read_hdf(cache_path, key='predictions_ls')
            predictions_sl = pd.read_hdf(cache_path, key='predictions_sl')
        except:
            predictions_full = None

    if predictions_full is None:
        trial_gen = genFactory.create(
            'just_short_match', input_noise=model_infos['input_noise'].head(1), batch_size=1,
            n_blocks=1)

        formatted_X = trial_gen.format_validation(X2)[0]
        cols = X2.index.names
        X2 = X2.reset_index().set_index(cols + ['cues'])
        index = X2.index

        predictions_full = models.apply(
            lambda m, X=formatted_X, i=X2.index:
                pd.Series(m.predict(X).squeeze(), index=i)).T
        predictions_full.to_hdf(cache_path, key='predictions')

        formatted_ls = trial_gen.format_validation(X_ls)[0]
        cols = X_ls.index.names
        X_ls = X_ls.reset_index().set_index(cols + ['cues'])

        predictions_ls = models.apply(
            lambda m, X=formatted_ls, i=X_ls.index:
                pd.Series(m.predict(X).squeeze(), index=i)).T
        predictions_ls.to_hdf(cache_path, key='predictions_ls')

        formatted_sl = trial_gen.format_validation(X_sl)[0]
        cols = X_sl.index.names
        X_sl = X_sl.reset_index().set_index(cols + ['cues'])

        predictions_sl = models.apply(
            lambda m, X=formatted_sl, i=X_sl.index:
                pd.Series(m.predict(X).squeeze(), index=i)).T
        predictions_sl.to_hdf(cache_path, key='predictions_sl')

    predictions = predictions_full.reindex(np.arange(71, 77), level='trial')
    pred_sl = predictions_sl.reindex(np.arange(10, 16), level='trial')[['sl_shaping']]

    cue_transitions = predictions.index.to_frame()['cues'].droplevel('cues').diff()
    cue_transitions = cue_transitions.reset_index().reset_index().set_index(
        ['index', 'trial', 'type', 'idx'])
    cue_times = pd.concat(
        list(map(cue_transitions[cue_transitions != 0
                                 ].dropna().reset_index().set_index(
                                     ['cues', 'trial', 'type'])['index'].xs,
                 (1, -1))), axis=1, keys=('start', 'stop'))/10

    cue_transitions = pred_sl.index.to_frame()['cues'].droplevel('cues').diff()
    cue_transitions = cue_transitions.reset_index().reset_index().set_index(
        ['index', 'trial', 'type', 'idx'])
    cue_sl = pd.concat(
        list(map(cue_transitions[cue_transitions != 0
                                 ].dropna().reset_index().set_index(
                                     ['cues', 'trial', 'type'])['index'].xs,
                 (1, -1))), axis=1, keys=('start', 'stop'))/10

    target = X2['response'].reindex(predictions.index.unique('trial'), level='trial')
    target_sl = X_sl['response'].reindex(pred_sl.index.unique('trial'), level='trial')

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
        #ax.plot(np.arange(predictions.shape[0])/10, predictions.values, c=SHAPING_COLORS[model_type],
        #    lw=0.7, label='response')
        ax.plot(np.arange(predictions.shape[0])/10, predictions.values, c='r',
            lw=1, label='response')
        jP.configure_spines(ax)
        ax.set_yticks([])
        ax.set_xlim(0, len(predictions)/10)


    PdfPlotter(path / f'sl_performance.pdf', fixed_margins=margins)
    plt.figure(figsize=(4.25, 1), dpi=dpi)
    gs = gridspec.GridSpec(1, 2)
    axs = map(plt.subplot, gs)
    ax_dict = {'sl_shaping': next(axs), 'sl_task': next(axs)}

    plot_model_behavior(
        pred_sl['sl_shaping'], ax_dict={'sl_shaping': ax_dict['sl_shaping']},
        cue_times=cue_sl, target=target_sl)
    plot_model_behavior(
        predictions['sl_shaping'], ax_dict={'sl_shaping': ax_dict['sl_task']},
        cue_times=cue_times, target=target)
    ax_dict['sl_shaping'].set_xlabel('Time (s)')
    ax_dict['sl_task'].set_xlabel('Time (s)')
    ax_dict['sl_shaping'].set_yticks([0, 1])
    ax_dict['sl_shaping'].set_yticklabels(['', ''])
    ax_dict['sl_shaping'].set_ylabel('Model\nResponse')
    jP.set_ylabel_position(ax_dict['sl_shaping'], nlines=2.8)
    ax_dict['sl_shaping'].text(
        0, 1, 'SL Shaping', transform=ax_dict['sl_shaping'].transAxes,
        ha='left', va='bottom', fontsize=6, color=SHAPING_COLORS['sl_only'])
    ax_dict['sl_task'].text(
        0, 1, 'SL + Full Task', transform=ax_dict['sl_task'].transAxes,
        ha='left', va='bottom', fontsize=6, color=SHAPING_COLORS['sl_only'])
    plt.show()

    margins_adj = jP.default_margins()
    margins_adj['top'] = 45
    PdfPlotter(path / f'sl_performance_4.75.pdf', fixed_margins=margins_adj)
    plt.figure(figsize=(4.75, 1), dpi=dpi)
    gs = gridspec.GridSpec(1, 2)
    axs = map(plt.subplot, gs)
    ax_dict = {'sl_shaping': next(axs), 'sl_task': next(axs)}

    plot_model_behavior(
        pred_sl['sl_shaping'], ax_dict={'sl_shaping': ax_dict['sl_shaping']},
        cue_times=cue_sl, target=target_sl)
    plot_model_behavior(
        predictions['sl_shaping'], ax_dict={'sl_shaping': ax_dict['sl_task']},
        cue_times=cue_times, target=target)
    ax_dict['sl_shaping'].set_xlabel('Time (s)')
    ax_dict['sl_task'].set_xlabel('Time (s)')
    ax_dict['sl_shaping'].set_yticks([0, 1])
    ax_dict['sl_shaping'].set_yticklabels(['', ''])
    ax_dict['sl_shaping'].set_ylabel('Model\nResponse')
    jP.set_ylabel_position(ax_dict['sl_shaping'], nlines=2.8)
    ax_dict['sl_shaping'].text(
        0, 1, 'SL Shaping', transform=ax_dict['sl_shaping'].transAxes,
        ha='left', va='bottom', fontsize=6, color=SHAPING_COLORS['sl_only'])
    ax_dict['sl_task'].text(
        0, 1, 'SL + Full Task', transform=ax_dict['sl_task'].transAxes,
        ha='left', va='bottom', fontsize=6, color=SHAPING_COLORS['sl_only'])
    plt.show()

    predictions = predictions_full.reindex(np.arange(72, 77), level='trial')
    pred_ls = predictions_ls.reindex(np.arange(11, 16), level='trial')[['ls_shaping_2']]

    cue_transitions = predictions.index.to_frame()['cues'].droplevel('cues').diff()
    cue_transitions = cue_transitions.reset_index().reset_index().set_index(
        ['index', 'trial', 'type', 'idx'])
    cue_times = pd.concat(
        list(map(cue_transitions[cue_transitions != 0
                                 ].dropna().reset_index().set_index(
                                     ['cues', 'trial', 'type'])['index'].xs,
                 (1, -1))), axis=1, keys=('start', 'stop'))/10

    cue_transitions = pred_ls.index.to_frame()['cues'].droplevel('cues').diff()
    cue_transitions = cue_transitions.reset_index().reset_index().set_index(
        ['index', 'trial', 'type', 'idx'])
    cue_ls = pd.concat(
        list(map(cue_transitions[cue_transitions != 0
                                 ].dropna().reset_index().set_index(
                                     ['cues', 'trial', 'type'])['index'].xs,
                 (1, -1))), axis=1, keys=('start', 'stop'))/10

    target = X2['response'].reindex(predictions.index.unique('trial'), level='trial')
    target_ls = X_sl['response'].reindex(pred_sl.index.unique('trial'), level='trial')

    PdfPlotter(path / f'ls_performance.pdf', fixed_margins=margins)
    plt.figure(figsize=(4.25, 1), dpi=dpi)
    gs = gridspec.GridSpec(1, 3)
    axs = map(plt.subplot, gs)
    ax_dict = {'ls_shaping': next(axs), 'ls_taskA': next(axs), 'ls_taskB': next(axs)}

    plot_model_behavior(
        pred_ls['ls_shaping_2'], ax_dict={'ls_shaping_2': ax_dict['ls_shaping']},
        cue_times=cue_ls, target=target_ls)
    plot_model_behavior(
        predictions['ls_shaping_1'], ax_dict={'ls_shaping_1': ax_dict['ls_taskB']},
        cue_times=cue_times, target=target)
    plot_model_behavior(
        predictions['ls_shaping_2'], ax_dict={'ls_shaping_2': ax_dict['ls_taskA']},
        cue_times=cue_times, target=target)
    ax_dict['ls_shaping'].set_xlabel('Time (s)')
    ax_dict['ls_taskA'].set_xlabel('Time (s)')
    ax_dict['ls_taskB'].set_xlabel('Time (s)')
    ax_dict['ls_shaping'].set_yticks([0, 1])
    ax_dict['ls_shaping'].set_yticklabels(['', ''])
    ax_dict['ls_shaping'].set_ylabel('Model\nResponse')
    jP.set_ylabel_position(ax_dict['ls_shaping'], nlines=2.8)
    ax_dict['ls_shaping'].text(
        0, 1, 'LS Shaping', transform=ax_dict['ls_shaping'].transAxes,
        ha='left', va='bottom', fontsize=6, color=SHAPING_COLORS['ls_only'])
    ax_dict['ls_taskA'].text(
        0, 1, 'LS + Full Task (A.)', transform=ax_dict['ls_taskA'].transAxes,
        ha='left', va='bottom', fontsize=6, color=SHAPING_COLORS['ls_only'])
    ax_dict['ls_taskB'].text(
        0, 1, 'LS + Full Task (B.)', transform=ax_dict['ls_taskB'].transAxes,
        ha='left', va='bottom', fontsize=6, color=SHAPING_COLORS['ls_only'])
    plt.show()

    PdfPlotter(path / f'ls_performance_4.75.pdf', fixed_margins=margins_adj)
    plt.figure(figsize=(4.75, 1), dpi=dpi)
    gs = gridspec.GridSpec(1, 3)
    axs = map(plt.subplot, gs)
    ax_dict = {'ls_shaping': next(axs), 'ls_taskA': next(axs), 'ls_taskB': next(axs)}

    plot_model_behavior(
        pred_ls['ls_shaping_2'], ax_dict={'ls_shaping_2': ax_dict['ls_shaping']},
        cue_times=cue_ls, target=target_ls)
    plot_model_behavior(
        predictions['ls_shaping_1'], ax_dict={'ls_shaping_1': ax_dict['ls_taskB']},
        cue_times=cue_times, target=target)
    plot_model_behavior(
        predictions['ls_shaping_2'], ax_dict={'ls_shaping_2': ax_dict['ls_taskA']},
        cue_times=cue_times, target=target)
    ax_dict['ls_shaping'].set_xlabel('Time (s)')
    ax_dict['ls_taskA'].set_xlabel('Time (s)')
    ax_dict['ls_taskB'].set_xlabel('Time (s)')
    ax_dict['ls_shaping'].set_yticks([0, 1])
    ax_dict['ls_shaping'].set_yticklabels(['', ''])
    ax_dict['ls_shaping'].set_ylabel('Model\nResponse')
    jP.set_ylabel_position(ax_dict['ls_shaping'], nlines=2.8)
    ax_dict['ls_shaping'].text(
        0, 1, 'LS Shaping', transform=ax_dict['ls_shaping'].transAxes,
        ha='left', va='bottom', fontsize=6, color=SHAPING_COLORS['ls_only'])
    ax_dict['ls_taskA'].text(
        0, 1, 'LS + Full Task (A.)', transform=ax_dict['ls_taskA'].transAxes,
        ha='left', va='bottom', fontsize=6, color=SHAPING_COLORS['ls_only'])
    ax_dict['ls_taskB'].text(
        0, 1, 'LS + Full Task (B.)', transform=ax_dict['ls_taskB'].transAxes,
        ha='left', va='bottom', fontsize=6, color=SHAPING_COLORS['ls_only'])
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
