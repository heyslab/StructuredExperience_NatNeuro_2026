# [Figure 3i]
import os
import datetime
import sys

sys.path.append('../../')

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
        [168],  index=pd.Index(['seed_run'],
        name='model_type'))

    margins = jP.default_margins()

    model_infos = models.apply(mdb.get_model).apply(pd.Series)
    model_infos = pd.concat(
        (model_infos, models.apply(mdb.get_model_attributes)), axis=1)

    path = Path('/analysis/ms_figures/seed_behavior')
    path.mkdir(exist_ok=True, parents=True)
    cache_path = Path('testing_data.hdf')

    dpi = 300
    jP.set_rcParams(plt)

    models = model_infos['path'].apply(tf.keras.models.load_model)
    
    trial_gen = genFactory.create(
        'just_short_match', input_noise=model_infos['input_noise'].head(1), batch_size=1,
        n_blocks=1)

    if cache_path.is_file():
        X2 = pd.read_hdf(cache_path, key='X')
    else:
        X2 = trial_gen.generate_trials(25)
        X2.to_hdf(cache_path, key='X')

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

    predictions = predictions_full.reindex(np.arange(0, 8), level='trial')

    cue_transitions = predictions.index.to_frame()['cues'].droplevel('cues').diff()
    cue_transitions = cue_transitions.reset_index().reset_index().set_index(
        ['index', 'trial', 'type', 'idx'])

    cue_times = pd.concat(
        list(map(cue_transitions[cue_transitions != 0
                                 ].dropna().reset_index().set_index(
                                     ['cues', 'trial', 'type'])['index'].xs,
                 (1, -1))), axis=1, keys=('start', 'stop'))/10

    target = X2['response'].reindex(predictions.index.unique('trial'), level='trial')

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


    margins_adj = margins.copy()
    margins_adj['top'] = 45
    PdfPlotter(path / f'altered_shaping.pdf', fixed_margins=margins_adj)
    plt.figure(figsize=(2, 1.35), dpi=dpi)
    ax = plt.gca()

    plot_model_behavior(
        predictions['seed_run'], ax_dict={'seed_run': ax}, cue_times=cue_times,
        target=target)

    ax.tick_params(axis='x')
    ax.set_xlabel('Time (s)')

    ax.text(
        -0.01, 0.5, 'NS-Altered', color='tab:purple',
        fontsize=5, va='center', ha='right', transform=ax.transAxes,
        rotation=90, ma='center')
    
    transform = matplotlib.transforms.blended_transform_factory(
        ax.figure.transFigure, ax.transAxes)
    ax.set_ylabel('Model Response')

    legend_lines = [matplotlib.lines.Line2D([0], [0], color='k'),
                    matplotlib.lines.Line2D([0], [0], color='r')]

    plt.show()

    margins_adj = margins.copy()
    margins_adj['top'] = 45
    PdfPlotter(path / f'altered_shaping_2.3.pdf', fixed_margins=margins_adj)
    plt.figure(figsize=(2.3, 1.35), dpi=dpi)
    ax = plt.gca()

    plot_model_behavior(
        predictions['seed_run'], ax_dict={'seed_run': ax}, cue_times=cue_times,
        target=target)

    ax.tick_params(axis='x')
    ax.set_xlabel('Time (s)')

    ax.text(
        -0.01, 0.5, 'NS-Altered', color='tab:purple',
        fontsize=5, va='center', ha='right', transform=ax.transAxes,
        rotation=90, ma='center')
    
    transform = matplotlib.transforms.blended_transform_factory(
        ax.figure.transFigure, ax.transAxes)
    ax.set_ylabel('Model Response')

    legend_lines = [matplotlib.lines.Line2D([0], [0], color='k'),
                    matplotlib.lines.Line2D([0], [0], color='r')]

    plt.show()
 
    
if __name__ == '__main__':
    main(sys.argv[1:])
