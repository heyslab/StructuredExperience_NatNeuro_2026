# [Figure 7c]
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
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),
    })

TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'LL': '#253D5B', 'S': 'k', 'L': 'r',
                'MM': '#DA627D', 'XL': '#96ACB7', 'super_long': '#96ACB7'}
SHAPING_COLORS = {'no_shaping': 'tab:blue', 'shaping': 'tab:orange', 'sl_only': 'tab:green', 'ls_only': 'tab:red'}

def generate_probe_input(task, input_noise=0.15):
    trials_gen = genFactory.create(
        task, input_noise=input_noise, batch_size=1, n_blocks=1)
    inputs = trials_gen.generate_trials(4)
    trial_type = inputs.index.unique('type')\
                       .intersection(('super_long', 'LL', 'MM'))\
                       .tolist().pop()

    inputs =  inputs.xs(trial_type, level='type', drop_level=False)
    trial_ids = inputs.index.unique('trial')[:4]
    return inputs.reindex(trial_ids, level='trial')

def main(argv):
    models = pd.Series(
        [132],  index=pd.Index(['shaping'],
        name='model_type'))

    #margins={'left': 80, 'right': 90, 'top': 45, 'bottom': 80}
    margins = jP.default_margins()
    margins['top'] = 60

    model_infos = models.apply(mdb.get_model).apply(pd.Series)
    model_infos = pd.concat(
        (model_infos, models.apply(mdb.get_model_attributes)), axis=1)

    path = Path('/analysis/ms_figures/probe_rnn_behavior')
    path.mkdir(exist_ok=True, parents=True)

    dpi = 300
    jP.set_rcParams(plt)

    models = model_infos['path'].apply(tf.keras.models.load_model)
    probe_input = pd.concat(
        list(map(generate_probe_input,
                 ('include_long_match', 'include_med_match', 'super_long'))),
        keys=pd.Index(['LL', 'MM', 'XL'], name='probe_type'))
    ll_resp = np.squeeze(
        models['shaping'].predict(np.expand_dims(probe_input.xs('LL', level='probe_type')[['light', 'cues']], 0)))
    mm_resp = np.squeeze(
        models['shaping'].predict(np.expand_dims(probe_input.xs('MM', level='probe_type')[['light', 'cues']], 0)))
    xl_resp = np.squeeze(
        models['shaping'].predict(np.expand_dims(probe_input.xs('XL', level='probe_type')[['light', 'cues']], 0)))
    predictions = pd.concat(
        list(map(pd.Series, (ll_resp, mm_resp, xl_resp))), keys=pd.Index(['LL', 'MM', 'XL'], name='probe_type'))
    idx = probe_input.reset_index()[['probe_type', 'trial', 'type', 'idx', 'cues', 'response']]
    predictions.index = pd.MultiIndex.from_frame(idx)
 
    def plot_cues(cue_times, ax):
        plot_args = {
            'color': TRIAL_COLORS[cue_times.index.unique('type')[0]],
            'alpha': 0.15,
            'zorder': -100}
        cue_times.apply(
            lambda x, ax=ax, plot_args=plot_args:
                ax.axvspan(*x, **plot_args), axis=1)

    def plot_model_behavior(predictions, ax, plot_cues=plot_cues):
        model_type = predictions.name
        cue_transitions = predictions.index.to_frame()['cues'].droplevel('cues').diff()
        cue_transitions = cue_transitions.reset_index().reset_index().set_index(
            ['index', 'trial', 'type', 'idx'])['cues']
        cue_times = pd.concat(
            list(map(cue_transitions[cue_transitions != 0
                                     ].dropna().reset_index().set_index(
                                         ['cues', 'trial', 'type'])['index'].xs,
                     (1, -1))), axis=1, keys=('start', 'stop'))/10
        cue_times.groupby('type').apply(plot_cues, ax=ax)
        target = predictions.index.get_level_values('response').to_series(index=predictions.index)

        ax.plot(np.arange(target.shape[0])/10, target.values, c='k',
            ls='-', label='target')
        #ax.plot(np.arange(predictions.shape[0])/10, predictions.values, c=SHAPING_COLORS[model_type],
        #    lw=0.7, label='response')
        ax.plot(np.arange(predictions.shape[0])/10, predictions.values, c='r',
            lw=1, label='response')
        jP.configure_spines(ax)
        ax.set_xlim(0, len(predictions)/10)
        ax.set_xticks([20, 40, 60])


    PdfPlotter(path / f'probe_performance_larger.pdf', fixed_margins=margins)
    plt.figure(figsize=(4.5, 1), dpi=dpi)
    gs = gridspec.GridSpec(1, 3)
    axs = map(plt.subplot, gs)
    ax_dict = {'LL': next(axs), 'MM': next(axs), 'XL': next(axs)}
    for probe_type in ax_dict.keys():
        plot_model_behavior(
            predictions[probe_type], ax_dict[probe_type])
        if probe_type != 'LL':
            ax_dict[probe_type].set_yticks([])
        ax_dict[probe_type].set_xlabel('Time (s)')
        ax_dict[probe_type].set_ylim([-0.2, 1])
        ax_dict[probe_type].set_yticks([0, 1])
        ax_dict[probe_type].set_title(probe_type, color=TRIAL_COLORS[probe_type])

    ax_dict['LL'].set_ylabel('Response\nRNN')
    jP.set_ylabel_position(ax_dict['LL'], nlines=2.35)
    plt.show()

    PdfPlotter(path / f'probe_performance_5.pdf', fixed_margins=margins)
    plt.figure(figsize=(5, 1), dpi=dpi)
    gs = gridspec.GridSpec(1, 3)
    axs = map(plt.subplot, gs)
    ax_dict = {'LL': next(axs), 'MM': next(axs), 'XL': next(axs)}
    for probe_type in ax_dict.keys():
        plot_model_behavior(
            predictions[probe_type], ax_dict[probe_type])
        if probe_type != 'LL':
            ax_dict[probe_type].set_yticks([])
        ax_dict[probe_type].set_xlabel('Time (s)')
        ax_dict[probe_type].set_ylim([-0.2, 1])
        ax_dict[probe_type].set_yticks([0, 1])
        ax_dict[probe_type].set_title(probe_type, color=TRIAL_COLORS[probe_type])

    ax_dict['LL'].set_ylabel('RNN\nResponse')
    jP.set_ylabel_position(ax_dict['LL'], nlines=2.35)
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])
