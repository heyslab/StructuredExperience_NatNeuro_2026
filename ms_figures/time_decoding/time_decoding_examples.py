# [Figure 2d,f]
import os
import sys

sys.path.append('../')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools as it
import argparse
import sklearn.metrics as metrics
import math
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from classes.datagen import tDNMSGenerator, genFactory
from classes.models import LeakyRNN
from classes.models import LeakyRNNCell

import models_database as mdb

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
    no_model_ids = [18]
    with_model_ids = [25]
    idx = pd.MultiIndex.from_tuples(
            list(zip(no_model_ids, it.repeat('no_shaping'))) +
            list(zip(with_model_ids, it.repeat('shaping'))), names=('model_id', 'model_type'))
    models = pd.Series(no_model_ids + with_model_ids,  index=idx)

    model_labels = [
        'No Shaping',
        'Shaping + Full Task'
    ]

    jP.set_rcParams(plt)

    margins = jP.default_margins()
    margins['left'] = 120
    margins['right'] = 170
    dpi = 300
    path = Path('/analysis/ms_figures/time_decoding')
    jP.make_folder(path)
    cache_file = Path('time_decoding.h5')

    def load_model(model, cache_file):
        model_name = f'model_{model}'
        predictions = pd.read_hdf(cache_file, key=model_name)
        return predictions
    predictions = pd.concat(list(map(load_model, models, it.repeat(cache_file))), axis=0)
    
    true = predictions.droplevel(0).reset_index().set_index(['model_id', 'type'])['time_bin']
    predicted = predictions.droplevel(0).reset_index().set_index(['model_id', 'type'])['SL']

    def plot_cm(ax, predictions, trial_type):
        true = predictions.index.get_level_values('time_bin')
        cm = metrics.confusion_matrix(true, predictions)
        cm = cm / np.sum(cm, axis=1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap', ('#000000', TRIAL_COLORS[trial_type]))
        im = ax.imshow(cm, vmin=0, vmax=0.5, cmap=cmap)
        ax.set_title(trial_type, color=TRIAL_COLORS[trial_type])
        ax.set_xticks([5,  15])
        ax.set_yticks([5, 10, 15])

    for model_type in ('no_shaping', 'shaping'):
        PdfPlotter(path / f'example_decoding_{model_type}.SL.pdf', fixed_margins=margins)
        plt.figure(figsize=(2.9, 1.25), dpi=dpi)
        gs = gridspec.GridSpec(1, 3)
        axs = list(map(plt.subplot, gs))
        for i, trial_type in enumerate(('SL', 'LS', 'SS')):
            plot_cm(
                axs[i],
                predictions.xs(model_type, level='model_type')\
                    .xs(trial_type, level='type')['SL'],
                trial_type)
        
        def format_axis(ax):
            [ax.spines[s].set_color(TRIAL_COLORS['SL']) for s in ax.spines]
        list(map(format_axis, axs))

        [axs[0].spines[s].set_linewidth(2) for s in axs[0].spines]
        [ax.set_yticks([]) for ax in axs[1:]]
        axs[1].set_xlabel('Predicted Times (s)')
        axs[0].set_ylabel('Actual Time (s)')
        axs[-1].text(
            1, 1.125,  'Trained on SL trials only', color=TRIAL_COLORS['SL'],
            fontsize=5, transform=axs[-1].transAxes, ha='right', va='top')

        ax = axs[-1]
        inax_position = ax.transAxes.transform([0.8, 0.2]) 
        infig_position = ax.figure.transFigure.inverted().transform(inax_position)
        color_scale = ax.figure.add_axes(
                list(infig_position) +
                [ax.get_position().width * 0.09, ax.get_position().height * 0.87]) 
        color_scale.imshow(
                np.array([np.linspace(0.15, 1, 50)]).T, cmap='Grays', aspect='auto', vmin=0, vmax=1)
        color_scale.set_xticks([])
        color_scale.yaxis.tick_right()
        color_scale.set_yticks([0, 50])
        color_scale.set_yticklabels(['>50%', '0'], fontsize=5)
        color_scale.set_ylabel('Probability', labelpad=-8, rotation=-90, fontsize=5)
        color_scale.yaxis.set_label_position("right")      
     
        plt.show()

    for model_type in ('no_shaping', 'shaping'):
        PdfPlotter(path / f'example_decoding_{model_type}.LS.pdf', fixed_margins=margins)
        plt.figure(figsize=(2.9, 1.25), dpi=dpi)
        gs = gridspec.GridSpec(1, 3)
        axs = list(map(plt.subplot, gs))
        for i, trial_type in enumerate(('LS', 'SL', 'SS')):
            plot_cm(
                axs[i],
                predictions.xs(model_type, level='model_type')\
                    .xs(trial_type, level='type')['LS'],
                trial_type)
        
        def format_axis(ax):
            [ax.spines[s].set_color(TRIAL_COLORS['LS']) for s in ax.spines]
        list(map(format_axis, axs))

        [axs[0].spines[s].set_linewidth(2) for s in axs[0].spines]
        [ax.set_yticks([]) for ax in axs[1:]]
        axs[1].set_xlabel('Predicted Times (s)')
        axs[0].set_ylabel('Actual Time (s)')
        axs[-1].text(
            1, 1.125,  'Trained on LS trials only', color=TRIAL_COLORS['LS'],
            fontsize=5, transform=axs[-1].transAxes, ha='right', va='top')

        ax = axs[-1]
        inax_position = ax.transAxes.transform([0.8, 0.2]) 
        infig_position = ax.figure.transFigure.inverted().transform(inax_position)
        color_scale = ax.figure.add_axes(
                list(infig_position) +
                [ax.get_position().width * 0.09, ax.get_position().height * 0.87]) 
        color_scale.imshow(
                np.array([np.linspace(0.15, 1, 50)]).T, cmap='Grays', aspect='auto', vmin=0, vmax=1)
        color_scale.set_xticks([])
        color_scale.yaxis.tick_right()
        color_scale.set_yticks([0, 50])
        color_scale.set_yticklabels(['>50%', '0'], fontsize=5)
        color_scale.set_ylabel('Probability', labelpad=-8, rotation=-90, fontsize=5)
        color_scale.yaxis.set_label_position("right")      
     
        plt.show()



if __name__ == '__main__':
    main(sys.argv[1:])
