# [Figure 3g]
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
import scipy

import matplotlib.gridspec as gridspec

from analysis_tools.mpl_helpers import PdfPlotter
import analysis_tools.jPlots as jP

from classes.models import LeakyRNN, LeakyRNNCell
from classes.datagen import genFactory
import models_database as mdb

TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'S': '#f89521'}
SHAPING_COLORS = {'no_shaping': 'tab:blue', 'shaping': 'tab:orange'}

def plot_training_compare_mse(ax, history_noshape, history, color='tab:red'):

    def plot_m_std(ax, hist, c, ls='-', win_std=5):
        m = hist.groupby('idx').mean()
        std = hist.groupby('idx').std()
        x = m.index
        m = m.rolling(win_std**2, center=True, min_periods=1, win_type='gaussian').mean(std=win_std)
        s = std
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.25)
        ax.plot(x, m, c=c, ls=ls)

    plot_m_std(ax, history_noshape, c='tab:blue')

    plot_m_std(ax, history, c=color)

    jP.configure_spines(ax)

    legend_lines = [matplotlib.lines.Line2D([0], [0], color='tab:blue', lw=1),
                    matplotlib.lines.Line2D([0], [0], color=color, lw=1, ls='--'),
                    matplotlib.lines.Line2D([0], [0], color=color, lw=1)]
    ax.legend(
        legend_lines, ['tDNMS No Pre-training', 'Shaping Task', 'tDNMS Pre-trained'],
        bbox_to_anchor=[1.0, 1.1], loc='upper right')
    ax.axvline(0, ls='--', color='k')
    ax.axvline(100, ls='--', color='k')
    ax.axvline(200, ls='--', color='k')
    ax.axvline(300, ls='--', color='k')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, -10))
    ax.set_yticks([0.02, 0.1])
    ax.set_xlabel('Training Block')
    ax.set_ylabel('Mean Squared\nError')


def main(argv):
    no_model_ids = [9, 18, 19, 24, 28, 29, 31, 32, 93, 109, 116, 119]
    seed_model_ids = [179, 180, 181, 182, 183, 184]

    models = pd.concat(list(map(pd.Series, (no_model_ids, seed_model_ids))),
                       keys=('no_shaping', 'seed_run'))
    models = models.droplevel(1)
    models.index.name = 'type'

    model_infos = models.apply(mdb.get_model).apply(pd.Series)
    model_infos = model_infos.reset_index().set_index(['type', 'model_id'])

    history = model_infos['path'].apply(
        lambda x: pd.read_hdf(x + '.dat.h5', key='history')['val_mean_squared_error'].reset_index())
    history = pd.concat(history.values, keys=model_infos.index)['val_mean_squared_error']
    history.index.names = ['type', 'model_id', 'idx']

    dpi = 300
    figsize = (2.5, 1.7)
    margins = jP.default_margins()
    path = Path('/analysis/ms_figures/seed_mse_evolution')
    path.mkdir(exist_ok=True, parents=True)
    jP.set_rcParams(plt)

    PdfPlotter(path / 'no_shaping.pdf', fixed_margins=margins)
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    plot_training_compare_mse(
        ax, history.xs('no_shaping'), 
        history.xs('seed_run'), color='tab:purple')
    jP.set_ylabel_position(ax, nlines=3)
    plt.show()

    adj_margins = margins.copy()
    adj_margins['top'] = 45
    PdfPlotter(path / 'no_shaping.ms_small.pdf', fixed_margins=adj_margins)
    plt.figure(figsize=(2, 1.35), dpi=dpi)
    ax = plt.gca()
    plot_training_compare_mse(
        ax, history.xs('no_shaping'), 
        history.xs('seed_run'), color='tab:purple')
    jP.set_ylabel_position(ax, nlines=3)
    color = 'tab:purple'
    legend_lines = [matplotlib.lines.Line2D([0], [0], color='tab:blue', lw=1),
                    matplotlib.lines.Line2D([0], [0], color='k', lw=1, ls='--'),
                    matplotlib.lines.Line2D([0], [0], color=color, lw=1)]
    ax.legend(
        legend_lines, ['NS', 'Intervention', 'NS-Altered'],
        bbox_to_anchor=[1.0, 1.1], loc='upper right', borderaxespad=0)
    ax.set_xlim(0, 1000)
    plt.show()

    print(scipy.stats.ttest_ind(*history.groupby('model_id').tail(1).groupby('type').apply(list).values))

if __name__ == '__main__':
    main(sys.argv[1:])
