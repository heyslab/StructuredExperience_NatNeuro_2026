# [Figure 1c]
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
SHAPING_COLORS = {'no_shaping': 'tab:blue', 'shaping': 'tab:orange'}

def plot_training_compare_mse(ax, history_noshape, base_history, history, color='tab:red'):

    def plot_m_std(ax, hist, c, ls='-', win_std=5):
        m = hist.groupby('idx').mean()
        std = hist.groupby('idx').std()
        x = m.index
        m = m.rolling(win_std**2, center=True, min_periods=1, win_type='gaussian').mean(std=win_std)
        s = std
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.25)
        ax.plot(x, m, c=c, ls=ls)

    plot_m_std(ax, history_noshape, c='tab:blue')
    plot_m_std(ax, base_history, c=color, ls='--')

    offset = base_history.index.get_level_values('idx').max()
    history_adj = history.reset_index(level='idx')
    history_adj['idx'] += offset + 1
    history_adj = history_adj.set_index('idx', append=True)[0]
    plot_m_std(ax, history_adj, c=color)

    jP.configure_spines(ax)

    legend_lines = [matplotlib.lines.Line2D([0], [0], color='tab:blue', lw=1),
                    matplotlib.lines.Line2D([0], [0], color=color, lw=1, ls='--'),
                    matplotlib.lines.Line2D([0], [0], color=color, lw=1)]
    ax.legend(
        legend_lines, ['tDNMS No Pre-training', 'Shaping Task', 'tDNMS Pre-trained'],
        bbox_to_anchor=[1.0, 1.1], loc='upper right')
    ax.axvspan(0, offset, color='k', alpha=0.1, zorder=-10)
    ax.axvline(offset, color='k', ls='--', zorder=-10, lw=0.5)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, -10))
    ax.set_yticks([0.02, 0.1])
    ax.set_xlabel('Training Block')
    ax.set_ylabel('Mean Squared\nError')


def main(argv):
    no_model_ids = [9, 18, 19, 24, 28, 29, 31, 32, 93, 109, 116, 119]
    with_model_ids = [20, 25, 21, 23, 27, 30, 76, 79, 82, 84, 132, 142]
    ls_model_ids = [62, 66, 67, 70, 63, 65, 68, 71, 75, 78, 81]
    sl_model_ids = [41, 47, 42, 45, 43, 46, 64, 80, 83, 85]

    models = pd.concat(list(map(pd.Series, (no_model_ids, with_model_ids, ls_model_ids, sl_model_ids))),
                       keys=('no_shaping', 'shaping', 'ls_shaping', 'sl_shaping'))
    models = models.droplevel(1)
    models.index.name = 'type'

    model_infos = models.apply(mdb.get_model).apply(pd.Series)
    model_infos = model_infos.reset_index().set_index(['type', 'model_id'])
    base_infos = model_infos.drop('no_shaping')['base_id'].apply(mdb.get_model).apply(pd.Series).droplevel('model_id')
    base_infos = base_infos.reset_index().set_index(['type', 'model_id'])

    history = model_infos['path'].apply(
        lambda x: pd.read_hdf(x + '.dat.h5', key='history')['val_mean_squared_error'])
    history.columns.name = 'idx'
    history = history.stack()
    base_history = base_infos['path'].apply(
        lambda x: pd.read_hdf(x + '.dat.h5', key='history')['val_mean_squared_error'])
    base_history.columns.name = 'idx'
    base_history = base_history.stack()

    dpi = 300
    figsize = (2.5, 1.7)
    margins = jP.default_margins()
    path = Path('/analysis/ms_figures/mse_evolution')
    jP.make_folder(path)
    jP.set_rcParams(plt)

    PdfPlotter(path / 'no_match_shaping.pdf', fixed_margins=margins)
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    plot_training_compare_mse(
        ax, history.xs('no_shaping'), base_history.reindex(model_infos.xs('shaping')['base_id'], level='model_id'),
        history.xs('shaping'), color='tab:orange')
    jP.set_ylabel_position(ax, nlines=3)
    plt.show()

    
    adj_margins = margins.copy()
    adj_margins['top'] = 45
    adj_margins['left'] = 80
    adj_margins['right'] = 45
    PdfPlotter(path / 'no_match_shaping.ms_lessSmall.pdf', fixed_margins=adj_margins)
    plt.figure(figsize=(2, 1.3), dpi=dpi)
    ax = plt.gca()
    plot_training_compare_mse(
        ax, history.xs('no_shaping'), base_history.reindex(model_infos.xs('shaping')['base_id'], level='model_id'),
        history.xs('shaping'), color='tab:orange')
    jP.set_ylabel_position(ax, nlines=3)
    ax.set_ylabel('')
    color = 'tab:orange'
    legend_lines = [matplotlib.lines.Line2D([0], [0], color='tab:blue', lw=1),
                    matplotlib.lines.Line2D([0], [0], color=color, lw=1, ls='--'),
                    matplotlib.lines.Line2D([0], [0], color=color, lw=1)]
    ax.legend(
        legend_lines, ['NS', 'Shaping', 'S/FT'],
        bbox_to_anchor=[1.0, 1.0], loc='upper right', borderaxespad=0)
    ax.set_xlim(0, 1000)
       
    ax.text(
        0.055, 0.55, 'Mean Squared\nError',
        transform=ax.figure.transFigure, rotation=90,
        ha='center', va='center')
    plt.show()

    adj_margins = margins.copy()
    adj_margins['top'] = 45
    adj_margins['right'] = 45
    PdfPlotter(path / 'no_match_shaping_2.5.pdf', fixed_margins=adj_margins)
    plt.figure(figsize=(2.5, 1.3), dpi=dpi)
    ax = plt.gca()
    plot_training_compare_mse(
        ax, history.xs('no_shaping'), base_history.reindex(model_infos.xs('shaping')['base_id'], level='model_id'),
        history.xs('shaping'), color='tab:orange')
    jP.set_ylabel_position(ax, nlines=2.5)
    ax.set_ylabel('Mean Squared\nError')
    color = 'tab:orange'
    legend_lines = [matplotlib.lines.Line2D([0], [0], color='tab:blue', lw=1),
                    matplotlib.lines.Line2D([0], [0], color=color, lw=1, ls='--'),
                    matplotlib.lines.Line2D([0], [0], color=color, lw=1)]
    ax.legend(
        legend_lines, ['NS', 'Shaping', 'S/FT'],
        bbox_to_anchor=[1.0, 1.0], loc='upper right', borderaxespad=0)
    ax.set_xlim(0, 1000)
    plt.show()


    PdfPlotter(path / 'no_match_shaping.ms_small.pdf', fixed_margins=margins)
    plt.figure(figsize=(2, 1.2), dpi=dpi)
    ax = plt.gca()
    plot_training_compare_mse(
        ax, history.xs('no_shaping'), base_history.reindex(model_infos.xs('shaping')['base_id'], level='model_id'),
        history.xs('shaping'), color='tab:orange')
    jP.set_ylabel_position(ax, nlines=3)
    color = 'tab:orange'
    legend_lines = [matplotlib.lines.Line2D([0], [0], color='tab:blue', lw=1),
                    matplotlib.lines.Line2D([0], [0], color=color, lw=1, ls='--'),
                    matplotlib.lines.Line2D([0], [0], color=color, lw=1)]
    ax.legend(
        legend_lines, ['NS', 'Shaping', 'S/FT'],
        bbox_to_anchor=[1.0, 1.1], loc='upper right')
    ax.set_xlim(0, 1000)
    plt.show()

    adj_margins = {'left': 95, 'right': 45, 'top': 45, 'bottom': 80}
    PdfPlotter(path / 'no_match_shaping.poster.pdf', fixed_margins=adj_margins)
    plt.figure(figsize=(2.365, 1.25), dpi=dpi)
    ax = plt.gca()
    plot_training_compare_mse(
        ax, history.xs('no_shaping'), base_history.reindex(model_infos.xs('shaping')['base_id'], level='model_id'),
        history.xs('shaping'), color='tab:orange')
    jP.set_ylabel_position(ax, nlines=2.5)
    plt.show()

    PdfPlotter(path / 'sl_shaping.pdf', fixed_margins=margins)
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    plot_training_compare_mse(
        ax, history.xs('no_shaping'), base_history.reindex(model_infos.xs('sl_shaping')['base_id'], level='model_id'),
        history.xs('sl_shaping'), color='tab:green')
    jP.set_ylabel_position(ax, nlines=3)
    color = 'tab:green'
    legend_lines = [matplotlib.lines.Line2D([0], [0], color='tab:blue', lw=1),
                    matplotlib.lines.Line2D([0], [0], color=color, lw=1, ls='--'),
                    matplotlib.lines.Line2D([0], [0], color=color, lw=1)]
    ax.legend(
        legend_lines, ['NS', 'SL Only', 'SL/FT'],
        bbox_to_anchor=[1.0, 1.1], loc='upper right')
    ax.set_xlim(0, 1000)
    plt.show()

    PdfPlotter(path / 'ls_shaping.pdf', fixed_margins=margins)
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()
    plot_training_compare_mse(
        ax, history.xs('no_shaping'), base_history.reindex(model_infos.xs('ls_shaping')['base_id'], level='model_id'),
        history.xs('ls_shaping'), color='tab:red')
    jP.set_ylabel_position(ax, nlines=3)
    color = 'tab:red'
    legend_lines = [matplotlib.lines.Line2D([0], [0], color='tab:blue', lw=1),
                    matplotlib.lines.Line2D([0], [0], color=color, lw=1, ls='--'),
                    matplotlib.lines.Line2D([0], [0], color=color, lw=1)]
    ax.legend(
        legend_lines, ['NS', 'LS Only', 'LS/FT'],
        bbox_to_anchor=[1.0, 1.1], loc='upper right')
    ax.set_xlim(0, 1000)
    plt.show()

    print(models.groupby('type').count())


if __name__ == '__main__':
    main(sys.argv[1:])
