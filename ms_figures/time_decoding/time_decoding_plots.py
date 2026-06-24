# [Figure 2e,g]
import os
import sys

sys.path.append('../../')

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
SHAPING_COLORS = {'no_shaping': 'tab:blue', 'shaping': 'tab:orange', 'ls_shaping': 'tab:green'}

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  
    "axes.facecolor":    (1.0, 1.0, 1.0, 0.0),  
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),
    })


def load_model(model_info):
    filename = Path(model_info['path']).parent /\
        f'model.{model_info["model_id"]}_time_decoding.h5'
    predictions = pd.read_hdf(filename)
    return predictions['circ_error']



def main(argv):
    model_ids = [9, 18, 19, 24, 28, 29, 31, 32, 93, 109, 116, 119]
    model_ids.extend([20, 22, 25, 21, 23, 26, 27, 30, 76, 79, 82, 84])

    model_infos = pd.DataFrame(list(map(mdb.get_model, model_ids)))
    model_infos['model_type'] = ['no_shaping' if x == 0 else 'shaping' for x in model_infos['base_id']]
    model_infos.set_index(['model_id', 'model_type'])

    error = pd.concat(model_infos.apply(load_model, axis=1).values, keys=model_infos['model_id'])
    error['model_type']= error.align(model_infos[['model_id', 'model_type']].set_index('model_id'), level='model_id')[1]['model_type']
    error = error.set_index('model_type', append=True)

    model_labels = [
        'No Shaping',
        'Shaping + Full Task'
    ]

    jP.set_rcParams(plt)

    margins = jP.default_margins()
    dpi = 300
    path = Path('/analysis/ms_figures/time_decoding')
    path.mkdir(exist_ok=True, parents=True)
    mse = (error**2).groupby(['model_id', 'model_type', 'type', 'split']).mean()

    mean_mse = mse.groupby(['model_type', 'type', 'model_id']).mean()
    mean_mse.columns.name = 'train_type'

    cols = list(mean_mse.columns)
    pvalues = mean_mse.stack().groupby(
        ['train_type', 'type', 'model_type']).apply(list).groupby(
            ['train_type', 'type']).apply(
                lambda x, scipy=scipy: scipy.stats.ttest_ind(*x)).apply(
                    pd.Series, index=['stat', 'p-value'])

    PdfPlotter(path / f'mse_comparison.pdf',
               fixed_margins=jP.default_margins())
    plt.figure(figsize=(3.5, 1.5), dpi=dpi)
    gs = gridspec.GridSpec(1, 3, width_ratios=(1, 1, 1), wspace=0.25)
    gss = gs[2].subgridspec(2, 2, width_ratios=(0.65, 1),
                            height_ratios=(1, 0.65))
    axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]
    axs = axs + [plt.subplot(gss[0, 1])]
    plot_order = list(mean_mse.index.unique('type'))
    mean_mse = mean_mse.reindex(plot_order, axis=0, level='type')
    mean_mse = mean_mse.reindex(plot_order + ['all'], axis=1)

    axs[0].imshow(mean_mse.xs('no_shaping', level='model_type').groupby('type').mean().T, vmax=20, vmin=0, cmap='inferno')
    im = axs[1].imshow(mean_mse.xs('shaping', level='model_type').groupby('type').mean().T, vmax=20, vmin=0, cmap='inferno')

    axs[2].set_xlim(-0.5, len(plot_order) - 0.5)
    axs[2].set_ylim(len(plot_order) + 0.5, -0.5)
    axs[2].grid(visible=True, c='k')
    axs[2].set_xticks(np.arange(len(plot_order)) + 0.5)
    axs[2].set_yticks(np.arange(len(plot_order)) + 0.5)
    axs[2].set_xticklabels([])
    axs[2].set_yticklabels([])
    axs[2].tick_params(axis='both', length=0)
    axs[2].set_title('Significant\nDifferences', fontsize=5)

    im2 = axs[2].imshow(
        np.log(pvalues['p-value']).unstack('type').reindex(
            plot_order, axis=1).reindex(plot_order + ['all'], axis=0), cmap='gray', alpha=0.25, vmax=0, vmin=-40)
    cbar2 = jP.color_bar(axs[2], im2)
    cbar2.set_ticks([])
    cbar2.set_label(r'log($p$)', rotation=-90, va='center')

    sig = jP.significance_symbols(pvalues['p-value'] * 12)
    sig = sig.unstack('type').reindex(plot_order, axis=1).reindex(plot_order + ['all'], axis=0)
    sig[sig == 'n.s.'] = ''
    sig = sig.fillna('')
    xy = np.meshgrid(np.arange(len((plot_order))), np.arange(len(plot_order) + 1))
    for x, y in zip(*map(np.ravel, xy)):
        axs[2].text(x, y, sig.iloc[y, x], va='center', ha='center')

    def setup_ax(ax, plot_order=plot_order):
        ax.set_yticks(np.arange(len(plot_order) + 1))
        ax.set_yticklabels(plot_order + ['All'])
        ax.set_xticks(np.arange(len(plot_order)))
        ax.set_xticklabels(plot_order)
        ax.set_xlabel('Testing Trials')

    setup_ax(axs[0])
    setup_ax(axs[1])
    axs[0].set_ylabel('Training Trials')
    axs[0].set_title(model_labels[0], color=SHAPING_COLORS['no_shaping'])
    axs[1].set_title('Shaping\n+ Full Task', color=SHAPING_COLORS['shaping'])
    axs[1].set_yticks([])
    cbar = jP.color_bar(axs[1], im)
    cbar.set_ticks([0, 20])
    cbar.set_label('Time Decoding\nError', rotation=-90, va='top')
    plt.show()

    with open(path / 'info_file.txt', 'w') as f:
        f.writelines(['Cross Context MSE:\n', str(pvalues), '\n\n'])

    error_list = mean_mse.drop('all', axis=1).stack()
    error_idx = error_list.index.to_frame()
    error_idx['same'] = error_idx['type'] == error_idx['train_type']
    error_list.index = pd.MultiIndex.from_frame(error_idx.drop('type', axis=1).drop('train_type', axis=1))

    margins={'left': 90, 'right': 45, 'top': 100, 'bottom': 80}
    PdfPlotter(path / f'mse_bar.pdf',
               fixed_margins=margins)
    plt.figure(figsize=(1.5, 1.5), dpi=dpi)
    ax = plt.subplot()

    def box_plotter(X, i, colors, ax):
        flierprops={'marker': '.', 'markersize': 2, 'mfc': 'k'}
        boxprops={'facecolor': next(colors), 'edgecolor': 'k'}
        medianprops={'color': 'k'}
        ax.boxplot(
            [X], positions=[next(i)], widths=0.5, flierprops=flierprops,
            boxprops=boxprops, patch_artist=True, medianprops=medianprops)

    error_groups = error_list.groupby(['same', 'model_type']).apply(list)
    error_groups = error_groups.reindex([True, False], level='same').reindex(
        ['no_shaping', 'shaping'], level='model_type')
    error_groups.apply(box_plotter, i=iter([0.15, 0.85, 1.85, 2.55]), colors=iter(['tab:blue', 'tab:orange']*2), ax=ax)

    pvalues = error_list.groupby(
        ['same', 'model_type']).apply(list).groupby('same').apply(
            lambda x: scipy.stats.ttest_ind(*x))
    jP.configure_spines(ax)
    ax.set_xticks([0.5, 2.2])
    ax.set_xticklabels(['Same\nContext', 'Cross-Context'])
    ax.set_ylabel('Time Decoding MSE')
    ax.set_yticks([5, 15, 25])
    ax.set_ylim(0, 25)

    sigs = jP.significance_symbols(pvalues.apply(pd.Series)[1])
    lbl_y = error_list.max() + jP.annotation_padding(ax, 0.08)
    jP.annotation(ax, (1.85, 2.55), lbl_y, sigs[False], va='bottom')

    lbl_y = error_list.xs(True, level='same').max() + jP.annotation_padding(ax, 0.08)
    jP.annotation(ax, (0.15, 0.85), lbl_y, sigs[True], va='bottom')

    ax.legend(handles=([
        matplotlib.patches.Patch(color='tab:blue', label='No Shaping'),
        matplotlib.patches.Patch(color='tab:orange', label='With Shaping')]),
        loc='lower right', bbox_to_anchor=(1.05, 1))

    plt.show()


    with open(path / 'info_file.txt', 'a') as f:
        f.writelines(['bar chart pvalues:\n', str(pvalues), '\n\n'])

    PdfPlotter(path / f'mse_comparison_nop.pdf',
               fixed_margins=jP.default_margins())
    plt.figure(figsize=(2.3, 1.35), dpi=dpi)
    gs = gridspec.GridSpec(1, 2, wspace=0.1)
    axs = [plt.subplot(gs[0]), plt.subplot(gs[1])]
    plot_order = list(mean_mse.index.unique('type'))
    mean_mse = mean_mse.reindex(plot_order, axis=0, level='type')
    mean_mse = mean_mse.reindex(plot_order + ['all'], axis=1)

    axs[0].imshow(
        mean_mse.xs('no_shaping', level='model_type').groupby('type').mean().T,
        vmax=20, vmin=0, cmap='inferno')
    im = axs[1].imshow(
        mean_mse.xs('shaping', level='model_type').groupby('type').mean().T,
        vmax=20, vmin=0, cmap='inferno')

    def setup_ax(ax, plot_order=plot_order):
        ax.set_yticks(np.arange(len(plot_order) + 1))
        ax.set_yticklabels(plot_order + ['All'])
        ax.set_xticks(np.arange(len(plot_order)))
        ax.set_xticklabels(plot_order)
        ax.set_xlabel('Testing Context')

    setup_ax(axs[0])
    setup_ax(axs[1])
    axs[0].set_ylabel('Decorder Training\nContext')
    axs[0].set_title(model_labels[0], color=SHAPING_COLORS['no_shaping'])
    axs[1].set_title('Shaping\n+ Full Task', color=SHAPING_COLORS['shaping'])
    axs[1].set_yticks([])
    cbar = jP.color_bar(axs[1], im)
    cbar.set_ticks([0, 20])
    cbar.set_label('Time Decoding\nError', rotation=-90, va='top')
    plt.show()

    margins={'left': 90, 'right': 45, 'top': 100, 'bottom': 80}
    PdfPlotter(path / f'mse_bar_small.pdf',
               fixed_margins=margins)
    plt.figure(figsize=(1.5, 1.35), dpi=dpi)
    ax = plt.subplot()

    def box_plotter(X, i, colors, ax):
        flierprops={'marker': '.', 'markersize': 2, 'mfc': 'k'}
        boxprops={'facecolor': next(colors), 'edgecolor': 'k'}
        medianprops={'color': 'k'}
        ax.boxplot(
            [X], positions=[next(i)], widths=0.5, flierprops=flierprops,
            boxprops=boxprops, patch_artist=True, medianprops=medianprops)

    error_groups = error_list.groupby(['same', 'model_type']).apply(list)
    error_groups = error_groups.reindex([True, False], level='same').reindex(
        ['no_shaping', 'shaping'], level='model_type')
    error_groups.apply(box_plotter, i=iter([0.15, 0.85, 1.85, 2.55]), colors=iter(['tab:blue', 'tab:orange']*2), ax=ax)

    pvalues = error_list.groupby(
        ['same', 'model_type']).apply(list).groupby('same').apply(
            lambda x: scipy.stats.ttest_ind(*x))
    jP.configure_spines(ax)
    ax.set_xticks([0.5, 2.2])
    ax.set_xticklabels(['Same\nContext', 'Cross-Context'])
    ax.set_ylabel('Time Decoding MSE')
    ax.set_yticks([5, 15, 25])
    ax.set_ylim(0, 25)

    sigs = jP.significance_symbols(pvalues.apply(pd.Series)[1])
    lbl_y = error_list.max() + jP.annotation_padding(ax, 0.08)
    jP.annotation(ax, (1.85, 2.55), lbl_y, sigs[False], va='bottom')

    lbl_y = error_list.xs(True, level='same').max() + jP.annotation_padding(ax, 0.08)
    jP.annotation(ax, (0.15, 0.85), lbl_y, sigs[True], va='bottom')

    ax.legend(handles=([
        matplotlib.patches.Patch(color='tab:blue', label='No Shaping'),
        matplotlib.patches.Patch(color='tab:orange', label='With Shaping')]),
        loc='lower right', bbox_to_anchor=(1.05, 1.03))

    plt.show()

    margins={'left': 120, 'right': 45, 'top': 100, 'bottom': 80}
    PdfPlotter(path / f'mse_bar_horiz.pdf',
               fixed_margins=margins)
    plt.figure(figsize=(2.9, 1.15), dpi=dpi)
    ax = plt.subplot()

    def box_plotter(X, i, colors, ax, **kwargs):
        flierprops={'marker': '.', 'markersize': 2, 'mfc': 'k'}
        boxprops={'facecolor': next(colors), 'edgecolor': 'k'}
        medianprops={'color': 'k'}
        ax.boxplot(
            [X], positions=[next(i)], widths=0.5, flierprops=flierprops,
            boxprops=boxprops, patch_artist=True, medianprops=medianprops,
            **kwargs)

    error_groups = error_list.groupby(['same', 'model_type']).apply(list)
    error_groups = error_groups.reindex([False, True], level='same').reindex(
        ['no_shaping', 'shaping'], level='model_type')
    error_groups.apply(
        box_plotter, i=iter([0.15, 0.85, 1.85, 2.55]),
        colors=iter(['tab:blue', 'tab:orange']*2), ax=ax, vert=False)

    pvalues = error_list.groupby(
        ['same', 'model_type']).apply(list).groupby('same').apply(
            lambda x: scipy.stats.ttest_ind(*x))
    jP.configure_spines(ax)
    ax.set_yticks([0.5, 2.2])
    ax.set_yticklabels(['Cross-\nContext', 'Same\nContext'])
    ax.set_xlabel('Time Decoding MSE')
    ax.set_xticks([5, 15, 25])
    ax.set_xlim(0, 25)

    sigs = jP.significance_symbols(pvalues.apply(pd.Series)[1])
    lbl_y = error_list.max() + jP.annotation_padding(ax, 0.08)
    jP.annotation_horiz(ax, (0.15, 0.85), lbl_y, sigs[False], ha='left')

    lbl_y = error_list.xs(True, level='same').max() + jP.annotation_padding(ax, 0.08)
    jP.annotation_horiz(ax, (1.85, 2.55), lbl_y, sigs[True], ha='left')

    ax.legend(handles=([
        matplotlib.patches.Patch(color='tab:blue', label='NS'),
        matplotlib.patches.Patch(color='tab:orange', label='S/FT')]),
        loc='lower right', bbox_to_anchor=(1.05, 0.9))

    plt.show()




    error_list.name = 'values'
    X = error_list.reset_index()
    model = ols("values ~ C(model_type)*C(same)", X).fit()
    aov_table = anova_lm(model, typ=2)
    print(aov_table)
    tukey = pairwise_tukeyhsd(
        endog=X['values'], groups=list(map(str, zip(X['model_type'],
        X['same']))), alpha=0.05)
    print(tukey)
    print(tukey.pvalues)


if __name__ == '__main__':
    main(sys.argv[1:])
