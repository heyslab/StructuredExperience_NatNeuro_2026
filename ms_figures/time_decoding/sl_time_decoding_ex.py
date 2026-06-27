# [Figure S8a,b,c]
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
SHAPING_COLORS = {'no_shaping': 'tab:blue', 'shaping': 'tab:orange', 'sl_shaping': 'tab:green'}

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  
    "axes.facecolor":    (1.0, 1.0, 1.0, 0.0),  
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),
    })

def main(argv):
    with_model_ids = [20, 22, 25, 21, 23, 26, 27, 30, 76, 79, 82, 84]
    no_model_ids = [9, 18, 19, 24, 28, 29, 31, 32, 93, 109, 116, 119]
    sl_model_ids = [41, 47, 42, 45, 43, 46, 64, 80, 83, 85]

    idx = pd.MultiIndex.from_tuples(
            list(zip(with_model_ids, it.repeat('shaping'))) +
            list(zip(sl_model_ids, it.repeat('sl_shaping'))), names=('model_id', 'model_type'))
    model_ids = pd.Series(with_model_ids + sl_model_ids,  index=idx)

    model_infos = model_ids.apply(mdb.get_model).apply(pd.Series)
    model_infos = pd.concat(
        (model_infos, model_ids.apply(mdb.get_model_attributes).drop('model_id', axis=1)), axis=1)

    model_labels = [
        'No Shaping',
        'Shaping + Full Task'
        'LS Only + Full Task'
    ]

    jP.set_rcParams(plt)

    margins = jP.default_margins()
    margins['left'] = 120
    margins['right'] = 170
    dpi = 300
    path = Path('/analysis/ms_figures/sl_time_decoding')
    path.mkdir(exist_ok=True, parents=True)
    info_file = path / 'sl_time_decoding.info.txt'
    with open(info_file, 'w') as f:
        pass

    def load_model(model_info):
        cache_file = Path(model_info['path']).parent \
            / f'model.{model_info["model_id"]}_time_decoding.h5'
        predictions = pd.read_hdf(cache_file)
        return predictions
    predictions = model_infos.apply(load_model, axis=1)
    predictions = pd.concat(predictions.values, keys=predictions.index)
    
    true = predictions.reset_index().set_index(['model_id', 'type'])['time_bin']

    def plot_cm(ax, predictions, trial_type):
        true = predictions.index.get_level_values('time_bin')
        cm = metrics.confusion_matrix(true, predictions)
        cm = cm / np.sum(cm, axis=1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap', ('#000000', TRIAL_COLORS[trial_type]))
        im = ax.imshow(cm, vmin=0, vmax=0.5, cmap=cmap)
        ax.set_title(trial_type, color=TRIAL_COLORS[trial_type])
        ax.set_xticks([5,  15])
        ax.set_yticks([5, 10, 15])

    for decoder in ('LS', 'SL'):
        PdfPlotter(path / f'example_decoding.{decoder}.pdf', fixed_margins=margins)
        plt.figure(figsize=(2.9, 1.25), dpi=dpi)
        gs = iter(gridspec.GridSpec(1, 1))
        for model_type in ('sl_shaping',):
            gss = next(gs).subgridspec(1, 3)
            axs = list(map(plt.subplot, gss))
            for i, trial_type in enumerate(('LS', 'SL', 'SS')):
                plot_cm(
                    axs[i],
                    predictions.xs(model_type, level='model_type')\
                        .xs(trial_type, level='type')['predictions'][decoder],
                    trial_type)
            
            def format_axis(ax):
                [ax.spines[s].set_color(TRIAL_COLORS[decoder]) for s in ax.spines]
            list(map(format_axis, axs))

            [axs[0].spines[s].set_linewidth(2) for s in axs[0].spines]
            [ax.set_yticks([]) for ax in axs[1:]]
            [ax.set_xticks([5, 10, 15]) for ax in axs]
            axs[-1].text(
                1, 1.01,  f'{decoder} time decoder', color=TRIAL_COLORS[decoder],
                fontsize=5, transform=axs[-1].transAxes, ha='right', va='bottom')
            axs[0].set_ylabel('SL/FT Time (s)')

        ax = axs[-1]
        axs[1].set_xlabel('Predicted Time (s)')
        inax_position = ax.transAxes.transform([0.9, 0.175]) 
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

    error = predictions['circ_error']
    mse = (error**2).groupby(['model_id', 'model_type', 'type', 'split']).mean()
    mse.columns.name = 'decode_type'

    mean_mse = mse.groupby(['model_type', 'type', 'model_id']).mean()
    error_list = mean_mse.drop('all', axis=1).stack()
    error_idx = error_list.index.to_frame()
    error_idx['same'] = error_idx['type'] == error_idx['decode_type']
    error_list.index = pd.MultiIndex.from_frame(error_idx.drop('type', axis=1).drop('decode_type', axis=1))
    error_list = error_list.drop('no_shaping')

    x = error_list.groupby(error_list.index.names).mean().unstack().diff(axis=1).abs()[True]
    margins=jP.default_margins()
    margins['bottom'] = 150
    #margins['top'] = 60
    PdfPlotter(path / f'mse_bar.horiz_small.pdf',
               fixed_margins=margins)
    plt.figure(figsize=(3.5, 1.5), dpi=dpi)
    ax = plt.subplot()

    def box_plotter(X, i, colors, ax):
        flierprops={'marker': '.', 'markersize': 2, 'mfc': 'k'}
        boxprops={'facecolor': next(colors), 'edgecolor': 'k'}
        medianprops={'color': 'k'}
        ax.boxplot(
            [X], positions=[next(i)], widths=0.5, flierprops=flierprops,
            boxprops=boxprops, patch_artist=True, medianprops=medianprops, vert=False)

    error_groups = x.groupby(['model_type']).apply(list)
    error_groups = error_groups.reindex(
        ['shaping', 'sl_shaping'], level='model_type')
    error_groups.apply(box_plotter, i=iter([1, 2]), colors=iter(['tab:orange', 'tab:green']), ax=ax)

    pvalues = error_list.groupby(
        ['same', 'model_type']).apply(list).groupby('same').apply(
            lambda x: scipy.stats.ttest_ind(*x))
    jP.configure_spines(ax, fix_xlabel=False)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['LS+SL', 'SL Only'])
    ax.set_xlabel(r'$\Delta$ (Between$-$Within Context)'+'\nTime Decoding MSE')
    ax.set_xticks([5, 10, 15])
    ax.set_xlim(0, 15)

    pval = scipy.stats.ttest_ind(x['sl_shaping'], x['shaping']).pvalue
    sigs = jP.significance_symbols(pd.Series([pval]))
    lbl_y = x.max() + jP.annotation_padding(ax, 0.08, axis='x')
    jP.annotation_horiz(ax, (1, 2), lbl_y, sigs[0], ha='left')
    plt.show()


    PdfPlotter(path / f'mse_bar.pdf',
               fixed_margins=jP.default_margins())
    plt.figure(figsize=(1.5, 2.62), dpi=dpi)
    ax = plt.subplot()

    def box_plotter(X, i, colors, ax):
        flierprops={'marker': '.', 'markersize': 2, 'mfc': 'k'}
        boxprops={'facecolor': next(colors), 'edgecolor': 'k'}
        medianprops={'color': 'k'}
        ax.boxplot(
            [X], positions=[next(i)], widths=0.5, flierprops=flierprops,
            boxprops=boxprops, patch_artist=True, medianprops=medianprops)

    error_groups = x.groupby(['model_type']).apply(list)
    error_groups = error_groups.reindex(
        ['sl_shaping', 'shaping'], level='model_type')
    error_groups.apply(box_plotter, i=it.count(), colors=iter(['tab:green', 'tab:orange']), ax=ax)

    pvalues = error_list.groupby(
        ['same', 'model_type']).apply(list).groupby('same').apply(
            lambda x: scipy.stats.ttest_ind(*x))
    jP.configure_spines(ax, fix_xlabel=False)
    ax.set_yticks([1, 3, 5, 7, 9, 11])
    ax.set_xticklabels(['SL/FT', 'S/FT'])
    ax.get_xticklabels()[0].set_color('tab:green')
    ax.get_xticklabels()[1].set_color('tab:orange')
    ax.set_ylabel(r'$\Delta$ (Between$-$Within Context)'+'\nTime Decoding MSE')
    ax.set_ylim(0, 11)

    pval = scipy.stats.ttest_ind(x['sl_shaping'], x['shaping'])
    sigs = jP.significance_symbols(pd.Series([pval.pvalue]))
    lbl_y = error_groups.apply(max).max() + jP.annotation_padding(ax, 0.08)
    jP.annotation(ax, (0, 1), lbl_y, sigs.values[0], va='bottom')
    jP.set_ylabel_position(ax, nlines=3)
    plt.show()
 

    with open(info_file, 'a') as f:
        f.write(str(pval))
        f.write('\n\n\n')

if __name__ == '__main__':
    main(sys.argv[1:])
