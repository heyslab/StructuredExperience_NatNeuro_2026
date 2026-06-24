# [Figure 2b]
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
from sklearn.decomposition import PCA
import itertools as it
import argparse
import sklearn
import scipy
   
import pingouin as pg

from classes.datagen import tDNMSGenerator, genFactory
from classes.models import LeakyRNN
from classes.models import LeakyRNNCell

import models_database as mdb

from analysis_tools.mpl_helpers import PdfPlotter
import analysis_tools.jPlots as jP
from analysis_tools.progressbar import ProgressBar

TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'LL': '#2b958c', 'S': '#f89521'}


def model_pca(model, X2, rnn_layer=0, p=ProgressBar.nocount()):
    p.increment()

    y = pd.DataFrame(
        model.layers[rnn_layer](np.expand_dims(X2[['light', 'odor']], 0))[0])
    y.index = X2.index
    pca = PCA(n_components=10)
    pca.fit(y)
    return pd.Series({'total': pca.explained_variance_, 'ratio': pca.explained_variance_ratio_})


def main(argv):
    path = Path('/analysis/ms_figures/explained_var')
    path.mkdir(exist_ok=True, parents=True)
    info_file = path / 'explained_var.info.txt'
    with open(info_file, 'w') as f:
        pass

    no_model_ids = [9, 18, 19, 24, 28, 29, 31, 32, 93, 109, 116, 119]
    with_model_ids = [20, 22, 25, 21, 23, 26, 27, 30, 76, 79, 82, 84]

    idx = pd.MultiIndex.from_tuples(
        list(zip(no_model_ids, it.repeat('no_shaping'))) +
        list(zip(with_model_ids, it.repeat('shaping'))),
        names=('model_id', 'model_type'))
    model_ids = pd.Series(no_model_ids + with_model_ids,  index=idx)

    model_infos = model_ids.apply(mdb.get_model).apply(pd.Series)
    model_attrs = model_ids.apply(mdb.get_model_attributes)
    model_infos = pd.concat(
        (model_infos,
         model_attrs[['gamma', 'epoc', 'input_noise', 'noise_level']]), axis=1)

    margins = jP.default_margins()
    dpi = 300
    jP.set_rcParams(plt)
    task = 'just_short_match'

    input_noise = model_infos.head(1)['input_noise']
    trials_gen = genFactory.create(
        task, input_noise=input_noise, batch_size=1, n_blocks=1)

    X2 = trials_gen.generate_trials(25)
    models = model_infos['path'].apply(tf.keras.models.load_model)

    p = ProgressBar(len(models))
    res = models.apply(model_pca, X2=X2, p=p)

    exp_var = res['total'].apply(pd.Series)
    mean = exp_var.groupby('model_type').mean().T
    std = exp_var.groupby('model_type').std().T

    mean = mean[:8]
    std = std[:8]
    PdfPlotter(path / 'var_explained.pdf', fixed_margins=jP.default_margins())
    plt.figure(figsize=(1.75, 1.775), dpi=dpi)
    ax = plt.gca()
    ax.fill_between(
        mean.index.get_level_values(0), mean['no_shaping']-std['no_shaping'],
        mean['no_shaping']+std['no_shaping'], color='tab:blue', alpha=0.25)
    ax.fill_between(
        mean.index.get_level_values(0), mean['shaping']-std['shaping'],
        mean['shaping']+std['shaping'], color='tab:orange', alpha=0.25)
    ax.plot(
        mean.index.get_level_values(0), mean['no_shaping'], c='tab:blue',
        marker='o', label='No Shaping', clip_on=False)
    ax.plot(
        mean.index.get_level_values(0), mean['shaping'], c='tab:orange',
        marker='o', label='Shaping+Full', clip_on=False)
    jP.configure_spines(ax)
    ax.set_xlim(-0.2, 6)
    ax.set_xticks([1, 3, 5, 7])
    ax.set_xticklabels(ax.get_xticks() + 1)
    ax.spines['bottom'].set_bounds(0, 7)
    ax.spines['left'].set_bounds(0, 25)
    ax.set_xlabel('# PC')
    ax.set_ylabel('Variance Explained (a.u.)')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.2))

    inax_position = ax.transAxes.transform([0.62, 0.3])
    infig_position = ax.figure.transFigure.inverted().transform(inax_position)
    ratio_ax = ax.figure.add_axes(
        list(infig_position) +
            [ax.get_position().width * 0.4, ax.get_position().height * 0.5])
 
    def box_plotter(X, i, ax, color, **kwargs):
        flierprops={'marker': '.', 'markersize': 2, 'mfc': 'k'}
        boxprops={'facecolor': color, 'edgecolor': 'k'}
        medianprops={'color': 'k'}
        ax.boxplot(
            [X], positions=[i], widths=0.5, flierprops=flierprops,
            boxprops=boxprops, patch_artist=True, medianprops=medianprops,
            **kwargs)

    exp_var = res['ratio'].apply(pd.Series)[[0, 1, 2]].xs('shaping', level='model_type')
    exp_var_ns = res['ratio'].apply(pd.Series)[[0, 1, 2]].xs('no_shaping', level='model_type')
    box_plotter(exp_var.sum(1) * 100, 0, ratio_ax, 'tab:orange')
    box_plotter(exp_var_ns.sum(1) * 100, 1, ratio_ax, 'tab:blue')
    jP.configure_spines(ratio_ax, fix_ylabel=False)
    ratio_ax.set_ylim(83, 100)
    ratio_ax.spines['left'].set_bounds(85, 100)
    ratio_ax.set_yticks([85, 100])
    ratio_ax.set_yticklabels([85, 100], fontsize=5)
    ratio_ax.spines['bottom'].set_bounds(0, 1)
    jP.percent_y(ratio_ax)
    ratio_ax.set_xticks([0, 1])
    ratio_ax.set_xticklabels(['S/FT', 'NS'], ha='right', rotation=45, rotation_mode='anchor', fontsize=5)
    ratio_ax.set_ylabel('% Variance\nPC 1-3', labelpad=-9, fontsize=6)
    pval = scipy.stats.ttest_ind(exp_var.sum(1), exp_var_ns.sum(1))
    sig = jP.significance_symbols([pval.pvalue])
    ylbl = max(exp_var.sum(1).max(), exp_var_ns.sum(1).max()) * 100 + jP.annotation_padding(ratio_ax, 0.08)
    jP.annotation(ratio_ax, (0, 1), ylbl, sig.values[0], va='bottom')

    plt.show()

    PdfPlotter(path / 'var_explained_2.pdf', fixed_margins=jP.default_margins())
    plt.figure(figsize=(2, 1.775), dpi=dpi)
    ax = plt.gca()
    ax.fill_between(
        mean.index.get_level_values(0), mean['no_shaping']-std['no_shaping'],
        mean['no_shaping']+std['no_shaping'], color='tab:blue', alpha=0.25)
    ax.fill_between(
        mean.index.get_level_values(0), mean['shaping']-std['shaping'],
        mean['shaping']+std['shaping'], color='tab:orange', alpha=0.25)
    ax.plot(
        mean.index.get_level_values(0), mean['no_shaping'], c='tab:blue',
        marker='o', label='No Shaping', clip_on=False)
    ax.plot(
        mean.index.get_level_values(0), mean['shaping'], c='tab:orange',
        marker='o', label='Shaping+Full', clip_on=False)
    jP.configure_spines(ax)
    ax.set_xlim(-0.2, 6)
    ax.set_xticks([1, 3, 5, 7])
    ax.set_xticklabels(ax.get_xticks() + 1)
    ax.spines['bottom'].set_bounds(0, 7)
    ax.spines['left'].set_bounds(0, 25)
    ax.set_xlabel('# PC')
    ax.set_ylabel('Variance Explained (a.u.)')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.2), borderaxespad=0)

    inax_position = ax.transAxes.transform([0.58, 0.3])
    infig_position = ax.figure.transFigure.inverted().transform(inax_position)
    ratio_ax = ax.figure.add_axes(
        list(infig_position) +
            [ax.get_position().width * 0.4, ax.get_position().height * 0.5])
    exp_var = res['ratio'].apply(pd.Series)[[0, 1, 2]]\
                          .xs('shaping', level='model_type')
    exp_var_ns = res['ratio'].apply(pd.Series)[[0, 1, 2]]\
                             .xs('no_shaping', level='model_type')
    box_plotter(exp_var.sum(1) * 100, 0, ratio_ax, 'tab:orange')
    box_plotter(exp_var_ns.sum(1) * 100, 1, ratio_ax, 'tab:blue')
    jP.configure_spines(ratio_ax, fix_ylabel=False)
    ratio_ax.set_ylim(83, 100)
    ratio_ax.spines['left'].set_bounds(85, 100)
    ratio_ax.set_yticks([85, 100])
    ratio_ax.set_yticklabels([85, 100], fontsize=5)
    ratio_ax.spines['bottom'].set_bounds(0, 1)
    jP.percent_y(ratio_ax)
    ratio_ax.set_xticks([0, 1])
    ratio_ax.set_xticklabels(
        ['S/FT', 'NS'], ha='right', rotation=45, rotation_mode='anchor',
        fontsize=5)
    ratio_ax.set_ylabel('% Variance\nPC 1-3', labelpad=-9, fontsize=6)
    pval = scipy.stats.ttest_ind(exp_var.sum(1), exp_var_ns.sum(1))
    sig = jP.significance_symbols([pval.pvalue])
    ylbl = max(exp_var.sum(1).max(), exp_var_ns.sum(1).max()) * \
           100 + jP.annotation_padding(ratio_ax, 0.08)
    jP.annotation(ratio_ax, (0, 1), ylbl, sig.values[0], va='bottom')

    plt.show()

    pcs = res['total'].apply(pd.Series)
    pcs.columns.name = 'PC'
    X = pcs.stack()
    X.name = 'values'
    X = X.reset_index()
    anova_res = pg.mixed_anova(
        data=X, dv='values', between='model_type', within='PC',
        subject='model_id', correction=False)
    print(anova_res)

    with open(info_file, 'a') as f:
        f.write(anova_res.to_string())
        f.write('\n\n\n')
        f.write(str(pval))
    print(pval)

if __name__ == '__main__':
    main(sys.argv[1:])
