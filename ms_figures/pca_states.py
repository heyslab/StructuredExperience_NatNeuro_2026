# [Figure 2a]
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

from classes.datagen import tDNMSGenerator, genFactory
from classes.models import LeakyRNN
from classes.models import LeakyRNNCell

import models_database as mdb

from analysis_tools.mpl_helpers import PdfPlotter
import analysis_tools.jPlots as jP

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  
    "axes.facecolor":    (1.0, 1.0, 1.0, 0.0),  
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),
    })

TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'LL': '#2b958c', 'S': '#f89521'}

def plot_pca(ax, pcs, dims=(1, 2)):
    def plot_trial(res, ax):
        trial_type = res.index.unique('type')[0]
        color = TRIAL_COLORS[trial_type]
        jP.plot_seg_colors(
            ax, *res.values.T, c=color, clip_on=False)

    pcs.groupby(['type', 'trial']).apply(
        plot_trial, ax=ax)
    jP.configure_spines(ax)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal', anchor='C')


def main(argv):
    models = pd.Series(
        [24, 130, 132],
        index=['no_shaping', 'only_shaping', 'shaping'])

    margins={'left': 80, 'right': 110, 'top': 45, 'bottom': 80}

    model_infos = models.apply(mdb.get_model).apply(pd.Series)
    model_infos = pd.concat(
        (model_infos, models.apply(mdb.get_model_attributes)), axis=1)

    path = Path('/analysis/ms_figures/pca')
    path.mkdir(exist_ok=True, parents=True)

    dpi = 300
    jP.set_rcParams(plt)

    models = model_infos['path'].apply(tf.keras.models.load_model)

    trial_gen = genFactory.create(
        'just_short_match', input_noise=model_infos['input_noise'].head(1), batch_size=1,
        n_blocks=1)
    X2 = trial_gen.generate_trials(25)

    rnn_layer = 0
    ys = pd.concat(
        list(map(lambda m, X2=X2, l=rnn_layer: pd.DataFrame(
            m.layers[l](np.expand_dims(X2[['light', 'odor']], 0))[0], index=X2.index), models)),
            keys=models.index.values)
    ys.index.names = ['model_type'] + ys.index.names[1:]

    def calc_pca(y, pcas):
        model_type = y.index.unique('model_type')[0]
        shapes = pcas[model_type].transform(y)
        return pd.DataFrame(shapes, index=y.index)
    pcas = ys.drop('only_shaping', level='model_type').groupby('model_type')\
             .apply(lambda x: PCA(n_components=3).fit(x))
    pcas['only_shaping'] = pcas['shaping']
    pcs = ys.groupby('model_type').apply(calc_pca, pcas=pcas)

    margins_adj = margins.copy()
    margins_adj['right'] = 100
    PdfPlotter(path / f'pca_3plots.pdf', fixed_margins=margins_adj)
    plt.figure(figsize=(3, 1.8), dpi=dpi)
    gs = gridspec.GridSpec(2, 3, wspace=0.05)
    axs = list(map(plt.subplot, gs))
    axs = [axs[0], axs[2], axs[3], axs[5], axs[1], axs[4]]

    plot_pca(axs[0], pcs.loc['no_shaping'].drop(2, axis=1))
    plot_pca(axs[2], pcs.loc['no_shaping'].drop(1, axis=1))

    plot_pca(axs[1], pcs.loc['shaping'].drop(2, axis=1))
    plot_pca(axs[3], pcs.loc['shaping'].drop(1, axis=1))

    plot_pca(axs[4], pcs.loc['only_shaping'].drop(2, axis=1))
    plot_pca(axs[5], pcs.loc['only_shaping'].drop(1, axis=1))
    
    for ax in (axs[0], axs[2]):
        ax.spines['left'].set_position(('outward', -5))
        ax.spines['left'].set_bounds([-2.5, 2.5])
        ax.set_yticks([-2.5, 2.5])

    for ax in (axs[1], axs[3], axs[4], axs[5]):
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)

    for ax in (axs[0], axs[1], axs[4]):
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)

    for ax in (axs[2], axs[3], axs[5]):
        ax.spines['bottom'].set_bounds([-5, 5])
        ax.set_xticks([-5, 0, 5])
        ax.set_xlabel('PC 1')

    axs[0].set_ylabel('PC 2')
    axs[2].set_ylabel('PC 3')

    legend_lines = pcs.index.unique('type').to_series().apply(
        lambda x: matplotlib.lines.Line2D(
            [0], [0], c=TRIAL_COLORS[x], lw=2, label=x))
    axs[1].legend(handles=list(legend_lines.values), loc='upper right', bbox_to_anchor=(1.45, 1.1))
    axs[1].set_title('Shaping + Full', color='tab:orange')
    axs[4].set_title('Only Shaping', color='tab:orange')
    axs[0].set_title('No Shaping', color='tab:blue')

    ax = axs[3]
    inax_position = ax.transAxes.transform([0.95, 1.15])
    infig_position = ax.figure.transFigure.inverted().transform(inax_position)
    color_scale = ax.figure.add_axes(
        list(infig_position) +
            [ax.get_position().width * 0.3, ax.get_position().height * 0.05])
    color_scale.imshow(
        [np.linspace(0.25, 1, 30)], cmap='Grays', aspect='auto', vmin=0, vmax=1)
    color_scale.set_axis_off()
    color_scale.text(
        0, 1, 'trial\nstart', ha='center', va='bottom', fontsize=5,
        transform=color_scale.transAxes)
    color_scale.text(
        1, 1, 'trial\nend', ha='center', va='bottom', fontsize=5,
        transform=color_scale.transAxes)

    plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])
