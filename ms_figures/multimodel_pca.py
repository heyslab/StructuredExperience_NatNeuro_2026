# [Figure S2a,b]
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
from analysis_tools.progressbar import ProgressBar

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
    jP.configure_spines(ax, fix_ylabel=True, fix_xlabel=False)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal', anchor='C')


def main(argv):
    no_model_ids = [9, 18, 19, 24, 28, 29, 31, 32, 93, 109, 116, 119]
    with_model_ids = [20, 22, 25, 21, 23, 26, 27, 30, 76, 79, 82, 84]
    input_noise = mdb.get_model_attributes(with_model_ids[0])['input_noise']

    task_name = 'just_short_match'
    trial_gen = genFactory.create(
        task_name, input_noise=input_noise, batch_size=1,
        n_blocks=1)
    X2 = trial_gen.generate_trials(15)
    formatted_X = trial_gen.format_validation(X2)[0]
    cols = X2.index.names
    X2 = X2.reset_index().set_index(cols + ['cues'])
    index = X2.index

    path = Path('/analysis/ms_figures/multi_pcas/')
    path.mkdir(exist_ok=True, parents=True)

    dpi = 300
    jP.set_rcParams(plt)

    for model_ids, label in zip((no_model_ids, with_model_ids), ('no_shaping', 'shaping')):
        models = pd.Series(model_ids, index=model_ids)

        margins={'left': 80, 'right': 100, 'top': 60, 'bottom': 80}
        model_infos = models.apply(mdb.get_model).apply(pd.Series)
        model_infos = pd.concat(
            (model_infos, models.apply(mdb.get_model_attributes)), axis=1)

        models[~model_infos['path'].apply(lambda x: Path(x).exists())]\
            .apply(lambda x, model_ids=model_ids: model_ids.remove(x))
        model_infos = model_infos.reindex(model_ids)

        models = model_infos['path'].apply(tf.keras.models.load_model)

        reverse_x = False

        input_noise = model_infos['input_noise'].head(1)
        rnn_layer = 0

        ys = pd.concat(
            list(map(lambda m, X2=X2, l=rnn_layer: pd.DataFrame(
                m.layers[l](np.expand_dims(X2[['light', 'odor']], 0))[0], index=X2.index), models)),
                keys=models.index.values)
        ys.index.names = ['model_id'] + ys.index.names[1:]

        def calc_pca(y, p=ProgressBar.nocount(), **pca_kwargs):
            pca = PCA(**pca_kwargs)
            shapes = pca.fit_transform(y)
            pca_res = pd.DataFrame(shapes, index=y.index)
            return pca_res

        p = ProgressBar(len(model_infos))
        print('calculating PCs')
        pcs = ys.groupby('model_id').apply(calc_pca, p,  n_components=3).droplevel(0)
        p.end()

        PdfPlotter(path / f'multiple_pca.{label}_7.pdf', fixed_margins=margins)
        plt.figure(figsize=(7, 1.25), dpi=dpi)
        gs = gridspec.GridSpec(1, len(model_ids), hspace=0.45)
        gs_iter = iter(gs)
        for i, model_id in enumerate(pcs.index.unique('model_id')):
            gss = next(gs_iter).subgridspec(2, 1)
            axs = list(map(plt.subplot, gss))

            plot_pca(axs[0], pcs.xs(model_id, level='model_id').drop(2, axis=1))
            plot_pca(axs[1], pcs.xs(model_id, level='model_id').drop(1, axis=1))

            if i == 0:
                for ax in axs:
                    ax.spines['left'].set_position(('outward', -5))
                    ax.spines['left'].set_bounds([-2.5, 2.5])
                    ax.set_yticks([-2.5, 2.5])
                axs[0].set_ylabel('PC 2')
                axs[1].set_ylabel('PC 3')

            else:
                for ax in axs:
                    ax.spines['left'].set_visible(False)
                    ax.set_yticks([])

            axs[0].spines['bottom'].set_visible(False)
            axs[0].set_xticks([])
            axs[1].spines['bottom'].set_bounds([-5, 5])
            axs[1].set_xticks([-5, 0, 5])
            axs[1].set_xlabel('PC 1')

        trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.figure.transFigure)
        if label == 'shaping':
            axs[0].text(1.2, 0.5, 'Shaping + Full Task', color='tab:orange', transform=trans, rotation=-90, va='center')
        elif label == 'no_shaping':
            axs[0].text(1.2, 0.5, 'No Shaping', color='tab:blue', transform=trans, rotation=-90, va='center')
        plt.show()
 

        PdfPlotter(path / f'multiple_pca.{label}.pdf', fixed_margins=margins)
        plt.figure(figsize=(6.5, 1.25), dpi=dpi)
        gs = gridspec.GridSpec(1, len(model_ids), hspace=0.45)
        gs_iter = iter(gs)
        for i, model_id in enumerate(pcs.index.unique('model_id')):
            gss = next(gs_iter).subgridspec(2, 1)
            axs = list(map(plt.subplot, gss))

            plot_pca(axs[0], pcs.xs(model_id, level='model_id').drop(2, axis=1))
            plot_pca(axs[1], pcs.xs(model_id, level='model_id').drop(1, axis=1))

            if i == 0:
                for ax in axs:
                    ax.spines['left'].set_position(('outward', -5))
                    ax.spines['left'].set_bounds([-2.5, 2.5])
                    ax.set_yticks([-2.5, 2.5])
                axs[0].set_ylabel('PC 2')
                axs[1].set_ylabel('PC 3')

            else:
                for ax in axs:
                    ax.spines['left'].set_visible(False)
                    ax.set_yticks([])

            axs[0].spines['bottom'].set_visible(False)
            axs[0].set_xticks([])
            axs[1].spines['bottom'].set_bounds([-5, 5])
            axs[1].set_xticks([-5, 0, 5])
            axs[1].set_xlabel('PC 1')

        trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.figure.transFigure)
        if label == 'shaping':
            axs[0].text(1.2, 0.5, 'Shaping + Full Task', color='tab:orange', transform=trans, rotation=-90, va='center')
        elif label == 'no_shaping':
            axs[0].text(1.2, 0.5, 'No Shaping', color='tab:blue', transform=trans, rotation=-90, va='center')
        plt.show()
 

if __name__ == '__main__':
    main(sys.argv[1:])
