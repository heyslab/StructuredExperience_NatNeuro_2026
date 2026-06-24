# [Figure 2c]
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
   
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from classes.datagen import tDNMSGenerator, genFactory
from classes.models import LeakyRNN
from classes.models import LeakyRNNCell

import models_database as mdb

from analysis_tools.mpl_helpers import PdfPlotter
import analysis_tools.jPlots as jP
from analysis_tools.progressbar import ProgressBar

TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'LL': '#2b958c', 'S': '#f89521'}

def model_tangling(model, X2, rnn_layer=0, p=ProgressBar.nocount(), delta_t=0.1):
    p.increment()
    y = pd.DataFrame(
        model.layers[rnn_layer](np.expand_dims(X2[['light', 'odor']], 0))[0])
    y.index = X2.index
    e = (y**2).sum(axis=1).mean() * 0.1

    def calc_t(t, y, e):
        denom = ((y.iloc[t] - y)**2).sum(axis=1)
        diff_y = y.diff() / delta_t
        num = ((diff_y.iloc[t] - diff_y)**2).sum(axis=1)
        return (num/(denom + e)).max()

    t = pd.Series(np.arange(len(y)), index=y.index)[1:].apply(calc_t, y=y, e=e)
    return t


def main(argv):
    path = Path('/analysis/ms_figures/tangling')
    path.mkdir(exist_ok=True, parents=True)

    no_model_ids = [24]
    with_model_ids = [132]

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

    X2 = trials_gen.generate_trials(15)
    models = model_infos['path'].apply(tf.keras.models.load_model)

    res = models.apply(model_tangling, X2=X2)

    PdfPlotter(path / 'tangling_examples.pdf', fixed_margins=jP.default_margins())
    plt.figure(figsize=(1.75, 1.775), dpi=dpi)
    ax = plt.gca()
    ax.annotate(
        '', (0, 0), (1, 1), xycoords=ax.transAxes, textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle='-', color='r', linestyle='--', linewidth=0.25))

    def plot_pts(X, ax):
        c = TRIAL_COLORS[X.index.unique('type')[0]]
        jP.plot_seg_colors(ax, *X.T.values, c, ls='', marker='o', ms=0.5, mew=0)

    res.T.groupby('trial').apply(plot_pts, ax=ax)
    ax.set_ylim(0, 50)
    ax.set_xlim(0, 50)
    jP.configure_spines(ax)

    legend_lines = res.columns.unique('type').to_series().apply(
        lambda x: matplotlib.lines.Line2D(
            [0], [0], c=TRIAL_COLORS[x], lw=2, label=x))
    ax.legend(handles=list(legend_lines.values), loc='upper left', bbox_to_anchor=(0, 1))

    inax_position = ax.transAxes.transform([0.14, 0.95])
    infig_position = ax.figure.transFigure.inverted().transform(inax_position)
    color_scale = ax.figure.add_axes(
        list(infig_position) +
            [ax.get_position().width * 0.3, ax.get_position().height * 0.025])
    color_scale.imshow(
        [np.linspace(0.25, 1, 30)], cmap='Grays', aspect='auto', vmin=0, vmax=1)
    color_scale.set_axis_off()
    color_scale.text(
        0, 1, 'trial\nstart', ha='center', va='bottom', fontsize=5,
        transform=color_scale.transAxes)
    color_scale.text(
        1, 1, 'trial\nend', ha='center', va='bottom', fontsize=5,
        transform=color_scale.transAxes)

    ax.set_ylabel('S/FT Trajectory Tangling', color='tab:orange')
    ax.set_xlabel('NS Trajectory Tangling', color='tab:blue')
    ax.set_xticks([10, 30, 50])
    ax.set_yticks([10, 30, 50])
    ax.yaxis.get_offset_text().set_position((-0.15, 0))
    ax.xaxis.get_offset_text().set_position((2, 0))

    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
