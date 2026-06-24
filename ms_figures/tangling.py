# [Figure S2c]
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
    return t.mean()


def main(argv):
    path = Path('/analysis/ms_figures/tangling')
    path.mkdir(exist_ok=True, parents=True)

    no_model_ids = [9, 18, 19, 24, 28, 29, 31, 32, 93, 109, 116, 119]
    with_model_ids = [20, 22, 25, 21, 23, 26, 27, 30, 76, 79, 82, 84]

    info_file = path / 'info_file.txt'
    with open(info_file, 'w') as f:
        pass

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

    p = ProgressBar(len(models))
    res = models.apply(model_tangling, X2=X2, p=p)

    PdfPlotter(path / 'tangling_plot.pdf', fixed_margins=jP.default_margins())
    plt.figure(figsize=(1.5, 2.62), dpi=dpi)
    ax = plt.gca()
    pval = scipy.stats.ttest_ind(*res.groupby('model_type').apply(list).values)
    def box_plotter(X, i, colors, ax):
        print(X)
        flierprops={'marker': '.', 'markersize': 2, 'mfc': 'k'}
        boxprops={'facecolor': next(colors), 'edgecolor': 'k'}
        medianprops={'color': 'k'}
        ax.boxplot(
            [X], positions=[next(i)], widths=0.5, flierprops=flierprops,
            boxprops=boxprops, patch_artist=True, medianprops=medianprops)

    res.reindex(['no_shaping', 'shaping'], level='model_type')\
       .groupby('model_type', sort=False).apply(box_plotter, it.count(), iter(['tab:blue', 'tab:orange']), ax)
    sigs = jP.significance_symbols([pval.pvalue])
    lbl_y = 40
    jP.annotation(ax, (0, 1), lbl_y, sigs.values[0], va='bottom')
    ax.set_ylim(0, 75)
    ax.set_yticks([20, 40, 60])
    jP.configure_spines(ax)
    ax.set_ylabel('Trajectory Tangling')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['NS', 'S/FT'])
    ax.get_xticklabels()[0].set_color('tab:blue')
    ax.get_xticklabels()[1].set_color('tab:orange')
    plt.show()

    print(pval)
    with open(info_file, 'a') as f:
        f.writelines(str(pval))


if __name__ == '__main__':
    main(sys.argv[1:])
