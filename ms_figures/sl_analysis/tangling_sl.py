# [Figure S8d]
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

sys.path.append('../')
from tangling import model_tangling

TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'LL': '#2b958c', 'S': '#f89521'}


def main(argv):
    path = Path('/analysis/ms_figures/sl_tangling')
    path.mkdir(exist_ok=True, parents=True)

    sl_model_ids = [41, 47, 42, 45, 43, 46, 64, 80, 83, 85]
    with_model_ids = [20, 22, 25, 21, 23, 26, 27, 30, 76, 79, 82, 84]

    info_file = path / 'info_file.txt'
    with open(info_file, 'w') as f:
        pass

    idx = pd.MultiIndex.from_tuples(
        list(zip(sl_model_ids, it.repeat('sl_shaping'))) +
        list(zip(with_model_ids, it.repeat('shaping'))),
        names=('model_id', 'model_type'))
    model_ids = pd.Series(sl_model_ids + with_model_ids,  index=idx)

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

    res.reindex(['sl_shaping', 'shaping'], level='model_type')\
       .groupby('model_type', sort=False).apply(box_plotter, it.count(), iter(['tab:green', 'tab:orange']), ax)
    sigs = jP.significance_symbols([pval.pvalue])
    ax.set_ylim(0, 35)
    ax.set_yticks([10, 20, 30])
    lbl_y = res.max() + jP.annotation_padding(ax, 0.08)
    jP.annotation(ax, (0, 1), lbl_y, sigs.values[0], va='bottom')
    jP.configure_spines(ax)
    ax.set_ylabel('Trajectory Tangling')
    ax.set_xticklabels(['SL/FT', 'S/FT'])
    ax.get_xticklabels()[0].set_color('tab:green')
    ax.get_xticklabels()[1].set_color('tab:orange')
    plt.show()

    print(pval)
    with open(info_file, 'a') as f:
        f.writelines(str(pval))


if __name__ == '__main__':
    main(sys.argv[1:])
