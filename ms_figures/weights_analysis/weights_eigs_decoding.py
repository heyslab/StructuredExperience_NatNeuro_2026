# [Figure 3f]
import os
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../fp_analysis')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as np

import numpy as onp
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import itertools as it
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from matplotlib.patches import Circle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from analysis_tools.mpl_helpers import PdfPlotter
import analysis_tools.jPlots as jP
import models_database as mdb

from classes.datagen import tDNMSGenerator, genFactory
from classes.models import LeakyRNN
from classes.models import LeakyRNNCell
import leaky_rnn

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  
    "axes.facecolor":    (1.0, 1.0, 1.0, 0.0),  
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0)})

SHAPING_COLORS = {
    'no_shaping': 'tab:blue',
    'shaping': 'tab:orange',
    'sl_only': 'tab:green',
    'ls_only': 'tab:red'}

def main(argv):
    no_model_ids   = [9,  18, 19, 24, 28, 29, 31, 32, 93, 109, 116, 119]
    with_model_ids = [20, 22, 25, 21, 23, 26, 27, 30, 76, 79,  82,  84]

    path = Path('/analysis/ms_figures/weights_eigs')
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

    models = model_infos['path'].apply(tf.keras.models.load_model)
    weights = pd.concat(models.apply(
        lambda x: x.layers[0].get_weights()[1]).apply(pd.DataFrame).values,
        keys=models.index)
    eigs = weights.groupby(['model_id', 'model_type']).apply(onp.linalg.eigvals)
    eigs = eigs.apply(lambda x: np.sort(x)[::-1])
    path = Path('/analysis/ms_figures/weights_eigs')
    path.mkdir(parents=True, exist_ok=True)
    PdfPlotter(path / 'eigs_decoding.pdf', fixed_margins=jP.default_margins())

    def format_dat(x):
        x = x.apply(lambda x: x[:4])
        x = x.apply(pd.Series)
        x = x.map(lambda x: (x.real, x.imag))
        return x.stack().apply(pd.Series).unstack()

    def decode_split(test_ids, eigs):
        train = format_dat(eigs.drop(list(test_ids), level='model_id'))
        test = format_dat(eigs.reindex(test_ids, level='model_id'))
        clf = RandomForestClassifier(max_depth=4, random_state=0)
        clf.fit(train.values, train.index.get_level_values('model_type'))
        clf.predict(test.values)
        labels = clf.predict(test.values)
        return pd.Series(labels, index=test_ids)

    def perform_decoding(eigs):
        ids = eigs.index\
                  .to_frame()\
                  .set_index('model_type', drop=True)['model_id']\
                  .groupby('model_type').apply(list)
        test_groups = ids.apply(pd.Series).T.apply(list, axis=1)
        test_groups.apply(decode_split, eigs=eigs)
        decoding_result = test_groups.apply(decode_split, eigs=eigs).T.stack().droplevel(1)
        decoding_result.index.name = 'model_id'
        decoding_result.name = 'prediction'
        return pd.merge(eigs.index.to_frame()['model_type'], decoding_result,
                        left_on='model_id', right_on='model_id')
    decoding_result = perform_decoding(eigs)

    def perform_shuffle(eigs):
        eigs_shuffle = eigs.copy()
        onp.random.shuffle(eigs_shuffle.values)
        shuffle_result = perform_decoding(eigs_shuffle)
        return (shuffle_result['model_type'] == shuffle_result['prediction']).mean()
    shuffle_check = pd.Series(np.arange(1000)).apply(lambda x, eigs=eigs: perform_shuffle(eigs))

    jP.set_rcParams(plt)
    path = Path('/analysis/ms_figures/weights_eigs')
    PdfPlotter(path / 'type_decoding.pdf', fixed_margins=jP.default_margins())

    plt.figure(figsize=(2, 1.25), dpi=300)
    ax = plt.gca()
    bins = np.linspace(0, 100, 24)
    hist, _ = np.histogram(shuffle_check.values * 100, bins=bins)

    ax.bar(bins[:-1] + np.diff(bins[:2])[0]/2, hist/len(shuffle_check) * 100,
           width=np.diff(bins[:2])[0], color='gray', alpha=0.3, ec='k')
    jP.configure_spines(ax)
    jP.percent_y(ax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))

    decoding_accuracy = (
        decoding_result['model_type'] == decoding_result['prediction']
        ).mean() * 100
    ax.axvline(decoding_accuracy, lw=2, color='tab:orange')
    arrowprops = dict(
        arrowstyle="-|>", connectionstyle="angle,angleA=0,angleB=90,rad=10")
    ax.annotate(
        f'{decoding_accuracy:.1f}%' + ' Classification\nAccuracy',
        (decoding_accuracy, 1), (-70, 5), textcoords='offset points',
        xycoords=ax.get_xaxis_transform(), arrowprops=arrowprops, fontsize=5)
    ax.text(0.75, 0.75, '***', fontsize=6, transform=ax.transAxes)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 15)
    ax.set_yticks([4, 8, 12])
    ax.set_xticks([20, 60, 100])
    ax.set_ylabel('Shuffle Percentage')
    ax.set_xlabel('Accuracy')
    plt.show()

    PdfPlotter(path / 'type_decoding_2.3.pdf', fixed_margins=jP.default_margins())
    plt.figure(figsize=(2.3, 1.25), dpi=300)
    ax = plt.gca()
    bins = np.linspace(0, 100, 24)
    hist, _ = np.histogram(shuffle_check.values * 100, bins=bins)

    ax.bar(bins[:-1] + np.diff(bins[:2])[0]/2, hist/len(shuffle_check) * 100,
           width=np.diff(bins[:2])[0], color='gray', alpha=0.3, ec='k')
    jP.configure_spines(ax)
    jP.percent_y(ax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))

    decoding_accuracy = (
        decoding_result['model_type'] == decoding_result['prediction']
        ).mean() * 100
    ax.axvline(decoding_accuracy, lw=2, color='tab:orange')
    arrowprops = dict(
        arrowstyle="-|>", connectionstyle="angle,angleA=0,angleB=90,rad=10")
    ax.annotate(
        f'{decoding_accuracy:.1f}%' + ' Classification\nAccuracy',
        (decoding_accuracy, 1), (-70, 5), textcoords='offset points',
        xycoords=ax.get_xaxis_transform(), arrowprops=arrowprops, fontsize=5)
    ax.text(0.75, 0.75, '***', fontsize=6, transform=ax.transAxes)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 15)
    ax.set_yticks([4, 8, 12])
    ax.set_xticks([20, 60, 100])
    ax.set_ylabel('Shuffle Percentage')
    ax.set_xlabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
