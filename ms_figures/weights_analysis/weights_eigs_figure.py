# [Figure 3d,e]
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
    jP.set_rcParams(plt)
    path = Path('/analysis/ms_figures/weights_eigs')
    path.mkdir(parents=True, exist_ok=True)

    no_model_ids   = [9,  18, 19, 24, 28, 29, 31, 32, 93, 109, 116, 119]
    with_model_ids = [20, 22, 25, 21, 23, 26, 27, 30, 76, 79,  82,  84]
    example_ids = [18, 20]

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
        lambda x: x.layers[0].get_weights()[1]).apply(pd.DataFrame).values, keys=models.index)
    eigs = weights.groupby(['model_id', 'model_type']).apply(onp.linalg.eigvals)
    eigs = eigs.apply(lambda x: np.sort(x)[::-1])
    print(eigs.apply(lambda x: x.max()))

    PdfPlotter(path / 'eigs.pdf', fixed_margins=jP.default_margins())
    plt.figure(figsize=(2.25, 1.25), dpi=300)
    gs = gridspec.GridSpec(1, 2)
    axs = list(map(plt.subplot, gs))

    def plotter(X, ax, c):
        circle = Circle((0, 0), 1, fill=False, edgecolor='k',  ls='--', lw=0.75)
        ax.add_patch(circle)
        ax.plot(np.real(X), np.imag(X), ls='', marker='o', c=c, ms=2, mec='k',
                mew=0.25)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(-1.5, 1.5)
        jP.configure_spines(ax)
        ax.set_xticks([-1, 1])
        ax.set_yticks([-1, 1])
        ax.set_aspect('equal', adjustable='box', anchor='SW')
        ax.axes.spines['left'].set_bounds(-1, 1)
        ax.axes.spines['bottom'].set_bounds(-1, 1)
        ax.set_xlabel(r'Re($\lambda$)')

    shaping_ex = eigs.reindex(example_ids, level='model_id')\
                     .xs('shaping', level='model_type').values[0]
    plotter(shaping_ex, ax=axs[0], c='tab:orange')
    noshaping_ex = eigs.reindex(example_ids, level='model_id')\
                       .xs('no_shaping', level='model_type').values[0]
    plotter(noshaping_ex, ax=axs[1], c='tab:blue')
    axs[0].set_ylabel(f'Im($\lambda$)')
    axs[1].set_yticks([])
    axs[1].axes.spines['left'].set_visible(False)

    legend_lines = [
        matplotlib.lines.Line2D([0], [0], ls='', marker='o', ms=2, mec='k', mew=0.25, color='tab:orange'),
        matplotlib.lines.Line2D([0], [0], ls='', marker='o', ms=2, mec='k', mew=0.25, color='tab:blue')]
    axs[1].legend(legend_lines, ('S/FT', 'NS'), loc='lower right', ncols=2, bbox_to_anchor=(1, 1))
    plt.show()

    PdfPlotter(path / 'all_eigs.pdf', fixed_margins=jP.default_margins())
    plt.figure(figsize=(2.25, 1.25), dpi=300)
    gs = gridspec.GridSpec(1, 2)
    axs = list(map(plt.subplot, gs))

    def histogram(X):
        hist, _, _ = np.histogram2d(
            np.imag(X), np.real(X),
            bins=(np.linspace(-1.5, 1.5, 75), np.linspace(-1.5, 1.5, 75))) 
        return np.clip(hist, 0, 1)

    def plotter_hist2d(X, ax, c):
        circle = Circle((0, 0), 1, fill=False, edgecolor='k',  ls='--', lw=0.5)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'cmap', ('#ffffff', c))
        ax.imshow(X, extent=(-1.5, 1.5, 1.5, -1.5), vmax=0.25, vmin=0, cmap=cmap)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(-1.5, 1.5)
        jP.configure_spines(ax)
        ax.set_xticks([-1, 1])
        ax.set_yticks([-1, 1])
        ax.set_aspect('equal', adjustable='box', anchor='SW')
        ax.axes.spines['left'].set_bounds(-1, 1)
        ax.axes.spines['bottom'].set_bounds(-1, 1)
        ax.set_xlabel(r'Re($\lambda$)')
        ax.add_patch(circle)

    H = eigs.apply(histogram)
    H = H.groupby('model_type').sum() / H.groupby('model_type').count()
    colors = ('tab:orange', 'tab:blue')
    list(map(plotter_hist2d, H[['shaping', 'no_shaping']], axs, colors))
    axs[1].set_yticks([])
    axs[1].axes.spines['left'].set_visible(False)
    axs[0].set_ylabel(f'Im($\lambda$)')

    key_pts = eigs.xs('shaping', level='model_type')\
                  .apply(pd.Series)[[0, 1, 2, 3]].agg([np.real, np.imag])\
                  .astype(np.float32).mean()
    arrowprops = dict(arrowstyle="-|>,head_length=0.3,head_width=0.1", facecolor='k')  
    for pt in key_pts.unstack().values:
        circle = matplotlib.patches.Circle(pt, 0.125, ec='tab:purple', lw=0.25, fill=False)
        axs[0].add_patch(circle)
        circle = matplotlib.patches.Circle(pt, 0.125, ec='tab:purple', lw=0.25, fill=False)
        axs[1].add_patch(circle)

    ax = axs[1]
    inax_position = ax.transAxes.transform([0.875, 0.277])
    infig_position = ax.figure.transFigure.inverted().transform(inax_position)
    color_scale = ax.figure.add_axes(
        list(infig_position) +
            [ax.get_position().width * 0.05, ax.get_position().height * 0.6])
    color_scale.imshow(
        np.expand_dims(np.linspace(1, 0, 30), -1), cmap='Grays', aspect='auto', vmin=0, vmax=1,extent=[0, 1, 0, 1])
    color_scale.set_yticks([0, 1])
    color_scale.set_yticklabels([0, '25%'], fontsize=5)
    color_scale.yaxis.set_label_position("right")
    color_scale.yaxis.tick_right()
    color_scale.set_ylabel('% RNNs', labelpad=-5, fontsize=6, rotation=-90)
    color_scale.set_xticks([])

    plt.show()

    
if __name__ == '__main__':
    main(sys.argv[1:])
