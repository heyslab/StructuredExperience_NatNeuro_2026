# [Figure 5d]
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

def plot_seg_colors3D(ax, x, y, z,  c, n=15, **kwargs):
    c_hsv = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(c))
    c1 = matplotlib.colors.hsv_to_rgb(c_hsv * [1, 0.5, 1])
    c3 = matplotlib.colors.hsv_to_rgb(c_hsv * [1, 1, 0.5])
    ls_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('tmp_list', [c1, c, c3], N=n)

    ax.set_prop_cycle('color', list(map(ls_cmap, np.arange(n))))
    def fmat_data(a, n=n):
        formatted = np.append(a, [np.nan] * (n-len(a)%n)).reshape(n, -1).T
        return np.concatenate((formatted, np.expand_dims(np.append(formatted[0, 1:], [np.nan]), 0)))
    x_formatted = fmat_data(x)
    y_formatted = fmat_data(y)
    z_formatted = fmat_data(z)
    print(x_formatted.shape)
    for i in range(x_formatted.shape[-1]):
        ax.plot3D(x_formatted[:, i], y_formatted[:, i], z_formatted[:, i],
                  **kwargs)
    return ax


def plot_seg_colors(ax, x, y, c, n=15, **kwargs):
    c_hsv = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(c))
    c1 = matplotlib.colors.hsv_to_rgb(c_hsv * [1, 0.25, 1])
    c3 = matplotlib.colors.hsv_to_rgb(c_hsv * [1, 1, 0.25])
    ls_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('tmp_list', [c1, c, c3], N=n)

    ax.set_prop_cycle('color', list(map(ls_cmap, np.arange(n))))
    x_formated = np.append(x, [np.nan] * (n-len(x)%n)).reshape(n, -1).T
    x_formated = np.concatenate((x_formated, np.expand_dims(np.append(x_formated[0, 1:], [np.nan]), 0)))
    y_formated = np.append(y, [np.nan] * (n-len(y)%n)).reshape(n, -1).T
    y_formated = np.concatenate((y_formated, np.expand_dims(np.append(y_formated[0, 1:], [np.nan]), 0)))
    ax.plot(x_formated, y_formated, **kwargs)
    return x, y



def plot_pca(ax, pcs, dims=(1, 2)):
    def plot_trial(res, ax):
        trial_type = res.index.unique('type')[0]
        color = TRIAL_COLORS[trial_type]
        plot_seg_colors(
            ax, *res.values.T, c=color, clip_on=False)

    pcs.groupby(['type', 'trial']).apply(
        plot_trial, ax=ax)
    jP.configure_spines(ax, fix_ylabel=False, fix_xlabel=False)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal', anchor='C')


def main(argv):
    models = pd.Series(
        [47, 68, 65],
        index=['sl_shaping', 'ls_shaping', 'ls_shaping2'])

    margins={'left': 80, 'right': 100, 'top': 45, 'bottom': 80}

    model_infos = models.apply(mdb.get_model).apply(pd.Series)
    model_infos = pd.concat(
        (model_infos, models.apply(mdb.get_model_attributes)), axis=1)

    path = Path('/analysis/ms_figures/shaping_tasks')
    jP.make_folder(path)

    dpi = 300
    jP.set_rcParams(plt)

    models = model_infos['path'].apply(tf.keras.models.load_model)

    trial_gen = genFactory.create(
        'just_short_match', input_noise=model_infos['input_noise'].head(1), batch_size=1,
        n_blocks=1)
    X2 = trial_gen.generate_trials(25)
    formatted_X = trial_gen.format_validation(X2)[0]
    cols = X2.index.names
    X2 = X2.reset_index().set_index(cols + ['cues'])
    index = X2.index

    reverse_x = False

    input_noise = model_infos['input_noise'].head(1)
    rnn_layer = 0

    ys = pd.concat(
        list(map(lambda m, X2=X2, l=rnn_layer: pd.DataFrame(
            m.layers[l](np.expand_dims(X2[['light', 'odor']], 0))[0], index=X2.index), models)),
            keys=models.index.values)
    ys.index.names = ['model_type'] + ys.index.names[1:]

    def calc_pca(y, **pca_kwargs):
        pca = PCA(**pca_kwargs)
        shapes = pca.fit_transform(y)
        pca_res = pd.DataFrame(shapes, index=y.index) # [np.array(dims) - 1]
        return pca_res

    pcs = ys.groupby('model_type').apply(calc_pca, n_components=3)

    margins = {'left': 80, 'right': 100, 'top': 45, 'bottom': 80}
    PdfPlotter(path / f'pca_3plots.pdf', fixed_margins=margins)
    plt.figure(figsize=(3.5, 2), dpi=dpi)
    gs = gridspec.GridSpec(2, 3, wspace=0.05)
    axs = list(map(plt.subplot, gs))
    axs = [axs[0], axs[2], axs[3], axs[5], axs[1], axs[4]]

    plot_pca(axs[0], pcs.loc['sl_shaping'].drop(2, axis=1))
    plot_pca(axs[2], pcs.loc['sl_shaping'].drop(1, axis=1))

    plot_pca(axs[1], pcs.loc['ls_shaping2'].drop(2, axis=1))
    plot_pca(axs[3], pcs.loc['ls_shaping2'].drop(1, axis=1))

    plot_pca(axs[4], pcs.loc['ls_shaping'].drop(2, axis=1))
    plot_pca(axs[5], pcs.loc['ls_shaping'].drop(1, axis=1))
    
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
        ax.set_ylim(-5, 10)

    for ax in (axs[2], axs[3], axs[5]):
        ax.spines['bottom'].set_bounds([-5, 5])
        ax.set_xticks([-5, 0, 5])
        ax.set_xlabel('PC 1')

    #for ax in (axs[1], axs[3], axs[4], axs[5]):
    for ax in axs:
        ax.set_xlim(-12, 12)

    axs[0].set_ylabel('PC 2')
    axs[2].set_ylabel('PC 3')

    legend_lines = pcs.index.unique('type').to_series().apply(
        lambda x: matplotlib.lines.Line2D(
            [0], [0], c=TRIAL_COLORS[x], lw=2, label=x))
    axs[1].legend(handles=list(legend_lines.values), loc='upper right', bbox_to_anchor=(1.37, 0.8))
    axs[1].set_title('LS + Full (B.)', color='tab:red', y=0.8)
    axs[4].set_title('LS + Full (A.)', color='tab:red', y=0.87)
    axs[0].set_title('SL + Full', color='tab:green', y=0.8)

    ax = axs[3]
    inax_position = ax.transAxes.transform([0.8, 0.8])
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

    margins = {'left': 80, 'right': 100, 'top': 45, 'bottom': 80}
    PdfPlotter(path / f'pca_3plots_vert.pdf', fixed_margins=margins)
    plt.figure(figsize=(2.25, 3), dpi=dpi)
    gs = gridspec.GridSpec(3, 2, wspace=0.01, hspace=0.01)
    axs = list(map(plt.subplot, gs))

    plot_pca(axs[0], pcs.loc['sl_shaping'].drop(2, axis=1))
    plot_pca(axs[1], pcs.loc['sl_shaping'].drop(1, axis=1))

    plot_pca(axs[2], pcs.loc['ls_shaping2'].drop(2, axis=1))
    plot_pca(axs[3], pcs.loc['ls_shaping2'].drop(1, axis=1))

    plot_pca(axs[4], pcs.loc['ls_shaping'].drop(2, axis=1))
    plot_pca(axs[5], pcs.loc['ls_shaping'].drop(1, axis=1))
    
    for ax in axs[0::2]:
        ax.spines['left'].set_position(('outward', -5))
        ax.spines['left'].set_bounds([-4, 4])
        ax.set_yticks([-4, 4])
        ax.set_ylabel('PC 2')

    for ax in axs[1::2]:
        #ax.set_yticks([])
        ax.yaxis.tick_right()
        ax.spines['right'].set_position(('outward', -5))
        ax.spines['right'].set_bounds([-4, 4])
        ax.set_yticks([-4, 4])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(True)
        ax.yaxis.set_label_position("right")
        ax.set_ylabel('PC 3')

    for ax in axs[:4]:
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)

    for ax in axs[4:]:
        ax.spines['bottom'].set_bounds([-5, 5])
        ax.set_xticks([-5, 0, 5])
        ax.set_xlabel('PC 1')

    #for ax in (axs[1], axs[3], axs[4], axs[5]):
    for ax in axs:
        ax.set_xlim(-11, 11)
        ax.set_ylim(-5, 5)

    legend_lines = pcs.index.unique('type').to_series().apply(
        lambda x: matplotlib.lines.Line2D(
            [0], [0], c=TRIAL_COLORS[x], lw=2, label=x))
    axs[5].legend(handles=list(legend_lines.values), loc='lower right', bbox_to_anchor=(1, 1))

    axs[0].text(0, 1.05, 'SL + Full', color='tab:green', ha='left', va='bottom', transform=axs[0].transAxes)
    axs[2].text(0, 1.05, 'LS + Full (A.)', color='tab:red', ha='left', va='bottom', transform=axs[2].transAxes)
    axs[4].text(0, 1.05, 'LS + Full (B.)', color='tab:red', ha='left', va='bottom', transform=axs[4].transAxes)

    ax = axs[3]
    inax_position = ax.transAxes.transform([0.45, 1.0])
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
