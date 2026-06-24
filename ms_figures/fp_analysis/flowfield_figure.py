# [Figure 4a,b,c]
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import datetime
import h5py

import jax
#jax.config.update("jax_platform_name", "cpu")
import jax.numpy as np
from jax import jacrev, random, vmap
from jax.example_libraries import optimizers

from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as onp
import sys
import time
from pathlib import Path
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import pandas as pd
import pickle

sys.path.append('../../')
import leaky_rnn

import fixed_point_finder.fixed_points as fp_optimize

from classes.datagen import tDNMSGenerator, genFactory
from classes.models import LeakyRNN
from classes.models import LeakyRNNCell
from analysis_tools.mpl_helpers import PdfPlotter
import models_database as mdb

from analysis_tools import jPlots as jP
#from jax.experimental import io_callback

#jax.config.update("jax_traceback_filtering", "off")
#jax.config.update("xla_python_client_preallocate", "false")

TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'S': '#f89521'}

def plot_pcs(axs, pca,  h_t, h_full, color='red'):

    hfull_pca = pca.transform(h_full[0])

    if h_t is not None:
        h_pca = pca.transform(h_t[0])
    alpha = 0.01

    size = 4
    marker_style = dict(marker='o', ms=size, mec='gray', color='k')
    ax = axs[0]
    ax.plot(hfull_pca[:, 0], hfull_pca[:, 1], c=color, alpha=0.25)
    #ax.plot(fp_pca[:, 0], fp_pca[:, 1], ls='', **marker_style)
    if h_t is not None:
        ax.plot(h_pca[:, 0], h_pca[:, 1], c='red')
        ax.plot(h_pca[0, 0], h_pca[0, 1], c='green', ms=2, ls='', marker='o')
        ax.plot(h_pca[-1, 0], h_pca[-1, 1], c='red', ms=2, ls='', marker='o')
    ax.set_ylim(-8, 8)
    ax.set_xlim(-8, 8)
    ax.set_xticks([])

    ax = axs[1]
    ax.plot(hfull_pca[:, 0], hfull_pca[:, 2], c=color, alpha=0.25)
    #ax.plot(fp_pca[:, 0], fp_pca[:, 2], ls='', **marker_style)
    if h_t is not None:
        ax.plot(h_pca[:, 0], h_pca[:, 2], c='red')
        ax.plot(h_pca[0, 0], h_pca[0, 2], c='green', ms=2, ls='', marker='o')
        ax.plot(h_pca[-1, 0], h_pca[-1, 2], c='red', ms=2, ls='', marker='o')
    ax.set_ylim(-8, 8)
    ax.set_xlim(-8, 8)



def main(args):
    jP.set_rcParams(plt)
    dpi = 300
    path = Path('/analysis/ms_figures/fps_analysis')
    path.mkdir(parents=True, exist_ok=True)

    fp_tol = 1e-6
    model_ids = pd.Series([18, 130, 132],
                          index=('no_shaping', 'only_shaping', 'shaping'))

    model_infos = model_ids.apply(mdb.get_model).apply(pd.Series)
    model_attrs = model_ids.apply(mdb.get_model_attributes)
    model_infos = pd.concat(
        (model_infos,
         model_attrs[['gamma', 'epoc', 'input_noise', 'noise_level']]), axis=1)

    def load_fps(path, fp_tol=fp_tol):
        with open(Path(path).parent / 'fps_info.pkl', 'rb') as f:
            fps = pickle.load(f)[fp_tol]
        return {'cue_on': fps['cue_on']['fps'], 'cue_off': fps['cue_off']['fps']}
    fps = model_infos['path'].apply(load_fps).apply(pd.Series).stack()

    models = model_infos['path'].apply(tf.keras.models.load_model)
    def create_params(model, gamma):
        w_in, w_r, b = model.layers[0].get_weights()
        w_out, b_out = model.layers[1].get_weights()
        h0 = onp.squeeze(model.layers[0](onp.zeros((1, 1, 2))).numpy())
        return leaky_rnn.rnn_params(h0, b, w_in, w_r, w_out, b_out, gamma, 0)
    model_params = pd.concat((models, model_infos['gamma']), axis=1)\
        .apply(lambda x: create_params(*x), axis=1)

    trials_gen = genFactory.create(
        'just_short_match', input_noise=0, batch_size=1, n_blocks=1)
    inputs = trials_gen.generate_trials(1)[['light', 'odor']]
    inputs = inputs.groupby(['type', 'idx']).head(1)
    rnn_res = model_params.apply(
        lambda x, inputs=np.expand_dims(inputs.values, 0):
            leaky_rnn.batched_rnn_run(x, inputs))\
                .apply(pd.Series, index=('h', 'o'))

    h_starts = rnn_res['h'].apply(lambda x: x[0, -1])
    pd.concat((model_params, h_starts), axis=1)\
        .apply(lambda x: x[0].update({'h0': x['h']}), axis=1)

    rnn_res = model_params.apply(
        lambda x, inputs=np.expand_dims(inputs.values, 0):
            leaky_rnn.batched_rnn_run(x, inputs))\
                .apply(pd.Series, index=('h', 'o'))\
                .map(np.squeeze)

    pcas = rnn_res['h'].apply(lambda x: PCA(n_components=3).fit(x))

    # Project the only shaping model into the PCA pf the fully trained model
    pcas['only_shaping'] = pcas['shaping']

    margins = jP.default_margins()
    margins['right'] = 250
    PdfPlotter(path / 'single_trajectories.pdf', fixed_margins=margins)
    fig = plt.figure(figsize=(6.5, 1.5), dpi=dpi)
    gs = iter(gridspec.GridSpec(1, 3, wspace=0.4))
    
    def plot_trial_pca(X, axs, pca, fps):
        c = TRIAL_COLORS[X.name]
        h_t = np.expand_dims(X.values, 0)
        plot_pcs(axs[::2], pca, None, h_t, color=c)
        plot_pcs(axs[1::2], pca, None, h_t, color=c)
        fp_pca = fps.apply(pca.transform)
        marker_style = {
            'ls': '', 'marker': 'x', 'color': 'k', 'zorder': 1e5,
            'ms': 3, 'mew': 0.8}
        axs[0].plot(*fp_pca['cue_on'].T[[0, 1]], **marker_style)
        axs[1].plot(*fp_pca['cue_off'].T[[0, 1]], **marker_style)
        axs[2].plot(*fp_pca['cue_on'].T[[0, 2]], **marker_style)
        axs[3].plot(*fp_pca['cue_off'].T[[0, 2]], **marker_style)

    def plot_trajectory(params, pca, x_star, axs, c):
        h_t, o_t = leaky_rnn.batched_rnn_run(
            params, np.tile(np.array(x_star), (1, 600, 1)))
        h_pca = pca.transform(h_t[0])

        axs[0].plot(h_pca[:, 0], h_pca[:, 1], c=c)
        axs[0].plot(h_pca[0, 0], h_pca[0, 1], c='green', ms=2, ls='', marker='o')
        axs[0].plot(h_pca[-1, 0], h_pca[-1, 1], c='red', ms=2, ls='', marker='o')

        axs[1].plot(h_pca[:, 0], h_pca[:, 2], c=c)
        axs[1].plot(h_pca[0, 0], h_pca[0, 2], c='green', ms=2, ls='', marker='o')
        axs[1].plot(h_pca[-1, 0], h_pca[-1, 2], c='red', ms=2, ls='', marker='o')

    for model_type in model_ids.index:
        gs0 = next(gs).subgridspec(2, 2)
        axs = list(map(plt.subplot, gs0))
        pd.DataFrame(rnn_res['h'][model_type], index=inputs.index).groupby('type')\
            .apply(plot_trial_pca, axs=axs, pca=pcas[model_type],
                   fps=fps[model_type])
        plot_trajectory(model_params[model_type], pca=pcas[model_type],
                        x_star=(0, 1), axs=axs[::2], c='k')
        plot_trajectory(model_params[model_type], pca=pcas[model_type],
                        x_star=(0, 0), axs=axs[1::2], c='r')

        list(map(jP.configure_spines, axs))
        list(map(lambda x: x.set_xlim(-9, 8), axs))

        axs[0].set_title('Cue On', fontsize=6, pad=0)
        axs[1].set_title('Cue Off', fontsize=6, pad=0, color='red')

        title_color = ('tab:blue' if model_type == 'no_shaping' else 'tab:orange')
        title = model_type.replace('_', ' ').title()
        if title == 'Shaping':
            title = 'Shaping + Full'
        axs[0].text(1.1, 1.35, title,
                    transform=axs[0].transAxes, fontsize=8, ha='center',
                    color=title_color)

        axs[0].set_ylabel('PC2')
        axs[2].set_ylabel('PC3')
        for ax in axs[::2]:
            ax.spines['left'].set_position(('outward', -5))
            ax.spines['left'].set_bounds([-2.5, 2.5])
            ax.set_yticks([-2.5, 2.5])

        for ax in axs[1::2]:
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)

        for ax in axs[:2]:
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)
            ax.set_ylim(-7, 7)
            ax.set_xlim(-9, 8)

        for ax in axs[2:]:
            ax.spines['bottom'].set_bounds([-5, 5])
            ax.set_xticks([-5, 5])
            ax.set_ylim(-5, 6)
            ax.set_xlabel('PC 1')

    fp_label = matplotlib.lines.Line2D(
        [], [], color='k', marker='x', linestyle='None',
        ms=3, mew=0.8, label='Fixed Point')
    start_label = matplotlib.lines.Line2D(
        [], [], color='green', marker='o', linestyle='None',
        ms=2, label='Start')
    end_label = matplotlib.lines.Line2D(
        [], [], color='red', marker='o', linestyle='None',
        ms=2, label='End (60s)')
    cue_on_label = matplotlib.lines.Line2D(
        [], [], color='k', linestyle='-',
        label='Cue On')
    cue_off_label = matplotlib.lines.Line2D(
        [], [], color='red', linestyle='-',
        label='Cue Off')
    axs[1].legend(
        handles=[fp_label, start_label, end_label, cue_on_label, cue_off_label],
        bbox_to_anchor=(1.25, 1), loc=2, borderaxespad=0., fontsize='x-small')
    plt.show()

    margins = jP.default_margins()
    margins['right'] = 250
    PdfPlotter(path / 'single_trajectories_7.pdf', fixed_margins=margins)
    fig = plt.figure(figsize=(7, 1.5), dpi=dpi)
    gs = iter(gridspec.GridSpec(1, 3, wspace=0.4))
    
    for model_type in model_ids.index:
        gs0 = next(gs).subgridspec(2, 2)
        axs = list(map(plt.subplot, gs0))
        pd.DataFrame(rnn_res['h'][model_type], index=inputs.index).groupby('type')\
            .apply(plot_trial_pca, axs=axs, pca=pcas[model_type],
                   fps=fps[model_type])
        plot_trajectory(model_params[model_type], pca=pcas[model_type],
                        x_star=(0, 1), axs=axs[::2], c='k')
        plot_trajectory(model_params[model_type], pca=pcas[model_type],
                        x_star=(0, 0), axs=axs[1::2], c='r')

        list(map(jP.configure_spines, axs))
        list(map(lambda x: x.set_xlim(-9, 8), axs))

        axs[0].set_title('Cue On', fontsize=6, pad=0)
        axs[1].set_title('Cue Off', fontsize=6, pad=0, color='red')

        title_color = ('tab:blue' if model_type == 'no_shaping' else 'tab:orange')
        title = model_type.replace('_', ' ').title()
        if title == 'Shaping':
            title = 'Shaping + Full'
        axs[0].text(1.1, 1.35, title,
                    transform=axs[0].transAxes, fontsize=8, ha='center',
                    color=title_color)

        axs[0].set_ylabel('PC2')
        axs[2].set_ylabel('PC3')
        for ax in axs[::2]:
            ax.spines['left'].set_position(('outward', -5))
            ax.spines['left'].set_bounds([-2.5, 2.5])
            ax.set_yticks([-2.5, 2.5])

        for ax in axs[1::2]:
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)

        for ax in axs[:2]:
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)
            ax.set_ylim(-7, 7)
            ax.set_xlim(-9, 8)

        for ax in axs[2:]:
            ax.spines['bottom'].set_bounds([-5, 5])
            ax.set_xticks([-5, 5])
            ax.set_ylim(-5, 6)
            ax.set_xlabel('PC 1')

    fp_label = matplotlib.lines.Line2D(
        [], [], color='k', marker='x', linestyle='None',
        ms=3, mew=0.8, label='Fixed Point')
    start_label = matplotlib.lines.Line2D(
        [], [], color='green', marker='o', linestyle='None',
        ms=2, label='Start')
    end_label = matplotlib.lines.Line2D(
        [], [], color='red', marker='o', linestyle='None',
        ms=2, label='End (60s)')
    cue_on_label = matplotlib.lines.Line2D(
        [], [], color='k', linestyle='-',
        label='Cue On')
    cue_off_label = matplotlib.lines.Line2D(
        [], [], color='red', linestyle='-',
        label='Cue Off')
    axs[1].legend(
        handles=[fp_label, start_label, end_label, cue_on_label, cue_off_label],
        bbox_to_anchor=(1.25, 1), loc=2, borderaxespad=0., fontsize='x-small')
    plt.show()
 

    margins = jP.default_margins()
    margins['top'] = 130
    for model_type in model_ids.index:
        PdfPlotter(path / f'{model_type}.flows.pdf',
                   fixed_margins=margins)
        fig = plt.figure(figsize=(2.375, 2), dpi=dpi)
        gs = iter(gridspec.GridSpec(2, 2))
        axs = list(map(plt.subplot, gs))
        pd.DataFrame(rnn_res['h'][model_type], index=inputs.index)\
            .groupby('type')\
            .apply(plot_trial_pca, axs=axs, pca=pcas[model_type],
                   fps=fps[model_type])

        h_full = rnn_res['h'][model_type]
        test_h = h_full + (onp.random.random(h_full.shape) * 2 - 1) * 3
        test_h, _ = fp_optimize.keep_unique_fixed_points(
            test_h, identical_tol=17, do_print=True)
        test_h = test_h[::12]

        arrowprops = dict(arrowstyle='-|>, head_width=0.1', shrinkA=0,
                          shrinkB=0, lw=0.3)
        for h in test_h:
            model_params[model_type]['h0'] = h

            x_star = np.array([0, 1])
            h_t, o_t = leaky_rnn.batched_rnn_run(
                model_params[model_type], np.tile(x_star, (1, 200, 1)))
            pcs  = pcas[model_type].transform(h_t[0])
            jP.plot_seg_colors(axs[0], *pcs.T[[0, 1]], c='#555555', n=5, lw=0.3)
            jP.plot_seg_colors(axs[2], *pcs.T[[0, 2]], c='#555555', n=5, lw=0.3)
            arrowprops['facecolor'] = '#555555'
            axs[0].annotate('', pcs[7, [0, 1]], pcs[6, [0, 1]],
                            arrowprops=arrowprops)
            axs[2].annotate('', pcs[7, [0, 2]], pcs[6, [0, 2]],
                            arrowprops=arrowprops)

            x_star = np.array([0, 0])
            h_t, o_t = leaky_rnn.batched_rnn_run(
                model_params[model_type], np.tile(x_star, (1, 200, 1)))
            pcs  = pcas[model_type].transform(h_t[0])
            jP.plot_seg_colors(axs[1], *pcs.T[[0, 1]], c='#aa0000', n=5, lw=0.3)
            jP.plot_seg_colors(axs[3], *pcs.T[[0, 2]], c='#aa0000', n=5, lw=0.3)
            arrowprops['facecolor'] = '#aa0000'
            axs[1].annotate('', pcs[7, [0, 1]], pcs[6, [0, 1]],
                            arrowprops=arrowprops)
            axs[3].annotate('', pcs[7, [0, 2]], pcs[6, [0, 2]],
                            arrowprops=arrowprops)


        list(map(jP.configure_spines, axs))
        list(map(lambda x: x.set_xlim(-9, 8), axs))

        axs[0].set_title('Cue On', fontsize=6, pad=0)
        axs[1].set_title('Cue Off', fontsize=6, pad=0, color='red')

        title_color = ('tab:blue' if model_type == 'no_shaping'
                       else 'tab:orange')
        title = model_type.replace('_', ' ').title()
        if title == 'Shaping':
            title = 'Shaping + Full'
        axs[0].text(1.1, 1.35, title,
                    transform=axs[0].transAxes, fontsize=8, ha='center',
                    color=title_color)

        axs[0].set_ylabel('PC2')
        axs[2].set_ylabel('PC3')
        for ax in axs[::2]:
            ax.spines['left'].set_position(('outward', -5))
            ax.spines['left'].set_bounds([-2.5, 2.5])
            ax.set_yticks([-2.5, 2.5])

        for ax in axs[1::2]:
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)

        for ax in axs[:2]:
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)
            ax.set_ylim(-7, 7)
            ax.set_xlim(-9, 8)

        for ax in axs[2:]:
            ax.spines['bottom'].set_bounds([-5, 5])
            ax.set_xticks([-5, 5])
            ax.set_ylim(-5, 6)
            ax.set_xlabel('PC 1')

        plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
