# [Figure S5a,b,c,d,e,f]
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
#jax.config.update("jax_platform_name", "cpu")
import jax.numpy as np

from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
import matplotlib
import matplotlib.pyplot as plt
import numpy as onp
import sys
from pathlib import Path
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import pandas as pd
import pickle
import itertools as it
from scipy.spatial import distance
import functools

sys.path.append('../../')
import leaky_rnn

from classes.datagen import tDNMSGenerator, genFactory
from classes.models import LeakyRNN
from classes.models import LeakyRNNCell
from analysis_tools.mpl_helpers import PdfPlotter
import models_database as mdb

from analysis_tools import jPlots as jP

TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'S': '#f89521'}

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  
    "axes.facecolor":    (1.0, 1.0, 1.0, 0.0),  
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0)})

def convert_color(c):
    c_hsv = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(c))
    c1 = matplotlib.colors.hsv_to_rgb(c_hsv * [0.95, 0.75, 1])
    c2 = matplotlib.colors.hsv_to_rgb(c_hsv * [1.05, 1, 0.75])
    return c1, c2

def distance_error_plot(model_type, predictions, y, long_cue_response, fps, path, dpi):
    if model_type == 'shaping':
        color = 'tab:orange'
    else:
        color = 'tab:blue'

    go_response = predictions.reindex(['LS', 'SL'], level='type').swaplevel(-1, 0)\
        .groupby('trial').apply(lambda x: x.loc['13s':'15s'].max())
    dist_func = functools.partial(distance.euclidean, fps['cue_on'][0])
    distances = y.xs('LS', level='type').groupby('trial').apply(lambda x: x.swaplevel(-1, 0).loc['8s']).apply(dist_func, axis=1)
    pts = pd.concat((distances.droplevel([0, 1]), long_cue_response), axis=1)
    margins = jP.default_margins()
    margins['right'] = 110
    margins['left'] = 140
    PdfPlotter(path / f'{model_type}.regression_3.25.pdf', fixed_margins=margins)
    plt.figure(figsize=(3.25, 2), dpi=dpi)
    gs = gridspec.GridSpec(1, 2, width_ratios=(2.75, 1))
    axs = list(map(plt.subplot, gs))
    ax = axs[0]

    c_hsv = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(TRIAL_COLORS['LS']))
    c1 = matplotlib.colors.hsv_to_rgb(c_hsv * [0.95, 0.75, 1])
    c2 = matplotlib.colors.hsv_to_rgb(c_hsv * [1, 1, 1])
    c3 = matplotlib.colors.hsv_to_rgb(c_hsv * [1.05, 1, 0.75])
    ls_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('tmp_list', [c1, c2, c3], N=10)
    ax.set_prop_cycle('color', list(map(ls_cmap, np.arange(10))))

    for segment in np.array_split(pts.sort_values(1).values, 10):
        ax.plot(*segment.T, ls='', marker='o')
    ax.set_xlim(2, 9)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Distance To Fixed Point')
    
    if model_type == 'shaping':
        label = 'Shaping + FT'
    elif model_type == 'no_shaping':
        label = 'No Shaping'
    ax.set_ylabel(f'{label}\nLS Response in ISI')
    jP.configure_spines(ax)
    jP.set_ylabel_position(ax, nlines=3)
    ax.text(0.02, 1, label, transform=ax.transAxes, color=color, ha='left', va='bottom')

    ax = axs[1]
    parts = ax.violinplot([long_cue_response.values, go_response.values], positions=[0, 1])
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
    parts['cmaxes'].set_edgecolor(color)
    parts['cmins'].set_edgecolor(color)
    parts['cbars'].set_edgecolor(color)

    jP.configure_spines(ax, fix_ylabel=False)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylim(0, 1)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.set_ylabel('Response Window Max (Go Trials)', rotation=-90, labelpad=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['LS ISI', 'Go Trial\nResp'], rotation_mode='anchor', rotation=45, ha='right')

    if long_cue_response.max() > go_response.min():
        ax.axhspan(go_response.min(), long_cue_response.max(), color='red', alpha=0.15, zorder=-1e3)

    plt.show()


def main(args):
    jP.set_rcParams(plt)
    dpi = 300
    path = Path('/analysis/ms_figures/fps_analysis')
    path.mkdir(parents=True, exist_ok=True)

    jP.set_rcParams(plt)
    path = Path('/analysis/ms_figures/fps_analysis')
    jP.make_folder(path)
    fp_tol = 1e-6
    model_ids = (9, 132)
    model_types = ('no_shaping', 'shaping')
    dpi = 300
    rnn_layer = 0

    for model_id, model_type in zip(model_ids, model_types):
        model_info = mdb.get_model(model_id)
        model_path = model_info['path']
        model_attrs = mdb.get_model_attributes(model_id)

        input_noise = model_attrs['input_noise']

        model = tf.keras.models.load_model(model_path)

        trials_gen = genFactory.create(
            model_info['task_name'], input_noise=input_noise, batch_size=1, n_blocks=1)
        X2 = trials_gen.generate_trials(100)

        time_idx = pd.to_timedelta(X2.index.get_level_values('idx')/10, unit='s')
        idx = X2.index.to_frame()
        idx['time'] = time_idx

        y = pd.DataFrame(model.layers[rnn_layer](onp.expand_dims(X2[['light', 'odor']], 0))[0])
        y.index = pd.MultiIndex.from_frame(idx)
        predictions = pd.Series(model.layers[1](y.values)[:, 0], index=y.index)
     
        def load_fps(path, fp_tol=fp_tol):
            with open(Path(path).parent / 'fps_info.pkl', 'rb') as f:
                fps = pickle.load(f)[fp_tol]
            return {'cue_on': fps['cue_on']['fps'], 'cue_off': fps['cue_off']['fps']}
        fps = load_fps(model_info['path'])

        long_cue_response = predictions.xs('LS', level='type').swaplevel(-1, 0)\
            .groupby('trial').apply(lambda x: x.loc['8s':'11s'].max())
        error_trials = long_cue_response\
            .where(long_cue_response > long_cue_response.quantile(0.9)).dropna()
        good_trials = long_cue_response\
            .where(long_cue_response < long_cue_response.quantile(0.1)).dropna()

        distance_error_plot(model_type, predictions, y, long_cue_response, fps, path, dpi)

        pca = PCA(n_components=3).fit(y)

        margins = jP.default_margins()
        margins['left'] = 90
        PdfPlotter(path / f'{model_type}.error_trials_4.pdf',
                   fixed_margins=margins)
        fig = plt.figure(figsize=(4, 1.5), dpi=dpi)
        gs_iter = iter(gridspec.GridSpec(1, 3, wspace=0.4))
        axs = []
        for gs in gs_iter:
            axs.extend(list(map(plt.subplot, gs.subgridspec(2, 1, hspace=0.2))))

        color = '#000000'
        color2 = '#ff0000'
        ls = '-'
        marker_style = {
            'ls': '', 'marker': 'x', 'zorder': 1e5,
            'ms': 3, 'mew': 0.8, 'clip_on': False}
        fps_pca = pca.transform(fps['cue_off'])
        fp_out = model.layers[1](fps['cue_off']).numpy().T[0]
        ax_iter = iter(axs)
        for ax, y_val in zip(ax_iter, [fps_pca[:, 1]] * 2 + [fps_pca[:, 2]] * 2 + [fp_out] * 2):
            ax.plot(fps_pca[:, 0], y_val, color=color2, **marker_style)

        fps_pca = pca.transform(fps['cue_on'])
        fp_out = model.layers[1](fps['cue_on']).numpy().T[0]
        ax_iter = iter(axs)
        for ax, y_val in zip(ax_iter, [fps_pca[:, 1]] * 2 + [fps_pca[:, 2]] * 2 + [fp_out] * 2):
            ax.plot(fps_pca[:, 0], y_val, color=color, **marker_style)
        
        h_error = y.reindex(error_trials.index, level='trial').groupby('trial')\
            .apply(lambda x: x.swaplevel(-1, 0).loc['3s':'11s'])
        h_error = h_error.droplevel(0).swaplevel(0, -1)
        h_error_pca = h_error.groupby('trial').apply(pca.transform)

        h_good = y.reindex(good_trials.index, level='trial').groupby('trial')\
            .apply(lambda x: x.swaplevel(-1, 0).loc['3s':'11s'])
        h_good = h_good.droplevel(0).swaplevel(0, -1)
        h_good_pca = h_good.groupby('trial').apply(pca.transform)

        h_error_out = predictions[h_error.index].groupby('trial').apply(list)
        h_good_out = predictions[h_good.index].groupby('trial').apply(list)

        c_good, c_error = convert_color(TRIAL_COLORS['LS'])
        def plot(x, ax, c):
            ax.plot(x[:, 0], x[:, 1], color=c, alpha=0.75)
            ax.plot(x[0, 0], x[0, 1], ls='', marker='o', color=c, mew=0.3, mec='k')
            ax.plot(x[50, 0], x[50, 1], ls='', marker='s', color=c, mew=0.3, mec='k')

        h_error_pca.apply(plot, ax=axs[0], c=c_error)
        h_good_pca.apply(plot, ax=axs[1], c=c_good)

        h_error_pca.apply(lambda x: x[:, [0, 2]]).apply(plot, ax=axs[2], c=c_error)
        h_good_pca.apply(lambda x: x[:, [0, 2]]).apply(plot, ax=axs[3], c=c_good)

        out_pc1 = pd.concat((h_error_pca.apply(lambda x: x[:,0]), h_error_out), axis=1)\
            .apply(lambda x: np.concatenate((np.array([x[0]]), np.array([x[1]])), axis=0).T, axis=1)
        out_pc1.apply(plot, ax=axs[4], c=c_error)

        out_pc1 = pd.concat((h_good_pca.apply(lambda x: x[:,0]), h_good_out), axis=1)\
            .apply(lambda x: np.concatenate((np.array([x[0]]), np.array([x[1]])), axis=0).T, axis=1)
        out_pc1.apply(plot, ax=axs[5], c=c_good)

        axs[0].text(-0.09, 0.025, 'PC 2', fontsize=8, rotation=90, transform=axs[0].transAxes, ha='right', va='center')
        axs[2].text(-0.09, 0.025, 'PC 3', fontsize=8, rotation=90, transform=axs[2].transAxes, ha='right', va='center')
        axs[4].text(-0.09, 0.025, 'Response', fontsize=8, rotation=90, transform=axs[4].transAxes, ha='right', va='center')

        for ax in axs:
            jP.configure_spines(ax, fix_ylabel=False)
            ax.set_xlim(-9, 8)
            ax.spines['left'].set_position(('outward', -5))

        for ax in axs[:-2]:
            ax.set_yticks([-3, 3])
            ax.spines['left'].set_bounds([-3, 3])

        for ax in axs[::2]:
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)

        for ax in axs[1::2]:
            ax.set_xticks([-5, 0, 5])
            ax.set_xlabel('PC 1')
            ax.spines['bottom'].set_bounds([-5, 5])

        axs[0].set_ylim(-5.6-2.8, 5.6)
        axs[1].set_ylim(-5.6-2.8, 5.6)

        axs[2].set_ylim(-4.4-2.2, 4.4)
        axs[3].set_ylim(-4.4-2.2, 4.4)

        for ax in axs[-2:]:
            ax.set_yticks([0, 1])
            ax.set_ylim(-0.25, 1)
            ax.spines['left'].set_bounds([0, 1])

        error_label = matplotlib.lines.Line2D(
            [], [], color=c_error, linestyle='-',
            label='Early Response')
        good_label = matplotlib.lines.Line2D(
            [], [], color=c_good, linestyle='-',
            label='No Early Response')
        fp_off_label = matplotlib.lines.Line2D(
            [], [], color='r', linestyle='', marker='x',
            label='Cue Off FP')
        fp_on_label = matplotlib.lines.Line2D(
            [], [], color='k', linestyle='', marker='x',
            label='Cue On FP')
        cue_start = matplotlib.lines.Line2D(
            [], [], color='k', linestyle='', marker='o',
            label='Cue Start')
        cue_stop = matplotlib.lines.Line2D(
            [], [], color='k', linestyle='', marker='s',
            label='Cue Stop')
     
        legend_lines = [error_label, good_label, fp_off_label, fp_on_label,
                        cue_start, cue_stop]
        axs[4].legend(
            'lower right',
            handles=legend_lines, bbox_to_anchor=(1, 1.75),
            ncols=3, fontsize='x-small')

        plt.show()
     
        short_cue_response = predictions.xs('LS', level='type').swaplevel(-1, 0)\
            .groupby('trial').apply(lambda x: x.loc['13s':'15s'].max())
        error2_trials = long_cue_response\
            .where(short_cue_response > short_cue_response.quantile(0.1)).dropna()

        h_error2 = y.reindex(error_trials.index, level='trial').groupby('trial')\
            .apply(lambda x: x.swaplevel(-1, 0).loc['13s':'15s'])
        h_error2 = h_error.droplevel(0).swaplevel(0, -1)

        cue_times = X2['cues'].xs('LS', level='type').groupby('idx').head(1)
        cue_transitions = pd.concat(
            (cue_times.where(cue_times.diff() > 0).dropna().reset_index()['idx'],
             cue_times.where(cue_times.diff() < 0).dropna().reset_index()['idx']),
            axis=1, keys=('start', 'stop')) / 10

        margins = jP.default_margins()
        PdfPlotter(path / f'{model_type}.error_response.pdf',
                   fixed_margins=margins)
        plt.figure(figsize=(3, 1.5), dpi=dpi)
        ax = plt.gca()
        color = TRIAL_COLORS['LS']
        cue_transitions.apply(lambda x, ax=ax, c='gray': ax.axvspan(*x, color=c, alpha=0.25), axis=1)
        jP.configure_spines(ax)

        error_predictions = predictions.reindex(error_trials.index, level='trial')
        good_predictions = predictions.reindex(good_trials.index, level='trial')

        error_predictions.groupby('trial').apply(
            lambda x, ax=ax, c=c_error:
                ax.plot(x.index.get_level_values('time').total_seconds(), x, color=c))
        good_predictions.groupby('trial').apply(
            lambda x, ax=ax, c=c_good:
                ax.plot(x.index.get_level_values('time').total_seconds(), x, color=c))

        ax.set_xlim(0, 20)
        if model_type == 'shaping':
            label = 'Shaping + FT'
        elif model_type == 'no_shaping':
            label = 'No Shaping'
        if model_type == 'shaping':
            color = 'tab:orange'
        else:
            color = 'tab:blue'
        ax.text(0, 1, label, transform=ax.transAxes, color=color, ha='left', va='bottom')

        error_label = matplotlib.lines.Line2D(
            [], [], color=c_error, linestyle='-',
            label='First Cue Response')
        good_label = matplotlib.lines.Line2D(
            [], [], color=c_good, linestyle='-',
            label='First Cue No Response')
        ax.legend(
            'bottom left',
            handles=[error_label, good_label], bbox_to_anchor=(-0.02, 1.3), loc=2, borderaxespad=0.,
            ncols=2)
        ax.set_ylabel('Response')
        ax.set_xlabel('Time (s)')
        ax.set_xticks([3, 6, 9, 12, 15, 18])
        ax.set_yticks([0, 0.5])
        plt.show()
        

if __name__ == '__main__':
    main(sys.argv[1:])
