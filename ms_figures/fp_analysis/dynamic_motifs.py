# [Figure 4d, S3a,b,c]
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
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

sys.path.append('../../')
import leaky_rnn

from classes.datagen import tDNMSGenerator, genFactory
from classes.models import LeakyRNN
from classes.models import LeakyRNNCell
from analysis_tools.mpl_helpers import PdfPlotter
import models_database as mdb

from analysis_tools import jPlots as jP

#jax.config.update("jax_traceback_filtering", "off")
#jax.config.update("xla_python_client_preallocate", "false")

TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'S': '#f89521'}
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
    inputs = inputs.groupby(['type', 'idx']).head(1).swaplevel(1, 0).reindex(['SS', 'SL', 'LS'], level='type')
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

    pd.concat((model_params, h_starts), axis=1)\
        .apply(lambda x: x[0].update({'h0': x['h']}), axis=1)
    no_cue = np.tile(np.array([0, 0]), (1, 400, 1))
    nullcline_nocue = model_params.apply(
        lambda x, inputs=no_cue:
            leaky_rnn.batched_rnn_run(x, inputs))\
                .apply(pd.Series, index=('h', 'o'))\
                .map(np.squeeze)
    h_vals = nullcline_nocue['h']
    h_vals['only_shaping'] = h_vals['only_shaping'][:200]
    nullcline_nocue['h'] = h_vals
    o_vals = nullcline_nocue['o']
    o_vals['only_shaping'] = o_vals['only_shaping'][:200]
    nullcline_nocue['o'] = o_vals

    pd.concat((model_params, h_starts), axis=1)\
        .apply(lambda x: x[0].update({'h0': x['h']}), axis=1)
    model_params['no_shaping']['h0']= rnn_res['h']['no_shaping'][200]
    cue = np.tile(np.array([0, 1]), (1, 400, 1))
    nullcline_cue = model_params.apply(
        lambda x, inputs=cue:
            leaky_rnn.batched_rnn_run(x, inputs))\
                .apply(pd.Series, index=('h', 'o'))\
                .map(np.squeeze)
    h_vals = nullcline_cue['h']
    h_vals['only_shaping'] = h_vals['only_shaping'][:200]
    nullcline_cue['h'] = h_vals
    o_vals = nullcline_cue['o']
    o_vals['only_shaping'] = o_vals['only_shaping'][:200]
    nullcline_cue['o'] = o_vals
 
    pcas = rnn_res['h'].apply(lambda x: PCA(n_components=3).fit(x))
    # Project the only shaping model into the PCA pf the fully trained model
    pcas['only_shaping'] = pcas['shaping']

    margins = jP.default_margins()
    margins['top'] = 140
    PdfPlotter(path / f'dynamic_motifs.pdf',
               fixed_margins=margins)
    fig = plt.figure(figsize=(1.75, 2), dpi=dpi)
    gs = iter(gridspec.GridSpec(3, 1))
    axs = list(map(plt.subplot, gs))
    legend_lines = []
    for model_type, color, color2, ls in zip(('only_shaping', 'shaping'),
                                             ('#880000', '#dd0000'),
                                             ('#888888', '#111111'),
                                             ('--', '-')):

        null_pcs = pcas[model_type].transform(nullcline_nocue['h'][model_type])
        axs[0].plot(null_pcs[:, 0], null_pcs[:, 1], color=color, ls=ls)
        axs[1].plot(null_pcs[:, 0], null_pcs[:, 2], color=color, ls=ls)
        axs[2].plot(null_pcs[:, 0], nullcline_nocue['o'][model_type], color=color, ls=ls)

        null_pcs = pcas[model_type].transform(nullcline_cue['h'][model_type])
        axs[0].plot(null_pcs[:, 0], null_pcs[:, 1], color=color2, ls=ls)
        axs[1].plot(null_pcs[:, 0], null_pcs[:, 2], color=color2, ls=ls)
        axs[2].plot(null_pcs[:, 0], nullcline_cue['o'][model_type], color=color2, ls=ls)

        marker_style = {
            'ls': '', 'marker': 'x', 'zorder': 1e5,
            'ms': 3, 'mew': 0.8}
        fps_pca = pcas[model_type].transform(fps[model_type]['cue_off'])
        fp_out = leaky_rnn.affine(model_params[model_type], fps[model_type]['cue_off']).T[0]
        axs[0].plot(fps_pca[:, 0], fps_pca[:, 1], color=color, **marker_style)
        axs[1].plot(fps_pca[:, 0], fps_pca[:, 2], color=color, **marker_style)
        axs[2].plot(fps_pca[:, 0], fp_out, color=color, **marker_style)

        fps_pca = pcas[model_type].transform(fps[model_type]['cue_on'])
        fp_out = leaky_rnn.affine(model_params[model_type], fps[model_type]['cue_on']).T[0]
        axs[0].plot(fps_pca[:, 0], fps_pca[:, 1], color=color2, **marker_style)
        axs[1].plot(fps_pca[:, 0], fps_pca[:, 2], color=color2, **marker_style)
        axs[2].plot(fps_pca[:, 0], fp_out, color=color2, **marker_style)

        if model_type == 'shaping':
            legend_label = 'Shaping + FT'
        else:
            legend_label = 'Only Shaping'

        legend_lines.append(
            matplotlib.lines.Line2D(
                [], [], color=color, linestyle=ls,
                label=f'{legend_label} Off'))
        legend_lines.append(
            matplotlib.lines.Line2D(
                [], [], color=color2, linestyle=ls,
                label=f'{legend_label} On'))

    legend_lines.append(
        matplotlib.lines.Line2D(
            [], [], color='k', marker='x', linestyle='None',
            ms=3, mew=0.8, label='Fixed Point'))

    list(map(jP.configure_spines, axs))
    list(map(lambda x: x.set_xlim(-9, 8), axs))

    axs[0].set_ylabel('PC 2')
    axs[1].set_ylabel('PC 3')
    axs[2].set_ylabel('Response')

    for ax in axs:
        ax.spines['left'].set_position(('outward', -5))

    for ax in axs[:2]:
        ax.set_yticks([-2.5, 2.5])
        ax.spines['left'].set_bounds([-2.5, 2.5])
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)

    axs[0].set_ylim(-7, 7)
    #axs[0].set_xlim(-9, 8)

    axs[1].set_ylim(-5, 6)

    axs[2].spines['bottom'].set_bounds([-5, 5])
    axs[2].spines['left'].set_bounds([0, 1])
    axs[2].set_xticks([-5, 0, 5])
    axs[2].set_ylim(-0.25, 1.25)
    axs[2].set_xlabel('PC 1')
    axs[2].set_yticks([0, 1])

    legend_lines.insert(2, legend_lines[-1])
    legend_lines.pop(-1)

    axs[0].legend(
        'bottom left',
        handles=legend_lines, bbox_to_anchor=(-0.2, 1.8), loc=2, borderaxespad=0.,
        ncols=2, fontsize='x-small')

    plt.show()

    cue_times = {
        'SL': [30, 50, 80, 130],
        'LS': [30, 80, 110, 130],
        'SS': [30, 50, 80, 100]}
    margins = jP.default_margins()
    margins['left'] = 175
    margins['right'] = 300
    arrowprops = dict(arrowstyle='-|>, head_width=0.3, head_length=1.15', shrinkA=0,
                      shrinkB=0, lw=0.3)
    for model_type, color, color2, ls in zip(('no_shaping', 'only_shaping', 'shaping'),
                                             ('#dd0000', '#880000', '#dd0000'),
                                             ('#111111', '#888888', '#111111'),
                                             it.repeat('-')):

        PdfPlotter(path / f'{model_type}.dynamic_motifs_7.pdf',
                   fixed_margins=margins)
        fig = plt.figure(figsize=(7, 2.5), dpi=dpi)
        gs = iter(gridspec.GridSpec(1, 3))
        for trial_type in inputs.index.unique('type'):
            gss = next(gs).subgridspec(3, 1)
            axs = list(map(plt.subplot, gss))
            null_pcs = pcas[model_type].transform(nullcline_nocue['h'][model_type])
            null_out = nullcline_nocue['o'][model_type]
            axs[0].plot(null_pcs[:, 0], null_pcs[:, 1], color=color, ls=ls)
            axs[1].plot(null_pcs[:, 0], null_pcs[:, 2], color=color, ls=ls)
            axs[2].plot(null_pcs[:, 0], null_out, color=color, ls=ls)

            zorder = axs[0].get_lines()[-1].get_zorder()
            arrowprops['facecolor'] = color
            if model_type == 'no_shaping':
                arrow_locs = [20]
            elif model_type == 'shaping':
                arrow_locs = [270, 350]
            else:
                arrow_locs = [75, 180]

            for arrow_loc in arrow_locs:
                axs[0].annotate('', null_pcs[arrow_loc, [0, 1]], null_pcs[arrow_loc-3, [0, 1]],
                                arrowprops=arrowprops, zorder=zorder)
                axs[1].annotate('', null_pcs[arrow_loc, [0, 2]], null_pcs[arrow_loc-3, [0, 2]],
                                arrowprops=arrowprops, zorder=zorder)
                axs[2].annotate('', (null_pcs[arrow_loc, 0], null_out[arrow_loc]), 
                                (null_pcs[arrow_loc-1, 0], null_out[arrow_loc-3]),
                                arrowprops=arrowprops, zorder=zorder)

            null_pcs = pcas[model_type].transform(nullcline_cue['h'][model_type])
            null_out = nullcline_cue['o'][model_type]
            axs[0].plot(null_pcs[:, 0], null_pcs[:, 1], color=color2, ls=ls)
            axs[1].plot(null_pcs[:, 0], null_pcs[:, 2], color=color2, ls=ls)
            axs[2].plot(null_pcs[:, 0], nullcline_cue['o'][model_type], color=color2, ls=ls)

            arrowprops['facecolor'] = color2
            zorder = axs[0].get_lines()[-1].get_zorder()
            if model_type == 'no_shaping':
                arrow_locs = [70]
            elif model_type == 'shaping':
                arrow_locs = [30, 90]
            else:
                arrow_locs = [40, 110]

            for arrow_loc in arrow_locs:
                axs[0].annotate('', null_pcs[arrow_loc, [0, 1]], null_pcs[arrow_loc-3, [0, 1]],
                                arrowprops=arrowprops, zorder=zorder)
                axs[1].annotate('', null_pcs[arrow_loc, [0, 2]], null_pcs[arrow_loc-3, [0, 2]],
                                arrowprops=arrowprops, zorder=zorder)
                axs[2].annotate('', (null_pcs[arrow_loc, 0], null_out[arrow_loc]),
                                (null_pcs[arrow_loc-3, 0], null_out[arrow_loc-3]),
                                arrowprops=arrowprops, zorder=zorder)

            marker_style = {
                'ls': '', 'marker': 'x', 'zorder': 1e5,
                'ms': 3, 'mew': 0.8}
            fps_pca = pcas[model_type].transform(fps[model_type]['cue_off'])
            fp_out = leaky_rnn.affine(model_params[model_type], fps[model_type]['cue_off']).T[0]
            axs[0].plot(fps_pca[:, 0], fps_pca[:, 1], color=color, **marker_style)
            axs[1].plot(fps_pca[:, 0], fps_pca[:, 2], color=color, **marker_style)
            axs[2].plot(fps_pca[:, 0], fp_out, color=color, **marker_style)

            fps_pca = pcas[model_type].transform(fps[model_type]['cue_on'])
            fp_out = leaky_rnn.affine(model_params[model_type], fps[model_type]['cue_on']).T[0]
            axs[0].plot(fps_pca[:, 0], fps_pca[:, 1], color=color2, **marker_style)
            axs[1].plot(fps_pca[:, 0], fps_pca[:, 2], color=color2, **marker_style)
            axs[2].plot(fps_pca[:, 0], fp_out, color=color2, **marker_style)

            pca = pd.DataFrame(pcas[model_type].transform(rnn_res['h'][model_type]), index=inputs.index)
            def plot_trial(X, ax, cue_times):
                c = TRIAL_COLORS[X.index.unique('type')[0]]
                jP.plot_seg_colors(ax, X[X.columns[0]], X[X.columns[1]], c=c, alpha=0.75)
                ax.plot(X[X.columns[0]][cue_times[::2]], X[X.columns[1]][cue_times[::2]], c=c, ls='', marker='o')
                ax.plot(X[X.columns[0]][cue_times[1::2]], X[X.columns[1]][cue_times[1::2]], c=c, ls='', marker='s')

            plot_trial(pca.xs(trial_type, level='type', drop_level=False), ax=axs[0], cue_times=cue_times[trial_type])
            plot_trial(pca[[0, 2]].xs(trial_type, level='type', drop_level=False), ax=axs[1], cue_times=cue_times[trial_type])
            pca['output'] = rnn_res['o'][model_type]
            plot_trial(pca[[0, 'output']].xs(trial_type, level='type', drop_level=False), ax=axs[2], cue_times=cue_times[trial_type])

            list(map(lambda x: x.set_xlim(-9, 8), axs))

            axs[0].set_title(trial_type, color=TRIAL_COLORS[trial_type])
            axs[0].set_ylabel('PC 2')
            axs[1].set_ylabel('PC 3')
            axs[2].set_ylabel('Response')

            for ax in axs:
                ax.spines['left'].set_position(('outward', -5))
                jP.configure_spines(ax)
                jP.set_ylabel_position(ax, nlines=5)

            for ax in axs[:2]:
                ax.set_yticks([-2.5, 2.5])
                ax.spines['left'].set_bounds([-2.5, 2.5])
                ax.set_xticks([])
                ax.spines['bottom'].set_visible(False)

            axs[0].set_ylim(-7, 7)
            #axs[0].set_xlim(-9, 8)
            axs[1].set_ylim(-5, 6)

            axs[2].spines['bottom'].set_bounds([-5, 5])
            axs[2].spines['left'].set_bounds([0, 1])
            axs[2].set_xticks([-5, 0, 5])
            axs[2].set_ylim(-0.25, 1.25)
            axs[2].set_xlabel('PC 1')
            axs[2].set_yticks([0, 1])
            axs[2].set_yticklabels([0.0, 1.0])


        title_color = ('tab:blue' if model_type == 'no_shaping'
                       else 'tab:orange')
        title = model_type.replace('_', ' ').title()
        if title == 'Shaping':
            title = 'Shaping + Full'
        axs[1].text(0.01, 0.5, title,
                    transform=axs[1].figure.transFigure, fontsize=8,
                    color=title_color, rotation=90, ha='left', va='center')

        fp_label = matplotlib.lines.Line2D(
            [], [], color='k', marker='x', linestyle='None',
            ms=3, mew=0.8, label='Fixed Point')
        start_label = matplotlib.lines.Line2D(
            [], [], color='k', marker='o', linestyle='None',
            ms=2, label='Cue On')
        end_label = matplotlib.lines.Line2D(
            [], [], color='k', marker='s', linestyle='None',
            ms=2, label='Cue Off')
        cue_on_label = matplotlib.lines.Line2D(
            [], [], color=color2, linestyle='-',
            label='Cue On Flow')
        cue_off_label = matplotlib.lines.Line2D(
            [], [], color=color, linestyle='-',
            label='Cue Off Flow')
        axs[0].legend(
            handles=[fp_label, start_label, end_label, cue_on_label, cue_off_label],
            bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
        plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])
