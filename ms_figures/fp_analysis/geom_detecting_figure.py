# [Figure S6a,b,c]
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
import scipy
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from pathlib import Path
import itertools as it
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

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

TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'LL': '#2b958c', 'S': '#f89521'}

SHAPING_COLORS = {
    'no_shaping': 'tab:blue',
    'shaping': 'tab:orange',
    'sl_only': 'tab:green',
    'ls_only': 'tab:red'}


class HandlerArrow(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        p = mpatches.FancyArrowPatch((xdescent, ydescent + height / 2),
                (xdescent + width, ydescent + height / 2),
                arrowstyle='-|>', mutation_scale=5, transform=trans,
                color=orig_handle.get_facecolor())
        return [p]



def main(argv):
    jP.set_rcParams(plt)

    no_model_ids   = [9,  18, 19, 24, 28, 29, 31, 32, 93, 109, 116, 119]
    with_model_ids = [20, 22, 25, 21, 23, 26, 27, 30, 76, 79,  82,  84]
    example_models = [18, 22]
    
    fp_tol = 1e-6

    idx = pd.MultiIndex.from_tuples(
        list(zip(no_model_ids, it.repeat('no_shaping'))) +
        list(zip(with_model_ids, it.repeat('shaping'))),
        names=('model_id', 'model_type'))
    model_ids = pd.Series(no_model_ids + with_model_ids,  index=idx)
    path = Path('/analysis/ms_figures/geom_detection_figures')
    path.mkdir(parents=True, exist_ok=True)

    info_file = path / 'info.txt'
    with open(info_file, 'w') as f:
        pass

    model_infos = model_ids.apply(mdb.get_model).apply(pd.Series)
    model_attrs = model_ids.apply(mdb.get_model_attributes)
    model_infos = pd.concat(
        (model_infos,
         model_attrs[['gamma', 'epoc', 'input_noise', 'noise_level']]), axis=1)

    models = model_infos['path'].apply(tf.keras.models.load_model)
    weights = pd.concat(
        models.apply(lambda x: x.layers[0].get_weights()[1]
        ).apply(pd.DataFrame).values, keys=models.index)
    input_weights = pd.concat(
        models.apply(lambda x: x.layers[0].get_weights()[0]
        ).apply(pd.DataFrame).values, keys=models.index)
    output_weights = pd.concat(
        models.apply(lambda x: x.layers[1].get_weights()[0]
        ).apply(pd.DataFrame).values, keys=models.index)

    def create_params(model, gamma):
        w_in, w_r, b = model.layers[0].get_weights()
        w_out, b_out = model.layers[1].get_weights()
        h0 = onp.squeeze(model.layers[0](onp.zeros((1, 1, 2))).numpy())
        return leaky_rnn.rnn_params(h0, b, w_in, w_r, w_out, b_out, gamma, 0)
    model_params = pd.concat((models, model_infos['gamma']), axis=1)\
        .apply(lambda x: create_params(*x), axis=1)

    def load_fps(path, fp_tol=fp_tol):
        with open(Path(path).parent / 'fps_info.pkl', 'rb') as f:
            fps = pickle.load(f)[fp_tol]
        return {'cue_on': fps['cue_on']['fps'], 'cue_off': fps['cue_off']['fps']}
    fps = model_infos['path'].apply(load_fps).apply(pd.Series).stack()
    fps.name = 'fps'

    trials_gen = genFactory.create(
        'just_short_match', input_noise=0, batch_size=1, n_blocks=1)
    inputs = trials_gen.generate_trials(10)[['light', 'odor']]
    trial_res = model_params.apply(
        lambda x, inputs=inputs.values[None, :]:
            leaky_rnn.batched_rnn_run(x, inputs))\
                .apply(pd.Series, index=('h', 'o'))
    pcas = trial_res['h'].apply(lambda x: PCA().fit(x[0]))
    pcas.name = 'pcas'

    fps_reduced = pd.merge(fps, pcas, on='model_id').set_index(fps.index).apply(
        lambda X: X['pcas'].transform(X['fps']), axis=1)

    h_starts = fps.xs('cue_off', level=-1).apply(lambda x: x[0])
    pd.concat((model_params, h_starts), axis=1, keys=('params', 'h'))\
        .apply(lambda x: x['params'].update({'h0': x['h']}), axis=1)
    cue_off = onp.zeros((1, 400, 2))
    cue_on = onp.zeros((1, 400, 2))
    cue_on[:, :, -1] = 1
    rnn_res = model_params.apply(
        lambda x, inputs=np.concatenate((cue_on, cue_off), axis=1):
            leaky_rnn.batched_rnn_run(x, inputs))\
                .apply(pd.Series, index=('h', 'o'))\
                .map(np.squeeze)

    h_reduced = pd.merge(trial_res[['h']], pcas, on='model_id')\
                  .set_index(trial_res.index).apply(lambda x: x['pcas'].transform(x['h'][0]), axis=1)
    h_reduced = pd.concat(
        h_reduced.apply(
            lambda x, inputs=inputs: pd.DataFrame(x, index=inputs.index)
            ).values, keys=h_reduced.index)
    

    print('calculating scores')
    score = []
    print(model_ids)
    for model_id in model_ids:
        print(model_id)
        m_PCA = pcas.loc[model_id]
        pca_vect = onp.zeros(128)
        pca_vect[0] = 3
        locations = np.meshgrid(np.arange(-8, 8, 0.5), np.arange(-8, 8, 0.5))
        locations = pd.DataFrame(list(map(it.chain.from_iterable, locations))).T
        locations = locations.set_index(pd.MultiIndex.from_frame(locations.astype(np.float32)))
        
        def calc_vector(X, h_reduced=h_reduced, m_PCA=m_PCA, model_id=model_id, model_params=model_params):
            pca_vect = onp.zeros(128)
            pca_vect[0] = X[0]
            pca_vect[1] = X[1]

            pt = h_reduced.loc[model_id].mean(0)
            h_starts = np.matmul(np.array([pt+pca_vect]), m_PCA.item().components_)[0] + m_PCA.item().mean_
            model_params[model_id].item()['h0'] = h_starts
            cue_off = onp.zeros((1, 1, 2))

            rnn_res = leaky_rnn.batched_rnn_run(model_params[model_id].item(), cue_off)
            res_reduced = m_PCA.item().transform(rnn_res[0][0])
            return res_reduced


        res = locations.apply(calc_vector, axis=1)
        directions = res.apply(lambda x: x[-1][:2]).apply(pd.Series) - locations
        directions.index=pd.MultiIndex.from_frame(locations.astype(np.float32))

        if model_id in example_models:
            model_type = model_ids.where(model_ids == model_id).dropna()\
                                  .index.get_level_values('model_type')[0]
            PdfPlotter(path / f'flow2d_off_{model_id}.pdf', fixed_margins=jP.default_margins())
            plt.figure(figsize=(2, 2), dpi=300)
            ax = plt.gca()
            def plot_trace(X, ax):
                trial_type = X.index.unique('type')[0]
                ax.plot(*X.values.T[:2], c=TRIAL_COLORS[trial_type], alpha=0.15, lw=0.25)

            h_reduced.loc[model_id].groupby('trial').apply(plot_trace, ax=ax)
            plt.quiver(
                locations[0].unstack(1), locations[1].unstack(1),
                directions[0].unstack(1).to_numpy(dtype=np.float32),
                directions[1].unstack(1).to_numpy(dtype=np.float32), color='k')
            ax.set_yticks([-5, 5])
            ax.set_xticks([-5, 5])
            ax.set_ylabel('PC 2')
            ax.set_xlabel('PC 1')

            ax.set_title(f'Cue Off - {model_type.replace("_", " ").capitalize()}',
                         color=SHAPING_COLORS[model_type])

            legend_arrow_off = mpatches.FancyArrowPatch(
                (0, 0), (1, 1), arrowstyle='-|>', mutation_scale=10, color='k')
            ax.legend([legend_arrow_off],
                    ['0.1s Cue Off'],
                    handler_map={mpatches.FancyArrowPatch: HandlerArrow()},
                    bbox_to_anchor=[0.95, 0.05], loc='lower right')
            plt.show()

        score.append(directions.sum())


    fps_reduced = pd.merge(fps, pcas, on='model_id').set_index(fps.index)\
                    .apply(lambda x: x['pcas'].transform(x['fps']), axis=1)
    pc3_loc = fps_reduced.apply(lambda x: [a[2] for a in x]).apply(onp.mean)
    pc3_diff = pc3_loc.unstack(2).diff(axis=1).dropna(axis=1).abs()

    scores = pd.concat(score, keys=model_ids.index)
    scores_total = (scores.unstack(-1)**2).sum(1)**0.5
    scores_total = scores_total / 0.1 / len(directions)
    data = pd.concat((scores_total, pc3_diff), axis=1)

    PdfPlotter(path / 'scatter.pdf', fixed_margins=jP.default_margins())
    plt.figure(figsize=(2, 2), dpi=300)
    ax = plt.gca()
    ax.plot(
        *data.xs('no_shaping', level='model_type').to_numpy(dtype=np.float32).T,
        color='tab:blue', ls='', marker='o', mec='k', ms=3, label='NS')
    ax.plot(
        *data.xs('shaping', level='model_type').to_numpy(dtype=np.float32).T,
        color='tab:orange', ls='', marker='o', mec='k', ms=3, label='S/FT')
    ax.set_ylim(-1, 8)
    ax.set_xlim(0, 1.75)
    jP.configure_spines(ax)

    ax.set_xlabel('Directional Bias, Cue Off (a.u.)')
    ax.set_ylabel(r'|$\Delta$ PC3|')
    ax.set_xticks([0.5, 1, 1.5])
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.legend(loc='lower left', ncols=2)

    directionality_pval = scipy.stats.ttest_ind(*scores_total.groupby('model_type').apply(list).values)
    dir_pval_symbol = jP.significance_symbols([directionality_pval.pvalue]).item()
    diff_pc3_pval = scipy.stats.ttest_ind(*pc3_diff['cue_on'].groupby('model_type').apply(list).values)
    pc3_pval_symbol = jP.significance_symbols([diff_pc3_pval.pvalue]).item()

    mean_dir = scores_total.groupby('model_type').mean().to_numpy(dtype=np.float32)
    arrowprops = dict(
        arrowstyle='-',
        shrinkA=0,
        shrinkB=0,
        connectionstyle="angle,angleA=0,angleB=90")
    ax.annotate(
        '', (mean_dir[1], 1), xytext=(mean_dir[1] + (mean_dir[0]-mean_dir[1])/2, 1.05),
        xycoords=ax.axes.get_xaxis_transform(),
        textcoords=ax.axes.get_xaxis_transform(), arrowprops=arrowprops)

    arrowprops = dict(
        arrowstyle='-',
        shrinkA=0,
        shrinkB=0,
        connectionstyle="angle,angleA=90,angleB=0")
    ax.annotate(
        '', (mean_dir[1] + (mean_dir[0]-mean_dir[1])/2, 1.05), xytext=(mean_dir[0], 1),
        xycoords=ax.axes.get_xaxis_transform(),
        textcoords=ax.axes.get_xaxis_transform(), arrowprops=arrowprops)
    
    ax.text(mean_dir[1] + (mean_dir[0]-mean_dir[1])/2, 1.1, dir_pval_symbol,
            clip_on=False, transform=ax.get_xaxis_transform(), ha='center',
            va='top', fontsize=5)

    arrowprops = dict(
        arrowstyle='-',
        shrinkA=0,
        shrinkB=0,
        connectionstyle="angle,angleA=0,angleB=90")
    ax.annotate(
        '', (mean_dir[1], 1), xytext=(mean_dir[1] + (mean_dir[0]-mean_dir[1])/2, 1.05),
        xycoords=ax.axes.get_xaxis_transform(),
        textcoords=ax.axes.get_xaxis_transform(), arrowprops=arrowprops)

    arrowprops = dict(
        arrowstyle='-',
        shrinkA=0,
        shrinkB=0,
        connectionstyle="angle,angleA=90,angleB=0")
    ax.annotate(
        '', (mean_dir[1] + (mean_dir[0]-mean_dir[1])/2, 1.05), xytext=(mean_dir[0], 1),
        xycoords=ax.axes.get_xaxis_transform(),
        textcoords=ax.axes.get_xaxis_transform(), arrowprops=arrowprops)
    
    ax.text(mean_dir[1] + (mean_dir[0]-mean_dir[1])/2, 1.1, pc3_pval_symbol,
            clip_on=False, transform=ax.get_xaxis_transform(), ha='center',
            va='top', fontsize=5)

    mean_dPC3 = pc3_diff.groupby('model_type').mean().to_numpy(dtype=np.float32).T[0]
    arrowprops = dict(
        arrowstyle='-',
        shrinkA=0,
        shrinkB=0,
        connectionstyle="angle,angleA=90,angleB=0")
    ax.annotate(
        '', (1, mean_dPC3[0]), xytext=(1.05, mean_dPC3[0] + (mean_dPC3[1]-mean_dPC3[0])/2),
        xycoords=ax.axes.get_yaxis_transform(),
        textcoords=ax.axes.get_yaxis_transform(), arrowprops=arrowprops)

    arrowprops = dict(
        arrowstyle='-',
        shrinkA=0,
        shrinkB=0,
        connectionstyle="angle,angleA=0,angleB=90")
    ax.annotate(
        '', (1.05, mean_dPC3[0] + (mean_dPC3[1]-mean_dPC3[0])/2), xytext=(1, mean_dPC3[1]),
        xycoords=ax.axes.get_yaxis_transform(),
        textcoords=ax.axes.get_yaxis_transform(), arrowprops=arrowprops)
    
    ax.text(1.1, mean_dPC3[0] + (mean_dPC3[1]-mean_dPC3[0])/2, '***',
            clip_on=False, transform=ax.get_yaxis_transform(), ha='right',
            va='center', fontsize=5, rotation=-90)

    arrowprops = dict(
        arrowstyle='-',
        shrinkA=0,
        shrinkB=0,
        connectionstyle="angle,angleA=0,angleB=90")
    ax.annotate(
        '', (mean_dir[1], 1), xytext=(mean_dir[1] + (mean_dir[0]-mean_dir[1])/2, 1.05),
        xycoords=ax.axes.get_xaxis_transform(),
        textcoords=ax.axes.get_xaxis_transform(), arrowprops=arrowprops)

    arrowprops = dict(
        arrowstyle='-',
        shrinkA=0,
        shrinkB=0,
        connectionstyle="angle,angleA=90,angleB=0")
    ax.annotate(
        '', (mean_dir[1] + (mean_dir[0]-mean_dir[1])/2, 1.05), xytext=(mean_dir[0], 1),
        xycoords=ax.axes.get_xaxis_transform(),
        textcoords=ax.axes.get_xaxis_transform(), arrowprops=arrowprops)
    
    ax.text(mean_dir[1] + (mean_dir[0]-mean_dir[1])/2, 1.1, '***',
            clip_on=False, transform=ax.get_xaxis_transform(), ha='center',
            va='top', fontsize=5)

    plt.show()


    scores = pd.concat(score, keys=model_ids.index)
    scores_total = (scores.unstack(-1)**2).sum(1)**0.5
    scores_total = scores_total / 0.1 / len(directions)
    data = pd.concat((scores_total, pc3_diff), axis=1)

    PdfPlotter(path / 'scatter.pdf', fixed_margins=jP.default_margins())
    plt.figure(figsize=(2, 2), dpi=300)
    ax = plt.gca()
    ax.plot(
        *data.xs('no_shaping', level='model_type').to_numpy(dtype=np.float32).T,
        color='tab:blue', ls='', marker='o', mec='k', ms=5, label='NS')
    ax.plot(
        *data.xs('shaping', level='model_type').to_numpy(dtype=np.float32).T,
        color='tab:orange', ls='', marker='o', mec='k', ms=5, label='S/FT')
    ax.set_ylim(0, 8)
    ax.set_xlim(0, 6)
    jP.configure_spines(ax)
    ax.set_xlabel('Cue Off - Directionality (a.u.)')
    ax.set_ylabel(r'|$\Delta$ PC3|')
    ax.set_xticks([2, 4, 6])
    ax.set_yticks([2, 4, 6, 8])
    ax.legend(loc='upper right')
    plt.show()

    
if __name__ == '__main__':
    main(sys.argv[1:])
