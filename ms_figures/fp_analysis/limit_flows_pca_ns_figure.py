# [Figure 4e]
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.25'

import sys
import jax
import jax.numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
import itertools as it
import numpy as onp
from pathlib import Path
import math
import pickle
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from sklearn.decomposition import PCA

sys.path.append('../../')
import leaky_rnn
from classes.datagen import tDNMSGenerator, genFactory
from classes.models import LeakyRNN
from classes.models import LeakyRNNCell

from analysis_tools.mpl_helpers import PdfPlotter
from analysis_tools.progressbar import ProgressBar
from analysis_tools import jPlots as jP
import models_database as mdb



TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'S': '#f89521'}

class HandlerArrow(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        p = mpatches.FancyArrowPatch((xdescent, ydescent + height / 2),
                (xdescent + width, ydescent + height / 2),
                arrowstyle='-|>', mutation_scale=5, transform=trans,
                color=orig_handle.get_facecolor())
        return [p]

def unit_vector(v):
    return v / onp.linalg.norm(v)


def angle_between(v1, v2):
    return onp.arccos(onp.clip(onp.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0))


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


def main(argv):
    jP.set_rcParams(plt)
    dpi = 300
    path = Path('/analysis/ms_figures/fps_schematic')
    path.mkdir(parents=True, exist_ok=True)

    step_time = 1
    step_size = int(step_time*10)
    fp_tol = 1e-6
    model_ids = pd.Series([9, 132],
                          index=('no_shaping', 'shaping'))

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
    inputs = trials_gen.generate_trials(2)[['light', 'odor']]
    rnn_res = model_params.apply(
        lambda x, inputs=np.expand_dims(inputs.values, 0):
            leaky_rnn.batched_rnn_run(x, inputs))\
                .apply(pd.Series, index=('h', 'o'))

    trials_gen = genFactory.create(
        'just_short_match', input_noise=0, batch_size=1, n_blocks=1)
    inputs = trials_gen.generate_trials(2)[['light', 'odor']]
    inputs = inputs.drop(0, level='trial')
    inputs = inputs.groupby(['type', 'idx']).head(1)

    def run_rnn(inputs, h0, model_params):
        model_params['h0'] = h0
        h_t, o_t = leaky_rnn.batched_rnn_run(
            model_params, np.expand_dims(inputs.values, 0))
        h_t = pd.DataFrame(h_t[0], index=inputs.index)
        h_t = h_t.set_index(pd.Index(o_t[0, :, 0], name='out'), append=True)
        return h_t
 
    h0 = fps['no_shaping']['cue_off'][0]
    trial_res = inputs.groupby(['type', 'idx']).head(1).groupby('type')\
                      .apply(run_rnn, h0=h0, model_params=model_params['no_shaping'])\
                      .droplevel(0)

    h_starts = trial_res.iloc[-1].values
    model_params['no_shaping']['h0'] = h_starts
    rnn_res = model_params.apply(
        lambda x:
            leaky_rnn.batched_rnn_run(x,  np.tile(np.array([0, 1]), (1, 800, 1))))\
                .apply(pd.Series, index=('h', 'o'))\
                .map(np.squeeze)
    h_t_on = pd.DataFrame(rnn_res['h']['no_shaping'])

    model_params['no_shaping']['h0'] = h_t_on.iloc[-1].values
    rnn_res = model_params.apply(
        lambda x:
            leaky_rnn.batched_rnn_run(x,  np.tile(np.array([0, 0]), (1, 800, 1))))\
                .apply(pd.Series, index=('h', 'o'))\
                .map(np.squeeze)
    h_t = pd.DataFrame(rnn_res['h']['no_shaping'])
    pca = PCA(n_components=128).fit(trial_res)
    #path_on = pca.transform(h_t_on)
    #path_off = pca.transform(h_t)

    fps_on_pc = pca.transform(fps['no_shaping']['cue_on'])
    fps_off_pc = pca.transform(fps['no_shaping']['cue_off'])
    ref_pts = (fps['no_shaping']['cue_on'][0], fps['no_shaping']['cue_off'][0])
    ref_pts_pc = list(map(lambda x, pca=pca: pca.transform([x])[0], ref_pts))

    def ref_dist(pt, refs):
        max_dist = math.dist(*refs)
        comp = refs[1]-refs[0]
        theta = onp.arctan2(comp[1], comp[0])
        theta2 = onp.arctan2(pt[1], pt[0])
        dtheta = theta2 - theta

        x = 1 - math.dist(pt, refs[0])/max_dist
        y = np.sin(dtheta)
        return x, y.item()

    def ref_array(pts, refs, ref_dist=ref_dist):
        return np.array(list(map(ref_dist, pts, it.repeat(refs))))

    dh_t = list(map(math.dist, h_t.values, h_t.shift(-1).values))
    distance_traveled = onp.append([0], onp.cumsum(dh_t)[:-1])
    distance_to_start = list(map(lambda x, h_t=h_t.values: math.dist(h_t[0], h_t[x]), np.arange(800)))

    #h0 = h_starts
    #trial_res = inputs.groupby(['type', 'idx']).head(1).groupby('type')\
    #                  .apply(run_rnn, h0=h0, model_params=model_params['shaping'])\
    #                  .droplevel(0)

    h_pc = pd.DataFrame(pca.transform(h_t.values))
    trial_info_pc = pd.DataFrame(pca.transform(trial_res), index=trial_res.index)
    polar_trial_info = pd.DataFrame(ref_array(trial_info_pc.values, ref_pts_pc), index=trial_info_pc.index)
    output_space = polar_trial_info[polar_trial_info.index.get_level_values('out') > 0.45]

    polar_h = pd.DataFrame(ref_array(h_pc.values, ref_pts_pc), index=h_pc.index)
    new_idx = pd.Series(np.arange(0, 1 + 1/15, 1/15))
    polar_h = polar_h.set_index(0, drop=False)
    h_pc.index = polar_h.index
    polar_h.index.name = 'dist'
    polar_h = polar_h.groupby('dist').mean()
    h_pc = h_pc.groupby('dist').mean()
    adjusted_polar = polar_h.reindex(polar_h.index.union(new_idx)).interpolate('values').bfill().reindex(new_idx)
    adjusted_h_pc = h_pc.reindex(h_pc.index.union(new_idx)).interpolate('values').bfill().reindex(new_idx) 

    polar_h2 = pd.DataFrame(ref_array(adjusted_h_pc.values, ref_pts_pc), index=adjusted_h_pc.index)

    def run_rnn(h0, model_params, step_size, steps=1, cue_on=True):
        model_params['no_shaping']['h0'] = h0.values
        x_star=(0, int(cue_on))
        h_t, _ = leaky_rnn.batched_rnn_run(
            model_params['no_shaping'], np.tile(np.array(x_star), (1, steps * step_size, 1)))
        if steps == 1:
            return h_t[0, -1]
        return h_t[0, np.arange(step_size, step_size * (steps+ 1), step_size)]

    adjusted_h = pca.inverse_transform(adjusted_h_pc)
    dh = adjusted_h.apply(run_rnn, model_params=model_params, step_size=step_size, cue_on=False, axis=1)
    dh_pc = pd.DataFrame(pca.transform(dh.apply(pd.Series)), index=adjusted_h.index)
    polar_dh_pc = pd.DataFrame(ref_array(dh_pc.values, ref_pts_pc), index=dh_pc.index)

    dh_on = adjusted_h.apply(run_rnn, model_params=model_params, step_size=step_size,
                             steps=5, cue_on=True, axis=1)
    dh_on = pd.concat(list(map(pd.DataFrame, dh_on.values)), keys=dh_on.index)
    dh_on.index.names = ('dist', 'step')
    dh_pc_on = pd.DataFrame(pca.transform(dh_on), index=dh_on.index)
    polar_dh_pc_on = pd.DataFrame(ref_array(dh_pc_on.values, ref_pts_pc), index=dh_pc_on.index)

    title_dict = {'SS': 'Short-Short', 'SL': 'Short-Long', 'LS': 'Long-Short'}

    PdfPlotter(path / f'flow_field_pca_ns_figure_{step_time}.pdf', fixed_margins=jP.default_margins())
    plt.figure(figsize=(3.25, 1.6), dpi=300)
    gs = gridspec.GridSpec(3, 1, hspace=0.05)
    ax = plt.gca()
    arrowprops = {'arrowstyle': '-|>', 'shrinkA': 0.1, 'shrinkB': 0.1, 'color': 'r', 'lw': 1}
    for t in polar_dh_pc_on.index.unique('dist'):
        arrowprops.update({'color': 'r'})
        ax.annotate('', (polar_dh_pc[0][t], polar_dh_pc[1][t]),
                    xytext=(adjusted_polar[0][t], adjusted_polar[1][t]),
                    textcoords=ax.transData, arrowprops=arrowprops)

        step0 = (adjusted_polar[0][t], adjusted_polar[1][t])
        for step in polar_dh_pc_on.index.unique('step'):
            arrowprops.update({'color': 'k'})
            next_step = (polar_dh_pc_on[0][t][step], polar_dh_pc_on[1][t][step])
            ax.annotate('', next_step, xytext=step0,
                        textcoords=ax.transData, arrowprops=arrowprops)
            step0 = next_step

    zorder = ('SL', 'LS', 'SS')
    for i, trial_type in enumerate(('SS', 'SL', 'LS')):
        polar_trial = polar_trial_info.xs(trial_type, level='type')
        itr2 = polar_trial.iloc[::step_size].iterrows()
        next(itr2)
        arrowprops.update({'color': TRIAL_COLORS[trial_type], 'lw': 1.5})
        for p1, p2 in zip(polar_trial.iloc[::step_size].iterrows(), itr2):
            ax.annotate('', p2[1].values + np.array([0, i/25-1/25]),
                        xytext=p1[1].values + np.array([0, i/25-1/25]),
                        textcoords=ax.transData, arrowprops=arrowprops,
                        zorder=100+zorder.index(trial_type))
            
    ax.axvspan(output_space[0].min(), 1, color='red', alpha=0.15, linewidth=0.25)
    ax.set_xlim(-0.1, 1)
    ax.set_ylim(-np.pi/2, np.pi/2)
    jP.configure_spines(ax)
    ax.spines['bottom'].set_bounds(0, 1)
    ax.spines['left'].set_bounds(-np.pi/4, np.pi/4)
    ax.set_yticks([-np.pi/4, np.pi/4])
    ax.set_yticklabels([r'$\pi$/4', r'$\pi$/4'])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(r'Vector Difference ($\angle$)')

    if i != 2:
        ax.set_xticklabels(['', '', '', '', ''])
    else:
        ax.set_xlabel('Norm. Distance To Cue On FP')

    #ax.text(0.025, 0.8, title_dict[trial_type], transform=ax.transAxes,
    #        color=TRIAL_COLORS[trial_type], fontweight='bold',
    #        fontsize=8)

    legend_arrow_off = mpatches.FancyArrowPatch((0, 0), (1, 1), arrowstyle='-|>',
                                       mutation_scale=10, color='r')
    legend_arrow_on = mpatches.FancyArrowPatch((0, 0), (1, 1), arrowstyle='-|>',
                                       mutation_scale=10, color='k')
    red_patch = mpatches.Patch(color='red', alpha=0.15, label='Red Category')
    ax.legend([legend_arrow_off, legend_arrow_on, red_patch],
            [f'{step_time}s Cue Off', f'{step_time}s Cue On', 'Response Area'],
            handler_map={mpatches.FancyArrowPatch: HandlerArrow()},
            bbox_to_anchor=[0, 0.9], loc='lower left',
            ncols=3, handletextpad=0.1)
    plt.suptitle('NS RNN - Leaky Integrator Method')
    plt.show()

    PdfPlotter(path / f'ns_figure_{step_time}_3.5.pdf', fixed_margins=jP.default_margins())
    plt.figure(figsize=(3.5, 1.6), dpi=300)
    gs = gridspec.GridSpec(3, 1, hspace=0.05)
    ax = plt.gca()
    arrowprops = {'arrowstyle': '-|>', 'shrinkA': 0.1, 'shrinkB': 0.1, 'color': 'r', 'lw': 1}
    for t in polar_dh_pc_on.index.unique('dist'):
        arrowprops.update({'color': 'r'})
        ax.annotate('', (polar_dh_pc[0][t], polar_dh_pc[1][t]),
                    xytext=(adjusted_polar[0][t], adjusted_polar[1][t]),
                    textcoords=ax.transData, arrowprops=arrowprops)

        step0 = (adjusted_polar[0][t], adjusted_polar[1][t])
        for step in polar_dh_pc_on.index.unique('step'):
            arrowprops.update({'color': 'k'})
            next_step = (polar_dh_pc_on[0][t][step], polar_dh_pc_on[1][t][step])
            ax.annotate('', next_step, xytext=step0,
                        textcoords=ax.transData, arrowprops=arrowprops)
            step0 = next_step

    zorder = ('SL', 'LS', 'SS')
    for i, trial_type in enumerate(('SS', 'SL', 'LS')):
        polar_trial = polar_trial_info.xs(trial_type, level='type')
        itr2 = polar_trial.iloc[::step_size].iterrows()
        next(itr2)
        arrowprops.update({'color': TRIAL_COLORS[trial_type], 'lw': 1.5})
        for p1, p2 in zip(polar_trial.iloc[::step_size].iterrows(), itr2):
            ax.annotate('', p2[1].values + np.array([0, i/25-1/25]),
                        xytext=p1[1].values + np.array([0, i/25-1/25]),
                        textcoords=ax.transData, arrowprops=arrowprops,
                        zorder=100+zorder.index(trial_type))
            
    ax.axvspan(output_space[0].min(), 1, color='red', alpha=0.15, linewidth=0.25)
    ax.set_xlim(-0.1, 1)
    ax.set_ylim(-np.pi/2, np.pi/2)
    jP.configure_spines(ax)
    ax.spines['bottom'].set_bounds(0, 1)
    ax.spines['left'].set_bounds(-np.pi/4, np.pi/4)
    ax.set_yticks([-np.pi/4, np.pi/4])
    ax.set_yticklabels([r'$\pi$/4', r'$\pi$/4'])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(r'Vector Difference ($\angle$)')

    if i != 2:
        ax.set_xticklabels(['', '', '', '', ''])
    else:
        ax.set_xlabel('Norm. Distance To Cue On FP')

    legend_arrow_off = mpatches.FancyArrowPatch((0, 0), (1, 1), arrowstyle='-|>',
                                       mutation_scale=10, color='r')
    legend_arrow_on = mpatches.FancyArrowPatch((0, 0), (1, 1), arrowstyle='-|>',
                                       mutation_scale=10, color='k')
    red_patch = mpatches.Patch(color='red', alpha=0.15, label='Red Category')
    ax.legend([legend_arrow_off, legend_arrow_on, red_patch],
            [f'{step_time}s Cue Off', f'{step_time}s Cue On', 'Response Area'],
            handler_map={mpatches.FancyArrowPatch: HandlerArrow()},
            bbox_to_anchor=[0.5, 0.9], loc='lower center',
            ncols=3, handletextpad=0.1)
    plt.suptitle('NS RNN - Leaky Integrator Method')
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
