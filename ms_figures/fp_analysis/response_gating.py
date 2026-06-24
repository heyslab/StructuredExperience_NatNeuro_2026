# [Figure S6c]
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

def unit_vector(v):
    return v / onp.linalg.norm(v)


def angle_between(v1, v2):
    return onp.arccos(onp.clip(onp.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0))


class HandlerArrow(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        p = mpatches.FancyArrowPatch((xdescent, ydescent + height / 2),
                (xdescent + width, ydescent + height / 2),
                arrowstyle='-|>', mutation_scale=5, transform=trans,
                color=orig_handle.get_facecolor())
        return [p]


def draw_arrows(ax, arrow_coords, arrowprops, alpha=1, **kwargs):
    for t in arrow_coords.index.get_level_values('theta'):
        steps = arrow_coords.index.get_level_values('step').unique().sort_values()

        step0 = (arrow_coords['theta'][t][steps[0]], arrow_coords['z'][t][steps[0]])
        for step in steps[1:]:
            next_step = (arrow_coords['theta'][t][step],
                         arrow_coords['z'][t][step])
            if np.abs(next_step[0] - step0[0]) > \
               np.abs(next_step[0] - 2*np.pi - step0[0]):
                altered_step = (next_step[0] - 2 * np.pi, next_step[1])
                arrow = ax.annotate('', altered_step, xytext=step0,
                                    textcoords=ax.transData, arrowprops=arrowprops,
                                    **kwargs)
            elif np.abs(next_step[0] - step0[0]) > \
                 np.abs(next_step[0] + 2*np.pi - step0[0]):
                altered_step = (next_step[0] + 2 * np.pi, next_step[1])
                arrow = ax.annotate('', altered_step, xytext=step0,
                                    textcoords=ax.transData, arrowprops=arrowprops,
                                    **kwargs)
            else:
                arrow = ax.annotate('', next_step, xytext=step0,
                                  textcoords=ax.transData, arrowprops=arrowprops,
                                  **kwargs)
            step0 = next_step


def draw_trial_arrows(ax, polar_trial_info, arrowprops,
                      zorder=('SS', 'SL', 'LS')):
    for i, trial_type in enumerate(('SS', 'SL', 'LS')):
        polar_trial = polar_trial_info.xs(trial_type, level='type')
        itr2 = polar_trial[['theta', 'z']].iloc[::5].iterrows()
        next(itr2)
        arrowprops.update({'color': TRIAL_COLORS[trial_type]})
        for p1, p2 in zip(polar_trial[['theta', 'z']].iloc[::5].iterrows(), itr2):
            if 'mask' in polar_trial.index.names:
                mask = p1[0][polar_trial.index.names.index('mask')]
                if mask == 1:
                    arrowprops.update({'alpha': 0.2})
                    arrowprops.update({'color': 'gray'})
                else:
                    arrowprops.update({'alpha': 1})
                    arrowprops.update({'color': TRIAL_COLORS[trial_type]})

            ax.annotate('', p2[1].values + np.array([0, (i*2)/10-2/10]),
                        xytext=p1[1].values+np.array([0, (i*2)/10-2/10]),
                        textcoords=ax.transData, arrowprops=arrowprops,
                        zorder=100+zorder.index(trial_type))

    if 'mask' in polar_trial.index.names:
        arrowprops.update({'alpha': 1})


def setup_axis(ax):
    ax.spines['bottom'].set_bounds(-np.pi, np.pi)
    ax.spines['left'].set_bounds(-1, 8)
    ax.set_xlabel('Limit Cycle Phase (Trial Structure "Timer")')
    ax.set_yticks([-1, 2, 5, 8])
    ax.set_xticks([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4,
                   np.pi/2, 3*np.pi/4, np.pi])
    ax.set_ylabel(r'PC3 (Cue Detection)')

    ax.set_xticklabels(
        [r'-$\pi$', r'-3$\pi$/4', r'-$\pi$/2', r'-$\pi$/4', 0,
         r'$\pi$/4', r'$\pi$/2', r'3$\pi$/4', r'$\pi$'])

    legend_arrow_off = mpatches.FancyArrowPatch(
        (0, 0), (1, 1), arrowstyle='-|>', mutation_scale=10, color='r')
    legend_arrow_on = mpatches.FancyArrowPatch(
        (0, 0), (1, 1), arrowstyle='-|>', mutation_scale=10, color='k')
    red_patch = mpatches.Patch(
        color='red', alpha=0.5, label='Red Category', lw=0.75, ec='k')
    ax.legend([legend_arrow_off, legend_arrow_on, red_patch],
            ['0.5s Cue Off', '0.5s Cue On', 'Response Area'],
            handler_map={mpatches.FancyArrowPatch: HandlerArrow()},
            bbox_to_anchor=[0, 0.9], loc='lower left', ncols=3, handletextpad=0.1)


def main(argv):
    jP.set_rcParams(plt)
    dpi = 300
    path = Path('/analysis/ms_figures/fps_schematic')
    path.mkdir(parents=True, exist_ok=True)

    step_time = 1
    step_size = int(10 * step_time)
    fp_tol = 1e-6
    output_cutoff = 0.45
    model_ids = pd.Series([132], index=('shaping',))

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

    # Find a good initial point to start at
    h0 = fps['shaping']['cue_off'][-1]
    trial_res = inputs.groupby(['type', 'idx']).head(1).groupby('type')\
                      .apply(run_rnn, h0=h0, model_params=model_params['shaping'])\
                      .droplevel(0)

    h_starts = trial_res.iloc[-1].values
    model_params['shaping']['h0'] = h_starts

    rnn_res = model_params.apply(
        lambda x:
            leaky_rnn.batched_rnn_run(x,  np.tile(np.array([0, 0]), (1, 800, 1))))\
                .apply(pd.Series, index=('h', 'o'))\
                .map(np.squeeze)
    h_t = pd.DataFrame(rnn_res['h']['shaping'])

    # Identify the limit cycle
    dh_t = list(map(math.dist, h_t.values, h_t.shift(-1).values))
    distance_traveled = onp.append([0], onp.cumsum(dh_t)[:-1])
    distance_to_start = list(map(
        lambda x, h_t=h_t.values: math.dist(h_t[0], h_t[x]), np.arange(800)))
    cycle_end = np.where(distance_to_start[1:] < distance_traveled[1:] / 100)[0][0]
    h_t = h_t.iloc[int(cycle_end):2*int(cycle_end)]

    h0 = h_starts
    trial_res = inputs.groupby(['type', 'idx']).head(1).groupby('type')\
                      .apply(run_rnn, h0=h0, model_params=model_params['shaping'])\
                      .droplevel(0)

    # calculate PCA on the RNN running the full task
    pca = PCA(n_components=128).fit(trial_res)

    h_pc = pd.DataFrame(pca.transform(h_t.values))
    pc_offset = h_pc.mean()
    h_pc = h_pc - pc_offset

    def pc_to_polar(h_pc, offset=0):
        theta = offset-h_pc.apply(lambda x: math.atan2(x[1], x[0]), axis=1)
        theta[theta >= np.pi] = theta[theta >= np.pi] - 2*np.pi
        polar_h = pd.DataFrame({
            'r': (h_pc[[0, 1]]**2).sum(axis=1)**0.5,
            'theta': theta,
            'z': h_pc[2]})
        return polar_h

    # convert to polar to identfy the offset (align start of the trial to -pi)
    trial_info_pc = pd.DataFrame(pca.transform(trial_res), index=trial_res.index) - pc_offset
    polar_trial_info = pc_to_polar(trial_info_pc)
    offset = np.pi - polar_trial_info.head(1)['theta'].values[0]

    polar_trial_info = pc_to_polar(trial_info_pc, offset)
    output_space = polar_trial_info[polar_trial_info.index.get_level_values('out') > 0.4]

    # determine starting h's for the cue on flow
    polar_h = pc_to_polar(h_pc, offset)
    new_idx = pd.Series(np.arange(-np.pi, np.pi, np.pi/10))
    polar_h = polar_h.set_index('theta', drop=False)
    adjusted_polar = polar_h.reindex(polar_h.index.union(new_idx))\
                            .interpolate('values').bfill().reindex(new_idx)
    h_pc.index = polar_h['theta']
    adjusted_h_pc = h_pc.reindex(h_pc.index.union(new_idx))\
                        .interpolate('values').bfill().reindex(new_idx) 

    def run_rnn(h0, model_params, step_size, steps=1, cue_on=True):
        model_params['shaping']['h0'] = h0.values
        x_star=(0, int(cue_on))
        h_t, _ = leaky_rnn.batched_rnn_run(
            model_params['shaping'], np.tile(np.array(x_star),
            (1, steps * step_size, 1)))
        return onp.vstack(
            (h0, h_t[0, np.arange(step_size, step_size * (steps+ 1), step_size)]))

    adjusted_h = pca.inverse_transform(adjusted_h_pc + pc_offset)
    dh = adjusted_h.apply(run_rnn, model_params=model_params, step_size=step_size,
                          cue_on=False, axis=1)
    dh = pd.concat(list(map(pd.DataFrame, dh.values)), keys=dh.index)
    dh.index.names = ('theta', 'step')
    dh_pc = pd.DataFrame(pca.transform(dh), index=dh.index) - pc_offset
    limit_flow = pc_to_polar(dh_pc, offset)

    # running the cue on model
    dh_on = adjusted_h.apply(run_rnn, model_params=model_params, step_size=step_size,
                             steps=15, cue_on=True, axis=1)
    dh_on = pd.concat(list(map(pd.DataFrame, dh_on.values)), keys=dh_on.index)
    dh_on.index.names = ('theta', 'step')
    dh_pc_on = pd.DataFrame(pca.transform(dh_on), index=dh_on.index) - pc_offset
    cue_on_flows = pc_to_polar(dh_pc_on, offset)

    output_space = \
        polar_trial_info[polar_trial_info.index.get_level_values('out') >
                         output_cutoff]
    def eval_out(X, output_cutoff=output_cutoff):
        return X.index.get_level_values('out').max() > output_cutoff

    def run_rnn_probe(h0, model_params, cue_on='S'):
        if cue_on == 'S':
            cue_x = onp.tile(np.array((0, 1)), (1, 20, 1))
        elif cue_on == 'L':
            cue_x = onp.tile((0, 1), (1, 50, 1))
        else:
            raise Exception('cue needs to be S or L')

        x = onp.hstack((cue_x, np.tile(np.array((0, 0)), (1, 70, 1))))
        model_params['shaping']['h0'] = h0.values
        h_t, o_t = leaky_rnn.batched_rnn_run(
            model_params['shaping'], x)

        h_t = onp.vstack((h0, h_t[0]))
        o_t = onp.hstack((np.array([0]), o_t[0, :, 0]))
        x = onp.hstack((np.array([1]), x[0, :, 1]))
        index = pd.MultiIndex.from_tuples(
            list(zip(x, o_t, it.count())), names=('cue', 'out', 'step'))
        h_t = pd.DataFrame(h_t, index=index)
        return h_t

    def calc_probe(cue_length, adjusted_h_pc=adjusted_h_pc, pca=pca,
                   pc_offset=pc_offset, #run_rnn_probe=run_rnn_probe,
                   model_params=model_params, #pc_to_polar=pc_to_polar,
                   offset=offset, eval_out=eval_out):
        probe_pc = adjusted_h_pc.copy()
        probe_pc = pd.concat([probe_pc]*9, keys=np.arange(-1, 8))
        probe_pc[2] = probe_pc.index.get_level_values(0)

        probe_h = pca.inverse_transform(probe_pc + pc_offset)
        probe_h.index.names = ('z_', 'theta_')
        dh = probe_h.apply(run_rnn_probe, model_params=model_params,
                           cue_on=cue_length, axis=1)
        dh = pd.concat(list(map(pd.DataFrame, dh.values)), keys=dh.index)
        dh_pc = pd.DataFrame(pca.transform(dh), index=dh.index) - pc_offset
        probe_flow = pc_to_polar(dh_pc, offset)

        return probe_flow.groupby(['z_', 'theta_'], as_index=True)\
                         .apply(eval_out).unstack('theta_')
    outs = pd.Series(
        ('S', 'L'), index=pd.Index(('S', 'L'), name='cue_length')
        ).apply(calc_probe)

    demos_pc = pd.concat(
        (trial_info_pc.xs('SS', level='type').xs(30, level='idx'),
         trial_info_pc.xs('LS', level='type').xs(110, level='idx'),
         trial_info_pc.xs('SL', level='type').xs(80, level='idx'))
        ).reset_index(drop=True)
    demos_pc.index.name = 'theta'
    demos_h = pca.inverse_transform(demos_pc + pc_offset)
    dh = demos_h.apply(run_rnn_probe, model_params=model_params,
                       cue_on='S', axis=1)
    dh = pd.concat(list(map(pd.DataFrame, dh.values)), keys=dh.index)
    dh_pc = pd.DataFrame(pca.transform(dh), index=dh.index) - pc_offset
    demo_flow = pc_to_polar(dh_pc, offset)
    demo_flow = demo_flow.set_index(
        pd.Index(demo_flow.index.get_level_values('cue'), name='mask'),
        append=True)

    annotation_pc = pd.concat(
        (trial_info_pc.xs('SS', level='type').xs(30, level='idx'),
         trial_info_pc.xs('SS', level='type').xs(50, level='idx'),
         trial_info_pc.xs('SL', level='type').xs(130, level='idx'),
         trial_info_pc.xs('SL', level='type').xs(199, level='idx')),
        keys=('cue1', 'cue2', 'no_cue1', 'no_cue2')
        )
    annotation_polar = pc_to_polar(annotation_pc, offset)

    PdfPlotter(path / f'pca_duration_gate_{step_time}.pdf',
               fixed_margins=jP.default_margins())
    plt.figure(figsize=(4.5, 2.5), dpi=300)
    ax = plt.gca()
    arrowprops = {'arrowstyle': '-|>', 'shrinkA': 0.05, 'shrinkB': 0.05,
                  'color': 'r', 'lw': 1}
    draw_arrows(ax, limit_flow, arrowprops)

    arrowprops.update({'color': 'k'})
    draw_arrows(
        ax, cue_on_flows.xs(cue_on_flows.index.unique('theta').min(),
                            level='theta', drop_level=False), arrowprops, 1)
    arrowprops.update({'color': 'rebeccapurple'})
    draw_arrows(
        ax, demo_flow.reindex(
            demo_flow.index.unique('step').sort_values()[::step_size],
            level='step').xs(1, level='mask').droplevel(['cue', 'out']), arrowprops,
            zorder=1e10)
    outs_sum = outs.apply(lambda x: x.astype(int)).sum()
    dt = outs_sum.columns.diff()[1]
    dz = outs_sum.index.diff()[1]
    extent = (outs_sum.columns.min() - dt/2,
              outs_sum.columns.max() + dt/2,
              outs_sum.index.max() + dz/2,
              outs_sum.index.min() - dz/2)
    colors = ['#F0F0F0', '#B7C3C7', '#FFE699']
    custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom_cmap", colors, N=3)

    ax.imshow(outs_sum, extent=extent, zorder=-1, interpolation='nearest',
              aspect='auto', cmap=custom_cmap)
    bounds = onp.array((
        output_space['theta'].min(),
        output_space['theta'].max(),
        output_space['z'].min(),
        output_space['z'].max()))
    verticies = [bounds[[0, 2]], bounds[[1, 2]], bounds[[1, 3]], bounds[[0, 3]]]

    poly = Polygon(verticies, closed=True, facecolor='red', edgecolor='k',
                   alpha=0.5, linewidth=0.75, zorder=1e5)
    ax.add_patch(poly)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-4, 10)
    jP.configure_spines(ax)
    setup_axis(ax)

    annotation_props = {'arrowstyle': '|-|,widthA=0.4,widthB=0.4',
                        'shrinkA': 0.2, 'shrinkB': 0.1,
                        'color': 'k', 'lw': 1}
    pts = annotation_polar.reindex(['cue1', 'cue2'], level=0)[['theta', 'z']].values
    ax.annotate('', (pts[0][0], -1.7), (pts[1][0], -1.5),
                textcoords=ax.transData, arrowprops=annotation_props)
    ax.text(np.mean(pts[:, 0]), -1.85, 'All Trial Types\nCue On', va='top',
            ha='center', fontsize=5)
    pts = annotation_polar.reindex(
        ['no_cue1', 'no_cue2'], level=0)[['theta', 'z']].values
    ax.annotate('', (pts[0][0], -1.7), (pts[1][0], -1.5),
                textcoords=ax.transData, arrowprops=annotation_props)
    ax.text(np.mean(pts[:, 0]), -1.85, 'All Trial Types\nCue Off', va='top',
                    ha='center', fontsize=5)


    for i, x in enumerate(demo_flow.reindex([0], level='step').iterrows()):
        ax.plot(x[1]['theta'], x[1]['z'], ls='', marker='x', color='k', ms=5,
                markeredgewidth=1.5, zorder=1e3)
        ax.annotate('i'*(i+1) + '.', (x[1]['theta'], x[1]['z']), xytext=(-8, -5),
                    textcoords='offset points', color='k')

    legend_arrow_off = mpatches.FancyArrowPatch(
        (0, 0), (1, 1), arrowstyle='-|>', mutation_scale=10, color='r')
    legend_arrow_2s = mpatches.FancyArrowPatch(
        (0, 0), (1, 1), arrowstyle='-|>', mutation_scale=10, color='rebeccapurple')
    legend_arrow_on = mpatches.FancyArrowPatch(
        (0, 0), (1, 1), arrowstyle='-|>', mutation_scale=10, color='k')
    red_patch = mpatches.Patch(
        color='red', alpha=0.5, label='Red Category', lw=0.75, ec='k')
    gray_patch = mpatches.Patch(
        color='#B7C3C7', alpha=0.5, lw=0)
    yellow_patch = mpatches.Patch(
        color='#FFE699', alpha=0.5, lw=0)
    ax.legend([legend_arrow_off, legend_arrow_2s, legend_arrow_on, gray_patch,
               red_patch, yellow_patch],
            [f'{step_time}s Cue Off', 'Short Cue', f'{step_time}s Cue On', 'L-Response',
             'Response Area', 'S-Response'],
            handler_map={mpatches.FancyArrowPatch: HandlerArrow()},
            ncols=3, handletextpad=0.1)

    ax.get_legend().set_bbox_to_anchor((0.93, 0.885))
    ax.get_legend().set_loc('lower right')
    ax.get_legend().borderaxespad = 0

    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
