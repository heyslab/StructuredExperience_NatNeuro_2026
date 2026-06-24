# [Figure 3a,b,c]
import os
import sys
sys.path.append('../')
sys.path.append('fp_analysis')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as np

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import itertools as it
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as onp
import matplotlib.gridspec as gridspec

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
    no_model_ids = [9, 18, 19, 24, 28, 29, 31, 32, 93, 109, 116, 119]
    with_model_ids = [20, 22, 25, 21, 23, 26, 27, 30, 76, 79, 82, 84]

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
    weights = pd.concat(models.apply(lambda x: x.layers[0].get_weights()[1]).apply(pd.DataFrame).values, keys=models.index)
    input_weights = pd.concat(models.apply(lambda x: x.layers[0].get_weights()[0]).apply(pd.DataFrame).values, keys=models.index)
    output_weights = pd.concat(models.apply(lambda x: x.layers[1].get_weights()[0]).apply(pd.DataFrame).values, keys=models.index)

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

    for trial_type in inputs.index.unique('type'):
        rnn_res = model_params.apply(
            lambda x, inputs=np.expand_dims(inputs.xs(trial_type).values, 0):
                leaky_rnn.batched_rnn_run(x, inputs))\
                    .apply(pd.Series, index=('h', 'o'))\
                    .map(np.squeeze)
                    
        sort_orders = rnn_res['h'].apply(lambda x: pd.DataFrame(x).idxmax(0).sort_values().index)

        def sort_weights(X, sorts):
            model_id = X.name[1]
            sorted_X = X.droplevel([0, 1]).reindex(sorts[model_id].item()).reindex(sorts[model_id].item(), axis=1)
            return sorted_X.reset_index(drop=True).pipe(lambda x: x.set_axis(range(x.shape[1]), axis=1))
            
        sorted_weights = weights.groupby(['model_type', 'model_id']).apply(sort_weights, sorts=sort_orders)
        sorted_weights.index.names = sorted_weights.index.names[:-1] + ['node']

        def sort_inputs(X, sorts, axis):
            model_id = X.name[1]
            sorted_X = X.droplevel([0, 1]).reindex(sorts[model_id].item(), axis=axis)
            return sorted_X.pipe(lambda x: x.set_axis(range(x.shape[1]), axis=1))
        sorted_output = output_weights.groupby(['model_type', 'model_id']).apply(sort_inputs, sorts=sort_orders, axis=0)
        sorted_input = input_weights.groupby(['model_type', 'model_id']).apply(sort_inputs, sorts=sort_orders, axis=1)
        path = Path('/analysis/ms_figures/weights_analysis')
        path.mkdir(exist_ok=True, parents=True)

        dpi = 300
        jP.set_rcParams(plt)
        
        def mask_eye(X):
            return X.mask(np.eye(len(X)).astype(bool))
        masked_weights = sorted_weights.groupby('model_id').apply(mask_eye)
        masked_weights.columns.name = 'node2'
        stacked_weights = masked_weights.stack()
        idx = stacked_weights.index.droplevel([0, 3, 4]).to_frame()
        idx['distance'] = stacked_weights.droplevel([0, 1, 2]).index.to_frame().diff(axis=1).dropna(axis=1)['node2'].values
        stacked_weights.index = pd.MultiIndex.from_frame(idx)

        means = stacked_weights.groupby(stacked_weights.index.names).mean()
        stds =  stacked_weights.groupby(stacked_weights.index.names).std()

        margins = jP.default_margins()
        PdfPlotter(path / f'weights_profile.{trial_type}.pdf', fixed_margins=margins)
        plt.figure(figsize=(1.5, 2), dpi=dpi)
        gs = gridspec.GridSpec(2, 1)
        axs = list(map(plt.subplot, gs))

        def plot_fn(x, ax, c):
            ax.plot(x.index.get_level_values('distance'), x.values, alpha=0.25,
                    c=c, lw=0.5)
            ax.axhline(0, ls='--', c='k', lw=0.5)
            ax.axvline(0, c='r', lw=0.5)
            ax.set_ylim(-0.05, 0.05)
            jP.configure_spines(ax)
            ax.spines['left'].set_bounds(-0.04, 0.04)
            ax.set_yticks([-0.04, 0.04])
            ax.set_xlim(-128, 128)
            ax.ticklabel_format(scilimits=(0,0))
        means.xs('shaping').groupby('model_id')\
             .apply(plot_fn, ax=axs[1], c='tab:orange')
        axs[1].plot(
            *means.xs('shaping').groupby('distance').mean().reset_index().values.T,
            c='tab:orange')
        axs[1].set_ylabel('Shaping\n+ Full Task', fontsize=5, color='tab:orange')
        jP.set_ylabel_position(axs[1], nlines=4)
        axs[0].set_ylabel('\nNo Shaping', fontsize=5, color='tab:blue')
        axs[0].text(
            0.03, 0.5, 'Connection Weights', ha='left', va='center', rotation=90,
            fontsize=7, transform=axs[0].figure.transFigure)
        means.xs('no_shaping').groupby('model_id').apply(plot_fn, ax=axs[0],
                 c='tab:blue')
        axs[0].plot(
            *means.xs('no_shaping').groupby('distance').mean().reset_index().values.T,
            c='tab:blue')
        axs[0].set_xticks([])
        axs[0].spines['bottom'].set_visible(False)
        axs[1].spines['bottom'].set_bounds(-100, 100)
        jP.set_ylabel_position(axs[0], nlines=4)
        plt.show()

        margins = jP.default_margins()
        PdfPlotter(path / f'weights_profile_2.pdf', fixed_margins=margins)
        plt.figure(figsize=(2, 2), dpi=dpi)
        gs = gridspec.GridSpec(2, 1)
        axs = list(map(plt.subplot, gs))

        def plot_fn(x, ax, c):
            ax.plot(x.index.get_level_values('distance'), x.values, alpha=0.25,
                    c=c, lw=0.5)
            ax.axhline(0, ls='--', c='k', lw=0.5)
            ax.axvline(0, c='r', lw=0.5)
            ax.set_ylim(-0.05, 0.05)
            jP.configure_spines(ax)
            ax.spines['left'].set_bounds(-0.04, 0.04)
            ax.set_yticks([-0.04, 0.04])
            ax.set_xlim(-128, 128)
            ax.ticklabel_format(axis='y', scilimits=(0,0))
        means.xs('shaping').groupby('model_id')\
             .apply(plot_fn, ax=axs[1], c='tab:orange')
        axs[1].plot(
            *means.xs('shaping').groupby('distance').mean().reset_index().values.T,
            c='tab:orange')
        axs[1].set_ylabel('Shaping\n+ Full Task', fontsize=5, color='tab:orange')
        jP.set_ylabel_position(axs[1], nlines=4)
        axs[0].set_ylabel('\nNo Shaping', fontsize=5, color='tab:blue')
        axs[0].text(
            0.03, 0.5, 'Connection Weights', ha='left', va='center', rotation=90,
            fontsize=7, transform=axs[0].figure.transFigure)
        means.xs('no_shaping').groupby('model_id').apply(plot_fn, ax=axs[0],
                 c='tab:blue')
        axs[0].plot(
            *means.xs('no_shaping').groupby('distance').mean().reset_index().values.T,
            c='tab:blue')
        axs[0].set_xticks([])
        axs[0].spines['bottom'].set_visible(False)
        axs[1].spines['bottom'].set_bounds(-100, 100)
        jP.set_ylabel_position(axs[0], nlines=4)
        axs[1].set_xlabel(r'$\Delta$ Position (LS Sort)')
        plt.show()
  
        mean_weights = sorted_weights.groupby(['model_type', 'node']).mean()
        mean_weights.columns = mean_weights.columns // 1
        mean_weights.index = pd.MultiIndex.from_tuples(
            list(zip(mean_weights.index.get_level_values('model_type'),
                     mean_weights.index.get_level_values('node') // 1)))
        mean_weights.columns.name = 'node'
        mean_weights.index.names = ('model_type', 'node')
        dpi = 300
        jP.set_rcParams(plt)
        margins = jP.default_margins()
        margins = jP.default_margins()
        margins['right'] *= 2
        vmax = 0.025
        vmin = -0.025

        sorted_input.index.names = sorted_input.index.names[:-1] + ['input']
        mean_input = sorted_input.groupby(['model_type', 'input']).mean()
        sorted_output.index.names = sorted_output.index.names[:-1] + ['node']
        mean_output = sorted_output.groupby(['model_type', 'node']).mean()
        margins = jP.default_margins()
        margins['left'] *= 1.25
        margins['bottom'] *= 1.25
        margins['right'] *= 2.25
        for model_type in model_ids.index.unique('model_type'):
            PdfPlotter(path / f'{model_type}.mean_weights.{trial_type}.pdf', fixed_margins=margins)
            plt.figure(figsize=(2.5, 2), dpi=dpi)
            gs = gridspec.GridSpec(
                2, 2, width_ratios=(10, 1), wspace=0, height_ratios=(1, 10),
                hspace=0.1)
            ax = plt.subplot(gs[0, 0])
            ax.imshow(mean_input.xs(model_type).values, interpolation='nearest',
                cmap=plt.get_cmap('inferno'), vmax=vmax, vmin=vmin,
                extent=(-0.5, 100, -0.5, 9.5))
            if model_type == 'shaping':
                label = 'Shaping + Full Task'
                color = 'tab:orange'
            else:
                label = 'No Shaping'
                color = 'tab:blue'
            ax.text(
                0, 1.1, label, color=color, transform=ax.transAxes, ha='left',
                va='bottom')
            ax.set_xticks([])
            ax.set_yticks([2, 7])
            ax.set_yticklabels(['Timed Cue', 'Start'], fontsize=5)
            ax.set_aspect('equal', anchor='SW')
     
            ax = plt.subplot(gs[1, 0])
            im = ax.imshow(
                mean_weights.xs(model_type).values, interpolation='nearest',
                cmap=plt.get_cmap('inferno'), vmax=vmax, vmin=vmin)
            ax.set_xlabel(f'Unit (Sorted by {trial_type})')
            jP.set_ylabel_position(ax, nlines=2)
            ax.set_yticks([])
            ax.set_ylabel(f'Unit (Sorted by {trial_type})')
            ax.set_yticks([20, 60, 100])
            ax.set_xticks([20, 60, 100])
            ax.set_aspect('equal', anchor='SW')

            ax = plt.subplot(gs[1, 1])
            im = ax.imshow(
                mean_output.xs(model_type).values, interpolation='nearest',
                cmap=plt.get_cmap('inferno'), vmax=vmax, vmin=vmin,
                extent=(-0.5, 5.5, -0.5, 127.5))
            ax.set_yticks([])
            ax.set_xticks([2.5])
            ax.set_xticklabels(
                ['Response'], fontsize=5, rotation_mode='anchor', ha='right',
                rotation=45)
            jP.color_bar(ax, im, spacing=2)
            ax.images[-1].colorbar.set_ticks([-0.02, 0, 0.02])
            ax.images[-1].colorbar.ax.ticklabel_format(scilimits=(0,0))
            ax.images[-1].colorbar.ax.get_yaxis()\
                .get_offset_text().set_position((3.5, 0))
            ax.images[-1].colorbar.ax.set_ylabel('Mean Connection Weight',
                                                 rotation=-90, labelpad=8)
     
            plt.show()
     
if __name__ == '__main__':
    main(sys.argv[1:])
