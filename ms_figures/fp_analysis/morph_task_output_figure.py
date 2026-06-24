# [Figure 4h,j]
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
from analysis_tools.progressbar import ProgressBar


TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'S': '#f89521'}
def main(args):
    jP.set_rcParams(plt)
    dpi = 300
    path = Path('/analysis/ms_figures/fps_analysis')
    path.mkdir(parents=True, exist_ok=True)

    fp_tol = 1e-6
    with_model_ids = [20, 22, 25, 21, 23, 26, 27, 30, 76, 79, 82, 84, 132]
    model_ids = pd.Series(with_model_ids,
                          index=pd.Index(with_model_ids, name='model_id'))

    model_infos = model_ids.apply(mdb.get_model).apply(pd.Series)
    model_attrs = model_ids.apply(mdb.get_model_attributes)
    model_infos = pd.concat(
        (model_infos,
         model_attrs[['gamma', 'epoc', 'input_noise', 'noise_level']]), axis=1)

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

    cue_times = {
        'SL': [30, 50, 80, 130],
        'LS': [30, 80, 110, 130],
        'SS': [30, 50, 80, 100]}
    margins = jP.default_margins()

    model_type = 'shaping'
    alphas = np.linspace(0, 1, 40)
    trial_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trial_merge', [TRIAL_COLORS['SL'], TRIAL_COLORS['SS']],
        N=len(alphas))
    outputs = pd.DataFrame([], index=models.index)
    p = ProgressBar(len(alphas))
    print('Calculating SS->SL')
    for i,a in  enumerate(alphas):
        p.increment()
        inputs = onp.zeros((1, 200, 2))
        cue_times = [30, 50, 80, 150 - int(50*a)]
        inputs[:, cue_times[0]:cue_times[1], 1] = 1
        inputs[:, cue_times[2]:cue_times[3], 1] = 1
    
        rnn_res = model_params.apply(
            lambda x, inputs=inputs:
                leaky_rnn.batched_rnn_run(x, inputs))\
                    .apply(pd.Series, index=('h', 'o'))\
                    .map(np.squeeze)
        rnn_res.name = cue_times[-1]
        out = rnn_res['o']
        out = out.apply(lambda x, cue_times=cue_times: x[cue_times[-1]:(cue_times[-1] + 20)])
        max_auc = pd.concat((out.apply(np.max),
                             out.apply(onp.trapz, dx=0.1)),
                             axis=1, keys=('max', 'auc'))
        max_auc.columns = pd.MultiIndex.from_tuples(
            list(zip(it.repeat(cue_times[-1]), max_auc.columns)), names=('length', 'metric'))
        outputs = pd.concat((outputs,
                             max_auc), axis=1)

    outputs.columns = pd.MultiIndex.from_tuples(outputs.columns, names=('length', 'metric'))
    outputs = outputs.map(float)

    outputs = outputs.sort_index(axis=1)
    mask = np.tile(outputs.columns.get_level_values('length') <= 130, (len(outputs),1))
    inbound_outs = outputs.mask(~mask).dropna(axis=1)
    outbound_outs = outputs.mask(mask).dropna(axis=1)
    n = inbound_outs.shape[1]//2
    trial_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trial_merge', [TRIAL_COLORS['SS'], TRIAL_COLORS['SL']],
        N=n/2)

    for metric in ('max', 'auc'):
        PdfPlotter(path / f'{model_type}.ss2sl.{metric}out.pdf',
                   fixed_margins=margins)
        plt.figure(figsize=(1.25, 1.75), dpi=dpi)
        ax = plt.gca()
        x_vals = (outputs.xs(metric, level='metric', axis=1).columns.get_level_values('length').to_series() - cue_times[-2])/10

        means = inbound_outs.xs(metric, level='metric', axis=1).mean(0).sort_index()
        std = inbound_outs.xs(metric, level='metric', axis=1).std(0).sort_index()
        jP.plot_seg_colors(ax, x_vals[:len(means)], means + std, c=None, y2=(means-std), n=n, cmap=trial_cmap, alpha=0.25)
        jP.plot_seg_colors(ax, x_vals[:len(means)], means, c=None, n=n, cmap=trial_cmap)

        means = outbound_outs.xs(metric, level='metric', axis=1).mean(0).sort_index()
        std = outbound_outs.xs(metric, level='metric', axis=1).std(0).sort_index()
        ax.fill_between(x_vals[-len(means):], means + std, y2=(means-std), color='k', alpha=0.15)
        ax.plot(x_vals[-len(means):], means, color='k', ls='--', alpha=0.5)
     
        jP.configure_spines(ax)
        ax.set_xlim(x_vals.min(), x_vals.max())

        if metric == 'max':
            ax.set_ylabel('Max Response')
        else:
            ax.set_ylabel('Response (AUC)')
        ax.set_xlabel('Second Cue Length (s)')
        ax.set_xticks([3, 5])
        ax.set_yticks([0.2, 0.6, 1.0])
        plt.show()
 
    trial_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trial_merge', [TRIAL_COLORS['LS'], TRIAL_COLORS['SS']],
        N=len(alphas))
    outputs = pd.DataFrame([], index=models.index)
    p = ProgressBar(len(alphas))
    print('Calculating SS->LS')
    for i,a in  enumerate(alphas):
        p.increment()
        inputs = onp.zeros((1, 200, 2))
        x = 100 - int(50*a)
        cue_times = [30, x, x+30, x+50]
        inputs[:, cue_times[0]:cue_times[1], 1] = 1
        inputs[:, cue_times[2]:cue_times[3], 1] = 1
    
        rnn_res = model_params.apply(
        lambda x, inputs=inputs:
            leaky_rnn.batched_rnn_run(x, inputs))\
                .apply(pd.Series, index=('h', 'o'))\
                .map(np.squeeze)
        rnn_res.name = cue_times[-1]
        max_auc = pd.concat((rnn_res['o'].apply(np.max),
                             rnn_res['o'].apply(onp.trapz, dx=0.1)),
                            axis=1, keys=('max', 'auc'))
        max_auc.columns = pd.MultiIndex.from_tuples(
            list(zip(it.repeat(cue_times[1]), max_auc.columns)), names=('length', 'metric'))
        outputs = pd.concat((outputs,
                             max_auc), axis=1)

    outputs.columns = pd.MultiIndex.from_tuples(outputs.columns, names=('length', 'metric'))
    outputs = outputs.map(float)

    outputs = outputs.sort_index(axis=1)
    mask = np.tile(outputs.columns.get_level_values('length') <= 80, (len(outputs),1))
    inbound_outs = outputs.mask(~mask).dropna(axis=1)
    outbound_outs = outputs.mask(mask).dropna(axis=1)
    n = inbound_outs.shape[1]//2
    trial_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trial_merge', [TRIAL_COLORS['SS'], TRIAL_COLORS['LS']],
        N=n/2)

    for metric in ('max', 'auc'):
        PdfPlotter(path / f'{model_type}.ss2ls.{metric}out.pdf',
                   fixed_margins=margins)
        plt.figure(figsize=(1.25, 1.75), dpi=dpi)
        ax = plt.gca()
        x_vals = (outputs.xs(metric, level='metric', axis=1).columns.get_level_values('length').to_series() - cue_times[0])/10

        means = inbound_outs.xs(metric, level='metric', axis=1).mean(0).sort_index()
        std = inbound_outs.xs(metric, level='metric', axis=1).std(0).sort_index()
        jP.plot_seg_colors(ax, x_vals[:len(means)], means + std, c=None, y2=(means-std), n=n, cmap=trial_cmap, alpha=0.25)
        jP.plot_seg_colors(ax, x_vals[:len(means)], means, c=None, n=n, cmap=trial_cmap)

        means = outbound_outs.xs(metric, level='metric', axis=1).mean(0).sort_index()
        std = outbound_outs.xs(metric, level='metric', axis=1).std(0).sort_index()
        ax.fill_between(x_vals[-len(means):], means + std, y2=(means-std), color='k', alpha=0.15)
        ax.plot(x_vals[-len(means):], means, color='k', ls='--', alpha=0.5)
     
        jP.configure_spines(ax)
        ax.set_xlim(x_vals.min(), x_vals.max())

        if metric == 'max':
            ax.set_ylabel('Max Response')
        else:
            ax.set_ylabel('Response (AUC)')
        ax.set_xlabel('First Cue Length (s)')
        ax.set_xticks([3, 5])
        ax.set_yticks([0.2, 0.6, 1.0])
        plt.show()
 

if __name__ == '__main__':
    main(sys.argv[1:])
