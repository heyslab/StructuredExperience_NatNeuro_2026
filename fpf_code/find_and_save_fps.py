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
import matplotlib.pyplot as plt
import numpy as onp
import sys
import time
from pathlib import Path
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import pandas as pd
import pickle
import argparse

sys.path.append('computation-thru-dynamics')
sys.path.append('../')
import leaky_rnn

import fixed_point_finder.decision as decision
import fixed_point_finder.fixed_points as fp_optimize
import fixed_point_finder.rnn as rnn
import fixed_point_finder.utils as utils

from classes.datagen import tDNMSGenerator, genFactory
from classes.models import LeakyRNN
from classes.models import LeakyRNNCell
from analysis_tools.mpl_helpers import PdfPlotter
import models_database as mdb

#from jax.experimental import io_callback

#jax.config.update("jax_traceback_filtering", "off")
#jax.config.update("xla_python_client_preallocate", "false")

TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'S': '#f89521'}

def plot_pcs(axs, pca, fps, h_t, h_full, color='red'):

    hfull_pca = pca.transform(h_full[0])

    max_fps_to_plot = 300
    if h_t is not None:
        h_pca = pca.transform(h_t[0])
    alpha = 0.01

    if len(fps) == 0:
        return

    hstars = np.reshape(fps, (-1, 128))
    fp_pca = pca.transform(hstars)

    size = 4
    marker_style = dict(marker='o', ms=size, mec='gray', color='k')
    ax = axs[0]
    ax.plot(hfull_pca[:, 0], hfull_pca[:, 1], c=color, alpha=0.25)
    ax.plot(fp_pca[:, 0], fp_pca[:, 1], ls='', **marker_style)
    if h_t is not None:
        ax.plot(h_pca[:, 0], h_pca[:, 1], c='red')
        ax.plot(h_pca[0, 0], h_pca[0, 1], c='green', ms=2, ls='', marker='o')
        ax.plot(h_pca[-1, 0], h_pca[-1, 1], c='red', ms=2, ls='', marker='o')
    ax.set_ylim(-8, 8)
    ax.set_xlim(-8, 8)
    ax.set_xticks([])


    ax = axs[1]
    ax.plot(hfull_pca[:, 0], hfull_pca[:, 2], c=color, alpha=0.25)
    ax.plot(fp_pca[:, 0], fp_pca[:, 2], ls='', **marker_style)
    if h_t is not None:
        ax.plot(h_pca[:, 0], h_pca[:, 2], c='red')
        ax.plot(h_pca[0, 0], h_pca[0, 2], c='green', ms=2, ls='', marker='o')
        ax.plot(h_pca[-1, 0], h_pca[-1, 2], c='red', ms=2, ls='', marker='o')
    ax.set_ylim(-8, 8)
    ax.set_xlim(-8, 8)



def find_fps(params, x_star, h_t, tol):
    rnn_fun = lambda h, params=params, x_star=x_star: leaky_rnn.rnn(params, h, x_star)
    batch_rnn_fun = vmap(rnn_fun, in_axes=(0,))

    fp_loss_fun = fp_optimize.get_fp_loss_fun(rnn_fun)
    total_fp_loss_fun = fp_optimize.get_total_fp_loss_fun(rnn_fun)

    fp_candidates = h_t # was batch x time x dim
    fp_candidates = np.reshape(fp_candidates, (-1, h_t.shape[-1])) # now batch * time x dim
    fp_candidates = np.tile(fp_candidates, (4, 1))

    #rnn_fun = lambda x: model.layers[0](onp.zeros((1, 30, 2)), initial_state=x)

    # Fixed point optimization hyperparams
    fp_num_batches = 40000
    fp_batch_size = 128
    fp_step_size = 2
    fp_decay_factor = 0.9999
    fp_decay_steps = 1
    fp_adam_b1 = 0.9
    fp_adam_b2 = 0.999
    fp_adam_eps = 1e-5
    fp_opt_print_every = 200

    fp_noise_var = 10
    fp_opt_stop_tol = 0.000001
    fp_tol = tol
    fp_unique_tol = 1
    fp_outlier_tol = 10.0

    fp_hps ={'num_batches': fp_num_batches,
             'step_size': fp_step_size,
             'decay_factor': fp_decay_factor,
             'decay_steps': fp_decay_steps,
             'adam_b1': fp_adam_b1, 'adam_b2': fp_adam_b2, 'adam_eps': fp_adam_eps,
             'noise_var': fp_noise_var,
             'fp_opt_stop_tol': fp_tol,
             'fp_tol': fp_tol,
             'unique_tol': fp_unique_tol,
             'outlier_tol': fp_outlier_tol,
             'opt_print_every': fp_opt_print_every}

    fps, fp_losses, fp_idxs, fp_opt_details = \
        fp_optimize.find_fixed_points(rnn_fun, fp_candidates, fp_hps, do_print=True)

    return fps



def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id')
    parser.add_argument('--tolerance', '-t', type=float, default=0.000001)
    args = parser.parse_args(argv)

    tol = args.tolerance
    model_id = args.model_id
    model_info = mdb.get_model(model_id)
    model_path = model_info['path']
    model_attrs = mdb.get_model_attributes(model_id)

    input_noise = model_attrs['input_noise']
    rnn_layer = 0

    model = tf.keras.models.load_model(model_path)
    w_in, w_r, b = model.layers[0].get_weights()
    w_out, b_out = model.layers[1].get_weights()
    h0 = onp.squeeze(model.layers[0](onp.zeros((1, 1, 2))).numpy())
    params = leaky_rnn.rnn_params(h0, b, w_in, w_r, w_out, b_out, model_attrs['gamma'], 0)

    trials_gen = genFactory.create(
        'just_short_match', input_noise=0, batch_size=1, n_blocks=1)

    X2 = trials_gen.generate_trials(1)[['light', 'odor']]
    h_t, o_t = leaky_rnn.batched_rnn_run(params, np.expand_dims(X2.values, 0))

    X2 = trials_gen.generate_trials(10)[['light', 'odor']]
    h_start = h_t[0, -1]
    params['h0'] = h_start
    h_t, o_t = leaky_rnn.batched_rnn_run(params, np.expand_dims(X2.values, 0))
    pca = PCA(n_components=3).fit(h_t[0])

    x_star = np.zeros(2)
    fps_cue_off = find_fps(params, x_star, h_t, tol)
    rnn_fun = lambda h, params=params, x_star=x_star: leaky_rnn.rnn(params, h, x_star)
    jacs_cue_off = fp_optimize.compute_jacobians(rnn_fun, fps_cue_off)
    eigs_cue_off = fp_optimize.compute_eigenvalue_decomposition(jacs_cue_off)

    x_star = np.array([0, 1])
    fps_cue_on = find_fps(params, x_star, h_t, tol)
    rnn_fun = lambda h, params=params, x_star=x_star: leaky_rnn.rnn(params, h, x_star)
    jacs_cue_on = fp_optimize.compute_jacobians(rnn_fun, fps_cue_on)
    eigs_cue_on = fp_optimize.compute_eigenvalue_decomposition(jacs_cue_on)

    path = Path(model_info['path']).parent
    PdfPlotter(path / f'fixed_pt_pca.{model_id}.pdf')
    fig = plt.figure(figsize=(8, 9))
    gs0 = iter(gridspec.GridSpec(3, 1, hspace=0.3))

    for trial_type in ('SL', 'LS', 'SS'):
        c = TRIAL_COLORS[trial_type]

        gs = next(gs0).subgridspec(2, 5)
        X_trial = X2.xs(trial_type, level='type')
        params['h0'] = h_start
        h_full, o_t = leaky_rnn.batched_rnn_run(params, np.expand_dims(X_trial.values, 0))
        axs = list(map(plt.subplot, (gs[0, 0], gs[1, 0])))
        plot_pcs(axs, pca, fps_cue_off, None, h_full, color=c)

        axs = list(map(plt.subplot, (gs[0, 1], gs[1, 1])))
        plot_pcs(axs, pca, fps_cue_on, None, h_full, color=c)

    plt.show()

    save_path = path / 'fps_info.pkl'
    if save_path.exists():
        with open(save_path, 'rb') as f:
            fps = pickle.load(f)
    else:
        fps = {}

    fps[tol] = {
        'cue_on': {'fps': fps_cue_on, 'jacobians': jacs_cue_on,
                   'eigenvalues': eigs_cue_on},
        'cue_off': {'fps': fps_cue_off, 'jacobians': jacs_cue_off,
                   'eigenvalues': eigs_cue_off}}

    with open(save_path, 'wb') as f:
        pickle.dump(fps, f)


if __name__ == '__main__':
    main(sys.argv[1:])
