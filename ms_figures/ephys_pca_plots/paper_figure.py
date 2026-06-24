# [Figure 6j,k]
import os
import sys

sys.path.append('../')

from sklearn.decomposition import PCA
from sklearn import metrics
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

from pathlib import Path
import numpy as np
import pandas as pd
import itertools as it
import argparse

from analysis_tools.progressbar import ProgressBar

from analysis_tools.mpl_helpers import PdfPlotter
import analysis_tools.jPlots as jP
from tensorly.decomposition import non_negative_parafac
from tensorly.cp_tensor import cp_normalize
import models_database as mdb

TRIAL_COLORS = {
    'LS': '#eb0d8c',
    'SL': '#2bace2',
    'SS': '#f89521',
    'LL': '#2b958c',
    'S' : '#f89521'}
SHAPING_COLORS = {'no_shaping': 'tab:blue', 'shaping': 'tab:orange'}

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  
    "axes.facecolor":    (1.0, 1.0, 1.0, 0.0),  
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),
    })


def calc_corrs(y):

    y_sq = y.unstack('idx')
    rho = pd.DataFrame(np.corrcoef(y_sq), index=y_sq.index, columns=y_sq.index)
    results = {}
    for trial_type in ('LS', 'SL', 'SS'):
        rho_type = rho.xs(trial_type, axis=1, level='trial_type').xs(trial_type, level='trial_type')
        results[trial_type] = rho_type.mask(
            np.triu(np.ones(len(rho_type))).astype(bool))\
                .stack(future_stack=True).stack(future_stack=True).mean()
    return pd.Series(results)

def plot_pca(ax, pcs, dims=(1, 2)):
    def plot_trial(res, ax):
        trial_type = res.index.unique('trial_type')[0]
        color = TRIAL_COLORS[trial_type]
        jP.plot_seg_colors(
            ax, *res.values.T, c=color, clip_on=False)

    pcs.groupby(['trial_type', 'split']).apply(
        plot_trial, ax=ax)
    jP.configure_spines(ax, fix_ylabel=False)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal', anchor='W')


def main(argv):
    jP.set_rcParams(plt)
    dpi = 300
    path = Path('/analysis/ms_figures/ephys_dim_updated_12.6')
    path.mkdir(exist_ok=True, parents=True)
    sl_shaping_paths = list(map(
        Path,
         (
          '/data1/dua/sl_spike_data/M2_2202025_g0',
          '/data1/dua/sl_spike_data/M2_2212025_g0',
          '/data1/dua/sl_spike_data/M1_2182025_g0',
          '/data1/dua/sl_spike_data/M1_2202025_g0',
          '/data1/dua/sl_spike_data/M4_2252025_g0',
          '/data1/dua/sl_spike_data/M4_2272025_g1',
          '/data1/dua/sl_spike_data/M5_2272025_g0',
          '/data1/dua/sl_spike_data/M5_3012025_g0')))

    shaping_paths = list(map(
        Path, 
        ('/data1/jack/tDNMS_EC_project/AC01_ephys/AC01_01292025_g0',
         '/heys-nas-LabData/jack/neuropix_data/j2_05092024/j2_05092024_g0',
         '/heys-nas-LabData/tDNMS_EC_project/data/AB02_ephys/AB02_08272024_g0',
         '/data1/jack/tDNMS_EC_project/AB02_ephys/AB02_08282024_g0',
         '/data1/jack/tDNMS_EC_project/AB04_ephys/AB04_09092024_g0')))
    model_types = pd.Series(
        ['shaping']*len(shaping_paths) + ['sl_shaping']*len(sl_shaping_paths),
        name='model_type').reset_index()
    model_types.columns= ['model_id'] + list(model_types.columns[1:])
    all_paths = pd.Series(shaping_paths + sl_shaping_paths,
                          index=pd.MultiIndex.from_frame(model_types))

    spikes = all_paths.apply(lambda x: pd.read_hdf('spikes_updated.h5', key=x.parts[-1]))
    def fix_spikes(spikes):
        fixed = spikes.T
        fixed.index.names = fixed.index.names[:-1] + ['idx']
        return fixed
    spikes = spikes.apply(fix_spikes)

    def find_good(spikes):
        corrs = spikes.fillna(0).apply(calc_corrs)
        good_units = corrs[corrs > 0.25].dropna(how='all', axis=1).columns
        trimmed = spikes[good_units].groupby('trial').apply(lambda x: x.iloc[4:36]).droplevel(0)
        return trimmed

    good_spikes = spikes.apply(find_good)


    def add_splits(X):
        trials = X.drop('SU', level='trial_type', errors='ignore').index.unique('trial')

        n_splits = 4
        split_trials = []
        size = len(trials) // n_splits
        while len(split_trials) < n_splits:
            np.random.seed(0)
            split_trials.append(np.random.choice(trials, size, replace=False))
            trials = trials.difference(split_trials[-1])
   
        key = pd.DataFrame(np.array(split_trials)).stack().droplevel(1).reset_index().set_index(0)
        key.index.name = 'trial'
        idx_frame = X.index.to_frame()
        idx_frame['split'] = key.reindex(idx_frame['trial']).values
        res = X.copy()
        res.index = pd.MultiIndex.from_frame(idx_frame)
        return res
    better_spikes = good_spikes.apply(add_splits)
    avgs = better_spikes.apply(lambda x: x.groupby(['trial_type','split', 'idx']).mean())
    res_s = pd.concat(avgs.xs('shaping', level='model_type').values, axis=1, keys=avgs.index).fillna(0)
    res_sl = pd.concat(avgs.xs('sl_shaping', level='model_type').values, axis=1, keys=avgs.index).drop('SU', errors='ignore')
    np.random.seed(0)
    res_sl = res_sl[np.random.choice(res_sl.columns, res_s.shape[1], replace=False)].fillna(0)

    clf = PCA(n_components=3)
    clf.fit(res_s)
    pcs = pd.DataFrame(clf.transform(res_s), index=res_s.index)

    clf_sl = PCA(n_components=3)
    clf_sl.fit(res_sl)
    pcs_sl = pd.DataFrame(clf_sl.transform(res_sl), index=res_sl.index)

    margins = jP.default_margins()
    margins_adj = margins.copy()
    margins_adj['right'] = 100
    margins_adj['left'] = 45
    PdfPlotter(path / f'pca_3plots.pdf', fixed_margins=margins_adj)
    plt.figure(figsize=(2.25, 1.8), dpi=dpi)
    gs = gridspec.GridSpec(2, 2, wspace=0, hspace=0)
    axs = list(map(plt.subplot, gs))

    plot_pca(axs[1], pcs.drop(2, axis=1))
    plot_pca(axs[3], pcs.drop(1, axis=1))

    plot_pca(axs[0], pcs_sl.drop(2, axis=1))
    plot_pca(axs[2], pcs_sl.drop(1, axis=1))

    for ax in (axs[0], axs[2]):
        ax.spines['left'].set_position(('outward', -5))
        ax.spines['left'].set_bounds([-4, 4])
        ax.set_yticks([-4, 4])

    for ax in (axs[1], axs[3]):
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)

    for ax in (axs[0], axs[1]):
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)

    for ax in (axs[2], axs[3]):
        ax.spines['bottom'].set_bounds([-8, 8])
        ax.set_xticks([-8, 0, 8])
        ax.set_xlabel('PC 1')

    axs[0].set_ylabel('PC 2')
    axs[2].set_ylabel('PC 3')

    legend_lines = pcs.index.unique('trial_type').to_series().apply(
        lambda x: matplotlib.lines.Line2D(
            [0], [0], c=TRIAL_COLORS[x], lw=2, label=x))
    axs[1].legend(handles=list(legend_lines.values), loc='upper right', bbox_to_anchor=(1.6, 1.1))
    axs[1].set_title('Shaping + Full', color='tab:orange')
    axs[0].set_title('SL-Only Shaping', color='tab:green')

    ax = axs[3]
    inax_position = ax.transAxes.transform([0.95, 1.15])
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

    margins = jP.default_margins()
    margins_adj = margins.copy()
    margins_adj['right'] =200
    PdfPlotter(path / f'pca_3plots_horiz.pdf', fixed_margins=margins_adj)
    plt.figure(figsize=(5, 1.5), dpi=dpi)
    gs = gridspec.GridSpec(1, 4, wspace=0.2, hspace=0)
    axs = list(map(plt.subplot, gs))

    plot_pca(axs[2], pcs.drop(2, axis=1))
    plot_pca(axs[3], pcs.drop(1, axis=1))

    plot_pca(axs[0], pcs_sl.drop(2, axis=1))
    plot_pca(axs[1], pcs_sl.drop(1, axis=1))

    for ax in axs:
        ax.spines['left'].set_position(('outward', -5))
        ax.spines['left'].set_bounds([-4, 4])
        ax.set_yticks([-4, 4])

        ax.spines['bottom'].set_bounds([-8, 8])
        ax.set_xticks([-8, 0, 8])
        ax.set_xlabel('PC 1')

    axs[0].set_ylabel('PC 2')
    axs[1].set_ylabel('PC 3')
    axs[2].set_ylabel('PC 2')
    axs[3].set_ylabel('PC 3')

    legend_lines = pcs.index.unique('trial_type').to_series().apply(
        lambda x: matplotlib.lines.Line2D(
            [0], [0], c=TRIAL_COLORS[x], lw=2, label=x))
    axs[-1].legend(handles=list(legend_lines.values), loc='upper right', bbox_to_anchor=(1.6, 1.3))
    axs[1].text(1, 1.1, 'Shaping + Full Task', color='tab:orange', transform=axs[2].transAxes, fontsize=7, ha='center')
    axs[0].text(1, 1.1, 'SL-Only Shaping', color='tab:green', transform=axs[0].transAxes, fontsize=7, ha='center')

    ax = axs[-1]
    inax_position = ax.transAxes.transform([1, 0.5])
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

    margins = jP.default_margins()
    margins_adj = margins.copy()
    margins_adj['right'] = 250
    PdfPlotter(path / f'pca_3plots_sizeAdj.pdf', fixed_margins=margins_adj)
    plt.figure(figsize=(3.75, 2), dpi=dpi)
    gs = gridspec.GridSpec(2, 2, wspace=0, hspace=0)
    axs = list(map(plt.subplot, gs))

    plot_pca(axs[1], pcs.drop(2, axis=1))
    plot_pca(axs[3], pcs.drop(1, axis=1))

    plot_pca(axs[0], pcs_sl.drop(2, axis=1))
    plot_pca(axs[2], pcs_sl.drop(1, axis=1))

    for ax in axs:
        ax.set_ylim(-10, 10)
        ax.set_xlim(-12, 12)
    
    for ax in (axs[0], axs[2]):
        ax.spines['left'].set_position(('outward', -5))
        ax.spines['left'].set_bounds([-4, 4])
        ax.set_yticks([-4, 4])

    for ax in (axs[1], axs[3]):
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)

    for ax in (axs[0], axs[1]):
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)

    for ax in (axs[2], axs[3]):
        ax.spines['bottom'].set_bounds([-8, 8])
        ax.set_xticks([-8, 0, 8])
        ax.set_xlabel('PC 1')

    axs[0].set_ylabel('PC 2')
    axs[2].set_ylabel('PC 3')

    legend_lines = pcs.index.unique('trial_type').to_series().apply(
        lambda x: matplotlib.lines.Line2D(
            [0], [0], c=TRIAL_COLORS[x], lw=2, label=x))
    axs[1].legend(handles=list(legend_lines.values), loc='upper right', bbox_to_anchor=(1.6, 1.1))
    axs[1].set_title('Shaping + Full', color='tab:orange')
    axs[0].set_title('SL-Only Shaping', color='tab:green')

    ax = axs[3]
    inax_position = ax.transAxes.transform([0.88, 1.65])
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


    exp_var = res_s.groupby('split')\
        .apply(lambda x, clf=PCA(n_components=3): clf.fit(x).explained_variance_ratio_).apply(pd.Series)
    exp_var_sl = res_sl.groupby('split')\
        .apply(lambda x, clf=PCA(n_components=3): clf.fit(x).explained_variance_ratio_).apply(pd.Series)

    def box_plotter(X, i, ax, color, **kwargs):

        flierprops={'marker': '.', 'markersize': 2, 'mfc': 'k'}
        boxprops={'facecolor': color, 'edgecolor': 'k'}
        medianprops={'color': 'k'}
        ax.boxplot(
            [X], positions=[i], widths=0.5, flierprops=flierprops,
            boxprops=boxprops, patch_artist=True, medianprops=medianprops,
            **kwargs)


    margins = jP.default_margins()
    margins_adj = margins.copy()
    margins_adj['left'] = 130
    PdfPlotter(path / 'explained_varSizeAdj.pdf', fixed_margins=margins_adj)
    plt.figure(figsize=(1.25, 2), dpi=dpi)
    ax = plt.gca()
    box_plotter(exp_var.sum(1) * 100, 0, ax, 'tab:orange')
    box_plotter(exp_var_sl.sum(1) * 100, 1, ax, 'tab:green')
    jP.configure_spines(ax)
    ax.set_ylim(65, 95)
    ax.spines['left'].set_bounds(67, 95)
    ax.set_yticks([70, 80, 90])
    ax.spines['bottom'].set_bounds(0, 1)
    jP.percent_y(ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['S/FT', 'SL/FT'], ha='right', rotation=45, rotation_mode='anchor')
    ax.set_ylabel('Variance PC 1-3')
    pval = scipy.stats.ttest_ind(exp_var.sum(1), exp_var_sl.sum(1))
    sig = jP.significance_symbols([pval.pvalue])
    ylbl = max(exp_var.sum(1).max(), exp_var_sl.sum(1).max()) * 100 + jP.annotation_padding(ax, 0.08)
    jP.annotation(ax, (0, 1), ylbl, sig.values[0], va='bottom')
    plt.show()


    margins = jP.default_margins()
    margins_adj = margins.copy()
    margins_adj['left'] = 130
    PdfPlotter(path / 'explained_var.pdf', fixed_margins=margins_adj)
    plt.figure(figsize=(1.5, 1.5), dpi=dpi)
    ax = plt.gca()
    box_plotter(exp_var.sum(1) * 100, 0, ax, 'tab:orange')
    box_plotter(exp_var_sl.sum(1) * 100, 1, ax, 'tab:green')
    jP.configure_spines(ax)
    ax.set_ylim(65, 95)
    ax.spines['left'].set_bounds(67, 95)
    ax.set_yticks([70, 80, 90])
    ax.spines['bottom'].set_bounds(0, 1)
    jP.percent_y(ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['S/FT', 'SL/FT'], ha='right', rotation=45, rotation_mode='anchor')
    ax.set_ylabel('Variance PC 1-3')
    pval = scipy.stats.ttest_ind(exp_var.sum(1), exp_var_sl.sum(1))
    sig = jP.significance_symbols([pval.pvalue])
    ylbl = max(exp_var.sum(1).max(), exp_var_sl.sum(1).max()) * 100 + jP.annotation_padding(ax, 0.08)
    jP.annotation(ax, (0, 1), ylbl, sig.values[0], va='bottom')
    plt.show()

    margins = jP.default_margins()
    margins_adj = margins.copy()
    margins_adj['left'] = 130
    PdfPlotter(path / 'explained_var_horiz.pdf', fixed_margins=margins_adj)
    plt.figure(figsize=(2.5, 1.5), dpi=dpi)
    ax = plt.gca()
    box_plotter(exp_var.sum(1) * 100, 0, ax, 'tab:orange', vert=False)
    box_plotter(exp_var_sl.sum(1) * 100, 1, ax, 'tab:green', vert=False)
    jP.configure_spines(ax)
    ax.set_xlim(65, 95)
    ax.spines['bottom'].set_bounds(67, 95)
    ax.set_xticks([70, 80, 90])
    ax.spines['left'].set_bounds(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['SL/FT', 'S/FT'], ha='right', rotation=45, rotation_mode='anchor')
    ax.set_xlabel('Variance Explained PC 1-3')
    pval = scipy.stats.ttest_ind(exp_var.sum(1), exp_var_sl.sum(1))
    sig = jP.significance_symbols([pval.pvalue])
    ylbl = max(exp_var.sum(1).max(), exp_var_sl.sum(1).max()) * 100 + jP.annotation_padding(ax, 0.08, axis='x')
    jP.annotation_horiz(ax, (0, 1), ylbl, sig.values[0], ha='left')
    plt.show()



if __name__ == '__main__':
    main(sys.argv[1:])


