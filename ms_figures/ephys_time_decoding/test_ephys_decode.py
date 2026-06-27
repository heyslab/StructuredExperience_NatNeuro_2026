# [Figure 6f,g,h, S9f,g]
import os
import sys

sys.path.append('../')

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

def main(argv):

    jP.set_rcParams(plt)
    dpi = 300

    sl_shaping_paths = list(map(
        Path,
         ('/data1/dua/sl_spike_data/M2_2202025_g0',
          '/data1/dua/sl_spike_data/M2_2212025_g0',
          '/data1/dua/sl_spike_data/M1_2182025_g0',
          '/data1/dua/sl_spike_data/M1_2202025_g0',
          '/data1/dua/sl_spike_data/M4_2252025_g0',
          '/data1/dua/sl_spike_data/M4_2272025_g1',
          '/data1/dua/sl_spike_data/M5_2272025_g0',
          '/data1/dua/sl_spike_data/M5_3012025_g0')))

    shaping_paths = list(map(
        Path, 
        ('/data3/jack/tDNMS_EC_project/AC01_ephys/AC01_01292025_g0',
         '/heys-nas-LabData/jack/neuropix_data/j2_05092024/j2_05092024_g0',
         '/data3/jack/tDNMS_EC_project/AB02_ephys/AB02_08272024_g0',
         '/data3/jack/tDNMS_EC_project/AB02_ephys/AB02_08282024_g0',
         '/data3/jack/tDNMS_EC_project/AB04_ephys/AB04_09092024_g0')))
    model_types = pd.Series(
        ['shaping']*len(shaping_paths) + ['sl_shaping']*len(sl_shaping_paths),
        name='model_type').reset_index()
    model_types.columns= ['model_id'] + list(model_types.columns[1:])
    all_paths = pd.Series(shaping_paths + sl_shaping_paths,
                          index=pd.MultiIndex.from_frame(model_types))
    predictions = (all_paths / 'analysis/time_decoding.h5')\
                      .apply(pd.read_hdf, key='model_0')
    predictions = pd.concat(
        predictions.values, keys=pd.MultiIndex.from_tuples(predictions.keys(),
        names=predictions.index.names))

    predictions = predictions * 0.5
    idx = predictions.index.to_frame()
    idx['time_bin'] *= 0.5
    predictions.index = pd.MultiIndex.from_frame(idx)

    def calc_error(predictions):
        circ_error = lambda a, b, n: np.min(
            np.array(list(map(
                lambda x, y: (x-y),
                it.repeat(a), (b, b+n, b-n)))) ** 2)
        err = predictions.groupby('time_bin').apply(
            lambda x, circ_error=circ_error: x.map(
                lambda a, circ_error=circ_error: circ_error(
                    a, x.index.get_level_values('time_bin'), 20)))
        squared_error = err.droplevel(0).sort_index()
        squared_error.columns.name = 'decode_type'

        return squared_error
    sq_err = calc_error(predictions)

    x = sq_err.groupby(['model_type', 'type']).mean().stack().xs('SL', level='type')
    print(x.unstack().T.div(x.unstack().T.xs('SL')))
    x = sq_err.groupby(['model_type', 'type']).mean().stack().xs('LS', level='type')
    print(x.unstack().T.div(x.unstack().T.xs('LS')))
    x = sq_err.groupby(['model_type', 'type']).mean().stack().xs('SS', level='type')
    print(x.unstack().T.div(x.unstack().T.xs('SS')))

    dpi = 300
    path = Path('/analysis/ms_figures/ephys')
    path.mkdir(exist_ok=True, parents=True)
    info_file = path / 'test_ephys_decode.info.txt'
    with open(info_file, 'w') as f:
        pass

    def plot_cm(ax, predictions, trial_type):
        true = predictions.index.get_level_values('time_bin')
        cm = metrics.confusion_matrix((true*2).astype(int), (predictions*2).astype(int))
        cm = cm / np.sum(cm, axis=1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'cmap', ('#000000', TRIAL_COLORS[trial_type]))
        im = ax.imshow(cm, vmin=0, vmax=0.2, cmap=cmap, extent=(0, 20, 0, 20))
        ax.set_title(trial_type, color=TRIAL_COLORS[trial_type])
        ax.set_xticks([5,  15])
        ax.set_yticks([5, 10, 15])
        arrowprops = dict(arrowstyle='-', color='w', shrinkA=0, shrinkB=0, lw=0.25, linestyle=(0, (5, 15)))
        ax.annotate('', (0, 1), (1, 0), xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=arrowprops)

    for model_type in ('sl_shaping', 'shaping'):
        margins = jP.default_margins()
        margins['left'] = 120
        margins['right'] = 170
  
        PdfPlotter(path / f'example_decoding_{model_type}.SL.pdf', fixed_margins=margins)
        plt.figure(figsize=(3.2, 1.25), dpi=dpi)
        gs = gridspec.GridSpec(1, 3)
        axs = list(map(plt.subplot, gs))
        for i, trial_type in enumerate(('SL', 'LS', 'SS')):
            plot_cm(
                axs[i],
                predictions.xs(model_type, level='model_type')\
                    .xs(trial_type, level='type')['SL'],
                trial_type)
        
        def format_axis(ax):
            [ax.spines[s].set_color(TRIAL_COLORS['SL']) for s in ax.spines]
        list(map(format_axis, axs))

        [axs[0].spines[s].set_linewidth(2) for s in axs[0].spines]
        [ax.set_yticks([]) for ax in axs[1:]]
        axs[1].set_xlabel('Predicted Times (s)')
        axs[0].set_ylabel('Actual Time (s)')
        axs[-1].text(
            1, 1.125,  'Trained on SL trials only', color=TRIAL_COLORS['SL'],
            fontsize=5, transform=axs[-1].transAxes, ha='right', va='top')

        ax = axs[-1]
        inax_position = ax.transAxes.transform([0.8, 0.2]) 
        infig_position = ax.figure.transFigure.inverted().transform(inax_position)
        color_scale = ax.figure.add_axes(
                list(infig_position) +
                [ax.get_position().width * 0.09, ax.get_position().height * 0.87]) 
        color_scale.imshow(
                np.array([np.linspace(0.15, 1, 50)]).T, cmap='Grays', aspect='auto', vmin=0, vmax=1)
        color_scale.set_xticks([])
        color_scale.yaxis.tick_right()
        color_scale.set_yticks([0, 50])
        color_scale.set_yticklabels(['>20%', '0'], fontsize=5)
        color_scale.set_ylabel('Probability', labelpad=-8, rotation=-90, fontsize=5)
        color_scale.yaxis.set_label_position("right")      
     
        plt.show()

        PdfPlotter(path / f'example_decoding_{model_type}.LS.pdf', fixed_margins=margins)
        plt.figure(figsize=(3.2, 1.25), dpi=dpi)
        gs = gridspec.GridSpec(1, 3)
        axs = list(map(plt.subplot, gs))
        for i, trial_type in enumerate(('LS', 'SL', 'SS')):
            plot_cm(
                axs[i],
                predictions.xs(model_type, level='model_type')\
                    .xs(trial_type, level='type')['LS'],
                trial_type)
        
        def format_axis(ax):
            [ax.spines[s].set_color(TRIAL_COLORS['LS']) for s in ax.spines]
        list(map(format_axis, axs))

        [axs[0].spines[s].set_linewidth(2) for s in axs[0].spines]
        [ax.set_yticks([]) for ax in axs[1:]]
        axs[1].set_xlabel('Predicted Times (s)')
        axs[0].set_ylabel('Actual Time (s)')
        axs[-1].text(
            1, 1.125,  'Trained on LS trials only', color=TRIAL_COLORS['LS'],
            fontsize=5, transform=axs[-1].transAxes, ha='right', va='top')

        ax = axs[-1]
        inax_position = ax.transAxes.transform([0.8, 0.2]) 
        infig_position = ax.figure.transFigure.inverted().transform(inax_position)
        color_scale = ax.figure.add_axes(
                list(infig_position) +
                [ax.get_position().width * 0.09, ax.get_position().height * 0.87]) 
        color_scale.imshow(
                np.array([np.linspace(0.15, 1, 50)]).T, cmap='Grays', aspect='auto', vmin=0, vmax=1)
        color_scale.set_xticks([])
        color_scale.yaxis.tick_right()
        color_scale.set_yticks([0, 50])
        color_scale.set_yticklabels(['>20%', '0'], fontsize=5)
        color_scale.set_ylabel('Probability', labelpad=-8, rotation=-90, fontsize=5)
        color_scale.yaxis.set_label_position("right")      
     
        plt.show()

    mean_mse = sq_err.groupby(['model_type', 'type', 'model_id']).mean()
    error_list = mean_mse.drop('all', axis=1).stack()
    error_idx = error_list.index.to_frame()
    error_idx['same'] = error_idx['type'] == error_idx['decode_type']
    error_list.index = pd.MultiIndex.from_frame(error_idx.drop('type', axis=1).drop('decode_type', axis=1))

    x = error_list.groupby(error_list.index.names).mean().unstack().diff(axis=1).abs()[True]

    margins={'left': 90, 'right': 45, 'top': 100, 'bottom': 80}
    margins=jP.default_margins()
    margins['bottom'] = 150
    PdfPlotter(path / f'mse_bar.pdf',
               fixed_margins=margins)
    plt.figure(figsize=(1.5, 2), dpi=dpi)
    ax = plt.subplot()

    def box_plotter(X, i, colors, ax):
        flierprops={'marker': '.', 'markersize': 2, 'mfc': 'k'}
        boxprops={'facecolor': next(colors), 'edgecolor': 'k'}
        medianprops={'color': 'k'}
        ax.boxplot(
            [X], positions=[next(i)], widths=0.5, flierprops=flierprops,
            boxprops=boxprops, patch_artist=True, medianprops=medianprops)

    error_groups = x.groupby(['model_type']).apply(list)
    error_groups = error_groups.reindex(
        ['shaping', 'sl_shaping'], level='model_type')
    error_groups.apply(box_plotter, i=iter([1, 2]), colors=iter(['tab:orange', 'tab:green']), ax=ax)

    pvalues = error_list.groupby(
        ['same', 'model_type']).apply(list).groupby('same').apply(
            lambda x: scipy.stats.ttest_ind(*x))
    jP.configure_spines(ax)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Shaping', 'SL Shaping'], rotation_mode='anchor', rotation=45, ha='right')
    ax.set_ylabel(r'$\Delta$ (Between$-$Within Context)'+'\nTime Decoding MSE')
    jP.set_ylabel_position(ax, nlines=2.5)
    ax.set_yticks([5, 10, 15])
    ax.set_ylim(0, 15)

    pval = scipy.stats.ttest_ind(x['sl_shaping'], x['shaping']).pvalue
    sigs = jP.significance_symbols(pd.Series([pval]))
    lbl_y = x.max() + jP.annotation_padding(ax, 0.08)
    jP.annotation(ax, (1, 2), lbl_y, sigs[0], va='bottom')


    #ax.legend(handles=([
    #    matplotlib.patches.Patch(color='tab:orange', label='LS Shaping'),
    #    matplotlib.patches.Patch(color='tab:green', label='Shaping')]),
    #    loc='upper left', bbox_to_anchor=(0, 1.05))

    plt.show()

    margins=jP.default_margins()
    margins['bottom'] = 150
    margins['top'] = 30
    PdfPlotter(path / f'mse_bar.horiz_small.pdf',
               fixed_margins=margins)
    plt.figure(figsize=(1.8, 1), dpi=dpi)
    ax = plt.subplot()

    def box_plotter(X, i, colors, ax):
        flierprops={'marker': '.', 'markersize': 2, 'mfc': 'k'}
        boxprops={'facecolor': next(colors), 'edgecolor': 'k'}
        medianprops={'color': 'k'}
        ax.boxplot(
            [X], positions=[next(i)], widths=0.5, flierprops=flierprops,
            boxprops=boxprops, patch_artist=True, medianprops=medianprops, vert=False)

    def box_plotter(X, i, colors, ax):
        c = next(colors)
        c_hsv = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(c))
        c1 = matplotlib.colors.hsv_to_rgb(c_hsv * [1, 0.5, 1])    
        c3 = matplotlib.colors.hsv_to_rgb(c_hsv * [1, 1, 0.25])    
        flierprops={'marker': '.', 'markersize': 2, 'mfc': 'k'}
        boxprops={'facecolor': c1, 'edgecolor': 'k'}
        pos = next(i)
        medianprops={'color': 'k'}
        ax.boxplot(
            [X], positions=[pos], widths=0.5, showfliers=False,
            boxprops=boxprops, patch_artist=True, medianprops=medianprops, vert=False)
        wiggle = np.linspace(-0.15, 0.15, len(X))
        ax.plot(X, wiggle + pos, ls='', c=c, marker='o', ms=2, zorder=1e10, mec=c3, mew=0.25)

    error_groups = x.groupby(['model_type']).apply(list)
    error_groups = error_groups.reindex(
        ['shaping', 'sl_shaping'], level='model_type')
    error_groups.apply(box_plotter, i=iter([1, 2]), colors=iter(['tab:orange', 'tab:green']), ax=ax)

    pvalues = error_list.groupby(
        ['same', 'model_type']).apply(list).groupby('same').apply(
            lambda x: scipy.stats.ttest_ind(*x))
    jP.configure_spines(ax, fix_xlabel=False)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['LS+SL', 'SL Only'])
    ax.set_xlabel(r'$\Delta$ (Between$-$Within Context)'+'\nTime Decoding MSE')
    ax.set_xticks([5, 10, 15])
    ax.set_xlim(0, 15)

    pval = scipy.stats.ttest_ind(x['sl_shaping'], x['shaping'])
    sigs = jP.significance_symbols(pd.Series([pval.pvalue]))
    lbl_y = x.max() + jP.annotation_padding(ax, 0.08, axis='x')
    jP.annotation_horiz(ax, (1, 2), lbl_y, sigs[0], ha='left')
    plt.show()

    margins=jP.default_margins()
    margins['bottom'] = 150
    margins['top'] = 30
    PdfPlotter(path / f'mse_bar_2.3.pdf',
               fixed_margins=margins)
    plt.figure(figsize=(2.3, 1), dpi=dpi)
    ax = plt.subplot()

    error_groups = x.groupby(['model_type']).apply(list)
    error_groups = error_groups.reindex(
        ['shaping', 'sl_shaping'], level='model_type')
    error_groups.apply(box_plotter, i=iter([1, 2]), colors=iter(['tab:orange', 'tab:green']), ax=ax)

    pvalues = error_list.groupby(
        ['same', 'model_type']).apply(list).groupby('same').apply(
            lambda x: scipy.stats.ttest_ind(*x))
    jP.configure_spines(ax, fix_xlabel=False)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['LS+SL', 'SL Only'])
    ax.set_xlabel(r'$\Delta$ (Between$-$Within Context)'+'\nTime Decoding MSE')
    ax.set_xticks([5, 10, 15])
    ax.set_xlim(0, 15)

    pval = scipy.stats.ttest_ind(x['sl_shaping'], x['shaping'])
    sigs = jP.significance_symbols(pd.Series([pval.pvalue]))
    lbl_y = x.max() + jP.annotation_padding(ax, 0.08, axis='x')
    jP.annotation_horiz(ax, (1, 2), lbl_y, sigs[0], ha='left')
    plt.show()

    print(pval)

    with open(info_file, 'a') as f:
        f.write(str(pval))


if __name__ == '__main__':
    main(sys.argv[1:])

