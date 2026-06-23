# [Figure 1d,f]
import sys
sys.path.append('../')

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path
import itertools as it
import tensorflow as tf
import numpy as np

from analysis_tools.mpl_helpers import PdfPlotter
from analysis_tools.progressbar import ProgressBar
import analysis_tools.jPlots as jP
import models_database as mdb

from classes.datagen import tDNMSGenerator, genFactory
from classes.models import LeakyRNN
from classes.models import LeakyRNNCell


plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  
    "axes.facecolor":    (1.0, 1.0, 1.0, 0.0),  
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0)})

TRIAL_COLORS = {
    'LS': '#eb0d8c',
    'SL': '#2bace2',
    'SS': '#f89521',
    'LL': '#2b958c',
    'S' : '#f89521'}

def plot_gen(counts, size=(2.2, 1.25), dpi=300):
    plt.figure(figsize=size, dpi=dpi)
    gs = iter(gridspec.GridSpec(1, 3, wspace=0.3, hspace=0.2))
    while True:
        gss = next(gs).subgridspec(len(counts) + 1, 1, height_ratios=[25] + counts.values.tolist())

        yield list(map(plt.subplot, gss))

def plot_unit(axs, y):

    def plot_trial_type_hm(y, axs_dict):
        trial_type = y.index.unique('type')[0]
        ax = axs_dict[trial_type]
        tc = y.groupby('idx').mean()
        axs_dict['tuning_curves'].plot(tc.index.get_level_values('idx')/10, tc, c=TRIAL_COLORS[trial_type])
        axs_dict['tuning_curves'].set_xlim(0, 20)

        rate_maps = y
        rate_maps = rate_maps.unstack('idx')
        rate_maps = rate_maps.sub(rate_maps.mean(axis=1), axis=0).div(rate_maps.std(axis=1), axis=0)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap', ('#ffffff', TRIAL_COLORS[trial_type]))
        axs_dict[trial_type].imshow(rate_maps, vmin=0, vmax=2, aspect='auto', cmap=cmap, extent=[0, 20, 0, len(y)])

    axs_dict = {k: v for (k, v) in
                zip(['tuning_curves'] + ['SS', 'LS', 'SL'], axs)}
    y.groupby('type').apply(
        plot_trial_type_hm, axs_dict=axs_dict)
    list(map(lambda x: x.spines['bottom'].set_visible(False), axs[:-1]))
    list(map(lambda x: x.spines['top'].set_visible(False), axs[2:]))
    list(map(lambda x: x.set_xticks([]), axs[:-1]))
    jP.configure_spines(axs[0])


def main(argv):
    model_id = 22
    model_info = mdb.get_model(model_id)
    model_path = model_info['path']
    model_attrs = mdb.get_model_attributes(model_id)

    path = Path('/analysis/ms_figures/example_units')
    jP.make_folder(path)
    jP.set_rcParams(plt)

    margins={'left': 80, 'right': 90, 'top': 45, 'bottom': 80}
    dpi = 300
    input_noise = model_attrs['input_noise']
    rnn_layer = 0

    model = tf.keras.models.load_model(model_path)

    trials_gen = genFactory.create(
        model_info['task_name'], input_noise=input_noise, batch_size=1, n_blocks=1)
    X2 = trials_gen.generate_trials(25)
    formatted_X = tDNMSGenerator.format_validation(X2)[0]
    res = model.predict(formatted_X).squeeze()
    predictions = pd.Series(res, index=X2.index)

    y = pd.DataFrame(model.layers[rnn_layer](np.expand_dims(X2[['light', 'odor']], 0))[0])
    y.index = X2.index
    cues = X2.groupby('type').apply(
        lambda x: x.loc[x.index.unique('trial')[0]]
        )[['trial_start', 'cues', 'response']].droplevel(0)
    n_types = len(cues.index.get_level_values('type').unique()) 

    plot_order = ['SL', 'LS', 'SS']
    unit_ids = [80, 98, 1]

    margins={'left': 80, 'right': 90, 'top': 45, 'bottom': 80}
    PdfPlotter(path / 'example_units_tdnms.pdf', fixed_margins=margins)
    plot_data = y[unit_ids].reindex(plot_order, level='type')
    counts = plot_data.groupby('trial').head(1).index.to_frame('type')['type'].value_counts()

    axs = plot_gen(counts, dpi=dpi)
    p = ProgressBar(len(unit_ids))
    for i, unit in enumerate(unit_ids):
        ax = next(axs)
        ax[0].set_title(f'Unit {i + 1}', pad=-1)
        p.increment()
        plot_unit(ax, y[unit])
        
        list(map(lambda x: x.set_yticks([]), ax))
        if i == 0:
            ax[0].set_ylabel('avg')
            jP.set_ylabel_position(ax[0], nlines=2.4)
            ax[2].set_ylabel('Trials')

        ax[-1].set_xlabel('Time (s)')
        x_transform = matplotlib.transforms.blended_transform_factory(
            ax[-1].transAxes, ax[-1].figure.dpi_scale_trans)
        ax[-1].xaxis.set_label_coords(
            0.5, 0.1, transform=x_transform)

        ax[0].set_axis_on()


    ax = ax[-1]
    inax_position = ax.transAxes.transform([0.95, 0.9]) 
    infig_position = ax.figure.transFigure.inverted().transform(inax_position)
    color_scale = ax.figure.add_axes(
            list(infig_position) +
            [ax.get_position().width * 0.03, ax.get_position().height * 1.1]) 
    color_scale.imshow(
            np.array([np.linspace(0.25, 1, 30)[::-1]]).T, cmap='Grays', aspect='auto', vmin=0, vmax=1)
    color_scale.set_axis_off() 
    color_scale.text(  
               1, 0, '0', ha='left', va='center', fontsize=5,   
               transform=color_scale.transAxes)
    color_scale.text( 
               1, 1, '2', ha='left', va='center', fontsize=5, 
                transform=color_scale.transAxes)
    color_scale.text( 
                4, 0.5, 'z-score\nrate', ha='left', va='center', fontsize=5, 
                transform=color_scale.transAxes, rotation=-90, ma='center')
    plt.show()


    margins={'left': 80, 'right': 90, 'top': 45, 'bottom': 80}
    PdfPlotter(path / 'example_units_tdnms_smaller.pdf', fixed_margins=margins)
    plot_data = y[unit_ids].reindex(plot_order, level='type')
    counts = plot_data.groupby('trial').head(1).index.to_frame('type')['type'].value_counts()

    axs = plot_gen(counts, dpi=dpi, size=(2, 1))
    p = ProgressBar(len(unit_ids))
    for i, unit in enumerate(unit_ids):
        ax = next(axs)
        ax[0].set_title(f'Unit {i + 1}', pad=-1)
        p.increment()
        plot_unit(ax, y[unit])
        
        list(map(lambda x: x.set_yticks([]), ax))
        if i == 0:
            ax[0].set_ylabel('avg')
            jP.set_ylabel_position(ax[0], nlines=2.4)
            ax[2].set_ylabel('Trials')

        ax[-1].set_xlabel('Time (s)')
        x_transform = matplotlib.transforms.blended_transform_factory(
            ax[-1].transAxes, ax[-1].figure.dpi_scale_trans)
        ax[-1].xaxis.set_label_coords(
            0.5, 0.1, transform=x_transform)

        ax[0].set_axis_on()


    ax = ax[-1]
    inax_position = ax.transAxes.transform([0.95, 0.9]) 
    infig_position = ax.figure.transFigure.inverted().transform(inax_position)
    color_scale = ax.figure.add_axes(
            list(infig_position) +
            [ax.get_position().width * 0.03, ax.get_position().height * 1.1]) 
    color_scale.imshow(
            np.array([np.linspace(0.25, 1, 30)[::-1]]).T, cmap='Grays', aspect='auto', vmin=0, vmax=1)
    color_scale.set_axis_off() 
    color_scale.text(  
               1, 0, '0', ha='left', va='center', fontsize=5,   
               transform=color_scale.transAxes)
    color_scale.text( 
               1, 1, '2', ha='left', va='center', fontsize=5, 
                transform=color_scale.transAxes)
    color_scale.text( 
                4, 0.5, 'z-score\nrate', ha='left', va='center', fontsize=5, 
                transform=color_scale.transAxes, rotation=-90, ma='center')
    plt.show()
   
    PdfPlotter(path / 'all_units.pdf', fixed_margins=margins)
    plt.figure(figsize=(2,1.7), dpi=dpi)
    gs = gridspec.GridSpec(1, 3)
    axs = list(map(plt.subplot, gs))

    def plot(y, axs_iter, sort=None):
        trial_type = y.index.unique('type')[0]
        if sort is None:
            sort = y.idxmax(axis=0).sort_values().index
        ax = next(axs_iter)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap', ('#ffffff', TRIAL_COLORS[trial_type]))
        ax.imshow(y.T.reindex(sort), cmap=cmap, aspect='auto', extent=(0, 20, 128, 0), interpolation='nearest')
        ax.set_title(trial_type, pad=5)

    sort =  y.xs('LS', level='type').groupby('idx').mean().idxmax(0).sort_values().index
    y.groupby(['type', 'idx']).mean().groupby('type').apply(plot, axs_iter=iter(axs), sort=sort)
    [v.set_linewidth(2) for k, v in axs[0].spines.items()]
    [v.set_color('#B200ED') for k, v in axs[0].spines.items()]
    axs[0].set_yticks([20, 120])
    axs[0].set_ylabel('RNN Unit #')
    y_transform = matplotlib.transforms.blended_transform_factory(
        axs[0].figure.dpi_scale_trans, axs[0].transAxes)
    axs[0].yaxis.set_label_coords(
        0.15, 0.5, transform=y_transform)

    axs[1].set_yticks([])
    axs[1].set_xlabel('Time (s)')
    x_transform = matplotlib.transforms.blended_transform_factory(
        axs[1].transAxes, axs[1].figure.dpi_scale_trans)
    axs[1].xaxis.set_label_coords( 
        0.5, 0.11, transform=x_transform) 

    axs[2].set_yticks([])
    list(map(lambda x: x.set_xticks([5, 15]), axs))

    plt.show()


if __name__ == '__main__':
     main(sys.argv[1:])
