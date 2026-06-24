# [Figure S1]
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import itertools as it

sys.path.append('../')
from classes.datagen import tDNMSGenerator, genFactory
from classes.models import LeakyRNN
from classes.models import LeakyRNNCell
from analysis_tools.mpl_helpers import PdfPlotter

from analysis_tools import jPlots as jP
from analysis_tools.progressbar import ProgressBar
import models_database as mdb

def main(args):
    jP.set_rcParams(plt)
    dpi = 300
    path = Path('/analysis/ms_figures/errors')
    path.mkdir(exist_ok=True, parents=True)
    rnn_layer = 0

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

    jP.set_rcParams(plt)
    task = 'just_short_match'

    input_noise = model_infos.head(1)['input_noise']
    trials_gen = genFactory.create(
        task, input_noise=input_noise, batch_size=1, n_blocks=1)

    X2 = trials_gen.generate_trials(15)
    models = model_infos['path'].apply(tf.keras.models.load_model)

    time_idx = pd.to_timedelta(X2.index.get_level_values('idx')/10, unit='s')
    idx = X2.index.to_frame()
    idx['time'] = time_idx

    def calc_predictions(model, X2, idx, p=ProgressBar.nocount()):
        p.increment()
        y = pd.DataFrame(model.layers[rnn_layer](np.expand_dims(X2[['light', 'odor']], 0))[0]) 
        predictions = pd.Series(model.layers[1](y.values)[:, 0], index=y.index)
        predictions.index = pd.MultiIndex.from_frame(idx)  
        return predictions

    p = ProgressBar(len(models), annotation='Calculating Predictions')
    predictions = models.apply(calc_predictions, X2=X2, idx=idx, p=p).T

    def calc_trial_max(X):
        trial_type = X.index.unique('type')
        if trial_type == 'SS':
            return pd.DataFrame([X.max(), None], index=('out', 'in'))
        elif trial_type == 'LS':
            adj_x = X.droplevel([0, 1, 2])
            max_out = max(adj_x.loc['0s':'11s'].max(), adj_x.loc['16s':'20s'].max())
            max_in = adj_x.loc['13s':'15s'].max()
            return  pd.Series([max_out, max_in], index=('out', 'in'))
        elif trial_type == 'SL':
            adj_x = X.droplevel([0, 1, 2])
            max_out = max(adj_x.loc['0s':'8s'].max(), adj_x.loc['16s':'20s'].max())
            max_in = adj_x.loc['13s':'15s'].max()
            return pd.Series([max_out, max_in], index=('out', 'in'))

    trial_maxs = predictions.stack().stack()\
                            .groupby(['model_id', 'model_type', 'trial']).apply(calc_trial_max)[0].unstack()
    maxes_list = trial_maxs.groupby(['model_id', 'model_type'])\
                           .apply(lambda x: pd.DataFrame(x.values.T, index=x.keys())).apply(list, axis=1) 
    fixed_trial_maxes = maxes_list.apply(lambda x: [a for a in x if np.isfinite(a)])

    step_size = 1
    margins = jP.default_margins()
    margins['bottom'] = 120
    PdfPlotter(path / 'boxplot.pdf', fixed_margins=margins)
    plt.figure(figsize=(6.5, 2), dpi=dpi)

    def violin_plot(X, ctr, ax, color, p=ProgressBar.nocount()):
        p.increment()
        offset = 0.4
        i = next(ctr)
        parts = ax.violinplot([X['out'], X['in']], positions=[i+0, i+offset], widths=0.25)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
        parts['cmaxes'].set_edgecolor(color)
        parts['cmins'].set_edgecolor(color)
        parts['cbars'].set_edgecolor(color)

        if max(X['out']) > min(X['in']):
            x_display = (ax.transData.transform((i-0.125, 0))[0],
                         ax.transData.transform((i+0.375, 0))[0])
            x_vals = (ax.transAxes.inverted().transform((x_display[0], 0))[0],
                      ax.transAxes.inverted().transform((x_display[1], 0))[0])
            ax.axhspan(min(X['in']), max(X['out']), xmin=x_vals[0], xmax=x_vals[1], color='red', alpha=0.15, zorder=-1e3)
        ax.set_xticks(list(ax.get_xticks()) + [i, i+offset])
    
    p = ProgressBar(len(model_ids))
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlim(-0.25, len(fixed_trial_maxes)/2 * step_size + 0.5)
    ctr = it.count(step=step_size)
    fixed_trial_maxes.xs('no_shaping', level='model_type', drop_level=False)\
                     .unstack().apply(violin_plot, ctr=ctr, ax=ax, axis=1,
                                      color='tab:blue', p=p)
    next(ctr)
    fixed_trial_maxes.xs('shaping', level='model_type', drop_level=False)\
                     .unstack().apply(violin_plot, ctr=ctr, ax=ax, axis=1,
                                      color='tab:orange', p=p)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(
        [['Error', 'Response'][i%2] for i, _ in enumerate(ax.get_xticks())],
        rotation=35, rotation_mode='anchor', ha='right')
    ax.set_ylabel('Trial Response Magnitude')
    jP.configure_spines(ax)

    bounds = ax.get_xlim()
    ax.text(np.diff(bounds)[0]/4, 1, 'No Shaping', color='tab:blue', transform=ax.transData, ha='center')
    ax.text(np.diff(bounds)[0] * 3/4, 1, 'Shaping + Full Task', color='tab:orange', transform=ax.transData, ha='center')
    plt.show()

    step_size = 1
    margins = jP.default_margins()
    margins['bottom'] = 120
    PdfPlotter(path / 'boxplot_7.pdf', fixed_margins=margins)
    plt.figure(figsize=(7, 2), dpi=dpi)
   
    p = ProgressBar(len(model_ids))
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlim(-0.25, len(fixed_trial_maxes)/2 * step_size + 0.5)
    ctr = it.count(step=step_size)
    fixed_trial_maxes.xs('no_shaping', level='model_type', drop_level=False)\
                     .unstack().apply(violin_plot, ctr=ctr, ax=ax, axis=1,
                                      color='tab:blue', p=p)
    next(ctr)
    fixed_trial_maxes.xs('shaping', level='model_type', drop_level=False)\
                     .unstack().apply(violin_plot, ctr=ctr, ax=ax, axis=1,
                                      color='tab:orange', p=p)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(
        [['Error', 'Response'][i%2] for i, _ in enumerate(ax.get_xticks())],
        rotation=35, rotation_mode='anchor', ha='right')
    ax.set_ylabel('Trial Response Magnitude')
    jP.configure_spines(ax)

    bounds = ax.get_xlim()
    ax.text(np.diff(bounds)[0]/4, 1, 'No Shaping', color='tab:blue', transform=ax.transData, ha='center')
    ax.text(np.diff(bounds)[0] * 3/4, 1, 'Shaping + Full Task', color='tab:orange', transform=ax.transData, ha='center')
    plt.show()



if __name__ == '__main__':
    main(sys.argv[1:])
