# [Figure S6d]
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import itertools as it
import statsmodels.formula.api as smf
from statsmodels.stats import multitest
import scipy.stats
import re
import itertools as it
import numpy as np
import pandas as pd


sys.path.append(f'../')
from classes.datagen import tDNMSGenerator, genFactory
from classes.models import LeakyRNN
from classes.models import LeakyRNNCell
from analysis_tools.mpl_helpers import PdfPlotter

from analysis_tools import jPlots as jP
from analysis_tools.progressbar import ProgressBar
import models_database as mdb


TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'S': '#f89521'}

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
    sorted_maxes = fixed_trial_maxes.apply(sorted)

    def calculate_cutoff(maxs):
        scores = []
        for i in range(1, len(maxs['out'])):
            out_corr = len(maxs['out']) - i + 1
            scores.append(
                np.sum(np.array(maxs['in']) >
                       maxs['out'][-i]) + out_corr)

        threshold = maxs['out'][-np.argmax(scores) - 1]
        score = np.max(scores)/(len(maxs['in']) + len(maxs['out']))
        if score == 1:
            threshold = (max(maxs['out']) + min(maxs['in'])) / 2
        return threshold, score

    def cutoff_helper(maxes, calculate_cutoff=calculate_cutoff):
        model_id = maxes.index.unique('model_id')[0]
        model_type = maxes.index.unique('model_type')[0]

        threshold, score = calculate_cutoff(maxes[model_id][model_type])
        return dict(model_id=model_id, model_type=model_type, threshold=threshold, score=score)

    cutoffs = pd.DataFrame(sorted_maxes.groupby('model_id').apply(cutoff_helper))[0]


    def run_model(model, X2, p=ProgressBar.nocount(), rnn_layer=0):
        p.increment()
        y = pd.DataFrame(model.layers[rnn_layer](np.expand_dims(X2[['light', 'odor']], 0))[0]) 
        predictions = pd.Series(model.layers[1](y.values)[:, 0], index=y.index)
        predictions.index = X2.index  
        return predictions


    def test_epoch(model, X2, cue_start, run_model=run_model):
        base_trial = X2.xs('LS', level='type').groupby('idx').head(1).droplevel(0)
        base_trial.loc[80:, 'cues'] = 0
        cue_end = cue_start + 10
        if cue_start < 80:
            base_trial.loc['cues'] = 0
        base_trial.loc[cue_start:cue_end, 'cues'] = 1

        base_trial = pd.concat([base_trial]*10, keys=pd.Index(np.arange(10), name='trial'))
        base_trial['odor'] = base_trial['cues'] + np.random.normal(scale=0, size=(len(base_trial),))
        pred = run_model(model, X2=base_trial)
        return pred.groupby('trial').max()

    test_epochs = [80, 85, 90, 95, 100, 105, 110]
    test_epochs = pd.Series(test_epochs, index=test_epochs)
    results = test_epochs.apply(
        lambda x,X2=X2, models=models: models.apply(test_epoch, X2=X2, cue_start=x))

    all_res = pd.concat(list(results.values), keys=list(results.index))
    per_error = (all_res.T > cutoffs.xs('threshold', level=1)\
                    .align(all_res, level='model_id')[0].values).mean(0) * 100
    per_error.index.names = ['epoch'] + per_error.index.names[1:]


    def box_plotter(X, i, colors, ax):
        c = next(colors)
        if c == 'tab:orange':
            c_light = '#ffbb78'
        else:
            c_light = '#aec7e8'
        flierprops={'marker': '.', 'markersize': 1.5, 'mfc': c, 'mec': c}
        boxprops={'facecolor': c_light, 'edgecolor': 'k'}
        medianprops={'color': c}
        ax.boxplot(
            [X], positions=[next(i)], widths=1.9, flierprops=flierprops,
            boxprops=boxprops, patch_artist=True, medianprops=medianprops)


    margins = jP.default_margins()
    PdfPlotter(path / '1s_boxplot.pdf', fixed_margins=margins)
    plt.figure(figsize=(2.5, 2.5), dpi=300)
    ax = plt.gca()
    per_error.xs('no_shaping', level='model_type')\
             .groupby('epoch').apply(
                box_plotter,
                iter((test_epochs-1).values), colors=it.repeat('tab:blue'), ax=ax)
    per_error.xs('shaping', level='model_type')\
             .groupby('epoch').apply(box_plotter, iter((test_epochs+1).values),
             colors=it.repeat('tab:orange'), ax=ax)
    ax.set_xticks([70, 80, 90, 100, 110, 120, 130])
    ax.set_xticklabels([7, 8, 9, 10, 11, 12, 13])
    ax.set_xlim(60, 130)
    ax.axvspan(30, 80, ymax=0.2, color=TRIAL_COLORS['LS'], alpha=0.25)
    ax.axvspan(110, 130, ymax=0.2, color='gray', alpha=0.25)
    ax.text(70, 0.7, 'L First Cue', color=TRIAL_COLORS['LS'],
            transform=ax.get_xaxis_transform(), fontsize=5, rotation=90,
            ha='center', va='center')
    ax.text(120, 0.7, 'Second Cue\nOmitted for Test',
            transform=ax.get_xaxis_transform(), fontsize=5, rotation=90,
            ha='center', va='center')
    jP.configure_spines(ax)
    jP.percent_y(ax)
    ax.set_yticks([20, 60, 100])
    ax.spines['left'].set_bounds(-1, 100)
    jP.annotation(ax, (79, 96), 105, '***')
    jP.annotation(ax, (100, 100), 105, '**')
    ax.set_ylim(-1, 115)
    ax.set_ylabel('Response to 1s Cue')
    ax.set_xlabel('Input Start Time (s)')
    plt.show()

    X = per_error.copy()
    X.name = 'values'
    X = X.reset_index()

    # Fit full model (ML) for likelihood-ratio test
    model = smf.mixedlm(
        f"values ~ C(epoch)*C(model_type)", X, groups='model_id'
        ).fit(reml=False)
    model_null = smf.mixedlm("values ~ C(epoch)", X, groups='model_id').fit(reml=False)

    # Likelihood-ratio test against intercept-only model
    LR = 2 * (model.llf - model_null.llf)
    df = len(model.params) - len(model_null.params)
    p_value = scipy.stats.chi2.sf(LR, df=df)
    print(f'Effect of model_type: {p_value}')
    p_values = per_error.groupby(['epoch','model_type']).apply(list)\
                        .groupby('epoch').apply(
                            lambda x, scipy=scipy: scipy.stats.ttest_ind(*x)
                        ).apply(pd.Series, index=('t', 'pvalue'))
    
    print(p_values)
    print(p_values['pvalue'].apply(lambda x: f'{x * 7:.4f}'))


if __name__ == '__main__':
    main(sys.argv[1:])
