# [Figure 5h]
import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import itertools as it

from analysis_tools.mpl_helpers import PdfPlotter
import analysis_tools.jPlots as jP
import models_database as mdb

from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols

SHAPING_COLORS = {
    'no_shaping': 'tab:blue',
    'shaping': 'tab:orange',
    'sl_only': 'tab:green',
    'ls_only': 'tab:red'}

def main(argv):
    no_model_ids = [9, 18, 19, 24, 28, 29, 31, 32, 93, 109, 116, 119]
    with_model_ids = [20, 22, 25, 21, 23, 26, 27, 30, 76, 79, 82, 84]
    ls_model_ids = [62, 66, 67, 70, 63, 65, 68, 71, 75, 78, 81]
    sl_model_ids = [41, 47, 42, 45, 43, 46, 64, 80, 83, 85]
    # examples: 41, 68, 65

    idx = pd.MultiIndex.from_tuples(
            list(zip(no_model_ids, it.repeat('no_shaping'))) +
            list(zip(with_model_ids, it.repeat('shaping'))) +
            list(zip(ls_model_ids, it.repeat('ls_only'))) +
            list(zip(sl_model_ids, it.repeat('sl_only'))), names=('model_id', 'model_type'))
    model_ids = pd.Series(no_model_ids + with_model_ids + ls_model_ids + sl_model_ids,  index=idx)

    model_infos = model_ids.apply(mdb.get_model).apply(pd.Series)
    model_infos = pd.concat(
        (model_infos, model_ids.apply(mdb.get_model_attributes)), axis=1)

    path = Path('/analysis/ms_figures/behavior')
    jP.make_folder(path)
    info_file = path / 'mse_comparison.info.txt'
    with open(info_file, 'w') as f:
        pass

    dpi = 300
    jP.set_rcParams(plt)
    
    error_list = model_infos['mse']
    error_list.name = 'values'
    X = error_list.reset_index()
    model = ols("values ~ C(model_type)", X).fit()
    aov_table = anova_lm(model, typ=1)
    print(aov_table)
    tukey = pairwise_tukeyhsd(
        endog=X['values'], groups=list(map(str, zip(X['model_type']))), alpha=0.05)
    print(tukey)
    with open(info_file, 'a') as f:
        f.write(aov_table.to_string())
        f.write('\n\n\n')
        f.write(str(tukey))
        f.write('\n\n\n')

    def box_plotter(X, order, ax, **kwargs):
        model_type = X.index.unique('model_type')[0]
        i = order.index(model_type)

        flierprops={'marker': '.', 'markersize': 2, 'mfc': 'k'}
        boxprops={'facecolor': SHAPING_COLORS[model_type], 'edgecolor': 'k'}
        medianprops={'color': 'k'}
        ax.boxplot(
            [X], positions=[i], widths=0.25, flierprops=flierprops,
            boxprops=boxprops, patch_artist=True, medianprops=medianprops,
            **kwargs)

    margins={'left': 110, 'right': 90, 'top': 90, 'bottom': 90}
    PdfPlotter(path / f'mse_comparison_bar.pdf',
               fixed_margins=margins)
    plt.figure(figsize=(2.25, 2.6), dpi=dpi)
    ax = plt.gca()
 
    plot_order = ['no_shaping', 'shaping', 'ls_only', 'sl_only']
    error_list.groupby('model_type').apply(
        box_plotter, order=plot_order, ax=ax, include_groups=True)
    ax.plot([2, 2], error_list.reindex([65, 68], level='model_id').values, ls='', marker='o', mec='k', mfc='tab:red')
    ax.annotate(
        '', xy=(2 + 0.05, error_list.xs(65, level='model_id')),
        xytext=(2+0.3, error_list.xs(65, level='model_id')),
        arrowprops=dict(facecolor='black', shrink=0, width=0.8, headwidth=3,
                        headlength=2))
    ax.annotate(
        '', xy=(2 + 0.05, error_list.xs(68, level='model_id')),
        xytext=(2+0.3, error_list.xs(68, level='model_id')),
        arrowprops=dict(facecolor='black', shrink=0, width=0.8, headwidth=3,
                        headlength=2))
    ax.text(2+0.35, error_list.xs(68, level='model_id'),'A.', fontsize=5, va='center')
    ax.text(2+0.35, error_list.xs(65, level='model_id'),'B.', fontsize=5, va='center')
    jP.configure_spines(ax)

    results = pd.DataFrame(tukey._results_table).map(lambda x: x.data).T.set_index(0).T
    results['group1'] = results['group1'].map(lambda x: x[2:-3])
    results['group2'] = results['group2'].map(lambda x: x[2:-3])
    results = results.set_index(['group1', 'group2'])
    pvalues = results['p-adj']
    lbl_y = [0.0225]
    def annotate(X, ax, order, lbl_y):
        if X['p-adj'] <= 0.05:
            jP.annotation(ax, (order.index(X.name[0]), order.index(X.name[1])), lbl_y[0], X[0], va='bottom')
            lbl_y[0] = lbl_y[0] + jP.annotation_padding(ax, 0.0035)
    pd.concat(
        (pvalues, jP.significance_symbols(pvalues)), axis=1).sort_values(
            'p-adj').apply(annotate, ax=ax, order=plot_order, lbl_y=lbl_y, axis=1)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['None', 'LS + SL', 'LS', 'SL'])
    ax.set_yticks([0.01, 0.02, 0.03])
    ax.set_ylim(0.009, 0.0325)
    ax.set_xlabel('Shaping Task')
    ax.ticklabel_format(axis='y', scilimits=(0, 0))
    ax.set_ylabel('Full Task MSE After Training')
    jP.set_ylabel_position(ax, nlines=2.7)
    plt.show()

    margins = jP.default_margins()
    margins['left'] = 180
    margins['top'] = 50
    margins['right'] = 45
    PdfPlotter(path / f'mse_comparison_bar_horiz.pdf',
               fixed_margins=margins)
    plt.figure(figsize=(3, 1.5), dpi=dpi)
    ax = plt.gca()
 
    plot_order = ['no_shaping', 'shaping', 'ls_only', 'sl_only']
    error_list.groupby('model_type').apply(
        box_plotter, order=plot_order, ax=ax, include_groups=True,
        vert=False)
    jP.configure_spines(ax)

    lbl_x = [0.0225]
    def annotate(X, ax, order, lbl_x):
        if X['p-adj'] <= 0.05:
            jP.annotation_horiz(ax, (order.index(X.name[0]), order.index(X.name[1])), lbl_x[0], X[0], ha='left')
            lbl_x[0] = lbl_x[0] + jP.annotation_padding(ax, 0.002, axis='x')

    pd.concat(
        (pvalues, jP.significance_symbols(pvalues)), axis=1).sort_values(
            'p-adj').apply(annotate, ax=ax, order=plot_order, lbl_x=lbl_x, axis=1)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['None', 'LS + SL', 'only LS', 'only SL'], ha='right')
    ax.set_xticks([0.01, 0.02, 0.03])
    ax.set_xlim(0.009, 0.0325)

    ax.plot(error_list.reindex([65, 68], level='model_id').values, [2, 2], ls='', marker='o', mec='k', mfc='tab:red')
    ax.annotate('', xy=(error_list.xs(65, level='model_id'), 2 + 0.1), xytext=(error_list.xs(65, level='model_id'), 2+0.3),
                arrowprops=dict(facecolor='black', shrink=0, width=0.8, headwidth=3, headlength=2))
    ax.annotate('', xy=(error_list.xs(68, level='model_id'), 2 + 0.1), xytext=(error_list.xs(68, level='model_id'), 2+0.3),
                arrowprops=dict(facecolor='black', shrink=0, width=0.8, headwidth=3, headlength=2))
    ax.text(error_list.xs(68, level='model_id'), 2+0.35,'A.', fontsize=5, ha='center')
    ax.text(error_list.xs(65, level='model_id'), 2+0.35,'B.', fontsize=5, ha='center')
 
    ax.set_ylabel('Shaping Task')
    ax.set_xlabel('Full Task MSE After Training')
    plt.show()



if __name__ == '__main__':
    main(sys.argv[1:])
