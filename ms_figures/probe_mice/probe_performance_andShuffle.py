# [Figure 7b]
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import itertools as it
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from analysis_tools.mpl_helpers import PdfPlotter
import analysis_tools.jPlots as jP

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  
    "axes.facecolor":    (1.0, 1.0, 1.0, 0.0),  
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),
    })

TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'LL': '#2b958c', 'S': '#f89521'}

def main(args):
    plot_path = Path('/analysis/ms_figures/probe_mice_andShuffle')
    plot_path.mkdir(exist_ok=True, parents=True)
    info_file = plot_path / 'probe_performance.info.txt'
    with open(info_file, 'w') as f:
        pass

    dpi = 300
    jP.set_rcParams(plt)
    data_path = Path('goPercentages.csv')
    shuffle_data_path = Path('shuffle_results.csv')
    shuffle_data = pd.read_csv(shuffle_data_path, index_col=0)
    go_percents = pd.read_csv(data_path)

    go_per = go_percents.set_index(['mouse', 'session', 'date'])
    go_per = go_per.drop('baseline', level='session') * 100

    probe_dat = go_per.droplevel('date')['ProbePercentGo']
    probe_dat = probe_dat.groupby('session').apply(list).apply(pd.Series).T

    nonprobe_dat = go_per[['LSpercentGo', 'SLpercentGo', 'SSpercentGo']]
    nonprobe_dat.columns = ['LS', 'SL', 'SS']

    margins = jP.default_margins()
    margins['top'] = 50
    PdfPlotter(plot_path / 'probe_behavior.pdf', fixed_margins=margins)
    plt.figure(figsize=(1.5, 1.8), dpi=dpi)
    
    def box_plotter(X, i, colors, ax):
        c = next(colors)
        flierprops={'marker': '.', 'markersize': 2, 'mfc': c, 'mec': c, 'clip_on': False}
        c_hsv = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(c))
        c1 = matplotlib.colors.hsv_to_rgb(c_hsv * [1, 0.5, 1])
        boxprops={'facecolor': c1, 'edgecolor': 'k', 'clip_on': False}
        medianprops={'color': c, 'clip_on': False}
        whiskerprops={'clip_on': False}
        capprops={'clip_on': False}
        count = next(i)
        ax.boxplot(
            [X.dropna()], positions=[count], widths=0.5, flierprops=flierprops,
            boxprops=boxprops, patch_artist=True, medianprops=medianprops,
            whiskerprops=whiskerprops, capprops=capprops)

    counts = it.count()
    ax = plt.gca()
    colors = [TRIAL_COLORS[k] for k in nonprobe_dat.columns]
    nonprobe_dat.apply(box_plotter, args=(counts, iter(colors), ax), axis=0)
    colors = ['#253D5B', '#DA627D', '#96ACB7']
    probe_dat.apply(box_plotter, args=(counts, iter(colors), ax), axis=0)
    ax.set_ylim(0, 100)
    jP.configure_spines(ax)
    ax.set_yticks([20, 60, 100])
    ax.set_xticklabels(nonprobe_dat.columns.tolist() + probe_dat.columns.tolist(), rotation=90)
    jP.percent_y(ax)
    y_pad = jP.annotation_padding(ax, 0.08)
    jP.annotation(ax, (0, 5), 100 + y_pad, '***', va='bottom')
    ax.annotate(
        '', (2, 100 + y_pad), xytext=(2, 90), textcoords='data',
        arrowprops={'arrowstyle': '<-', 'shrinkA': 0, 'shrinkB': 0})
    ax.set_ylim(0, 110)
    ax.set_ylabel('Trials w/ Response')
    ax.spines['left'].set_bounds(0, 100)
    plt.show()

    margins = jP.default_margins()
    margins['top'] = 50
    PdfPlotter(plot_path / 'probe_behavior_larger.pdf', fixed_margins=margins)
    plt.figure(figsize=(2, 2.5), dpi=dpi)
    counts = it.count()
    ax = plt.gca()
    colors = [TRIAL_COLORS[k] for k in nonprobe_dat.columns]
    nonprobe_dat.apply(box_plotter, args=(counts, iter(colors), ax), axis=0)
    colors = ['#253D5B', '#DA627D', '#96ACB7']
    probe_dat.apply(box_plotter, args=(counts, iter(colors), ax), axis=0)
    ax.set_ylim(0, 100)
    jP.configure_spines(ax)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_xticklabels(nonprobe_dat.columns.tolist() + probe_dat.columns.tolist(), rotation=90)
    jP.percent_y(ax)
    y_pad = jP.annotation_padding(ax, 0.08)
    jP.annotation(ax, (0, 5), 100 + y_pad, '***', va='bottom')
    ax.annotate(
        '', (2, 100 + y_pad), xytext=(2, 90), textcoords='data',
        arrowprops={'arrowstyle': '<-', 'shrinkA': 0, 'shrinkB': 0})
    ax.set_ylim(0, 110)
    ax.set_ylabel('Trials w/ Response')
    ax.spines['left'].set_bounds(0, 100)

    def draw_shuffle(X, i, ax):
        x = next(i)
        ax.plot([x-0.25, x+0.25], [X['mean'] * 100, X['mean'] * 100], color='k', alpha=0.25, solid_capstyle='butt')
        ax.fill_between(
            [x-0.25, x+0.25], *np.tile(X[['0.05', '0.95']] * 100, (2, 1)).T,
            color='k', alpha=0.25, ec=None, lw=0)

    counts = it.count()
    shuffle_data.reindex(
        nonprobe_dat.columns.tolist() + probe_dat.columns.tolist(), axis=1
        ).apply(draw_shuffle, i=counts, ax=ax)
    plt.show()

    X1 = probe_dat.stack()
    X1 = X1.droplevel(0)
    X1.index.name = 'trial_type'
    X2 = nonprobe_dat.reset_index(drop=True).stack().droplevel(0)
    X2.index.name = 'trial_type'

    X = pd.concat((X1, X2))
    X.name = 'values'
    X = X.reset_index()
    tukey = pairwise_tukeyhsd(
        endog=X['values'], groups=list(map(str, zip(X['trial_type']))), alpha=0.05) 

    fmt_probe = go_per.droplevel('date')['ProbePercentGo']  
    fmt_probe.index.names = ['mouse', 'type']
    fmt_probe.name='values'
    fmt_nonprobe = nonprobe_dat.stack().droplevel('session').droplevel('date')
    fmt_nonprobe.index.names = ['mouse', 'type']
    fmt_nonprobe.name='values'
    X = pd.concat((fmt_probe, fmt_nonprobe))
    X = X.reset_index()
    model = ols("values ~ C(type)*C(mouse)", X).fit()
    aov_table = anova_lm(model, typ=2)
    print(aov_table)

    with open(info_file, 'a') as f:
        f.write(aov_table.to_string())
        f.write('\n\n\n')
        f.write(str(tukey))


if __name__ == '__main__':
    main(sys.argv[1:])
