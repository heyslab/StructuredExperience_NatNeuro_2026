# [Figure 6b,d]
import sys 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import pandas as pd
import numpy as np
import scipy
from scipy.stats import ttest_ind

from analysis_tools import jPlots as jP
from analysis_tools.mpl_helpers import PdfPlotter

plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  
    "axes.facecolor":    (1.0, 1.0, 1.0, 0.0),  
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),
    })

TRIAL_COLORS = {'LS': '#eb0d8c', 'SL': '#2bace2', 'SS': '#f89521', 'LL': '#2b958c', 'S': 'k', 'L': 'r'}

def avg_plot(ax, per_avg, ctrl_avg, bbox_to_anchor=(1.1, 0)):
    ax.errorbar(
        per_avg.index, per_avg['mean'], yerr=per_avg['SD'], label='SL Shaping',
        fmt='-o', c='tab:green', capsize=2)
    ax.errorbar(
        ctrl_avg.index, ctrl_avg['mean'], yerr=ctrl_avg['SD'],
        label='LS+SL Shaping', fmt='-o', c='tab:orange', capsize=2)
    jP.configure_spines(ax)
    ax.set_ylim(40, 80)
    jP.percent_y(ax)
    ax.set_xlabel('Day')
    ax.set_ylabel('Performance (% Correct)')
    ax.legend(loc='lower right', bbox_to_anchor=bbox_to_anchor)
    #ax.set_title('Average Task Performance by Day')
    ax.spines['bottom'].set_bounds(1, 8)
    ax.set_yticks([40, 50, 60, 70, 80])
    ax.set_xticks([2, 4, 6, 8])


def type_plot(per_type, ctrl_type, axs):
    trial_types = per_type.index.get_level_values('trial_type').unique()
    day = per_type.index.get_level_values('day').unique()
    
    for ax, trial_type in zip(axs, trial_types):
        type_sl = per_type.xs(trial_type, level='trial_type')
        type_s = ctrl_type.xs(trial_type, level='trial_type')

        ax.errorbar(
            day, type_sl['mean'], yerr=type_sl['SD'], fmt='-o', linestyle='-',
            capsize=2, c=TRIAL_COLORS[trial_type], label=f'SL Shaping', clip_on=False, mec='k', mew=0.2)
        ax.errorbar(
            day, type_s['mean'], yerr=type_s['SD'], fmt='-o', linestyle='--',
            capsize=2, c=TRIAL_COLORS[trial_type], label=f'SL + LS Shaping', alpha=0.5, clip_on=False, mec='k', mew=0.2)
        ax.set_ylabel(trial_type, color=TRIAL_COLORS[trial_type])
        ax.set_ylim(0, 100)
        jP.percent_y(ax)
        ax.set_yticks([0, 40, 80])

    legend_lines = [matplotlib.lines.Line2D([0], [0], color='black'),
                    matplotlib.lines.Line2D([0], [0], ls='--', color='black', alpha=0.5)]
    axs[0].legend(
        legend_lines, ['SL Shaping', 'SL + LS Shaping'], loc='lower right',
        bbox_to_anchor=(1.1, 0))
    axs[0].set_xticks([])
    axs[0].spines['bottom'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].set_xticks([])


def type_plot_horiz(per_type, ctrl_type, axs):
    trial_types = per_type.index.get_level_values('trial_type').unique()
    day = per_type.index.get_level_values('day').unique()
    
    for ax, trial_type in zip(axs, trial_types):
        type_sl = per_type.xs(trial_type, level='trial_type')
        type_s = ctrl_type.xs(trial_type, level='trial_type')
        label = ' - '.join(trial_type)
        label = label.replace('S', 'Short')
        label = label.replace('L', 'Long')

        ax.errorbar(
            day, type_sl['mean'], yerr=type_sl['SD'], fmt='-o', linestyle='-',
            capsize=2, c=TRIAL_COLORS[trial_type], label=f'SL Shaping', clip_on=False, mec='k', mew=0.2)
        ax.errorbar(
            day, type_s['mean'], yerr=type_s['SD'], fmt='-o', linestyle='--',
            capsize=2, c=TRIAL_COLORS[trial_type], label=f'SL + LS Shaping', alpha=0.5, clip_on=False, mec='k', mew=0.2)
        ax.text(0, 1, label, color=TRIAL_COLORS[trial_type], transform=ax.transAxes, ha='left', va='bottom')
        ax.set_ylim(0, 100)
        jP.percent_y(ax)
        ax.set_yticks([0, 40, 80])
        ax.set_xlabel('Day')
        ax.set_xlim(0.8, 8)

    axs[0].set_ylabel('Percent Correct')
    legend_lines = [matplotlib.lines.Line2D([0], [0], color='black'),
                    matplotlib.lines.Line2D([0], [0], ls='--', color='black', alpha=0.5)]
    axs[0].legend(
        legend_lines, ['SL Shaping', 'SL + LS Shaping'], loc='lower left',
        bbox_to_anchor=(0, 0))
    axs[1].set_yticks([])
    axs[2].set_yticks([])
    axs[1].spines['left'].set_visible(False)
    axs[2].spines['left'].set_visible(False)


def main(args):
    data_path = Path(__file__).resolve().parent
    plot_path = Path('/analysis/ms_figures/mouse_behavior')
    plot_path.mkdir(exist_ok=True)
    info_file = plot_path / 'performance_plot.info.txt'
    with open(info_file, 'w') as f:
        pass

    jP.set_rcParams(plt)
    margins = jP.default_margins()
    dpi = 300

    # Reads in average performance/day for SL shaping mice
    per_avg = pd.read_csv(data_path / 'sl_only_avg.csv')
    per_avg.columns = ['day', 'M1', 'M2', 'M4', 'M5', 'mean', 'SD']
    per_avg = per_avg.set_index('day')

    # Reads in performance by trial type for SL shaping mice
    per_type = pd.read_csv(data_path / 'sl_by_type.csv')
    per_type.columns = ['day', 'trial_type', 'M1', 'M2', 'M4', 'M5', 'mean', 'SD']
    per_type = per_type.set_index(['day', 'trial_type'])

    # Reads in avg performance/day and by trial type for Full Shaping mice (Erin's data)
    ctrl_avg = pd.read_csv(data_path / 'tdnms_control.csv')
    ctrl_avg = ctrl_avg.set_index('day')
    ctrl_type = pd.read_csv(data_path / 'ctrl_by_type.csv')
    ctrl_type = ctrl_type.set_index(['day', 'trial_type'])

    PdfPlotter(plot_path / 'per_avg.pdf', fixed_margins=jP.default_margins())
    plt.figure(figsize=(1.8, 1.75), dpi=300)
    ax = plt.subplot()
    avg_plot(ax, per_avg, ctrl_avg)
    plt.show()

    PdfPlotter(plot_path / 'per_avg_2.3.pdf', fixed_margins=jP.default_margins())
    plt.figure(figsize=(2.3, 1.75), dpi=300)
    ax = plt.subplot()
    avg_plot(ax, per_avg, ctrl_avg, bbox_to_anchor=(1, 0))
    plt.show()

    t_stat, p = ttest_ind(per_avg['mean'], ctrl_avg['mean'], equal_var=False)
    print('Welch\'s independent t-test')
    print("t-statistic:", t_stat)
    print("p-value:", p)
    with open(info_file, 'a') as f:
        f.write(f'p-value: {p}' + '\n')
        f.write(f't-stat: {t_stat}' + '\n\n\n')

    PdfPlotter(plot_path / 'per_type.pdf', fixed_margins=jP.default_margins())
    fig = plt.figure(figsize=(2, 3), dpi=300)
    gs = gridspec.GridSpec(3, 1)
    axs = list(map(plt.subplot, gs))
    list(map(jP.configure_spines, axs))
    type_plot(per_type, ctrl_type, axs)
    axs[-1].set_xlabel('Day')
    axs[0].set_title('Average Performance by Day')
    plt.show()

    PdfPlotter(plot_path / 'per_type_horiz.pdf', fixed_margins=jP.default_margins())
    fig = plt.figure(figsize=(4.7, 1.25), dpi=300)
    gs = gridspec.GridSpec(1, 3, wspace=0.1)
    axs = list(map(plt.subplot, gs))
    list(map(jP.configure_spines, axs))
    type_plot_horiz(per_type, ctrl_type, axs)
    plt.show()

    PdfPlotter(plot_path / 'per_type_horiz_5.2.pdf', fixed_margins=jP.default_margins())
    fig = plt.figure(figsize=(5.2, 1.25), dpi=300)
    gs = gridspec.GridSpec(1, 3, wspace=0.1)
    axs = list(map(plt.subplot, gs))
    list(map(jP.configure_spines, axs))
    type_plot_horiz(per_type, ctrl_type, axs)
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
