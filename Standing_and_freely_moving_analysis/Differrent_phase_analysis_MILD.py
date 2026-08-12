# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.rcParams['svg.fonttype']='none'
plt.rcParams['path.simplify']=False
plt.rcParams['axes.unicode_minus']=False

BEFORE_COLOR='#E6C280'
MIDDLE_COLOR='#8FA8C4'
AFTER_COLOR='#E3A8B8'
POINT_COLOR='black'

YLIM_CONFIG={
    'firing_rate':(-40,120),
    'half_width_std':(0.001,0.0025),
    'maximum_displacement':(-20,100),
    'peak_to_valley_std':(0.0008,0.002),
    'peak_trough_ratio_std':(0,0.5),
    'recovery_slope_std':(-0.5000000,2000000),
    'repolarization_slope_std':(0,2500000),
}

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def select_units(unit_summary,filter_type):
    if filter_type=='only_middle':
        return unit_summary[(unit_summary['group']=='MILD')&(unit_summary['middle_neuron'])]
    elif filter_type=='both':
        return unit_summary[
            (unit_summary['group']=='MILD')&
            (unit_summary['before_neuron'])&
            (unit_summary['middle_neuron'])&
            (unit_summary['after_neuron'])
        ]
    else:
        return unit_summary[
            (unit_summary['group']=='MILD')&
            (unit_summary['middle_neuron'])&
            (unit_summary['after_neuron'])
        ]

def plot_pie_component(summary,title,filename,color):
    n_unit=len(summary)
    n_neuron=int(summary.iloc[:,0].sum())
    n_noise=n_unit-n_neuron
    plt.figure(figsize=(6,5))
    ax=plt.gca()
    ax.set_clip_on(False)
    patches,texts,autotexts=plt.pie(
        [n_neuron,n_noise],
        labels=[f'neuron: {n_neuron}',f'noise: {n_noise}'],
        autopct='%1.1f%%',
        colors=[color,'#E6E6E6']
    )
    for p in patches:
        p.set_clip_on(False)
    for t in texts+autotexts:
        t.set_clip_on(False)
    plt.title(title)
    ensure_folder('figures/component')
    plt.savefig(
        f'figures/component/{filename}.svg',
        bbox_inches='tight',
        transparent=True
    )
    plt.close()

def plot_phase_components(unit_summary):
    mild=unit_summary[unit_summary['group']=='MILD']
    plot_pie_component(mild[['before_neuron']],
                       'Before Phase Component',
                       'before_component',
                       BEFORE_COLOR)
    plot_pie_component(mild[['middle_neuron']],
                       'Middle Phase Component',
                       'middle_component',
                       MIDDLE_COLOR)
    plot_pie_component(mild[['after_neuron']],
                       'After Phase Component',
                       'after_component',
                       AFTER_COLOR)
    plot_pie_component(mild[['all_neuron']],
                       'All Phase Component',
                       'all_component',
                       'mediumseagreen')

def plot_middle_reference_retention_components(unit_summary):
    middle_ref=unit_summary[
        (unit_summary['group']=='MILD')&
        (unit_summary['middle_neuron'])
    ]
    if middle_ref.empty:
        return
    plot_pie_component(
        middle_ref[['before_neuron']],
        'Before Retention on Middle Neurons',
        'middle_reference_before_retention',
        BEFORE_COLOR
    )
    plot_pie_component(
        middle_ref[['after_neuron']],
        'After Retention on Middle Neurons',
        'middle_reference_after_retention',
        AFTER_COLOR
    )

def get_compare_columns(stat_name,comparison):
    if comparison=='before_middle':
        return f'before_{stat_name}',f'middle_{stat_name}','Before','Middle'
    elif comparison=='middle_after':
        return f'middle_{stat_name}',f'after_{stat_name}','Middle','After'
    elif comparison=='before_after':
        return f'before_{stat_name}',f'after_{stat_name}','Before','After'
    else:
        raise ValueError('comparison must be before_middle, middle_after or before_after')

def statistical_test(first,second,paired=True):
    if len(first)<3 or len(second)<3:
        return np.nan,'NA','ns'
    if paired:
        diff=second.values-first.values
        _,p_norm=stats.shapiro(diff)
        _,p_t=stats.ttest_rel(second,first)
        if p_norm<0.05:
            _,p=stats.wilcoxon(diff)
            test='Wilcoxon'
        else:
            p=p_t
            test='Paired t-test'
    else:
        _,p_norm1=stats.shapiro(first)
        _,p_norm2=stats.shapiro(second)
        if p_norm1>0.05 and p_norm2>0.05:
            _,p=stats.ttest_ind(second,first)
            test='Independent t-test'
        else:
            _,p=stats.mannwhitneyu(second,first,alternative='two-sided')
            test='Mann-Whitney U'
    if p<0.001:
        sig='***'
    elif p<0.01:
        sig='**'
    elif p<0.05:
        sig='*'
    else:
        sig='ns'
    return p,test,sig

def set_axis(stat_name):
    if stat_name in YLIM_CONFIG:
        plt.ylim(YLIM_CONFIG[stat_name])

def add_stat_bar(ax,p,sig,test):
    y_min,y_max=ax.get_ylim()
    y=y_max-(y_max-y_min)*0.05
    ax.plot([0,1],[y,y],color='black',linewidth=1,clip_on=False)
    ax.plot([0,0],[y-(y_max-y_min)*0.01,y],color='black',linewidth=1,clip_on=False)
    ax.plot([1,1],[y-(y_max-y_min)*0.01,y],color='black',linewidth=1,clip_on=False)
    ax.text(0.5,y+(y_max-y_min)*0.02,f'p={p:.4f} {sig}',ha='center',fontsize=10,clip_on=False)
    ax.annotate(test,xy=(0.98,0.98),xycoords='axes fraction',
                ha='right',va='top',fontsize=8,clip_on=False,
                bbox=dict(boxstyle='round,pad=0.3',facecolor='white',alpha=0.7))

def format_axes(ax):
    ax.set_clip_on(False)
    for collection in ax.collections:
        collection.set_clip_on(False)
    for line in ax.lines:
        line.set_clip_on(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_clip_on(False)

def plot_three_phase_comparison(unit_summary,stat_name,filter_type='both',comparison='before_middle',paired=True):
    sub_df=select_units(unit_summary,filter_type)
    if sub_df.empty:
        print(f'No data: {stat_name} {filter_type}')
        return

    col1,col2,label1,label2=get_compare_columns(stat_name,comparison)

    if col1 not in sub_df.columns or col2 not in sub_df.columns:
        return

    first=sub_df[col1].copy()
    second=sub_df[col2].copy()

    valid=first.notna()&second.notna()
    first=first[valid]
    second=second[valid]

    if len(first)<3:
        return

    p,test,sig=statistical_test(first,second,paired)

    plot_data=pd.DataFrame({
        'phase':[label1]*len(first)+[label2]*len(second),
        'value':np.concatenate([first.values,second.values])
    })

    folder=f'figures/{filter_type}/{comparison}'
    ensure_folder(folder)

    plt.figure(figsize=(6,8))
    ax=plt.gca()
    ax.set_clip_on(False)

    palette={
        'Before':BEFORE_COLOR,
        'Middle':MIDDLE_COLOR,
        'After':AFTER_COLOR
    }

    sns.boxplot(
        data=plot_data,
        x='phase',
        y='value',
        hue='phase',
        palette=palette,
        showfliers=False,
        legend=False,
        ax=ax
    )

    sns.stripplot(
        data=plot_data,
        x='phase',
        y='value',
        color=POINT_COLOR,
        alpha=0.5,
        jitter=0.15,
        ax=ax
    )

    format_axes(ax)
    set_axis(stat_name)
    add_stat_bar(ax,p,sig,test)

    ax.set_ylabel(stat_name.replace('_',' ').title())
    ax.set_xlabel('')
    ax.set_title(f'MILD {stat_name}\n{label1} vs {label2}\nFilter:{filter_type}')

    filename=f'{stat_name}_p_{p:.4f}.svg'

    plt.savefig(
        os.path.join(folder,filename),
        bbox_inches='tight',
        transparent=True
    )

    plot_data.to_csv(
        os.path.join(folder,f'{stat_name}_data.csv'),
        index=False
    )

    plt.close()

    print(f'{stat_name} | {filter_type} | {comparison} | n={len(first)} | {test}: p={p:.4f} {sig}')


def plot_phase_locations_stat_on_selected_neurons(unit_summary,stat_name,phase,filter_type='both'):
    sub_df=select_units(unit_summary,filter_type)
    col=f'{phase}_{stat_name}'

    if col not in sub_df.columns:
        return

    data=sub_df[col].dropna()

    if len(data)<3:
        return

    folder=f'figures/{filter_type}/location'
    ensure_folder(folder)

    color=BEFORE_COLOR if phase=='before' else MIDDLE_COLOR if phase=='middle' else AFTER_COLOR

    plt.figure(figsize=(6,8))
    ax=plt.gca()
    ax.set_clip_on(False)

    sns.boxplot(
        y=data,
        color=color,
        showfliers=False,
        ax=ax
    )

    sns.stripplot(
        y=data,
        color=POINT_COLOR,
        alpha=0.4,
        jitter=0.15,
        ax=ax
    )

    format_axes(ax)
    set_axis(stat_name)

    ax.set_ylabel(stat_name)
    ax.set_title(f'{phase.capitalize()} {stat_name}\nFilter:{filter_type}')

    plt.savefig(
        f'{folder}/{phase}_{stat_name}.svg',
        bbox_inches='tight',
        transparent=True
    )

    pd.DataFrame({'value':data}).to_csv(
        f'{folder}/{phase}_{stat_name}.csv',
        index=False
    )

    plt.close()

def plot_global_matrix_three_phase(unit_summary,features,filter_type='both',comparison='before_middle'):
    sub_df=select_units(unit_summary,filter_type)

    if sub_df.empty:
        return

    storage=[]

    for feature in features:
        col1,col2,label1,label2=get_compare_columns(feature,comparison)

        if col1 not in sub_df.columns or col2 not in sub_df.columns:
            continue

        first=sub_df[col1]
        second=sub_df[col2]

        valid=first.notna()&second.notna()
        first=first[valid]
        second=second[valid]

        if len(first)<3:
            continue

        ratio=pd.Series(second.values/first.values)

        storage.append(
            pd.DataFrame({
                'Feature':[feature]*len(ratio),
                'Ratio':ratio.values
            })
        )

    if not storage:
        return

    df=pd.concat(storage,ignore_index=True)

    folder=f'figures/{filter_type}/matrix'
    ensure_folder(folder)

    plt.figure(figsize=(max(len(features)*1.8,8),6))
    ax=plt.gca()
    ax.set_clip_on(False)

    sns.boxplot(
        data=df,
        x='Feature',
        y='Ratio',
        color=MIDDLE_COLOR,
        showfliers=False,
        ax=ax
    )

    sns.stripplot(
        data=df,
        x='Feature',
        y='Ratio',
        color=POINT_COLOR,
        alpha=0.25,
        jitter=0.15,
        ax=ax
    )

    format_axes(ax)

    plt.axhline(
        1,
        color='#D9534F',
        linestyle='--',
        linewidth=1.5
    )

    plt.ylabel(f'{comparison} ratio')
    plt.xticks(rotation=30,ha='right')
    plt.title(f'{comparison} Matrix\nFilter:{filter_type}')

    plt.tight_layout()

    plt.savefig(
        f'{folder}/matrix_{comparison}.svg',
        bbox_inches='tight',
        transparent=True
    )

    df.to_csv(
        f'{folder}/matrix_{comparison}.csv',
        index=False
    )

    plt.close()


if __name__=='__main__':
    warnings.filterwarnings('ignore')
    ensure_folder('figures')

    unit_summary=pd.read_csv('results/unit_summary_MILD_Three_0.1.csv')

    plot_phase_components(unit_summary)
    plot_middle_reference_retention_components(unit_summary)

    print('Starting MILD three-phase comparison analysis...')

    stats_list=[
        'firing_rate',
        'half_width',
        'peak_to_valley',
        'peak_trough_ratio',
        'repolarization_slope',
        'recovery_slope',
        'peak_to_valley_median',
        'peak_trough_ratio_median',
        'half_width_median',
        'repolarization_slope_median',
        'recovery_slope_median',
        'amplitude_between_peak_and_trough_median',
        'peak_to_valley_std',
        'peak_trough_ratio_std',
        'half_width_std',
        'repolarization_slope_std',
        'recovery_slope_std',
        'amplitude_between_peak_and_trough_std',
        'x_median',
        'y_median',
        'x_std',
        'y_std',
        'x_ptp',
        'y_ptp',
        'maximum_displacement'
    ]

    location_features=[
        'x_std',
        'y_std',
        'x_ptp',
        'y_ptp',
        'maximum_displacement'
    ]

    filters=[
        'only_middle',
        'both',
        'all'
    ]

    comparisons=[
        'before_middle',
        'middle_after',
        'before_after'
    ]

    print('Generating three phase comparison plots...')

    for filter_type in filters:
        for stat_name in stats_list:
            for comparison in comparisons:
                plot_three_phase_comparison(
                    unit_summary,
                    stat_name,
                    filter_type=filter_type,
                    comparison=comparison,
                    paired=True
                )

    print('Generating location plots...')

    for filter_type in filters:
        for stat_name in location_features:
            for phase in ['before','middle','after']:
                plot_phase_locations_stat_on_selected_neurons(
                    unit_summary,
                    stat_name,
                    phase,
                    filter_type
                )

    print('Generating matrix plots...')

    matrix_features=[
        'half_width_std',
        'peak_to_valley_std',
        'peak_trough_ratio_std',
        'repolarization_slope_std',
        'recovery_slope_std',
        'maximum_displacement'
    ]

    for filter_type in filters:
        for comparison in comparisons:
            plot_global_matrix_three_phase(
                unit_summary,
                matrix_features,
                filter_type,
                comparison
            )

    print('All analysis completed.')