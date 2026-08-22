# -*- coding:utf-8 -*-
import os,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

root_folder=r'Z:\WXY'
cell_bundle_folder=r'H:\WXY\MILD-DATA\CU\Cell_boundle'
save_folder=r'H:\WXY\MILD-DATA\CU\Refractory_Result'
os.makedirs(save_folder,exist_ok=True)

def refrac(spikes,fs):
    if len(spikes)<2:return np.nan
    bad=np.where(np.diff(spikes/fs)*1000<1)[0]
    return 0 if len(bad)==0 else len(np.unique(np.r_[bad,bad+1]))/len(spikes)

def calculate_refractory_violation():
    result=[]
    for f in os.listdir(cell_bundle_folder):
        if not f.endswith('cell_bundle_OK_plot.csv'):continue
        m=re.search(r'WXY_(.*?)_(\d+)_cell',f)
        if not m:continue
        system,mouse=m.groups()
        print('Processing:',system,mouse)
        bundle=pd.read_csv(os.path.join(cell_bundle_folder,f),index_col=0)

        for unit in bundle:
            for t,uid in bundle[unit].items():
                phy=os.path.join(root_folder,system,mouse,str(t),'analysis','sorting_result','phy')
                if not os.path.exists(phy):continue
                sorting=se.read_phy(phy,exclude_cluster_groups=['noise','mua'])
                uid=int(uid)
                if uid not in sorting.get_unit_ids():continue
                result.append({'Mouse':mouse,'Unit':unit,'Recording':t,'Violation_rate':refrac(sorting.get_unit_spike_train(uid),sorting.get_sampling_frequency())})

    result=pd.DataFrame(result)
    result.to_csv(os.path.join(save_folder,'All_tracked_units_refractory_violation.csv'),index=False,encoding='utf-8-sig')

    unit_rate=result.groupby(['Mouse','Unit']).Violation_rate.mean()*100
    unit_rate.to_csv(os.path.join(save_folder,'Unit_mean_refractory_violation.csv'))

    print('Mean ± SD:',unit_rate.mean(),unit_rate.std(),'N:',len(unit_rate))
    return unit_rate


def plot_refractory_distribution():
    df=pd.read_csv(r'H:\WXY\MILD-DATA\CU\Refractory_Result\All_tracked_units_refractory_violation.csv')
    data=(df.groupby(['Mouse','Unit']).Violation_rate.mean()*100).dropna()

    pd.DataFrame({'Mean_violation_rate(%)':[data.mean()],'SD_violation_rate(%)':[data.std()],'N_units':[len(data)]}).to_csv(os.path.join(save_folder,'Refractory_violation_summary.csv'),index=False)

    fig,ax=plt.subplots(figsize=(8,6))
    ax.hist(data,bins=np.linspace(0,100,101),weights=np.ones(len(data))/len(data),color='#c8d82b')
    ax.set(xlabel='Spikes Violating Refractory Period (%)',ylabel='Proportion',xlim=(0,20))

    axins=inset_axes(ax,width='45%',height='45%',loc='upper right')
    axins.hist(data,bins=np.linspace(0,10,21),weights=np.ones(len(data))/len(data),color='#c8d82b')
    axins.set_xlim(0,10);axins.set_ylim(0,0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder,'Refractory_violation_distribution.svg'),transparent=True,bbox_inches='tight')
    plt.close()
    print('Mean ± SD:',data.mean(),data.std(),'N:',len(data))

if __name__=='__main__':
    #calculate_refractory_violation()
    plot_refractory_distribution()