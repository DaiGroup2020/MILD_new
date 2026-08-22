# -*- coding:utf-8 -*-
import os,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spikeinterface.full as si

input_folder=r'H:\WXY\MILD-DATA\Cell_boundle'
output_folder=r'H:\WXY\MILD-DATA\Pearson_Result_All'
data_root=r'Z:\WXY'
os.makedirs(output_folder,exist_ok=True)

plt.rcParams['svg.fonttype']='none'
plt.rcParams['font.family']='Arial'
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['pdf.fonttype']=42

def pearson(totalwaveform):
    return np.array([np.corrcoef(i) for i in totalwaveform])

def get_week_labels(n):
    weeks=[2,4,6,8,10,12] if n==6 else list(range(2,14)) if n==12 else list(range(1,n+1))
    return [f'{i}w' for i in weeks]

def load_bundle(file):
    df=pd.read_csv(file)
    return df.iloc[:,0],df.iloc[:,1:]

def parse_animal(filename):
    m=re.search(r'WXY_(.*?)_(\d+)_cell',filename)
    return m.group(1),m.group(2)

def get_main_waveform(waveform,unit_id):
    template=waveform.get_template(unit_id=int(unit_id),mode='median')
    main=np.zeros(template.shape[0])
    for ch in range(template.shape[1]):
        if np.max(abs(template[:,ch]))>np.max(abs(main)):main=template[:,ch]
    return main

def generate_unit_pearson(bundle_file):
    filename=os.path.basename(bundle_file)
    system,mouse=parse_animal(filename)
    print('Processing:',system,mouse)
    root=os.path.join(data_root,system,mouse)
    save_folder=os.path.join(output_folder,mouse)
    os.makedirs(save_folder,exist_ok=True)
    recording_time,units=load_bundle(bundle_file)
    week_labels=get_week_labels(len(recording_time))

    for unit_name in units.columns:
        print('Unit:',unit_name)
        waveforms=[]
        for t,unit_id in zip(recording_time,units[unit_name]):
            folder=os.path.join(root,str(t),'analysis','sorting_result','waveform_from_phy')
            waveform=si.load_waveforms(folder,with_recording=False)
            waveforms.append(get_main_waveform(waveform,unit_id))

        p=pearson(np.array([waveforms]))[0]

        csv=os.path.join(save_folder,unit_name+'_Pearson.csv')
        pd.DataFrame(p,index=week_labels,columns=week_labels).to_csv(csv,encoding='utf-8-sig')

        plt.figure(figsize=(6,5))
        plt.imshow(p,cmap='viridis',vmin=0.9,vmax=1)
        plt.colorbar(label='Pearson Similarity')
        plt.xticks(range(len(week_labels)),week_labels,rotation=45)
        plt.yticks(range(len(week_labels)),week_labels)
        plt.title(mouse+' '+unit_name)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder,unit_name+'_Pearson.svg'),dpi=300,bbox_inches='tight')
        plt.close()
        print('Saved:',csv)



def load_pearson_files(folder):
    return [os.path.join(r,f) for r,_,fs in os.walk(folder) for f in fs if f.endswith('_Pearson.csv')]

def generate_mean_pearson(result_folder):
    weeks=[f'{i}w' for i in range(2,14)]
    files=[os.path.join(r,f) for r,_,fs in os.walk(result_folder) for f in fs if f.endswith('_Pearson.csv')]
    mean_matrix=np.full((12,12),np.nan)

    for i,w1 in enumerate(weeks):
        for j,w2 in enumerate(weeks):
            vals=[]
            for f in files:
                df=pd.read_csv(f,index_col=0)
                if w1 in df.index and w2 in df.columns:
                    vals.append(df.loc[w1,w2])
            if vals:
                mean_matrix[i,j]=np.mean(vals)

    pd.DataFrame(mean_matrix,index=weeks,columns=weeks).to_csv(
        os.path.join(result_folder,'Mean_Pearson_2-13w.csv'),
        encoding='utf-8-sig'
    )

    # 完整矩阵
    plt.figure(figsize=(7,6))
    plt.imshow(mean_matrix,cmap='viridis',vmin=0.9,vmax=1)
    plt.colorbar(label='Pearson Similarity')
    plt.xticks(range(12),weeks,rotation=45)
    plt.yticks(range(12),weeks)
    plt.title('Mean Pearson Similarity (2-13 weeks)')
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder,'Mean_Pearson_2-13w.svg'),dpi=300,bbox_inches='tight')
    plt.close()

    # 去除对角线
    diag_matrix=mean_matrix.copy()
    np.fill_diagonal(diag_matrix,np.nan)
    cmap=plt.cm.viridis.copy()
    cmap.set_bad('white')

    plt.figure(figsize=(7,6))
    plt.imshow(diag_matrix,cmap=cmap,vmin=0.9,vmax=1)
    plt.colorbar(label='Pearson Similarity')
    plt.xticks(range(12),weeks,rotation=45)
    plt.yticks(range(12),weeks)
    plt.title('Mean Pearson Similarity (Diagonal removed)')
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder,'Mean_Pearson_2-13w_Diagonal_removed.svg'),dpi=300,bbox_inches='tight')
    plt.close()

    print('Saved:',os.path.join(result_folder,'Mean_Pearson_2-13w.csv'))

if __name__=='__main__':
    #for f in os.listdir(input_folder):
    #    if f.endswith('cell_bundle_OK_plot.csv'):
    #        generate_unit_pearson(os.path.join(input_folder,f))

    generate_mean_pearson(r'H:\WXY\MILD-DATA\Pearson_Result_All')