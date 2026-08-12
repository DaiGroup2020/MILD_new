import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

root_folder_input=r"Z:"
experimenter="WXY"
acquisition_system="blackrock"
mouse_id="5125"
save_dir=r"H:\WXY\MILD-DATA\blackrock\5125"

cell_bundle_path=os.path.join(save_dir,"WXY_blackrock_5125_cell_bundle_OK_plot.csv")
'''
time_mapping={
    '20250613-1759':2,
    '20250621-1023':3,
    '20250627-1227':4,
    '20250709-1030':5,
    '20250717-1010':6,
    '20250725-0948':7,
    '20250731-1124':8,
    '20250808-1420':9,
    '20250811-1637':10,
    '20250818-1117':11,
    '20250826-1118':12,
    '20250901-1433':13
}
'''
time_mapping={
    '20250807-1521':2,
    '20250814-1420':3,
    '20250821-1426':4,
    '20250828-1447':5,
    '20250904-1517':6,
    '20250911-1812':7,
    '20250918-1624':8,
    '20250925-1628':9,
    '20251009-1536':10,
    '20251016-1551':11,
    '20251030-1428':12,
    '20251106-1627':13
}


Z_MAX=0.05

plt.rcParams['svg.fonttype']='none'
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42
plt.rcParams['font.family']='Arial'

cell_bundle_df=pd.read_csv(cell_bundle_path,index_col=0)
unit_names=list(cell_bundle_df.columns)
unit_num=len(unit_names)

print("Tracking unit number:",unit_num)

unit_colors=plt.cm.viridis(np.linspace(0,1,unit_num))


def find_unit_position(recording_time,target_unit_id):
    phy_folder=os.path.join(root_folder_input,experimenter,acquisition_system,mouse_id,recording_time,"analysis","sorting_result","phy")
    if not os.path.exists(phy_folder):
        return None
    sorting=se.read_phy(phy_folder,exclude_cluster_groups=["noise","mua"])
    unit_ids=sorting.get_unit_ids()
    position=np.where(unit_ids==target_unit_id)[0]
    if len(position)==0:
        return None
    return int(position[0])


def load_ccgs(recording_time):
    ccgs_path=os.path.join(root_folder_input,experimenter,acquisition_system,mouse_id,recording_time,"analysis","sorting_result","waveform_from_phy","correlograms","ccgs.npy")
    if not os.path.exists(ccgs_path):
        print("Missing:",ccgs_path)
        return None
    return np.asarray(np.load(ccgs_path))


def get_unit_frequency(recording_time,unit_name):
    if recording_time not in cell_bundle_df.index:
        return None

    target_unit_id=cell_bundle_df.loc[recording_time,unit_name]

    if pd.isna(target_unit_id):
        return None

    target_unit_id=int(target_unit_id)

    position=find_unit_position(recording_time,target_unit_id)

    if position is None:
        return None

    ccgs=load_ccgs(recording_time)

    if ccgs is None or position>=ccgs.shape[0]:
        return None

    z=ccgs[position,position,:]

    if np.sum(z)>0:
        z=z/np.sum(z)

    return z


os.makedirs(save_dir,exist_ok=True)

for unit_index,unit_name in enumerate(unit_names):

    fig=plt.figure(figsize=(15,20),dpi=300)
    ax=fig.add_subplot(111,projection='3d')
    ax.grid(False)

    unit_color=unit_colors[unit_index]
    day_index=0
    valid_month=[]
    x=None

    for recording_time,month in time_mapping.items():

        z=get_unit_frequency(recording_time,unit_name)

        if z is None:
            continue

        if x is None:
            x=np.linspace(-50,50,len(z))

        y=np.full_like(x,day_index)
        z_bottom=np.min(z)

        vertices=np.column_stack((x,y,z))
        vertices=np.vstack((vertices,np.column_stack((x,y,np.full(x.shape,z_bottom)))))

        faces=[]

        for i in range(len(x)-1):
            faces.append([i,i+1,len(x)+i+1,len(x)+i])

        poly3d=Poly3DCollection([vertices[f] for f in faces],color=unit_color,alpha=0.3)

        ax.add_collection3d(poly3d)

        ax.plot(x,y,z,color=unit_color,linewidth=2)

        valid_month.append(month)
        day_index+=1


    ax.set_xlabel("Shift (ms)")
    ax.set_ylabel("Recording time (Month)")
    ax.set_zlabel("Frequency")

    ax.set_yticks(np.arange(day_index))
    ax.set_yticklabels([f"M{i}" for i in valid_month])

    ax.set_zlim(0,Z_MAX)
    ax.set_zticks(np.linspace(0,Z_MAX,2))

    ax.view_init(elev=30,azim=-60)
    ax.set_box_aspect([5,10,2])

    ax.xaxis.set_pane_color((0.85,0.85,0.85,1))
    ax.yaxis.set_pane_color((0.85,0.85,0.85,1))
    ax.zaxis.set_pane_color((0.85,0.85,0.85,1))

    ax.xaxis.pane.set_edgecolor("black")
    ax.yaxis.pane.set_edgecolor("black")
    ax.zaxis.pane.set_edgecolor("black")

    ax.set_title(unit_name,fontsize=16)

    save_path=os.path.join(save_dir,f"cellbundle-ccgs_{unit_name}.svg")

    plt.savefig(save_path,format="svg",dpi=300,transparent=True)

    plt.close()

    print("Saved:",save_path)

print("Finished.")