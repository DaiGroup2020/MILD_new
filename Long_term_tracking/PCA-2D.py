import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize,LinearSegmentedColormap,rgb_to_hsv,hsv_to_rgb,to_rgb
from matplotlib.cm import ScalarMappable
import os

data_folder=r"H:\WXY\MILD-DATA\blackrock\5125"
raw_path=os.path.join(data_folder,"WXY_blackrock_5125_spikeform_cluster.csv")
center_path=os.path.join(data_folder,"WXY_blackrock_5125_template_cluster.csv")

FIG_SIZE=(5,5)

target_units=[
    "Unit_1",
    "Unit_7",
    "Unit_8",
    "Unit_10",
    "Unit_11",
    "Unit_13"
]
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

# RGB 0-255
base_color=(177,221,206)

plt.rcParams['svg.fonttype']='none'
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42
plt.rcParams['font.family']='Arial'

def load_and_prep(path):
    df=pd.read_csv(path)
    df.columns=['Time','UnitName','PC1','PC2','PC3']
    df['Month']=df['Time'].map(time_mapping)
    return df.dropna(subset=['Month'])

def normalize_rgb(color):
    if isinstance(color,(tuple,list)):
        return np.array(color)/255
    else:
        return np.array(to_rgb(color))

def generate_unit_colors(base_color,n_units):
    rgb=normalize_rgb(base_color)
    hsv=rgb_to_hsv(rgb.reshape(1,1,3))[0,0]

    colors=[]

    # 固定色相，只改变饱和度和亮度
    sat_values=np.linspace(0.7,1.25,n_units)
    val_values=np.linspace(0.55,1.0,n_units)

    for s,v in zip(sat_values,val_values):
        new_hsv=hsv.copy()

        new_hsv[0]=hsv[0]
        new_hsv[1]=np.clip(hsv[1]*s,0,1)
        new_hsv[2]=np.clip(hsv[2]*v,0.25,1)

        colors.append(
            tuple(hsv_to_rgb(new_hsv))
        )

    return colors

def lighten_color(color,amount=0.55):
    rgb=np.array(color)
    return tuple(rgb+(1-rgb)*amount)

def create_unit_cmap(unit_color):
    return LinearSegmentedColormap.from_list(
        "unit_gradient",
        [
            lighten_color(unit_color),
            unit_color
        ]
    )

def confidence_ellipse(x,y,ax,color,n_std=2):
    if len(x)<5:
        return

    cov=np.cov(x,y)

    if np.linalg.det(cov)<=0:
        return

    vals,vecs=np.linalg.eigh(cov)

    order=vals.argsort()[::-1]
    vals=vals[order]
    vecs=vecs[:,order]

    theta=np.degrees(
        np.arctan2(vecs[1,0],vecs[0,0])
    )

    width=2*n_std*np.sqrt(vals[0])
    height=2*n_std*np.sqrt(vals[1])

    ellipse=patches.Ellipse(
        xy=(np.mean(x),np.mean(y)),
        width=width,
        height=height,
        angle=theta,
        fill=False,
        linewidth=2,
        edgecolor=color
    )

    ax.add_patch(ellipse)

def main():

    df_raw=load_and_prep(raw_path)
    df_center=load_and_prep(center_path)

    all_target=df_raw[
        df_raw['UnitName'].isin(target_units)
    ]

    x_min=all_target['PC1'].min()
    x_max=all_target['PC1'].max()
    y_min=all_target['PC2'].min()
    y_max=all_target['PC2'].max()

    x_pad=(x_max-x_min)*0.1
    y_pad=(y_max-y_min)*0.1

    months=sorted(time_mapping.values())

    norm=Normalize(
        vmin=min(months),
        vmax=max(months)
    )

    unit_colors=generate_unit_colors(
        base_color,
        len(target_units)
    )

    unit_color_map=dict(
        zip(
            target_units,
            unit_colors
        )
    )

    print("Unit colors:")
    for k,v in unit_color_map.items():
        print(k,v)

    for unit_name in target_units:

        fig,ax=plt.subplots(
            figsize=FIG_SIZE,
            dpi=300
        )

        cmap=create_unit_cmap(
            unit_color_map[unit_name]
        )

        u_raw=df_raw[
            df_raw['UnitName']==unit_name
        ]

        u_center=df_center[
            df_center['UnitName']==unit_name
        ].sort_values('Month')

        if u_raw.empty:
            print("Skip:",unit_name)
            continue

        for month in months:

            data=u_raw[
                u_raw['Month']==month
            ]

            if data.empty:
                continue

            color=cmap(
                norm(month)
            )

            confidence_ellipse(
                data['PC1'].values,
                data['PC2'].values,
                ax,
                color
            )

            center=u_center[
                u_center['Month']==month
            ]

            if not center.empty:

                ax.scatter(
                    center['PC1'],
                    center['PC2'],
                    s=35,
                    color=color,
                    edgecolors='white',
                    linewidths=0.8,
                    zorder=5
                )

        ax.set_xlim(
            x_min-x_pad,
            x_max+x_pad
        )

        ax.set_ylim(
            y_min-y_pad,
            y_max+y_pad
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        ax.set_title(
            f"Cluster compactness: {unit_name}",
            fontsize=14,
            fontweight='bold'
        )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

        sm=ScalarMappable(
            cmap=cmap,
            norm=norm
        )

        sm.set_array([])

        cbar=fig.colorbar(
            sm,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

        cbar.set_label("Month")

        save_path=os.path.join(
            data_folder,
            f"PCA_compactness_{unit_name}.svg"
        )

        plt.savefig(
            save_path,
            format="svg",
            dpi=300,
            transparent=True
        )

        plt.close()

        print("Saved:",save_path)

if __name__=="__main__":
    main()