# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 23:12:38 2025

Combined functionality of touchField_nomalization.py and ZSCORE_TouchField.py.
Includes two Touch Field heatmap methods:
1) Normalized averaged firing rate heatmaps (using the global minimum as baseline)
2) Normalized Z-Score heatmaps based on firing rate data (using the global minimum as baseline)

Author: ZhaoYan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# Global font configuration: use sans-serif with Arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

#%% Method 1: Normalized heatmaps based on firing rate data

def load_and_process_firing_rate_data():
    """
    Load and process firing rate data
    """
    print("=== Method 1: Normalized heatmap based on firing rate data ===")
    
    # Load firing rate data
    npy_file = r"D:\desktop\SourceData\fig5Code and data\fig5g-h\touch_field-events_ave_firing_rate.npy"
    data = np.load(npy_file, allow_pickle=True)
    
    date_index = data.shape[1]
    unit_index = data.shape[0]
    
    print(f"Firing rate data shape: {unit_index} units × {date_index} dates")
    
    # Compute mean and std for each position
    all_ave_touch_field = []
    all_std_touch_field = []
    
    for i in range(unit_index):
        single_ave_touch_field = []
        single_std_touch_field = []
        for j in range(date_index):
            aves = []
            stds = []
            data_temp = np.array(data[i][j])
            for k in range(0, len(data_temp), 8):
                # Take the current 8 elements (one stim_type block)
                segment = data_temp[k:k+8]
                np.array(segment)
                # Compute mean and std
                ave = np.mean(segment)
                std = np.std(segment)    
                # Append results
                aves.append(ave)
                stds.append(std)
            single_ave_touch_field.append(np.array(aves))
            single_std_touch_field.append(np.array(stds))
        all_ave_touch_field.append(single_ave_touch_field)
        all_std_touch_field.append(single_std_touch_field)
    
    return all_ave_touch_field, all_std_touch_field, unit_index, date_index

def plot_firing_rate_heatmaps(all_ave_touch_field, all_std_touch_field, unit_index, date_index):
    """
    Plot heatmaps based on firing rate data
    """
    # Compute global minimum as normalization baseline
    print("Computing global minimum as normalization baseline...")
    all_values = []
    for i in range(unit_index):
        for j in range(date_index):
            all_values.extend(all_ave_touch_field[i][j])
    global_min = np.min(all_values)
    print(f"Global minimum firing rate: {global_min}")
    
    # Normalize using global minimum
    all_normalized_touch_field = []    
    for i in range(unit_index):
        single_normalized_touch_field = []
        for j in range(date_index):
            # Normalization: (value - global_min) / global_min
            normalized_values = (all_ave_touch_field[i][j] - global_min) / global_min
            single_normalized_touch_field.append(normalized_values)
        all_normalized_touch_field.append(single_normalized_touch_field)
    
    # Also compute z-scores for comparison
    all_zs_touch_field = []    
    for i in range(unit_index):
        single_zs_touch_field = []
        for j in range(date_index):
            zs = stats.zscore(all_ave_touch_field[i][j])
            single_zs_touch_field.append(zs)
        all_zs_touch_field.append(single_zs_touch_field)
    
    # 1) Original firing rate heatmaps
    print("Plotting original firing rate heatmaps...")
    fig, axs = plt.subplots(date_index, unit_index, figsize=(20, 15))
    
    for i in range(unit_index):
        for j in range(date_index):
            ax = axs[j][i]
            position_data = all_ave_touch_field[i][j].reshape(3,3)
            cax = ax.imshow(position_data, cmap='jet', interpolation='gaussian')
            fig.colorbar(cax, ax=ax, shrink=0.8, pad=0.05)
            ax.set_title('unit_index{},date_index{}'.format(i, j))
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.suptitle('Original Firing Rate Heatmap', fontsize=16, y=0.98)
    plt.show()
    
    # 2) Normalized heatmaps (white to red)
    print("Plotting normalized heatmaps...")
    fig, axs = plt.subplots(date_index, unit_index, figsize=(20, 15))
    
    # Custom colormap: white to red
    colors = ['white', 'red']
    n_bins = 256
    #custom_cmap = LinearSegmentedColormap.from_list('white_to_red', colors, N=n_bins)
    
    for i in range(unit_index):
        for j in range(date_index):
            ax = axs[j][i]
            normalized_data = all_normalized_touch_field[i][j].reshape(3,3)
            cax = ax.imshow(normalized_data, cmap='jet', interpolation='gaussian')
            fig.colorbar(cax, ax=ax, shrink=0.8, pad=0.05)
            ax.set_title('unit_index{},date_index{}'.format(i, j))
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.suptitle('Normalized Heatmap (Global Min as Baseline)', fontsize=16, y=0.98)
    plt.show()
    
    # 3) Z-score heatmaps (for comparison)
    print("Plotting z-score heatmaps for comparison...")
    fig, axs = plt.subplots(date_index, unit_index, figsize=(20, 15))
    
    for i in range(unit_index):
        for j in range(date_index):
            ax = axs[j][i]
            zs = all_zs_touch_field[i][j].reshape(3,3)
            cax = ax.imshow(zs, cmap='jet', interpolation='gaussian')
            fig.colorbar(cax, ax=ax, shrink=0.8, pad=0.05)
            ax.set_title('unit_index{},date_index{}'.format(i, j))
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.suptitle('Original Z-score Heatmap (for comparison)', fontsize=16, y=0.98)
    plt.show()
    
    # Statistics output
    print("\n=== Firing rate statistics ===")
    print(f"Data shape: {unit_index} units × {date_index} dates")
    print(f"Each unit/date has 9 positions")
    print(f"Global minimum firing rate: {global_min:.6f}")
    
    # Normalized values statistics
    all_normalized_values = []
    for i in range(unit_index):
        for j in range(date_index):
            all_normalized_values.extend(all_normalized_touch_field[i][j])
    
    print(f"Normalized value range: [{np.min(all_normalized_values):.6f}, {np.max(all_normalized_values):.6f}]")
    print(f"Normalized mean: {np.mean(all_normalized_values):.6f}")
    print(f"Normalized std: {np.std(all_normalized_values):.6f}")
    
    # Z-score statistics
    all_zs_values = []
    for i in range(unit_index):
        for j in range(date_index):
            all_zs_values.extend(all_zs_touch_field[i][j])
    
    print(f"Z-score value range: [{np.min(all_zs_values):.6f}, {np.max(all_zs_values):.6f}]")
    print(f"Z-score mean: {np.mean(all_zs_values):.6f}")
    print(f"Z-score std: {np.std(all_zs_values):.6f}")



#%% Main

def main():
    """
    Main entry — run the two Touch Field heatmap methods
    """
    print("Starting Touch Field heatmap analysis...")
    
    # Method 1: Normalized heatmap based on firing rate data
    all_ave_touch_field, all_std_touch_field, unit_index, date_index = load_and_process_firing_rate_data()
    plot_firing_rate_heatmaps(all_ave_touch_field, all_std_touch_field, unit_index, date_index)
    

    
    print("\nAll analyses complete!")

# Run main when executed as a script
if __name__ == "__main__":
    main()
