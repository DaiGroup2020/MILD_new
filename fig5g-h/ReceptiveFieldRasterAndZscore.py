# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 23:12:38 2025

Touch-receptive-field PSTH and Z-score computation and visualization — unit-first version.

This script extracts data in the order of unit first, then date. Given a
`unit_index`, it displays PSTH rasters and Z-score heatmaps for stimulus types
1–9 across all recording dates for that unit.

Each figure contains all dates for that unit, producing subplots with
columns: date0, date1, ..., dateN (total N+1 columns).

This revision uses a rainbow colormap and automatically scales the color range
based on the data range for each subplot.

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

# Load data
rasters = np.load(r"D:\desktop\SourceData\fig5Code and data\fig5g-h\touch_field-raster.npy", allow_pickle=True)
windows = np.load(r"D:\desktop\SourceData\fig5Code and data\fig5g-h\touch_field-events.npy", allow_pickle=True)

# Segment first to shape (unit_num, date_num, types, repeats, (x, y)), then
# find the maximum length among all 1D arrays
max_len = max(max(len(arr) for arr in row) for row in windows)

# Pad windows with edge values to equalize lengths when needed
padded_windows = [
    [np.pad(arr, (0, max_len - len(arr)), mode='edge') for arr in row]
    for row in windows
]
windows = np.asanyarray(padded_windows)

#%% PSTH extraction: clip rasters by windows and prepare plotting format

event_type_num = 9
repeat_time = 8
pre_stimuli_time_s = 1  # Defines raster clipping boundary (pre-stimulus)
post_stimuli_time_s = 1

psth = np.empty((rasters.shape[0], rasters.shape[1], event_type_num, repeat_time), dtype=object)
psth_timeWindow = np.empty((windows.shape[0], windows.shape[1], event_type_num, repeat_time), dtype=object)

for unit_index in range(rasters.shape[0]):
    for date_index in range(rasters.shape[1]):
        single_spike_time = rasters[unit_index, date_index]
        current_windows = windows[unit_index, date_index]
        pairs = [current_windows[i:i+2] for i in range(0, len(current_windows), 2)]
        types = [pairs[i:i+repeat_time] for i in range(0, len(pairs), repeat_time)]
        event_type_windows = np.asanyarray(types, dtype=object)
        
        for m in range(event_type_num):
            for n in range(repeat_time):
                time_window = event_type_windows[m][n]
                start_time = time_window[0] - pre_stimuli_time_s
                end_time = time_window[1] + post_stimuli_time_s
                raster_time = single_spike_time[(single_spike_time >= start_time) & (single_spike_time <= end_time)]
                psth[unit_index, date_index, m, n] = raster_time
                psth_timeWindow[unit_index, date_index, m, n] = time_window

# Rearrange dimensions from (unit, date, stimtype, repeat) to (unit, stimtype, date, repeat)
rearranged_psth = np.empty((psth.shape[0], psth.shape[2], psth.shape[1], psth.shape[3]), dtype=object)
rearranged_psth_timeWindow = np.empty((psth_timeWindow.shape[0], psth_timeWindow.shape[2], psth_timeWindow.shape[1], psth_timeWindow.shape[3]), dtype=object)

for i in range(psth.shape[0]):
    for j in range(psth.shape[2]):
        for k in range(psth.shape[1]):
            for l in range(psth.shape[3]):
                rearranged_psth[i, j, k, l] = psth[i, k, j, l]
                rearranged_psth_timeWindow[i, j, k, l] = psth_timeWindow[i, k, j, l]

#%% Plot PSTH — unit-first version

def plot_psth_by_unit(unit_index, psth_result, psth_timeWindow_result):
    """
    Plot PSTH rasters for the specified `unit_index` across all dates and
    stimulus types 1–9. Each row corresponds to a stimulus type and each
    column corresponds to a date (date0...dateN).

    Parameters:
    unit_index: int — index of the unit to analyze
    psth_result: numpy array — PSTH data (unit, stimtype, date, repeat)
    psth_timeWindow_result: numpy array — time window data (unit, stimtype, date, repeat)
    """
    
    event_types = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # stimulus types 1–9
    frequency = 20
    
    num_units = psth_result.shape[0]
    num_stimtypes = psth_result.shape[1]
    num_dates = psth_result.shape[2]
    num_repeats = psth_result.shape[3]
    
    # Create canvas: 9 rows (stimtypes 1–9) × num_dates columns
    fig, axes = plt.subplots(nrows=9, ncols=num_dates, figsize=(4*num_dates, 25))
    colormap = cm.get_cmap('tab10', num_units)  # tab10 colormap
    
    # Handle single-date case for axes shape
    if num_dates == 1:
        axes = axes.reshape(-1, 1)
    
    # Iterate over all stim types (1–9)
    for stim_type_index in range(9):
        # Iterate over all dates
        for date_index in range(num_dates):
            ax = axes[stim_type_index, date_index]
            
            # Extract raster data and time windows
            raster_data = psth_result[unit_index, stim_type_index, date_index]
            time_windows = psth_timeWindow_result[unit_index, stim_type_index, date_index]
            
            # Plot repeats
            for i in range(num_repeats):
                # Current repeat raster and time window
                current_raster = raster_data[i]
                current_time_window = time_windows[i]
                
                # Align raster on time axis
                aligned_raster = current_raster - current_time_window[0]
                x1 = current_time_window[0] - current_time_window[0]
                x2 = current_time_window[1] - current_time_window[0]
                
                # Offset for current repeat
                offset = i
                
                # Draw eventplot
                ax.eventplot(aligned_raster, lineoffsets=offset, linelengths=0.8, linewidths=0.5, colors=colormap(unit_index))
            
            # Draw square wave indicator — optimized size/placement
            t = np.linspace(-1, 2, 1000)  # time axis
            square_wave = np.where(((t % (1 / frequency)) < (0.5 / frequency)) & ((t >= x1) & (t < x2)), 1, 0)
            # Optimize square wave height and vertical placement
            square_wave_height = 0.6
            square_wave_position = -1.8
            ax.step(t, square_wave * square_wave_height + square_wave_position, where='post', color='r', linewidth=1, alpha=0.8)
            
            # Aesthetic tweaks
            # ax.set_title(f'Unit {unit_index}, Date {date_index}, Position {stim_type_index + 1}')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xlim(-0.2, 1.2)
            ax.set_ylim(-2.5, num_repeats - 0.5)  # leave room for square wave
            # Hide tick labels but keep ticks
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    
    # Layout adjustments
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.5, top=0.95)
    plt.suptitle(f'Unit {unit_index} - Raster Analysis', fontsize=16, y=0.99)
    plt.show()

#%% Compute and plot Z-score — unit-first version, rainbow colormap with auto scaling

def plot_zscore_by_unit(unit_index, psth_result, psth_timeWindow_result):
    """
    Plot Z-score heatmaps for the specified `unit_index` across all dates and
    stimulus types 1–9. Each row corresponds to a stimulus type and each
    column corresponds to a date (date0...dateN). Uses a rainbow colormap and
    automatically adjusts the color range based on data.

    Parameters:
    unit_index: int — index of the unit to analyze
    psth_result: numpy array — PSTH data (unit, stimtype, date, repeat)
    psth_timeWindow_result: numpy array — time window data (unit, stimtype, date, repeat)
    """
    
    # Parameter settings (kept consistent with original logic)
    trials_on_time_s = 1  # fill duration when start==end to equalize lengths
    interval_lines = 3  # spacing between dates (not drawn, reserved if needed)
    bin_time_s = 0.1  # histogram bin width (s)
    frequency = 20  # square wave frequency
    
    num_units = psth_result.shape[0]
    num_stimtypes = psth_result.shape[1]
    num_dates = psth_result.shape[2]
    num_repeats = psth_result.shape[3]
    
    # Use rainbow colormap
    custom_cmap = cm.get_cmap('rainbow')
    
    # Create canvas: 9 rows (stimtypes 1–9) × num_dates columns
    fig, axes = plt.subplots(nrows=9, ncols=num_dates, figsize=(4*num_dates, 25))
    colormap = cm.get_cmap('tab10', num_units)  # tab10 colormap
    
    # Handle single-date case for axes shape
    if num_dates == 1:
        axes = axes.reshape(-1, 1)
    
    # Iterate over all stim types (1–9)
    for stim_type_index in range(9):
        # Iterate over all dates
        for date_index in range(num_dates):
            ax = axes[stim_type_index, date_index]
            
            # Equalize Z-score lengths across repeats by equalizing bin_edges
            # (strictly following original logic)
            max_length = 0
            time_windows = psth_timeWindow_result[unit_index, stim_type_index, date_index]
            
            for i in range(num_repeats):
                current_time_window = time_windows[i]
                if current_time_window[0] == current_time_window[1]:
                    current_time_window[1] = current_time_window[0] + trials_on_time_s
                start = current_time_window[0] - pre_stimuli_time_s
                end = current_time_window[1] + post_stimuli_time_s
                bin_edges = np.arange(start, end + bin_time_s, bin_time_s)
                if len(bin_edges) - 1 > max_length:
                    max_length = len(bin_edges) - 1
            
            all_z_scores = []
            
            # Extract raster data and time windows
            raster_data = psth_result[unit_index, stim_type_index, date_index]
            time_windows = psth_timeWindow_result[unit_index, stim_type_index, date_index]
            
            # Initialize aligned_bin_edges (for imshow extents) and square wave params
            aligned_bin_edges = None
            x1 = 0
            x2 = 1
            
            # Iterate repeats (strictly following original logic)
            for i in range(num_repeats):
                # Current repeat raster (includes pre/post stimulus regions)
                current_raster = raster_data[i]
                current_time_window = time_windows[i]
                if current_time_window[0] == current_time_window[1]:
                    current_time_window[1] = current_time_window[0] + trials_on_time_s
                
                start = current_time_window[0] - pre_stimuli_time_s
                end = current_time_window[1] + post_stimuli_time_s
                baseline_time_window = [current_time_window[0] - pre_stimuli_time_s, current_time_window[0]]
                
                # Build bin_edges
                bin_edges = np.arange(start, end + bin_time_s, bin_time_s)
                # Ensure equal length across repeats; extend edges if needed
                while len(bin_edges) < max_length + 1:
                    bin_edges = np.append(bin_edges, bin_edges[-1] + bin_time_s)
                
                count_hist, bin_edges = np.histogram(current_raster, bins=bin_edges)
                firingrate_hist = count_hist / bin_time_s
                
                index = np.where((bin_edges >= baseline_time_window[0]) & (bin_edges <= baseline_time_window[1]))
                # Use count_hist or firing rate for baseline; raster timestamps are not averaged
                baseline_data = count_hist[index]
                
                # If baseline_data is all zeros, mean and std are zero -> NaNs
                # Ensure at least one non-zero to avoid NaNs
                if np.all(baseline_data == 0):
                    baseline_data[-1] = 1
                mean_baseline = np.mean(baseline_data)
                std_baseline = np.std(baseline_data)
                
                z_scores = (count_hist - mean_baseline) / std_baseline
                
                # Align bin_edges relative to trial onset for plotting
                aligned_bin_edges = bin_edges - current_time_window[0]
                z_scores_2d = z_scores.reshape(1, -1)
                
                x1 = current_time_window[0] - current_time_window[0]
                x2 = current_time_window[1] - current_time_window[0]
                
                # Offset for current repeat (row index)
                offset = i
                
                # Append current z_scores with its row offset
                all_z_scores.append((z_scores_2d, offset))
            
            # Build 2D matrix for the heatmap
            max_offset = num_repeats
            
            z_scores_matrix = np.zeros((max_offset, max_length))
            
            # Fill matrix rows with z-scores
            for z_scores_2d, offset in all_z_scores:
                z_scores_matrix[offset:offset + 1, :] = z_scores_2d
            
            # Determine data range per subplot for color scaling
            z_scores_flat = z_scores_matrix.flatten()
            # Remove NaNs
            z_scores_flat = z_scores_flat[~np.isnan(z_scores_flat)]
            
            if len(z_scores_flat) > 0:
                # Compute actual spread (keep symmetric around 0)
                vmin = np.percentile(z_scores_flat, 5)
                vmax = np.percentile(z_scores_flat, 95)
                abs_max = max(abs(vmin), abs(vmax))
                vmin = -8
                vmax = 8
            else:
                # Default range when no valid data
                vmin = -8
                vmax = 8
            
            # Use aligned_bin_edges from the last repeat to set heatmap extents
            if aligned_bin_edges is not None:
                cax = ax.imshow(z_scores_matrix, aspect='auto', cmap=custom_cmap, 
                               extent=[aligned_bin_edges[0], aligned_bin_edges[-1], 0, max_offset], 
                               vmin=vmin, vmax=vmax)
            else:
                # Fallback extents when aligned_bin_edges is None
                cax = ax.imshow(z_scores_matrix, aspect='auto', cmap=custom_cmap, 
                               extent=[-1, 2, 0, max_offset], 
                               vmin=vmin, vmax=vmax)
            
            # Per-subplot colorbar
            colorbar = fig.colorbar(cax, ax=ax)
            colorbar.set_label('Z-score')
            
            # Draw square wave indicator — optimized to avoid occluding data
            t = np.linspace(-1, 2, 1000)
            square_wave = np.where(((t % (1 / frequency)) < (0.5 / frequency)) & ((t >= x1) & (t < x2)), 1, 0)
            # Optimize height and vertical placement
            square_wave_height = 0.6
            square_wave_position = -1.8
            ax.step(t, square_wave * square_wave_height + square_wave_position, where='post', color='r', linewidth=1, alpha=0.8)
            
            # Aesthetic tweaks
            # ax.set_title(f'Unit {unit_index}, Date {date_index}, Position {stim_type_index + 1}')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xlim(-0.2, 1.2)
            ax.set_ylim(-2.5, num_repeats)  # leave room for square wave
            # Hide tick labels but keep ticks
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    
    # Layout adjustments
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.5, top=0.95)
    plt.suptitle(f'Unit {unit_index} - Z-score Analysis (All Dates) - Rainbow Light', fontsize=16, y=0.98)
    plt.show()


#%% Batch processing helper (optional)

def batch_process_all_units():
    """
    Batch process and plot for all units
    """
    num_units = rearranged_psth.shape[0]
    
    for unit_index in range(num_units):
        print(f"\nProcessing unit index: {unit_index}")
        
        # Plot PSTH rasters
        plot_psth_by_unit(unit_index, rearranged_psth, rearranged_psth_timeWindow)
        
        # Plot Z-score heatmaps
        plot_zscore_by_unit(unit_index, rearranged_psth, rearranged_psth_timeWindow)
        
        print(f"Unit {unit_index} done.")

# Uncomment the line below to batch process all units
batch_process_all_units()




