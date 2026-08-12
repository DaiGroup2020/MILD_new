## Figure 5 Code and Data — Analysis and Visualization

This repository contains Python scripts and NumPy data used to reproduce Figure 5 analyses. There are two major modules:

- `fig5c-d/`: Frequency tuning analyses and 3D visualizations.
- `fig5g-h/`: Touch receptive field analyses (rasters, Z-scores, and heatmaps).

The code assumes Windows paths and uses `matplotlib` for visualization.

---

## Directory Layout

```
fig5Code and data/
  ├─ fig5c-d/
  │   ├─ 5c-events_firing_rate.npy
  │   ├─ 5c-tuning_curve-events.npy
  │   ├─ 5c-tuning_curve-raster.npy
  │   ├─ 5d-events_firing_rate.npy
  │   ├─ 5d-tuning_curve-events.npy
  │   ├─ 5d-tuning_curve-raster.npy
  │   └─ FrequencyTuningCode.py
  └─ fig5g-h/
      ├─ ReceptiveFieldRasterAndZscore.py
      ├─ TouchReceptiveFieldHeatmap.py
      ├─ touch_field-raster.npy
      ├─ touch_field-events.npy
      └─touch_field-events_ave_firing_rate.npy
      
```

---

## Environment

- OS: Windows 10+
- Python: 3.8–3.11
- Packages: `numpy`, `matplotlib`, `scipy`

Install:

```bash
pip install numpy matplotlib scipy
```

Note: Scripts set `matplotlib` to use the `Arial` font. If `Arial` is unavailable, change to an installed font (e.g., `SimHei`).

---

## Data Structures (All .npy files)

Unless otherwise noted, data files are NumPy object arrays with shape `[num_units, num_dates]`. Each cell stores a 1D array of values for that unit/date across stimulus event types and repeats.

- Frequency tuning data (Fig. 5c–d):
  - `5d-events_firing_rate.npy` or `5c-events_firing_rate.npy`: per-event or averaged firing rates; 1D sequences ordered by event type with repeats. Event type count is 10 by default.
  - `5d-tuning_curve-events.npy`, `5d-tuning_curve-raster.npy` and their 5c counterparts: aligned event timing windows and spike rasters; used by PSTH workflows.
- Touch receptive field data (Fig. 5g–h):
  - `touch_field-raster.npy`: `[unit, date]` where each element is a 1D array of spike timestamps (seconds).
  - `touch_field-events.npy`: `[unit, date]` aligned event start/end times; events are segmented into 9 positions with 8 repeats each (types=9, repeats=8).
  - `touch_field-events_ave_firing_rate.npy`: `[unit, date]` where each 1D sequence length is divisible by 8; every group of 8 forms one position. 9 positions map to a 3×3 grid.

---

## Module: fig5c-d (Frequency Tuning)

### Script: `FrequencyTuningCode.py`

Performs two analyses on frequency tuning data arranged as `[unit, date]` with 1D sequences of repeated event-type blocks:

1) Spikes-per-cycle analysis with variability-based capping, followed by per-unit 3D tracks across dates.
2) Aggregation of tuning curves (mean and std across repeats) and 3D visualization of repeats with mean ± std overlays per date.

Key defaults and assumptions:

- Event types: 10 (`DEFAULT_EVENT_TYPES = 10`)
- Frequencies (x-axis): `[2, 5, 10, 20, 40, 60, 80, 180, 240, 480]`
- Input data shape: `[num_units, num_dates]`; each cell is a 1D sequence whose indices map to event types by `index % num_event_types`.

Data loading:

- The script uses an absolute default path `DEFAULT_DATA_FILE`. Update it to the correct file, for example:

```python
DEFAULT_DATA_FILE = r"D:\desktop\SourceData\fig5Code and data\fig5c-d\5d-events_firing_rate.npy"
```

Core functions:

- `load_data(path)`: loads the `[unit, date]` object array.
- `compute_spikes_per_cycle(data, frequencies) -> (ave_spc, capped_spc)`: partitions by event type, divides by frequency to get spikes per cycle, computes mean/std across repeats, and applies capping rule `mean + std >= 0.9 → 1.0`.
- `aggregate_tuning_curves(data, num_event_types) -> (ave, std, raw_blocks)`: aggregates mean/std per event type and slices raw data into repeat-length blocks.
- `plot_unit_tracks_3d(data_3d, x_values, title)`: 3D line plots for each unit across dates.
- `plot_repeats_with_mean_std_3d(raw_repeat_blocks, x_values, title)`: plots all repeats with mean ± std bands per date.

How to run:

```bash
python fig5c-d/FrequencyTuningCode.py
```

Expected outputs:

- 3D figures per analysis showing frequency tuning across dates and repeats.

Notes:

- If your source file is `5c-events_firing_rate.npy`, ensure that it contains repeated blocks of length 10 (one per event type) so that `% 10` indexing is valid. Adjust `DEFAULT_EVENT_TYPES` and `DEFAULT_FREQUENCIES` if your dataset differs.

---

## Module: fig5g-h (Touch Receptive Field)

This module has two complementary scripts. See the detailed `fig5g-h/README.md` for more.

### Script: `ReceptiveFieldRasterAndZscore.py`

Builds PSTH-like rasters and Z-score heatmaps per unit across stimulus positions (1–9) and dates, using spike rasters and event windows.

Key assumptions:

- Positions: 9 (`event_type_num = 9`)
- Repeats per position: 8 (`repeat_time = 8`)
- Windowing: pre = 1 s, post = 1 s
- Histogram bin width for Z-score: 0.1 s; baseline window is `[start−1s, start]`. If an event has `start == end`, the script sets `end = start + 1 s`.

I/O:

- Inputs: `touch_field-raster.npy`, `touch_field-events.npy`
- Output: Two figures per unit (rasters and Z-score heatmaps), arranged with rows = positions (1–9), columns = dates.

How to run:

```bash
python fig5g-h/ReceptiveFieldRasterAndZscore.py
```

Optional (single unit in interactive mode):

```python
from fig5g-h.ReceptiveFieldRasterAndZscore import (
    plot_psth_by_unit,
    plot_zscore_by_unit,
    rearranged_psth,
    rearranged_psth_timeWindow,
)
unit_index = 0
plot_psth_by_unit(unit_index, rearranged_psth, rearranged_psth_timeWindow)
plot_zscore_by_unit(unit_index, rearranged_psth, rearranged_psth_timeWindow)
```

### Script: `TouchReceptiveFieldHeatmap.py`

Generates three 3×3 heatmap variants from averaged firing-rate data per unit/date:

1) Original average firing-rate heatmap
2) Globally normalized heatmap using the global minimum as baseline: `(value - global_min) / global_min`
3) Z-score heatmap of positions

Assumptions:

- Every 8 values form one stimulus-type block; 9 blocks form 9 positions → 3×3 grid per unit/date.

I/O:

- Input: `touch_field-events_ave_firing_rate.npy`
- Output: Three heatmap sets (date × unit) and printed stats (global minimum, normalized/Z-score ranges/means/stds).

How to run:

```bash
python fig5g-h/TouchReceptiveFieldHeatmap.py
```

---

## Usage Checklist

1. Install dependencies: `pip install numpy matplotlib scipy`.
2. Ensure `.npy` paths inside scripts match your local directory. Update any absolute paths (e.g., `DEFAULT_DATA_FILE` in `FrequencyTuningCode.py`).
3. Run the desired script(s) from the repository root or their respective subfolders.

---

## Troubleshooting

- File not found: Update `np.load(...)` absolute paths in the scripts to match your environment.
- Font issues: Change `plt.rcParams['font.sans-serif']` to a font installed on your system.
- Color scaling/visibility: Tweak figure sizes, colormaps, or Z-score `vmin/vmax` parameters in the plotting functions.

---

## Citation

For academic use, please cite appropriately and credit the author.


