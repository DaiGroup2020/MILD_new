"""
Frequency tuning analysis and visualization

This module loads per-event firing-rate style data arranged as a 2D array
with shape (num_units, num_dates), where each cell stores a 1D array of
values ordered by repeated presentations of event types.

It provides two analyses:
1) Spikes-per-cycle normalization and capping based on a variability rule,
   then 3D visualization per unit across dates.
2) Aggregation of tuning curves (mean and std across repeats) and 3D
   visualization of repeats with mean ± std overlays across dates.

All comments and docstrings are in English for journal submission.
"""

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_DATA_FILE = (
    r"D:\desktop\SourceData\fig5Code and data\fig5d\6f-events_firing_rate.npy"
)

# The 10 event frequencies used by the dataset; order must match the data.
DEFAULT_FREQUENCIES: List[float] = [2, 5, 10, 20, 40, 60, 80, 180, 240, 480]

# Number of distinct event types per repeat block
DEFAULT_EVENT_TYPES = 10


# -----------------------------------------------------------------------------
# Data IO
# -----------------------------------------------------------------------------

def load_data(file_path: str) -> np.ndarray:
    """Load the numpy array containing tuning data.

    The expected object is a 2D array with shape (num_units, num_dates); each
    cell is a 1D array containing repeated event blocks ordered by event type.
    """
    data = np.load(file_path, allow_pickle=True)
    return data


# -----------------------------------------------------------------------------
# Computations
# -----------------------------------------------------------------------------

def compute_spikes_per_cycle(
    data: np.ndarray,
    frequencies: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute spikes per cycle per condition and an optionally capped version.

    For each unit/date cell, elements are partitioned by event type using
    index % num_event_types. Each raw value is divided by the corresponding
    frequency to obtain spikes per cycle. For each event type, we compute the
    mean and std across repeats. A capped version is produced where the mean is
    set to 1 if mean + 1*std >= 0.9.

    Returns
    -------
    ave_spc : ndarray
        Array with shape (num_units, num_dates, num_event_types) containing the
        mean spikes per cycle per condition.
    capped_spc : ndarray
        Same shape; values capped to 1 under the rule (mean + std >= 0.9).
    """
    num_units = data.shape[0]
    num_dates = data.shape[1]
    num_event_types = len(frequencies)

    ave_spc: List[List[np.ndarray]] = []
    capped_spc: List[List[np.ndarray]] = []

    for unit_idx in range(num_units):
        unit_ave: List[np.ndarray] = []
        unit_capped: List[np.ndarray] = []
        for date_idx in range(num_dates):
            raw = np.asarray(data[unit_idx][date_idx])
            per_type_values: List[List[float]] = [[] for _ in range(num_event_types)]
            for k, value in enumerate(raw):
                et = k % num_event_types
                spc = float(value) / float(frequencies[et])
                per_type_values[et].append(spc)

            means: List[float] = []
            capped_means: List[float] = []
            for values in per_type_values:
                m = float(np.mean(values)) if len(values) > 0 else 0.0
                s = float(np.std(values)) if len(values) > 0 else 0.0
                c = 1.0 if (m + s) >= 0.9 else m
                means.append(m)
                capped_means.append(c)

            unit_ave.append(np.asarray(means, dtype=float))
            unit_capped.append(np.asarray(capped_means, dtype=float))
        ave_spc.append(unit_ave)
        capped_spc.append(unit_capped)

    return np.asarray(ave_spc, dtype=float), np.asarray(capped_spc, dtype=float)


def aggregate_tuning_curves(
    data: np.ndarray,
    num_event_types: int,
) -> Tuple[np.ndarray, np.ndarray, List[List[List[np.ndarray]]]]:
    """Aggregate tuning curves per unit/date across repeats.

    For each unit/date:
      - Partition by event type (index % num_event_types) to compute mean/std
        across repeats, yielding arrays of length num_event_types.
      - Slice raw data into repeat blocks of length num_event_types,
        preserving per-repeat row vectors for visualization.

    Returns
    -------
    ave : ndarray of shape (num_units, num_dates, num_event_types)
    std : ndarray of shape (num_units, num_dates, num_event_types)
    raw_blocks : nested lists indexed as [unit][date] -> list of repeat arrays
    """
    num_units = data.shape[0]
    num_dates = data.shape[1]

    all_ave: List[List[np.ndarray]] = []
    all_std: List[List[np.ndarray]] = []
    raw_blocks: List[List[List[np.ndarray]]] = []

    for unit_idx in range(num_units):
        unit_ave: List[np.ndarray] = []
        unit_std: List[np.ndarray] = []
        unit_raw_blocks: List[List[np.ndarray]] = []

        for date_idx in range(num_dates):
            raw = np.asarray(data[unit_idx][date_idx])

            # Group values by event type across repeats
            per_type_values: List[List[float]] = [[] for _ in range(num_event_types)]
            for k, value in enumerate(raw):
                et = k % num_event_types
                per_type_values[et].append(float(value))

            means = [float(np.mean(v)) if len(v) > 0 else 0.0 for v in per_type_values]
            stds = [float(np.std(v)) if len(v) > 0 else 0.0 for v in per_type_values]

            # Slice raw into repeat-length blocks
            blocks: List[np.ndarray] = [
                raw[i : i + num_event_types] for i in range(0, len(raw), num_event_types)
            ]

            unit_ave.append(np.asarray(means, dtype=float))
            unit_std.append(np.asarray(stds, dtype=float))
            unit_raw_blocks.append([np.asarray(b, dtype=float) for b in blocks])

        all_ave.append(unit_ave)
        all_std.append(unit_std)
        raw_blocks.append(unit_raw_blocks)

    return np.asarray(all_ave, dtype=float), np.asarray(all_std, dtype=float), raw_blocks


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_unit_tracks_3d(
    data_3d: np.ndarray,
    x_values: List[float],
    title: str,
) -> None:
    """Plot 3D tracks for each unit across dates.

    Parameters
    ----------
    data_3d : ndarray
        Shape (num_units, num_dates, num_event_types).
    x_values : list of float
        X-axis values per event type (e.g., frequencies).
    title : str
        Figure title.
    """
    if isinstance(data_3d, list):
        data_3d = np.asarray(data_3d, dtype=float)

    num_units = data_3d.shape[0]
    num_dates = data_3d.shape[1]

    colors = plt.cm.viridis(np.linspace(0, 1, num_units))
    fig = plt.figure(figsize=(18, 10))

    # One subplot per unit
    for u in range(num_units):
        ax = fig.add_subplot(int(np.ceil(num_units / 2.0)), 2, u + 1, projection='3d')
        ax.set_title(f"Unit {u}")
        for d in range(num_dates):
            y = np.full(len(x_values), d, dtype=float)
            z = data_3d[u, d]
            ax.plot(
                x_values,
                y,
                z,
                color=colors[u],
                marker='o',
                markersize=5,
                linewidth=1.5,
            )
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Date Index')
        ax.set_zlabel('Response')
        ax.set_yticks(list(range(num_dates)))
        ax.grid(False)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_repeats_with_mean_std_3d(
    raw_repeat_blocks: List[List[List[np.ndarray]]],
    x_values: List[float],
    title: str,
) -> None:
    """Plot per-unit 3D repeats per date with mean ± std overlays.

    raw_repeat_blocks is indexed as [unit][date] -> list of repeat arrays
    where each repeat array has length num_event_types.
    """
    num_units = len(raw_repeat_blocks)
    for u in range(num_units):
        per_dates = raw_repeat_blocks[u]
        days = [f"Day {i+1}" for i in range(len(per_dates))]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
    
        offsets = np.arange(len(per_dates), dtype=float)
        for d_idx, repeats in enumerate(per_dates):
            if not repeats:
                continue
            # Convert to 2D array: (num_repeats, num_event_types)
            R = np.vstack([np.asarray(r, dtype=float) for r in repeats])
            mean_values = np.mean(R, axis=0)
            std_values = np.std(R, axis=0)

            # Plot individual repeats
            for row in R:
                ax.plot(x_values, [offsets[d_idx]] * len(row), row, color='black', alpha=0.25)

            # Plot mean
            ax.plot(x_values, [offsets[d_idx]] * len(mean_values), mean_values, label=f"{days[d_idx]} Mean", linewidth=2)

            # Plot mean ± std as bands using two lines
            x = np.asarray(x_values, dtype=float)
            y = np.full_like(x, offsets[d_idx], dtype=float)
            lower = mean_values - std_values
            upper = mean_values + std_values
            ax.plot(x, y, lower, color='gray', alpha=0.6, linewidth=1)
            ax.plot(x, y, upper, color='gray', alpha=0.6, linewidth=1)

        ax.set_title(f"{title} - Unit {u}")
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Date Index')
        ax.set_zlabel('Response')
        ax.set_yticks(list(range(len(per_dates))))
        ax.legend()
        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    # Load data
    data = load_data(DEFAULT_DATA_FILE)

    # Analysis 1: spikes per cycle (mean and capped) and 3D tracking plots
    ave_spc, capped_spc = compute_spikes_per_cycle(data, DEFAULT_FREQUENCIES)
   
    plot_unit_tracks_3d(
        capped_spc,
        x_values=DEFAULT_FREQUENCIES,
        title='Spikes per Cycle (Capped by Variability Rule) across Dates',
    )

    # Analysis 2: aggregate tuning curves and visualize repeats with mean ± std
    ave, std, raw_blocks = aggregate_tuning_curves(data, num_event_types=DEFAULT_EVENT_TYPES)
    plot_repeats_with_mean_std_3d(
        raw_repeat_blocks=raw_blocks,
        x_values=DEFAULT_FREQUENCIES,
        title='Tuning Curves (Repeats, Mean ± Std) across Dates',
    )


if __name__ == "__main__":
    main()
            

