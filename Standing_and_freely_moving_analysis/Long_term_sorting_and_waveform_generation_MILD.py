# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings('ignore')
import pprint
from timeit import default_timer
import numpy as np
import pandas as pd
import shutil
from scipy.signal import resample_poly
import probeinterface as pi
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
import spikeinterface as si
import spikeinterface.exporters as sexp
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
from function_utils import _compute_fwhm_basic
from pprint import pprint
from spikeinterface.postprocessing.template_metrics import get_peak_to_valley, get_peak_trough_ratio, get_half_width, \
    get_repolarization_slope, get_recovery_slope, get_trough_and_peak_idx
import json
from scipy.spatial.distance import pdist, squareform




MILD_mouse_list = ['601', '602', '610']
recording_id_checklist = {'601': '20260519-0000',  '602': '20260519-0000', '610': '20260519-0000'}
recording_start_phase_checklist = {'601': 'before', '602': 'before', '610': 'before'}
# 修改后：填入你实际想切分的两个时间点（单位：秒）
recording_phase_change_time_checklist = {
    '601': [600, 1200],
    '602': [600, 1200],
    '610': [600, 1200]
}


phase_change_margin = 1.50

def get_amplitude_between_peak_and_trough(template):

    trough_idx, peak_idx = get_trough_and_peak_idx(template)
    amplitude = template[peak_idx] - template[trough_idx]
    return amplitude

def read_sorting_metadata(root_folder_input, experimenter, acquisition_system, mouse_id, recording_time, display=False):
    if display:
        time_start = default_timer()
        print('============================================================================')
        print('data: ', experimenter, acquisition_system, mouse_id, recording_time)
        print('CHECK SORTING METADATA MODULE')
    if not root_folder_input.endswith('/'):
        root_folder_input = root_folder_input + '/'
    sorting_result_folder = root_folder_input + '/' + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                            '/' + recording_time + '/analysis/sorting_result/'
    sorting_metadata_path = sorting_result_folder + 'sorting_metadata.npy'
    if os.path.isfile(sorting_metadata_path):
        sorting_metadata = np.load(sorting_metadata_path, allow_pickle=True).item()
    else:
        sorting_metadata = None
        if display:
            print(sorting_metadata_path + ' does not exist, please sort data before checking sorting metadata')
    if display:
        print('Sorting metadata is:')
        pprint.pprint(sorting_metadata)
        time_end = default_timer()
        print('Module time usage(s): ', time_end - time_start)
    return sorting_metadata


def single_recording_all_phase_sorting_and_waveform_generation(root_folder_input, experimenter, acquisition_system,
                                                               mouse_id, recording_time, phase_change_times,
                                                               start_phase):
    time_start = default_timer()
    print('============================================================================')
    print('data: ', experimenter, acquisition_system, mouse_id, recording_time)
    print('WAVEFORM EXTRACTION MODULE (Manual 3-Phase Splitting with Margin)')
    print('Routing...')
    if not root_folder_input.endswith('/'):
        root_folder = root_folder_input + '/'
    else:
        root_folder = root_folder_input
    raw_data_folder = root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + '/' + recording_time + '/raw_data/'
    sorting_result_folder = root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + '/' + recording_time + '/analysis/sorting_result/'

    assert len(phase_change_times) == 2, "请在 checklist 中提供两个时间点，例如 [600, 1200]"
    time1, time2 = phase_change_times[0], phase_change_times[1]

    # ---- 1. 创建文件夹 ----
    paths = {
        'waveform': sorting_result_folder + 'waveform/',
        'before_waveform': sorting_result_folder + 'before_waveform/',
        'middle_waveform': sorting_result_folder + 'middle_waveform/',
        'after_waveform': sorting_result_folder + 'after_waveform/',
        'before_phy': sorting_result_folder + 'before_phy/',
        'middle_phy': sorting_result_folder + 'middle_phy/',
        'after_phy': sorting_result_folder + 'after_phy/'
    }
    for p in paths.values():
        if not os.path.isdir(p):
            os.makedirs(p)

    sorting_metadata = read_sorting_metadata(root_folder_input, experimenter, acquisition_system, mouse_id,
                                             recording_time, display=False)

    # ---- 2. 数据读取与预处理（保持原样） ----
    print('reading recordings...')
    raw_data_list = os.listdir(raw_data_folder)
    if acquisition_system == 'intan':
        recording_data_list = [i for i in raw_data_list if i.endswith('.rhd') or i.endswith('.rhs')]
        recording = si.concatenate_recordings(
            [se.read_intan(raw_data_folder + r, stream_id='0') for r in recording_data_list])
    elif acquisition_system == 'blackrock':
        recording_data_list = [i for i in raw_data_list if i.endswith('.ns6')]
        recording = si.concatenate_recordings([se.read_blackrock(raw_data_folder + r) for r in recording_data_list])
    elif acquisition_system == 'alpha':
        recording = se.read_alphaomega(raw_data_folder)
    else:
        raise Exception('acquisition_system currently should either be intan or blackrock')

    sampling_frequency = recording.get_sampling_frequency()
    n_sample = recording.get_num_samples()

    probe_layout, impedance_list = sorting_metadata['layout'], sorting_metadata['impedance']
    impedance_threshold, auto_bad_channel_detection_flag = sorting_metadata['impedance_threshold'], sorting_metadata[
        'auto_bad_channel_detection']
    probe = pi.Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=probe_layout, shapes='circle', shape_params={'radius': 10})
    probe.set_device_channel_indices(np.array(range(probe_layout.shape[0])))
    recording = recording.set_probe(probe)

    recording_filtered = spre.bandpass_filter(spre.notch_filter(recording, 50, dtype='float64'), freq_min=300,
                                              freq_max=5000)
    bad_channels_ids_auto = spre.detect_bad_channels(recording_filtered, method='mad')[
        0] if auto_bad_channel_detection_flag else []
    bad_channel_ids = list(
        set(bad_channels_ids_auto) | set(recording_filtered.get_channel_ids()[impedance_list > impedance_threshold]))
    recording_removed = recording_filtered.remove_channels(bad_channel_ids)
    if "inter_sample_shift" in recording_removed.get_property_keys():
        recording_removed = spre.phase_shift(recording_removed)
    recording_cmr = spre.common_reference(recording_removed, operator='median', reference='global')

    # ---- 3. 全局 Waveform 提取（保持原样） ----
    sorting = se.read_phy(paths['before_phy'].replace('before_phy/', 'phy/'), exclude_cluster_groups=["noise", 'mua'])
    waveform = si.WaveformExtractor.create(recording_cmr, sorting, paths['waveform'], remove_if_exists=True)
    waveform.set_params(ms_before=3., ms_after=4., return_scaled=True)
    waveform.run_extract_waveforms(n_jobs=-1, chunk_size=30000)
    sqm.compute_quality_metrics(waveform)
    spost.compute_spike_amplitudes(waveform)
    spost.compute_isi_histograms(waveform, window_ms=100, bin_ms=1)
    spost.compute_correlograms(waveform, window_ms=100.0, bin_ms=1)
    spost.compute_unit_locations(waveform, method='center_of_mass', radius_um=300)
    spost.compute_template_metrics(waveform)
    compute_waveform_metrics(waveform)
    compute_unit_waveform_location(waveform)
    del waveform

    # ---- 4. 带有 Margin 扣除的核心切片逻辑 ----
    # 计算物理时间轴上的各个边界帧点
    seg1_end = int((time1 - phase_change_margin) * sampling_frequency)
    seg2_start = int((time1 + phase_change_margin) * sampling_frequency)
    seg2_end = int((time2 - phase_change_margin) * sampling_frequency)
    seg3_start = int((time2 + phase_change_margin) * sampling_frequency)

    # 边界安全性检查，防止由于 margin 过大导致越界
    seg1_end = max(0, seg1_end)
    seg2_start = max(seg1_end, seg2_start)
    seg2_end = max(seg2_start, seg2_end)
    seg3_start = min(n_sample, max(seg2_end, seg3_start))

    # 根据 start_phase 标签分配 before/middle/after 的时序
    if start_phase == 'before':
        f_before_start, f_before_end = 0, seg1_end
        f_middle_start, f_middle_end = seg2_start, seg2_end
        f_after_start, f_after_end = seg3_start, n_sample
    elif start_phase == 'after':
        f_after_start, f_after_end = 0, seg1_end
        f_middle_start, f_middle_end = seg2_start, seg2_end
        f_before_start, f_before_end = seg3_start, n_sample
    else:
        raise Exception('unknown start phase label: ', start_phase)

    # 内部数据提取复用函数
    def extract_phase_waveforms(slice_name, start_f, end_f):
        # 针对极短区间或由于输入错误导致的无效区间做一层保护
        if start_f >= end_f:
            print(f"Warning: {slice_name} phase range [{start_f} to {end_f}] is empty. Skipped.")
            return

        print(f'generating {slice_name} sorting and waveform [{start_f} to {end_f}]...')
        seg_sorting = sorting.frame_slice(start_frame=start_f, end_frame=end_f)
        seg_recording = recording_cmr.frame_slice(start_frame=start_f, end_frame=end_f)

        w_folder = paths[f'{slice_name}_waveform']
        phy_folder = paths[f'{slice_name}_phy']

        seg_waveform = si.WaveformExtractor.create(seg_recording, seg_sorting, w_folder, remove_if_exists=True)
        seg_waveform.set_params(ms_before=3., ms_after=4., return_scaled=True)
        seg_waveform.run_extract_waveforms(n_jobs=-1, chunk_size=30000)

        sqm.compute_quality_metrics(seg_waveform)
        spost.compute_spike_amplitudes(seg_waveform)
        spost.compute_isi_histograms(seg_waveform, window_ms=100, bin_ms=1)
        spost.compute_correlograms(seg_waveform, window_ms=100.0, bin_ms=1)
        spost.compute_unit_locations(seg_waveform, method='center_of_mass', radius_um=300)
        spost.compute_template_metrics(seg_waveform)
        compute_waveform_metrics(seg_waveform)
        compute_unit_waveform_location(seg_waveform)

        try:
            sexp.export_to_phy(seg_waveform, compute_pc_features=True, compute_amplitudes=True,
                               output_folder=phy_folder + 'phy', remove_if_exists=True, n_jobs=-1, copy_binary=True)
        except Exception as e:
            print(e)
            print(f'export to phy failed for {slice_name}, usually caused by NaN unit sliced.')

        del seg_waveform
        del seg_sorting

    # ---- 5. 顺序执行三段提取 ----
    extract_phase_waveforms('before', f_before_start, f_before_end)
    extract_phase_waveforms('middle', f_middle_start, f_middle_end)
    extract_phase_waveforms('after', f_after_start, f_after_end)

    del sorting
    print('done')
    time_end = default_timer()
    print('Module time usage(s): ', time_end - time_start)

def compute_waveform_metrics(waveform):


    unit_ids = waveform.unit_ids
    sampling_frequency = waveform.sampling_frequency
    upsampling_factor = 10
    features_df = pd.DataFrame()
    extremum_channels_ids = spost.get_template_extremum_channel(waveform)
    # all_templates = waveform.get_all_templates()
    for unit_index, unit_id in enumerate(unit_ids):
        wfs = waveform.get_waveforms(unit_id)
        features_temp_list = []
        chan_ids = np.array(extremum_channels_ids[unit_id])
        if chan_ids.ndim == 0:
            chan_ids = [chan_ids]
        chan_ind = waveform.channel_ids_to_indices(chan_ids)


        for i in range(len(wfs)):
            wf = wfs[i]
            main_waveform = wf[:, chan_ind].flatten()
            waveform_upsampled = resample_poly(main_waveform, up=upsampling_factor, down=1)
            sampling_frequency_up = upsampling_factor * sampling_frequency
            features_temp = {}
            features_temp['peak_to_valley'] = get_peak_to_valley(waveform_upsampled,
                                                            sampling_frequency=sampling_frequency_up)
            features_temp['peak_trough_ratio'] = get_peak_trough_ratio(waveform_upsampled,
                                                            sampling_frequency=sampling_frequency_up)
            features_temp['half_width'] = get_half_width(waveform_upsampled,
                                                            sampling_frequency=sampling_frequency_up)
            features_temp['repolarization_slope'] = get_repolarization_slope(waveform_upsampled,
                                                            sampling_frequency=sampling_frequency_up)
            features_temp['recovery_slope'] = get_recovery_slope(waveform_upsampled,
                                                            sampling_frequency=sampling_frequency_up)
            features_temp['amplitude_between_peak_and_trough'] = get_amplitude_between_peak_and_trough(waveform_upsampled)
            features_temp_list.append(features_temp)

        features_temp_df = pd.DataFrame(features_temp_list)
        feature = pd.concat([features_temp_df.median().add_suffix('_median'), features_temp_df.std().add_suffix('_std')])
        feature['n_spikes_calc'] = len(wfs)

        features_df = pd.concat([features_df, feature], axis=1)

    features_df = features_df.T
    features_df = features_df.set_index(unit_ids)

    waveform_metrics_folder = waveform.folder / 'waveform_metrics'
    if not os.path.isdir(waveform_metrics_folder):
        os.makedirs(waveform_metrics_folder)
    features_df.to_csv(waveform_metrics_folder / 'metrics.csv')

    return features_df



def compute_unit_waveform_location(waveform, peak_sign="neg", radius_um=75, feature="ptp"):

    unit_ids = waveform.sorting.unit_ids

    recording = waveform.recording
    contact_locations = recording.get_channel_locations()

    assert feature in ["ptp", "mean", "energy", "peak_voltage"], f"{feature} is not a valid feature"

    sparsity = si.compute_sparsity(waveform, peak_sign=peak_sign, method="radius", radius_um=radius_um)
    unit_locations_dict = {}
    metrics_df = pd.DataFrame()

    for unit_id in unit_ids:
        # waveforms = waveform.get_waveforms(unit_id)
        waveforms = waveform.get_waveforms(unit_id=unit_id)
        unit_locations_dict[str(unit_id)] = []

        for wf in waveforms:
            chan_inds = sparsity.unit_id_to_channel_indices[unit_id]
            local_contact_locations = contact_locations[chan_inds, :]

            if feature == "ptp":
                wf_data = (wf[:, chan_inds]).ptp(axis=0)
            elif feature == "mean":
                wf_data = (wf[:, chan_inds]).mean(axis=0)
            elif feature == "energy":
                wf_data = np.linalg.norm(wf[:, chan_inds], axis=0)
            elif feature == "peak_voltage":
                wf_data = wf[waveform.nbefore, chan_inds]

            # center of mass
            com = np.sum(wf_data[:, np.newaxis] * local_contact_locations, axis=0) / np.sum(wf_data)
            unit_locations_dict[str(unit_id)].append(com.tolist())

        if len(unit_locations_dict[str(unit_id)]) > 0:
            distance_matrix = squareform(pdist(unit_locations_dict[str(unit_id)], 'euclidean'))
            max_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
            max_dist = distance_matrix[max_idx]
            temp = pd.Series(np.hstack([np.median(unit_locations_dict[str(unit_id)], axis=0),
                                           np.std(unit_locations_dict[str(unit_id)], axis=0),
                                           np.ptp(unit_locations_dict[str(unit_id)], axis=0),
                                        max_dist]),
                                index=['x_median', 'y_median', 'x_std', 'y_std', 'x_ptp', 'y_ptp', 'maximum_displacement'])
        else:
            temp = pd.Series(7*[np.nan],index=['x_median', 'y_median', 'x_std', 'y_std', 'x_ptp', 'y_ptp',
                                               'maximum_displacement'])
        metrics_df = pd.concat([metrics_df, temp], axis=1)
    metrics_df = metrics_df.T
    metrics_df = metrics_df.set_index(unit_ids)
    unit_waveform_locations_folder = waveform.folder / 'unit_waveform_locations'
    if not os.path.isdir(unit_waveform_locations_folder):
        os.makedirs(unit_waveform_locations_folder)
    metrics_df.to_csv(unit_waveform_locations_folder / 'metrics.csv')
    with open(str(unit_waveform_locations_folder / 'unit_locations.json'), 'w', encoding='utf-8') as f:
        json.dump(unit_locations_dict, f, indent=2)
    return unit_locations_dict





if __name__ == "__main__":

    root_folder = 'H:/data/'  # This should be your own pseudo folder
    experimenter = 'WXY'
    acquisition_system = 'blackrock'

    print("Processing MILD group: ", MILD_mouse_list)
    for mouse_id in MILD_mouse_list:
        recording_time = recording_id_checklist[mouse_id]

        # 此时这里获取到的会是你在最上面配置的列表，比如 [600, 1200]
        phase_change_time = recording_phase_change_time_checklist[mouse_id]
        start_phase = recording_start_phase_checklist[mouse_id]

        single_recording_all_phase_sorting_and_waveform_generation(root_folder, experimenter, acquisition_system,
                                                                   mouse_id, recording_time, phase_change_time,
                                                                   start_phase)
    print("MILD group done!")

