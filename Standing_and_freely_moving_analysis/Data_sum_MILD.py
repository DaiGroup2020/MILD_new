# -*- coding: utf-8 -*-
import numpy as np
import spikeinterface as si
import os
import pandas as pd
import spikeinterface.extractors as se

if __name__ == "__main__":
    isi_violations_ratio_threshold = 0.1
    firing_rate_threshold = 0.1
    half_width_lower_threshold = 0.10 / 1000
    half_width_upper_threshold = 0.75 / 1000

    root_folder = 'H:/data/'
    experimenter = 'WXY'
    acquisition_system = 'blackrock'

    MILD_mouse_list = ['601', '602', '603']
    recording_id_checklist = {
        '601': '20260519-0000',
        '602': '20260519-0000',
        '603': '20260519-0000',

    }

    unit_summary = pd.DataFrame()

    for group_name, group_list in zip(['MILD'], [MILD_mouse_list]):
        for mouse_id in group_list:
            recording_time = recording_id_checklist[mouse_id]
            sorting_result_folder = f"{root_folder}/{experimenter}/{acquisition_system}/{mouse_id}/{recording_time}/analysis/sorting_result/"

            print(f"Processing metrics for {group_name} - Mouse {mouse_id}...")

            # 加载全局波形提取器以获取标准的 unit_ids
            waveform_folder = sorting_result_folder + 'waveform/'
            waveform = si.load_waveforms(waveform_folder, with_recording=False)
            unit_ids = waveform.unit_ids


            # --- 辅助读取函数：将 Unit ID 设为索引，防止行数不一致时错位 ---
            def load_phase_metrics(phase_dir):
                base_path = f"{sorting_result_folder}{phase_dir}/"
                try:
                    q_metric = pd.read_csv(base_path + 'quality_metrics/metrics.csv').rename(
                        columns={'Unnamed: 0': 'unit_id'})
                    t_metric = pd.read_csv(base_path + 'template_metrics/metrics.csv').rename(
                        columns={'Unnamed: 0': 'unit_id'})
                    w_metric = pd.read_csv(base_path + 'waveform_metrics/metrics.csv').rename(
                        columns={'Unnamed: 0': 'unit_id'})
                    l_metric = pd.read_csv(base_path + 'unit_waveform_locations/metrics.csv').rename(
                        columns={'Unnamed: 0': 'unit_id'})

                    # 以 unit_id 为核心横向合并单期数据
                    df_phase = q_metric.merge(t_metric, on='unit_id') \
                        .merge(w_metric, on='unit_id') \
                        .merge(l_metric, on='unit_id')
                    df_phase.set_index('unit_id', inplace=True)
                    return df_phase
                except Exception as e:
                    print(f"Warning: Failed to load some metrics in {phase_dir}. Error: {e}")
                    return pd.DataFrame(index=unit_ids)


            # --- 载入 4 个区间的全部指标数据 ---
            df_all = load_phase_metrics('waveform')
            df_before = load_phase_metrics('before_waveform')
            df_middle = load_phase_metrics('middle_waveform')
            df_after = load_phase_metrics('after_waveform')

            # --- 构建当前小鼠的 Summary 基础表 ---
            new_summary = pd.DataFrame(index=unit_ids)
            new_summary['group'] = group_name
            new_summary['mouse'] = mouse_id
            new_summary['recording'] = recording_time
            new_summary['unit'] = unit_ids

            # 1. 批量填充 ALL 全局区间的所有指标，并进行神经元筛选
            new_summary = new_summary.join(df_all.reindex(unit_ids).add_prefix('all_'))

            new_summary['all_isi_violations_ratio_good'] = new_summary[
                                                               'all_isi_violations_ratio'] <= isi_violations_ratio_threshold
            new_summary['all_firing_rate_good'] = new_summary['all_firing_rate'] >= firing_rate_threshold
            new_summary['all_half_width_good'] = (new_summary['all_half_width'] >= half_width_lower_threshold) & \
                                                 (new_summary['all_half_width'] <= half_width_upper_threshold)
            new_summary['all_neuron'] = new_summary['all_isi_violations_ratio_good'] & \
                                        new_summary['all_firing_rate_good'] & \
                                        new_summary['all_half_width_good']

            # 2. 批量填充 BEFORE 区间指标与筛选
            new_summary = new_summary.join(df_before.reindex(unit_ids).add_prefix('before_'))

            new_summary['before_isi_violations_ratio_good'] = new_summary[
                                                                  'before_isi_violations_ratio'] <= isi_violations_ratio_threshold
            new_summary['before_firing_rate_good'] = new_summary['before_firing_rate'] >= firing_rate_threshold
            new_summary['before_half_width_good'] = (new_summary['before_half_width'] >= half_width_lower_threshold) & \
                                                    (new_summary['before_half_width'] <= half_width_upper_threshold)
            new_summary['before_neuron'] = new_summary['before_isi_violations_ratio_good'] & \
                                           new_summary['before_firing_rate_good'] & \
                                           new_summary['before_half_width_good']

            # 3. 批量填充 MIDDLE 区间指标与筛选
            new_summary = new_summary.join(df_middle.reindex(unit_ids).add_prefix('middle_'))

            new_summary['middle_isi_violations_ratio_good'] = new_summary[
                                                                  'middle_isi_violations_ratio'] <= isi_violations_ratio_threshold
            new_summary['middle_firing_rate_good'] = new_summary['middle_firing_rate'] >= firing_rate_threshold
            new_summary['middle_half_width_good'] = (new_summary['middle_half_width'] >= half_width_lower_threshold) & \
                                                    (new_summary['middle_half_width'] <= half_width_upper_threshold)
            new_summary['middle_neuron'] = new_summary['middle_isi_violations_ratio_good'] & \
                                           new_summary['middle_firing_rate_good'] & \
                                           new_summary['middle_half_width_good']

            # 4. 批量填充 AFTER 区间指标与筛选
            new_summary = new_summary.join(df_after.reindex(unit_ids).add_prefix('after_'))

            new_summary['after_isi_violations_ratio_good'] = new_summary[
                                                                 'after_isi_violations_ratio'] <= isi_violations_ratio_threshold
            new_summary['after_firing_rate_good'] = new_summary['after_firing_rate'] >= firing_rate_threshold
            new_summary['after_half_width_good'] = (new_summary['after_half_width'] >= half_width_lower_threshold) & \
                                                   (new_summary['after_half_width'] <= half_width_upper_threshold)
            new_summary['after_neuron'] = new_summary['after_isi_violations_ratio_good'] & \
                                          new_summary['after_firing_rate_good'] & \
                                          new_summary['after_half_width_good']

            # 合并到大总表中
            unit_summary = pd.concat([unit_summary, new_summary], axis=0, ignore_index=True)

    # 创建保存结果的文件夹（如果不存在）
    if not os.path.exists('results'):
        os.makedirs('results')

    # 保存结果
    unit_summary.to_csv('results/unit_summary.csv', index=False)
    print("All phase metrics merged successfully! Saved to 'results/unit_summary.csv'.")