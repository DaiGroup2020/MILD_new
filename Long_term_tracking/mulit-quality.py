import warnings

warnings.filterwarnings('ignore')
import spikeinterface as si
import spikeinterface.extractors as se
import json
import spikeinterface.postprocessing as spost
import shutil
import pprint
from timeit import default_timer
import matplotlib
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.ticker as ticker
import os
import spikeinterface.qualitymetrics as sqm
import pandas as pd

def read_sorting_metadata(root_folder_input, experimenter, acquisition_system, mouse_id, recording_time, display=False):
    assert not root_folder_input.startswith(
        'Z:'), 'This is study version code, DO NOT USE REAL DATA ON Z DISK! Change your root to your pseudo dataset.'
    if display:
        time_start = default_timer()
        print('============================================================================')
        print('data: ', experimenter, acquisition_system, mouse_id, recording_time)
        print('CHECK SORTING METADATA MODULE')
    if not root_folder_input.endswith('/'):
        root_folder_input = root_folder_input + '/'
    sorting_result_folder = root_folder_input + experimenter + '/' + acquisition_system + '/' + mouse_id + \
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


def compute_waveform_duration(waveform,unit_id,fs):
    template=waveform.get_template(
        unit_id=unit_id,
        mode='median'
    )

    channel=np.argmax(
        np.ptp(template,axis=0)
    )

    wf=template[:,channel]

    trough_idx=np.argmin(wf)

    peak_idx=np.argmax(
        wf[trough_idx:]
    )+trough_idx

    duration=(peak_idx-trough_idx)/fs*1000

    return duration


def Spike_tracking_quality(root_folder_input, experimenter, acquisition_system, mouse_id):
    time_start = default_timer()
    print('============================================================================')
    print('Animal: ', experimenter, acquisition_system, mouse_id)
    print('TRACE SPIKES PLOT')
    print('Gathering materials...')
    cell_bundle_path = os.path.dirname(os.path.realpath(
        __file__)) + '/results/' + experimenter + '_' + acquisition_system + '_' + mouse_id + '_cell_bundle_OK_plot.csv'
    assert os.path.isfile(
        cell_bundle_path), cell_bundle_path + ' does not exist, please bundle cells for this animal before tracing cluster.'
    if not root_folder_input.endswith('/'):
        root_folder = root_folder_input + '/'
    else:
        root_folder = root_folder_input
    cell_bundle_df = pd.read_csv(cell_bundle_path, index_col=0)
    recording_time_list = cell_bundle_df.index
    recording_number = len(recording_time_list)
    print('Recoring number: ', recording_number)
    assert recording_number >= 2, 'please give more than 1 recording times'
    result_folder = os.path.dirname(os.path.realpath(__file__)) + '/results/'
    unit_names = cell_bundle_df.columns
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    unit_number = len(unit_names)
    # print('Unit number: ', unit_number)
    # print('unit_names:',unit_names)
    SNR_chains=[]
    Amp_chains = []
    FR_chains = []
    X_tracking_chains = []
    Y_tracking_chains = []
    for unit_name in unit_names:
        SNR_chain=[]
        Amp_chain = []
        FR_chain = []
        X_chain = []  # 新增：存储单个 Unit 在不同时间的 X 坐标
        Y_chain = []  # 新增：存储单个 Unit 在不同时间的 Y 坐标
        template_feature_chain = []
        recordingtime_SNR_chain = []
        print('unit_name:', unit_name)
        for recording_ind in range(recording_number):
            recording_time = recording_time_list[recording_ind]
            unit_id = cell_bundle_df.loc[recording_time, unit_name]
            print('Unit_id:',unit_id)
            assert not root_folder_input.startswith(
                'Z:'), 'This is study version code, DO NOT USE REAL DATA ON Z DISK! Change your root to your pseudo dataset.'
            time_start = default_timer()
            # print('============================================================================')
            # print('data: ', experimenter, acquisition_system, mouse_id, recording_time)
            # print('SINGLE RECORDING FIGURE GENERATION MODULE')
            # print('Routing...')
            if not root_folder_input.endswith('/'):
                source_root_folder = root_folder_input + '/'
            else:
                source_root_folder = root_folder_input
            source_sorting_result_folder = source_root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                                   '/' + recording_time + '/analysis/sorting_result/'
            assert os.path.isdir(
                source_sorting_result_folder), source_sorting_result_folder + ' does not exist, please sort spikes and post curate before extracting waveform'
            source_sorting_metadata_path = source_sorting_result_folder + 'sorting_metadata.npy'
            assert os.path.isfile(
                source_sorting_metadata_path), source_sorting_metadata_path + ' does not exist, please run sort_spikes before post curation.'
            source_post_curation_folder = source_sorting_result_folder + 'phy/'
            assert os.path.isdir(
                source_post_curation_folder), source_post_curation_folder + ' does not exist, please run sort_spikes before post curation, or maybe no units are detected'
            source_waveform_folder = source_sorting_result_folder + 'waveform_from_phy/'
            assert os.path.isdir(
                source_waveform_folder), source_waveform_folder + ' does not exist, please run sort_spikes before post curation, or maybe no units are detected'
            sorting_metadata = read_sorting_metadata(root_folder_input, experimenter, acquisition_system, mouse_id,
                                                    recording_time, display=False)
            assert sorting_metadata[
             'waveform_extraction_accomplished'], 'Waveform extraction is not finished yet. Please extract waveform before figure generation'

            root_folder = source_root_folder
            sorting_result_folder = source_sorting_result_folder
            sorting_metadata_path = source_sorting_metadata_path
            post_curation_folder = source_post_curation_folder
            waveform_folder = source_waveform_folder

            if os.path.exists(waveform_folder + 'sorting_preserve.json'):
                os.remove(waveform_folder + 'sorting_preserve.json')
            os.rename(waveform_folder + 'sorting.json', waveform_folder + 'sorting_preserve.json')
            with open(waveform_folder + 'sorting_preserve.json', 'r') as json_preserve:
                json_data = json.load(json_preserve)
                # original_address = r"[A-Z]:\\\\.*?\\\\analysis\\\\sorting_result\\\\phy"
                original_address = r"(([A-Z]:)|(\\\\\\\\\d*?\.\d*?\.\d*?\.\d*?))\\\\.*?\\\\analysis\\\\sorting_result\\\\phy"
                new_address = post_curation_folder[:-1]
                new_address = re.sub(r"\\", r"\\\\\\\\", new_address)
                new_address = re.sub(r"/", r"\\\\\\\\", new_address)
                json_string_new = re.sub(original_address, new_address, str(json_data))
                json_data_new = ast.literal_eval(json_string_new)
                with open(waveform_folder + 'sorting.json', 'w') as json_file_new:
                    json.dump(json_data_new, json_file_new, indent=4)
            # print('post_curation_folder:', post_curation_folder)
            # print('waveform_folder:', waveform_folder)
            # print('reading sorting and waveform results...')
            sorting = se.read_phy(post_curation_folder, exclude_cluster_groups=["noise", 'mua'])
            unit_ids = sorting.get_unit_ids()
            # print('unit_ids:', unit_ids)
            # print('unit_id:', unit_id)
            unit_id_int = int(unit_id)
            position = np.where(unit_ids == unit_id_int)[0]

            #waveform = si.WaveformExtractor.load_from_folder(waveform_folder)
            #SNR = sqm.compute_snrs(waveform_extractor=waveform, unit_ids=unit_ids)

            waveform = si.load_waveforms(waveform_folder)
            SNR = sqm.compute_snrs(waveform)
            firing_rate = sqm.compute_firing_rates(waveform)
            # --- 新增：计算神经元位置 ---
            # 使用 center_of_mass 方法获取更精确的重心坐标
            unit_locations = spost.compute_unit_locations(waveform, method='center_of_mass')
            # 找到当前 unit_id 对应的索引
            unit_idx = np.where(waveform.unit_ids == unit_id_int)[0][0]
            # 提取坐标 (x, y)
            X_chain.append(unit_locations[unit_idx][0])
            Y_chain.append(unit_locations[unit_idx][1])

            amplitude_medians = sqm.compute_amplitude_medians(waveform)
            template_feature = spost.compute_template_metrics(waveform)
            template_feature = pd.DataFrame(template_feature)
            duration = compute_waveform_duration(waveform, unit_id_int, sorting.get_sampling_frequency())
            # print(template_feature)
            # print(template_feature.iloc[position])
            #template_feature_chain.append(template_feature.iloc[position])
            feature_row=template_feature.iloc[position].to_dict()
            feature_row["duration"] = duration
            template_feature_chain.append(feature_row)

            SNR_chain.append(SNR[unit_id_int])
            FR_chain.append(firing_rate[unit_id_int])
            Amp_chain.append(amplitude_medians[unit_id_int])
            # print(SNR_chain)
            # print(FR_chain)
            # print(Amp_chain)
        SNR_chains.append(SNR_chain)
        FR_chains.append(FR_chain)
        Amp_chains.append(Amp_chain)
        # 将该 Unit 的完整时间链条加入总表
        X_tracking_chains.append(X_chain)
        Y_tracking_chains.append(Y_chain)

        # print(template_feature_chain)
        #template_feature_chain=np.array(template_feature_chain)
        # print(template_feature_chain)
        #template_feature_chain_2d = template_feature_chain.reshape((-1, template_feature_chain.shape[2]))
        # print(template_feature_chain_2d)
        # print("shape:", template_feature_chain_2d.shape)
        #template_feature_chain_2d_df = pd.DataFrame(template_feature_chain_2d, columns=['peak_to_valley','peak_trough_ratio','halfwidth','repolarization_slope','recovery_slope'], index=recording_time_list)
        template_feature_chain_2d_df = pd.DataFrame(template_feature_chain, index=recording_time_list)
        template_feature_chain_2d_df.to_csv( result_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_' + unit_name + '_template_feature.csv')

        #template_feature_chain_2d_df.to_csv(result_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_' + unit_name + '_template_feature.csv')
    SNR_chains =  np.transpose(SNR_chains)
    FR_chains = np.transpose(FR_chains)
    Amp_chains = np.transpose(Amp_chains)
    print(SNR_chains)
    print(FR_chains)
    print(Amp_chains)
    SNR_chains_df = pd.DataFrame(SNR_chains, columns=unit_names, index=recording_time_list)
    FR_chains_df = pd.DataFrame(FR_chains, columns=unit_names, index=recording_time_list)
    Amp_chains_df = pd.DataFrame(Amp_chains, columns=unit_names, index=recording_time_list)
    # print('Saving SNR csv...')
    SNR_chains_df.to_csv(
        result_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_SNR.csv')
    FR_chains_df.to_csv(
        result_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_FiringRate.csv')
    Amp_chains_df.to_csv(
        result_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_Amplitude.csv')

    # --- 保存位置追踪数据 ---
    # 转置数据以匹配 (Index=时间, Columns=神经元) 的格式
    X_tracking_df = pd.DataFrame(np.transpose(X_tracking_chains), columns=unit_names, index=recording_time_list)
    Y_tracking_df = pd.DataFrame(np.transpose(Y_tracking_chains), columns=unit_names, index=recording_time_list)

    # 保存为 Excel 文件（支持多 Sheet）
    location_file_path = result_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_Location_Tracking.xlsx'

    with pd.ExcelWriter(location_file_path) as writer:
        X_tracking_df.to_excel(writer, sheet_name='Unit_X_Position')
        Y_tracking_df.to_excel(writer, sheet_name='Unit_Y_Position')

    print('done')
    time_end = default_timer()
    print('Module time usage(s): ', time_end - time_start)


def All_units_quality(root_folder_input, experimenter, acquisition_system, mouse_id):
    time_start = default_timer()
    print('============================================================================')
    print('Animal: ', experimenter, acquisition_system, mouse_id)
    print('TRACE SPIKES PLOT')
    print('Gathering materials...')
    # 指定要扫描的目录路径
    directory_path = root_folder_input + '/' + experimenter + '/' + acquisition_system + '/' + mouse_id
    # 获取所有子目录名称
    subdirectories = list_subdirectories(directory_path)
    # 打印结果
    print("Subdirectories in the specified directory:")
    for subdir in subdirectories:
        print(subdir)
    recording_time_list = subdirectories
    recording_number = len(recording_time_list)
    print('Recoring number: ', recording_number)
    assert recording_number >= 2, 'please give more than 1 recording times'
    result_folder = os.path.dirname(os.path.realpath(__file__)) + '/results/'
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    recordingtime_SNR_chain = []
    recordingtime_firing_rate_chain = []
    recordingtime_amplitude_medians_chain = []

    for recording_ind in range(recording_number):
        recording_time = recording_time_list[recording_ind]
        assert not root_folder_input.startswith(
            'Z:'), 'This is study version code, DO NOT USE REAL DATA ON Z DISK! Change your root to your pseudo dataset.'
        time_start = default_timer()
        # print('============================================================================')
        # print('data: ', experimenter, acquisition_system, mouse_id, recording_time)
        # print('SINGLE RECORDING FIGURE GENERATION MODULE')
        # print('Routing...')
        if not root_folder_input.endswith('/'):
                source_root_folder = root_folder_input + '/'
        else:
            source_root_folder = root_folder_input
        source_sorting_result_folder = source_root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                                           '/' + recording_time + '/analysis/sorting_result/'
        assert os.path.isdir(
            source_sorting_result_folder), source_sorting_result_folder + ' does not exist, please sort spikes and post curate before extracting waveform'
        source_sorting_metadata_path = source_sorting_result_folder + 'sorting_metadata.npy'
        assert os.path.isfile(
            source_sorting_metadata_path), source_sorting_metadata_path + ' does not exist, please run sort_spikes before post curation.'
        source_post_curation_folder = source_sorting_result_folder + 'phy/'
        assert os.path.isdir(
            source_post_curation_folder), source_post_curation_folder + ' does not exist, please run sort_spikes before post curation, or maybe no units are detected'
        source_waveform_folder = source_sorting_result_folder + 'waveform_from_phy/'
        assert os.path.isdir(
            source_waveform_folder), source_waveform_folder + ' does not exist, please run sort_spikes before post curation, or maybe no units are detected'
        sorting_metadata = read_sorting_metadata(root_folder_input, experimenter, acquisition_system, mouse_id,
                                                    recording_time, display=False)
        assert sorting_metadata[
            'waveform_extraction_accomplished'], 'Waveform extraction is not finished yet. Please extract waveform before figure generation'

        root_folder = source_root_folder
        sorting_result_folder = source_sorting_result_folder
        sorting_metadata_path = source_sorting_metadata_path
        post_curation_folder = source_post_curation_folder
        waveform_folder = source_waveform_folder

        if os.path.exists(waveform_folder + 'sorting_preserve.json'):
            os.remove(waveform_folder + 'sorting_preserve.json')
        os.rename(waveform_folder + 'sorting.json', waveform_folder + 'sorting_preserve.json')
        with open(waveform_folder + 'sorting_preserve.json', 'r') as json_preserve:
            json_data = json.load(json_preserve)
            # original_address = r"[A-Z]:\\\\.*?\\\\analysis\\\\sorting_result\\\\phy"
            original_address = r"(([A-Z]:)|(\\\\\\\\\d*?\.\d*?\.\d*?\.\d*?))\\\\.*?\\\\analysis\\\\sorting_result\\\\phy"
            new_address = post_curation_folder[:-1]
            new_address = re.sub(r"\\", r"\\\\\\\\", new_address)
            new_address = re.sub(r"/", r"\\\\\\\\", new_address)
            json_string_new = re.sub(original_address, new_address, str(json_data))
            json_data_new = ast.literal_eval(json_string_new)
            with open(waveform_folder + 'sorting.json', 'w') as json_file_new:
                json.dump(json_data_new, json_file_new, indent=4)
        #waveform = si.load_waveforms(waveform_folder, with_recording=False)
        #SNR = sqm.compute_snrs(waveform)

        waveform = si.load_waveforms(waveform_folder)
        SNR = sqm.compute_snrs(waveform)


        firing_rate = sqm.compute_firing_rates(waveform)
        amplitude_medians = sqm.compute_amplitude_medians(waveform)
        SNR_df = pd.DataFrame(list(SNR.items()), columns=['unit_id', 'value'])
        SNR_df_to_numpy = SNR_df['value'].values
        recordingtime_SNR = np.transpose(SNR_df_to_numpy)
        recordingtime_SNR_chain.append(recordingtime_SNR)
        firing_rate_df = pd.DataFrame(list(firing_rate.items()), columns=['unit_id', 'value'])
        firing_rate_df_to_numpy = firing_rate_df['value'].values
        recordingtime_firing_rate = np.transpose(firing_rate_df_to_numpy)
        recordingtime_firing_rate_chain.append(recordingtime_firing_rate)
        amplitude_medians_df = pd.DataFrame(list(amplitude_medians.items()), columns=['unit_id', 'value'])
        amplitude_medians_df_to_numpy = amplitude_medians_df['value'].values
        recordingtime_amplitude_medians = np.transpose(amplitude_medians_df_to_numpy)
        recordingtime_amplitude_medians_chain.append(recordingtime_amplitude_medians)
    recordingtime_SNR_chain_df = pd.DataFrame(recordingtime_SNR_chain,
                                                  index=recording_time_list)
    recordingtime_SNR_chain_df.to_csv(
        result_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_All_unit' + '_SNR.csv')
    recordingtime_firing_rate_chain_df = pd.DataFrame(recordingtime_firing_rate_chain,
                                                  index=recording_time_list)
    recordingtime_firing_rate_chain_df.to_csv(
        result_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_All_unit' + '_FR.csv')
    recordingtime_amplitude_medians_chain_df = pd.DataFrame(recordingtime_amplitude_medians_chain,
                                                  index=recording_time_list)
    recordingtime_amplitude_medians_chain_df.to_csv(
        result_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_All_unit' + '_Amp.csv')


from spikeinterface.postprocessing import get_template_extremum_channel


def Export_Channel_Density_CSV(root_folder_input, experimenter, acquisition_system, mouse_id):
    """
    遍历所有录制时间点，统计每个通道上的 Unit 数量，并输出为 CSV 文件。
    """
    print('============================================================================')
    print(f'Generating Density Report for {mouse_id}...')
    directory_path = os.path.join(root_folder, experimenter, acquisition_system, mouse_id)
    recording_time_list = [name for name in os.listdir(directory_path) if
                           os.path.isdir(os.path.join(directory_path, name))]

    result_folder = os.path.dirname(os.path.realpath(__file__)) + '/results/'
    all_density_data = []

    for recording_time in recording_time_list:
        # 1. 构造波形文件夹路径 (复用你的路径逻辑)
        source_sorting_result_folder = os.path.join(root_folder_input, experimenter, acquisition_system, mouse_id,
                                                    recording_time, 'analysis/sorting_result/')
        waveform_folder = os.path.join(source_sorting_result_folder, 'waveform_from_phy/')

        if not os.path.isdir(waveform_folder):
            print(f"⚠️ 跳过 {recording_time}: 找不到波形文件夹")
            continue

        # 2. 加载波形并计算每个 Unit 归属的通道
        try:
            waveform = si.load_waveforms(waveform_folder, with_recording=False)
            # 获取单位与最大振幅通道的对应关系 {unit_id: channel_id}
            unit_max_chan = get_template_extremum_channel(waveform, peak_sign='neg')

            # 3. 统计当前时间点的通道密度
            current_counts = {'recording_time': recording_time}
            for unit_id, chan_id in unit_max_chan.items():
                # 提取通道 ID 中的数字
                try:
                    chan_idx = int(re.findall(r'\d+', str(chan_id))[0])
                except:
                    chan_idx = chan_id

                current_counts[chan_idx] = current_counts.get(chan_idx, 0) + 1

            all_density_data.append(current_counts)
            print(f"  - {recording_time}: 处理完成")

        except Exception as e:
            print(f"❌ 处理 {recording_time} 时出错: {e}")

    # 4. 构建 DataFrame 并格式化
    if all_density_data:
        df_density = pd.DataFrame(all_density_data)
        # 设置时间为第一列索引
        df_density.set_index('recording_time', inplace=True)
        # 填充缺失值为0，转为整数
        df_density = df_density.fillna(0).astype(int)
        # 对列名（通道序号）按数字大小升序排列
        df_density = df_density.reindex(sorted(df_density.columns), axis=1)

        # 5. 保存文件
        save_name = f"{experimenter}_{acquisition_system}_{mouse_id}_All_unit_Density.csv"
        save_path = os.path.join(result_folder, save_name)
        df_density.to_csv(save_path)

        print('============================================================================')
        print(f"✅ 任务完成！Density CSV 已保存至:\n{save_path}")
    else:
        print("❌ 未提取到任何密度数据，请检查路径和 waveform 文件。")


def list_subdirectories(directory):
    try:
        subdirs = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
        return subdirs
    except Exception as e:
        print(f"Error: {e}")
        return []


root_folder = 'H:/data'
experimenter = 'WXY'
acquisition_system = 'blackrock'
mouse_id = '132'



#Spike_tracking_quality(root_folder, experimenter, acquisition_system, mouse_id)
All_units_quality(root_folder, experimenter, acquisition_system, mouse_id)
#Export_Channel_Density_CSV(root_folder, experimenter, acquisition_system, mouse_id)

