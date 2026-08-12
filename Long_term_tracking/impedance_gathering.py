import numpy as np
import os
import spikeinterface as si
import spikeinterface.extractors as se
from intanutil.read_header import read_header as read_intan_header
import pandas as pd
#from load_intan_rhs import read_header

if __name__ == '__main__':

    root_folder_input = 'H:/data'
    experimenter = 'WXY'
    acquisition_system = 'blackrock'
    mouse_id = '5132'


    # 获取当前目录中所有文件夹的名称
    folders = [folder for folder in os.listdir('H:/data/WXY/blackrock/5132') if os.path.isdir(os.path.join('H:/data/WXY/blackrock/5132', folder))]

    # 将文件夹名存储在一个数组中
    recording_time_list = []
    for folder in folders:
        recording_time_list.append(folder)
    print(recording_time_list)

    recording_time_list_1 = [int(str(value)[:8]) for value in recording_time_list]
    #print(recording_time_list_1)

    from datetime import datetime
    # 将日期字符串转换为日期对象
    dates = [datetime.strptime(str(date), "%Y%m%d") for date in recording_time_list_1]
    # 获取第一个日期作为基准日期
    base_date = dates[0]
    # 计算每个日期与基准日期的差值，并生成新的数组
    date_differences = [(date - base_date).days for date in dates]

    # print(date_differences)

    print('============================================================================')
    print('IMPEDANCE MODULE')
    print('reading impedance...')
    impedance_array = np.array([])
    if not root_folder_input.endswith('/'):
        source_root_folder = root_folder_input + '/'
    else:
        source_root_folder = root_folder_input

    for recording_time in recording_time_list:
        source_probe_layout_file_path = source_root_folder + experimenter + '/probe_layout.csv'
        source_raw_data_folder = source_root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                                 '/' + recording_time + '/raw_data/'
        assert os.path.isfile(
            source_probe_layout_file_path), source_probe_layout_file_path + ' does not exist, please create this file under experimenter  folder before sort spikes'
        assert os.path.isdir(
            source_raw_data_folder), source_raw_data_folder + 'does not exist, please put recording data in this path before sorting'
        root_folder = source_root_folder
        probe_layout_file_path = source_probe_layout_file_path
        raw_data_folder = source_raw_data_folder
        print('data: ', experimenter, acquisition_system, mouse_id, recording_time)

        raw_data_list = os.listdir(raw_data_folder)

        if acquisition_system == 'intan':
            recording_data_list = [i for i in raw_data_list if i.endswith('.rhd')]
            assert len(recording_data_list) >= 1, 'no .rhs files detected in folder: ' + raw_data_folder
            recording = si.concatenate_recordings([se.read_intan(raw_data_folder + recording, stream_id='0') for recording in recording_data_list])
            channel_num = len(recording.get_channel_ids())
            impedance_file = open(raw_data_folder + recording_data_list[0], 'rb')
            intan_header = read_intan_header(impedance_file)
            impedance_list = np.array(
                [intan_header['amplifier_channels'][i]['electrode_impedance_magnitude'] / 1e6 for i in
                 range(channel_num)])

        elif acquisition_system == 'blackrock':
            recording_data_list = [i for i in raw_data_list if i.endswith('.ns6')]
            assert len(recording_data_list) >= 1, 'no .ns6 files detected in folder: ' + raw_data_folder
            recording = si.concatenate_recordings([se.read_blackrock(raw_data_folder + recording) for recording in recording_data_list])
            channel_num = len(recording.get_channel_ids())
            assert 'impedance' in raw_data_list, 'no impedance file detected, please save impedance file from blackrock before sorting'
            impedance_file = pd.read_csv(raw_data_folder + 'impedance')
            temp = impedance_file[6:].iloc[:, 0].str.split('\t', expand=True)
            temp = temp.iloc[:channel_num, 1].str.split(' kOhm', expand=True).iloc[:, 0].to_list()
            impedance_list = np.array([int(imp) / 1e3 for imp in temp])
        else:
            recording = None
            channel_num = None
            impedance_list = None
            raise Exception('acquisition_system currently should either be intan or blackrock')
        if impedance_array.size == 0:
            impedance_array = impedance_list
        else:
            impedance_array = np.vstack((impedance_array, impedance_list))

    # , index = date_differences
    impedance_pd = pd.DataFrame(impedance_array, columns=range(impedance_array.shape[1]))
    impedance_pd.insert(loc=0, column='Raw Recording Time', value=recording_time_list)
    impedance_pd.insert(loc=1, column='Recording Time', value=date_differences)
    save_path = os.path.join(source_root_folder, experimenter, acquisition_system, mouse_id,
                             experimenter + '_' + acquisition_system + '_' + mouse_id + '_impedance.csv')
    impedance_pd.to_csv(save_path)
    #impedance_pd.to_csv(experimenter+'_'+acquisition_system+'_'+mouse_id+'_'+'impedance.csv')


    impedance_filtered = impedance_pd.copy()

    # 明确指定阻抗通道列（第3列开始）
    impedance_data_cols = impedance_filtered.columns[2:]


    # 将阻抗 <0.1 或 >3.0 MΩ 的值替换为空（NaN）
    impedance_filtered[impedance_data_cols] = impedance_filtered[impedance_data_cols].mask(
        (impedance_filtered[impedance_data_cols] < 0.1) |
        (impedance_filtered[impedance_data_cols] > 3.0)
    )
    # 插入“Valid Channels”列（记录有效阻抗通道数）
    valid_channel_count = impedance_filtered[impedance_data_cols].notna().sum(axis=1)
    impedance_filtered.insert(
        loc=2,  # 插在 Recording Time 后面
        column="Valid Channels",
        value=valid_channel_count
    )

    # 计算每行的均值和方差（只对有效通道计算）
    mean_imp = impedance_filtered[impedance_data_cols].mean(axis=1, skipna=True)
    var_imp = impedance_filtered[impedance_data_cols].var(axis=1, skipna=True)

    # 在最后追加列
    impedance_filtered["Mean Impedance"] = mean_imp
    impedance_filtered["Variance Impedance"] = var_imp

    save_path = os.path.join(source_root_folder, experimenter, acquisition_system, mouse_id,
                             experimenter + '_' + acquisition_system + '_' + mouse_id + '_impedance_filtered.csv')
    impedance_filtered.to_csv(save_path)

