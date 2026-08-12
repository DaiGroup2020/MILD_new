from function_utils import get_channel_inds
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
import numpy as np
import spikeinterface as si
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from timeit import default_timer
import os
from pprint import pprint
import json
import re
import ast
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt


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


def cell_bundling(root_folder_input, experimenter, acquisition_system, mouse_id, recording_time_list,
                  similarity_threshold=0.5):
    time_start = default_timer()
    recording_time_list.sort()
    print('============================================================================')
    print('Animal: ', experimenter, acquisition_system, mouse_id)
    print('CELL BUNDLING MODULE')
    print('Gathering materials...')
    if not root_folder_input.endswith('/'):
        root_folder = root_folder_input + '/'
    else:
        root_folder = root_folder_input
    recording_number = len(recording_time_list)
    assert recording_number >= 2, 'please give more than 1 recording times'
    result_folder = os.path.dirname(os.path.realpath(__file__)) + '/results/'
    layout = None
    sorting_metadata_list = []
    unit_ids_list = []
    waveform_list = []

    for recording_time in recording_time_list:
        assert os.path.isdir(root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                             '/' + recording_time), root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                                                    '/' + recording_time + ' does not exist, please make sure data exist'
        sorting_result_folder = root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                                '/' + recording_time + '/analysis/sorting_result/'
        assert os.path.isdir(
            sorting_result_folder), sorting_result_folder + ' does not exist, please sort spikes and post curate before extracting waveform'
        sorting_metadata_path = sorting_result_folder + 'sorting_metadata.npy'
        assert os.path.isfile(
            sorting_metadata_path), sorting_metadata_path + ' does not exist, please run sort_spikes before post curation.'
        post_curation_folder = sorting_result_folder + 'phy/'
        assert os.path.isdir(
            post_curation_folder), post_curation_folder + ' does not exist, please run sort_spikes before post curation, or maybe no units are detected'
        waveform_folder = sorting_result_folder + 'waveform_from_phy/'
        assert os.path.isdir(
            waveform_folder), waveform_folder + ' does not exist, please run sort_spikes before post curation, or maybe no units are detected'
        sorting_metadata = read_sorting_metadata(root_folder_input, experimenter, acquisition_system, mouse_id,
                                                 recording_time, display=False)
        assert sorting_metadata[
            'waveform_extraction_accomplished'], 'Waveform extraction is not finished yet. Please extract waveform before figure generation'
        if layout is None:
            layout = sorting_metadata['layout']
        else:
            assert layout.tolist() == sorting_metadata[
                'layout'].tolist(), 'Layout from ' + recording_time + ' not match with those from previous data'
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
        unit_ids = se.read_phy(post_curation_folder, exclude_cluster_groups=["noise", 'mua']).get_unit_ids()
        waveform = si.load_waveforms(waveform_folder, with_recording=False)
        sorting_metadata_list.append(sorting_metadata)
        unit_ids_list.append(unit_ids)
        waveform_list.append(waveform)
        os.remove(waveform_folder + 'sorting.json')
        os.rename(waveform_folder + 'sorting_preserve.json', waveform_folder + 'sorting.json')
    print('Calculating channel inds mask...')
    common_channel_locations = set.intersection(
        *[set(map(tuple, waveform.get_channel_locations())) for waveform in waveform_list])
    common_channel_locations = [list(common_channel_location) for common_channel_location in common_channel_locations]
    common_channel_num = np.array(common_channel_locations).shape[0]
    assert common_channel_num >= 1, 'No common channel found. Bundling failed'
    print('Number of common channels: ' + str(common_channel_num))
    channel_inds_list = list(map(lambda waveform: get_channel_inds(waveform, common_channel_locations), waveform_list))

    print('Calculating unit chain...')
    unit_chain = np.array([unit_ids_list[0]],
                          dtype=np.float64)  # each row represent one recording time and each column represent one unit
    for recording_ind in range(1, recording_number):
        templates_pre = waveform_list[recording_ind - 1].get_all_templates(mode='median')[:, :,
                        channel_inds_list[recording_ind - 1]]
        templates_post = waveform_list[recording_ind].get_all_templates(mode='median')[:, :,
                         channel_inds_list[recording_ind]]
        templates_flatten_pre = templates_pre.reshape(templates_pre.shape[0], -1)
        templates_flatten_post = templates_post.reshape(templates_post.shape[0], -1)
        '''
        # === 🛠️ 针对采样率不一致导致的维度不匹配的硬核补救代码 ===
        from scipy.signal import resample

        # 假设原本的 templates 结构是 (num_units, num_samples, num_channels)
        # 在被拉平前，它们的维度可能形如：X 轴有 140 个点，Y 轴有 210 个点

        # 检查两个时间点提取的波形时域长度（num_samples）是否一致
        # 注意：请根据你代码中 flatten 之前的真实变量名进行调整
        # 假设未拉平前的变量名叫 templates_pre 和 templates_post

        # 这里以你报错的信息为例进行对齐：
        # 目标：将所有 Y.shape[1] 强行变换为与 X.shape[1] 一致的维度（或者反过来）

        # 如果你的代码直接拿到了拉平后的 templates_flatten_pre (Shape: N x 9940) 
        # 和 templates_flatten_post (Shape: M x 14910)
        # 我们可以根据通道数（71）把它们还原回（电极通道 × 时间点），重采样后再拉平：

        num_channels = 71

        # 还原 pre 的时间点数
        time_pts_pre = templates_flatten_pre.shape[1] // num_channels  # 9940 / 71 = 140
        # 还原 post 的时间点数
        time_pts_post = templates_flatten_post.shape[1] // num_channels  # 14910 / 71 = 210

        if time_pts_pre != time_pts_post:
            print(f"⚠️ 检测到采样点数不一致 ({time_pts_pre} vs {time_pts_post})，正在启动时域线性重采样...")

            # 将 post (210个点) 重采样降低到 pre (140个点) 的维度
            # 先 reshape 变成 (num_units, time_pts, num_channels)
            num_units_post = templates_flatten_post.shape[0]
            reshaped_post = templates_flatten_post.reshape(num_units_post, time_pts_post, num_channels)

            # 在时域（axis=1）上进行重采样，使其时间点数变成 time_pts_pre (140)
            resampled_post = resample(reshaped_post, time_pts_pre, axis=1)

            # 重新拉平，此时它的维度会完美变成 num_units_post × 9940
            templates_flatten_post = resampled_post.reshape(num_units_post, -1)
            print("✅ 重采样完成，维度已完美对齐！")
        '''


        similarity = cosine_similarity(templates_flatten_pre, templates_flatten_post)
        similarity_df = pd.DataFrame(similarity, index=unit_ids_list[recording_ind - 1],
                                     columns=unit_ids_list[recording_ind])
        similarity_df.to_csv(
            result_folder + f"similarity_{recording_time_list[recording_ind - 1]}_to_{recording_time_list[recording_ind]}.csv")

        unit_inds_pre, unit_inds_post = linear_sum_assignment(-similarity)
        quality_mask = (similarity[[unit_inds_pre], [unit_inds_post]] >= similarity_threshold)[0]
        unit_inds_pre = unit_inds_pre[quality_mask]
        unit_inds_post = unit_inds_post[quality_mask]
        unit_ids_pre = unit_ids_list[recording_ind - 1]
        unit_ids_post = unit_ids_list[recording_ind]
        unit_ids_matched_pre = unit_ids_pre[unit_inds_pre]
        unit_ids_matched_post = unit_ids_post[unit_inds_post]
        # first add one row, which represent previous found units would link to which unit id in new recording
        new_row = [unit_ids_matched_post[
                       np.where(unit_ids_matched_pre == id_pre)[0][0]] if id_pre in unit_ids_matched_pre else np.nan for
                   id_pre in
                   unit_chain[-1, :].tolist()]
        unit_chain = np.vstack([unit_chain, new_row])
        # second add new columns, which represent new discovered units in new recording
        new_unit_inds = np.setdiff1d(unit_ids_post, unit_ids_matched_post)
        new_columns = np.vstack((unit_chain.shape[0] - 1) * [np.full(len(new_unit_inds), np.nan)] + [new_unit_inds])
        unit_chain = np.hstack((unit_chain, new_columns))
        '''
        for unit_ind_pre, unit_ind_post in zip(unit_inds_pre, unit_inds_post):
            print(str(unit_ids_pre[unit_ind_pre]) + ' from data1 matches ' + str(unit_ids_post[unit_ind_post]) + ' from data2')
        '''
    unit_chain_df = pd.DataFrame(unit_chain, columns=['Unit_' + str(i) for i in range(unit_chain.shape[1])],
                                 index=recording_time_list)
    print('Saving cell bundle csv...')
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    unit_chain_df.to_csv(result_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_cell_bundle.csv')
    print('done')
    time_end = default_timer()
    print('Module time usage(s): ', time_end - time_start)


def trace_cluster(root_folder_input, experimenter, acquisition_system, mouse_id, max_spikeform_num=200,
                  time_scale_factor=0.5, elev_angle=30, azim_angle=-60,save_plot=True):
    time_start = default_timer()
    print('============================================================================')
    print('Animal: ', experimenter, acquisition_system, mouse_id)
    print('TRACE CLUSTER MODULE')
    print('Gathering materials...')
    cell_bundle_path = os.path.dirname(os.path.realpath(
        __file__)) + '/results/' + experimenter + '_' + acquisition_system + '_' + mouse_id + '_cell_bundle_OK.csv'
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
    unit_number = len(unit_names)
    print('Unit number: ', unit_number)
    waveform_list = []
    for recording_time in recording_time_list:
        assert os.path.isdir(root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                             '/' + recording_time), root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                                                    '/' + recording_time + ' does not exist, please make sure data exist'
        sorting_result_folder = root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                                '/' + recording_time + '/analysis/sorting_result/'
        assert os.path.isdir(
            sorting_result_folder), sorting_result_folder + ' does not exist, please sort spikes and post curate before extracting waveform'
        post_curation_folder = sorting_result_folder + 'phy/'
        assert os.path.isdir(
            post_curation_folder), post_curation_folder + ' does not exist, please run sort_spikes before post curation, or maybe no units are detected'
        waveform_folder = sorting_result_folder + 'waveform_from_phy/'
        assert os.path.isdir(
            waveform_folder), waveform_folder + ' does not exist, please run sort_spikes before post curation, or maybe no units are detected'
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
        waveform = si.load_waveforms(waveform_folder, with_recording=False)
        waveform_list.append(waveform)
        os.remove(waveform_folder + 'sorting.json')
        os.rename(waveform_folder + 'sorting_preserve.json', waveform_folder + 'sorting.json')
    print('Calculating channel inds mask...')
    common_channel_locations = set.intersection(
        *[set(map(tuple, waveform.get_channel_locations())) for waveform in waveform_list])
    common_channel_locations = [list(common_channel_location) for common_channel_location in common_channel_locations]
    common_channel_num = np.array(common_channel_locations).shape[0]
    assert common_channel_num >= 1, 'No common channel found. Bundling failed'
    print('Number of common channels: ' + str(common_channel_num))
    channel_inds_list = list(map(lambda waveform: get_channel_inds(waveform, common_channel_locations), waveform_list))
    print('Training PCA projector...')
    training = None
    for recording_ind in range(1, recording_number):
        templates = waveform_list[recording_ind].get_all_templates(mode='median')[:, :,
                    channel_inds_list[recording_ind]]
        templates_faltten = templates.reshape(templates.shape[0], -1)
        if training is None:
            training = templates_faltten
        else:
            training = np.vstack([training, templates_faltten])
    projector = PCA(n_components=3)
    projector.fit(training)
    del training
    print('Transforming templates and waveforms...')
    templates_dict = {}
    spikeforms_dict = {}
    for recording_ind in range(recording_number):
        recording_time = recording_time_list[recording_ind]
        templates_dict[recording_time] = {}
        spikeforms_dict[recording_time] = {}
        for unit_name in unit_names:
            unit_id = cell_bundle_df.loc[recording_time, unit_name]
            if not pd.isna(unit_id):
                template = waveform_list[recording_ind].get_all_templates(unit_ids=[int(unit_id)], mode='median')[:, :,
                           channel_inds_list[recording_ind]]
                template_flatten = template.reshape(template.shape[0], -1)
                templates_dict[recording_time][unit_name] = projector.transform(template_flatten)
                spikeforms = waveform_list[recording_ind].get_waveforms(unit_id=int(unit_id))[:, :,
                             channel_inds_list[recording_ind]]
                if spikeforms.shape[0] > max_spikeform_num:
                    random.seed(0)
                    spikeforms = spikeforms[random.sample(range(spikeforms.shape[0]), max_spikeform_num), :, :]
                spikeforms_flatten = spikeforms.reshape(spikeforms.shape[0], -1)
                spikeforms_dict[recording_time][unit_name] = projector.transform(spikeforms_flatten)
    print('Saving trace cluster csv...')
    templates_df = pd.DataFrame([], columns=['RecordingTime', 'UnitName', 'PC1', 'PC2', 'PC3'])
    spikeforms_df = pd.DataFrame([], columns=['RecordingTime', 'UnitName', 'PC1', 'PC2', 'PC3'])
    for recording_time in recording_time_list:
        if recording_time in templates_dict.keys() and recording_time in spikeforms_dict.keys():
            for unit_name in unit_names:
                if unit_name in templates_dict[recording_time].keys() and unit_name in spikeforms_dict[
                    recording_time].keys():
                    new_templates = templates_dict[recording_time][unit_name]
                    new_templates_np = np.hstack(
                        (
                        np.array((new_templates.shape[0]) * [recording_time, unit_name]).reshape(-1, 2), new_templates))
                    new_templates_df = pd.DataFrame(new_templates_np,
                                                    columns=['RecordingTime', 'UnitName', 'PC1', 'PC2', 'PC3'])
                    templates_df = pd.concat([templates_df, new_templates_df])

                    new_spikeforms = spikeforms_dict[recording_time][unit_name]
                    new_spikeforms_np = np.hstack(
                        (np.array((new_spikeforms.shape[0]) * [recording_time, unit_name]).reshape(-1, 2),
                         new_spikeforms))
                    new_spikeforms_df = pd.DataFrame(new_spikeforms_np,
                                                     columns=['RecordingTime', 'UnitName', 'PC1', 'PC2', 'PC3'])
                    spikeforms_df = pd.concat([spikeforms_df, new_spikeforms_df])
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    templates_df.to_csv(
        result_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_template_cluster.csv',
        index=False)
    spikeforms_df.to_csv(
        result_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_spikeform_cluster.csv',
        index=False)
    time_space = np.std(templates_df['PC1'].astype(float)) * time_scale_factor
    del templates_df
    del spikeforms_df
    print('Plotting PC cluster trace...')
    color_map = matplotlib.cm.get_cmap('gist_rainbow')
    shuffle_color_selector = np.array(range(unit_number)) / unit_number
    random.seed(0)
    random.shuffle(shuffle_color_selector)
    unit_color_map = color_map(shuffle_color_selector)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    for recording_ind, recording_time in enumerate(recording_time_list):
        z_value = recording_ind * time_space
        if recording_time in templates_dict.keys() and recording_time in spikeforms_dict.keys():
            for unit_name, unit_color in zip(unit_names, unit_color_map):
                if unit_name in templates_dict[recording_time].keys() and unit_name in spikeforms_dict[
                    recording_time].keys():
                    # plotting points
                    x = spikeforms_dict[recording_time][unit_name][:, 0]
                    y = spikeforms_dict[recording_time][unit_name][:, 1]
                    z = np.full(len(x), z_value)
                    ax.scatter(x, y, z, facecolor=unit_color, edgecolor=unit_color, marker='.')
                    x_center = templates_dict[recording_time][unit_name][:, 0]
                    y_center = templates_dict[recording_time][unit_name][:, 1]
                    z_center = np.full(len(x_center), z_value)
                    ax.scatter(x_center, y_center, z_center, facecolor='none', edgecolor='k', marker='o', s=50)
                    # plotting lines
                    if recording_ind >= 1:
                        recording_time_pre = recording_time_list[recording_ind-1]
                        if (recording_time_pre in templates_dict.keys()) and (unit_name in templates_dict[recording_time_pre].keys()):
                            x_center_pre = templates_dict[recording_time_pre][unit_name][:, 0]
                            y_center_pre = templates_dict[recording_time_pre][unit_name][:, 1]
                            z_center_pre = np.full(len(x_center_pre), z_value-time_space)
                            x_line = np.concatenate([x_center_pre, x_center])
                            y_line = np.concatenate([y_center_pre, y_center])
                            z_line = np.concatenate([z_center_pre, z_center])
                            ax.plot(x_line, y_line, z_line, color=unit_color)
    z_ticks = np.linspace(0, (recording_number-1)*time_space, recording_number)
    z_tick_labels = recording_time_list
    ax.set_zticks(z_ticks)
    ax.set_zticklabels(z_tick_labels)
    ax.zaxis.line.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='z', which='both', bottom=False, top=False, color='none')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(False)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('Unit Cluster Tracing')

    # 设置视角
    ax.view_init(elev=elev_angle, azim=azim_angle)

    # 美化图形
    ax.set_xlabel('PC 1', labelpad=15)
    ax.set_ylabel('PC 2', labelpad=15)
    ax.set_zlabel('Recording Time', labelpad=20)
    ax.set_title(f'Unit Cluster Tracing - {experimenter} {acquisition_system} {mouse_id}', pad=20)

    # 添加图例（可选）
    custom_legend = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, alpha=0.4),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markeredgecolor='k', markersize=10)
    ]
    ax.legend(custom_legend, ['Individual Waveforms', 'Template Centers'], loc='upper right')

    # 保存或显示图像
    if save_plot:
        output_image = f"{result_folder}{experimenter}_{acquisition_system}_{mouse_id}_PCA_trace_cluster.png"
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"✅ Saved PCA plot with data to: {output_image}")
    else:
        plt.show()

    plt.show()

def trace_unit_location(root_folder_input, experimenter, acquisition_system, mouse_id):
    print('============================================================================')
    print('UNIT LOCATION TRACE MODULE')

    root_folder = root_folder_input if root_folder_input.endswith('/') else root_folder_input + '/'

    cell_bundle_path = os.path.dirname(os.path.realpath(__file__)) + '/results/' + experimenter + '_' + acquisition_system + '_' + mouse_id + '_cell_bundle_OK.csv'
    assert os.path.isfile(cell_bundle_path), cell_bundle_path + ' does not exist'

    cell_bundle_df = pd.read_csv(cell_bundle_path, index_col=0)
    recording_time_list, unit_names = cell_bundle_df.index, cell_bundle_df.columns

    x_df = pd.DataFrame(index=recording_time_list, columns=unit_names)
    y_df = pd.DataFrame(index=recording_time_list, columns=unit_names)

    for recording_time in recording_time_list:
        print('Processing:', recording_time)

        sorting_folder = root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + '/' + recording_time + '/analysis/sorting_result/'
        location_path = sorting_folder + 'waveform_from_phy/unit_locations/unit_locations.npy'

        assert os.path.isfile(location_path), location_path + ' does not exist'

        locations = np.load(location_path)

        sorting = se.read_phy(
            sorting_folder + 'phy/',
            exclude_cluster_groups=['noise', 'mua']
        )

        unit_ids = sorting.get_unit_ids()

        for unit_name in unit_names:
            unit_id = cell_bundle_df.loc[recording_time, unit_name]

            if not pd.isna(unit_id):
                idx = np.where(unit_ids == int(unit_id))[0][0]
                x_df.loc[recording_time, unit_name] = locations[idx, 0]
                y_df.loc[recording_time, unit_name] = locations[idx, 1]

    result_folder = os.path.dirname(os.path.realpath(__file__)) + '/results/'
    name = experimenter + '_' + acquisition_system + '_' + mouse_id

    x_path = result_folder + name + '_unit_x_location.csv'
    y_path = result_folder + name + '_unit_y_location.csv'

    x_df.to_csv(x_path)
    y_df.to_csv(y_path)

    print('Saved:\n', x_path, '\n', y_path)

    return x_df, y_df