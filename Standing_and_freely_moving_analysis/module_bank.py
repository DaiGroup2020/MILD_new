from function_utils import get_channel_inds, fit_ellipse
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
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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
        similarity = cosine_similarity(templates_flatten_pre, templates_flatten_post)
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
                  time_scale_factor=0.5):
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
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
    plt.show()

def trace_isi(root_folder_input, experimenter, acquisition_system, mouse_id, unit_color_map=None):
    time_start = default_timer()
    print('============================================================================')
    print('Animal: ', experimenter, acquisition_system, mouse_id)
    print('TRACE ISI MODULE')
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
    print('Ploting trace ISI...')
    if unit_color_map is None:
        color_map = matplotlib.cm.get_cmap('gist_rainbow')
        shuffle_color_selector = np.array(range(unit_number)) / unit_number
        random.seed(0)
        random.shuffle(shuffle_color_selector)
        unit_color_map = color_map(shuffle_color_selector)
    fig = plt.figure()
    unit_plots = []
    for unit_name_ind, unit_name in enumerate(unit_names):
        unit_color = unit_color_map[unit_name_ind]
        # Create a subplot for the unit
        unit_subplot = fig.add_subplot(1, unit_number, unit_name_ind + 1, projection='3d')
        unit_subplot.grid(False)
        for recording_ind, recording_time in enumerate(recording_time_list):
            unit_id = cell_bundle_df.at[recording_time, unit_name]
            unit_ids = waveform_list[recording_ind].sorting.get_unit_ids()
            unit_ind = np.argwhere(unit_ids == unit_id)[0][0]
            (hist_all, edge) = spost.compute_isi_histograms(waveform_list[recording_ind], load_if_exists=True)
            x = np.array([(a + edge[i + 1]) / 2.0 for i, a in enumerate(edge[0:-1])])
            hist_all = hist_all / np.sum(hist_all, axis=1)[:, np.newaxis] * 100

            z = hist_all[unit_ind, :]
            y = np.full_like(x, recording_ind)
            # 创建多边形面的顶点
            vertices = np.column_stack((x, y, z))
            vertices = np.vstack((vertices, np.column_stack((np.flip(x), np.flip(y), np.full(x.shape, 0)))))
            poly3d = Poly3DCollection([vertices], color=unit_color, alpha=0.3)
            unit_subplot.add_collection3d(poly3d)
            unit_subplot.plot(x, y, z, color=unit_color, label=f'Record {recording_time}')
            unit_plots.append(unit_subplot)
        unit_plots[-1].set_xlabel('Interval s')
        unit_plots[-1].set_ylabel('Record')
        unit_plots[-1].set_zlabel('Probability')
        unit_plots[-1].set_title(unit_name)
        unit_plots[-1].set_axis_off()
    plt.show()


def trace_autocorr(root_folder_input, experimenter, acquisition_system, mouse_id, unit_color_map=None):
    time_start = default_timer()
    print('============================================================================')
    print('Animal: ', experimenter, acquisition_system, mouse_id)
    print('TRACE AUTOCORR MODULE')
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
    print('Ploting trace autocorr...')
    if unit_color_map is None:
        color_map = matplotlib.cm.get_cmap('gist_rainbow')
        shuffle_color_selector = np.array(range(unit_number)) / unit_number
        random.seed(0)
        random.shuffle(shuffle_color_selector)
        unit_color_map = color_map(shuffle_color_selector)
    fig = plt.figure()
    unit_plots = []
    for unit_name_ind, unit_name in enumerate(unit_names):
        unit_color = unit_color_map[unit_name_ind]
        # Create a subplot for the unit
        unit_subplot = fig.add_subplot(1, unit_number, unit_name_ind + 1, projection='3d')
        unit_subplot.grid(False)
        for recording_ind, recording_time in enumerate(recording_time_list):
            unit_id = cell_bundle_df.at[recording_time, unit_name]
            unit_ids = waveform_list[recording_ind].sorting.get_unit_ids()
            unit_ind = np.argwhere(unit_ids == unit_id)[0][0]

            (ccg, edge) = spost.compute_correlograms(waveform_list[recording_ind], load_if_exists=True)
            x = np.array([(a + edge[i + 1]) / 2.0 for i, a in enumerate(edge[0:-1])])
            z = ccg[unit_ind, unit_ind, :]
            z = z / np.max(z)
            y = np.full_like(x, recording_ind)
            # 创建多边形面的顶点
            vertices = np.column_stack((x, y, z))
            vertices = np.vstack((vertices, np.column_stack((np.flip(x), np.flip(y), np.full(x.shape, 0)))))
            poly3d = Poly3DCollection([vertices], color=unit_color, alpha=0.3)
            unit_subplot.add_collection3d(poly3d)
            unit_subplot.plot(x, y, z, color=unit_color, label=f'Record {recording_time}')
            unit_plots.append(unit_subplot)
        unit_plots[-1].set_xlabel('Time s')
        unit_plots[-1].set_ylabel('Record')
        unit_plots[-1].set_zlabel('Probability')
        unit_plots[-1].set_title(unit_name)
        unit_plots[-1].set_axis_off()
    plt.show()


def trace_corr(root_folder_input, experimenter, acquisition_system, mouse_id, unit_color_map=None):
    time_start = default_timer()
    print('============================================================================')
    print('Animal: ', experimenter, acquisition_system, mouse_id)
    print('TRACE CORR MODULE')
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
    print('Ploting trace corr...')
    if unit_color_map is None:
        color_map = matplotlib.cm.get_cmap('gist_rainbow')
        shuffle_color_selector = np.array(range(unit_number ** 2)) / (unit_number ** 2)
        random.seed(0)
        random.shuffle(shuffle_color_selector)
        unit_color_map = color_map(shuffle_color_selector)
    fig = plt.figure()
    unit_plots = []
    for unit_name_ind1, unit_name1 in enumerate(unit_names):
        for unit_name_ind2, unit_name2 in enumerate(unit_names):
            fig_ind = unit_name_ind1*unit_number + unit_name_ind2
            pair_color = unit_color_map[fig_ind]
            # Create a subplot for the unit
            unit_subplot = fig.add_subplot(unit_number, unit_number, fig_ind + 1, projection='3d')
            unit_subplot.grid(False)
            for recording_ind, recording_time in enumerate(recording_time_list):
                unit_ids = waveform_list[recording_ind].sorting.get_unit_ids()
                unit_id1 = cell_bundle_df.at[recording_time, unit_name1]
                unit_ind1 = np.argwhere(unit_ids == unit_id1)[0][0]
                unit_id2 = cell_bundle_df.at[recording_time, unit_name2]
                unit_ind2 = np.argwhere(unit_ids == unit_id2)[0][0]

                (ccg, edge) = spost.compute_correlograms(waveform_list[recording_ind], load_if_exists=True)
                x = np.array([(a + edge[i + 1]) / 2.0 for i, a in enumerate(edge[0:-1])])
                z = ccg[unit_ind1, unit_ind2, :]
                z = z / np.max(z)
                y = np.full_like(x, recording_ind)
                # 创建多边形面的顶点
                vertices = np.column_stack((x, y, z))
                vertices = np.vstack((vertices, np.column_stack((np.flip(x), np.flip(y), np.full(x.shape, 0)))))
                poly3d = Poly3DCollection([vertices], color=pair_color, alpha=0.3)
                unit_subplot.add_collection3d(poly3d)
                unit_subplot.plot(x, y, z, color=pair_color, label=f'Record {recording_time}')
                # unit_subplot.set_axis_off()
                unit_plots.append(unit_subplot)
            # unit_plots[-1].set_xlabel('Time s')
            # unit_plots[-1].set_ylabel('Record')
            # unit_plots[-1].set_zlabel('Probability')
            # unit_plots[-1].set_title(unit_name)
            # unit_plots[-1].set_axis_off()
    plt.show()


def trace_template(root_folder_input, experimenter, acquisition_system, mouse_id, unit_color_map=None, space_factor=0.1):
    time_start = default_timer()
    print('============================================================================')
    print('Animal: ', experimenter, acquisition_system, mouse_id)
    print('TRACE TEMPLATE MODULE')
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
    print('Ploting trace autocorr...')
    if unit_color_map is None:
        color_map = matplotlib.cm.get_cmap('gist_rainbow')
        shuffle_color_selector = np.array(range(unit_number)) / unit_number
        random.seed(0)
        random.shuffle(shuffle_color_selector)
        unit_color_map = color_map(shuffle_color_selector)
    fig = plt.figure()
    unit_plots = []
    for unit_name_ind, unit_name in enumerate(unit_names):
        unit_color = unit_color_map[unit_name_ind]
        # Create a subplot for the unit
        unit_subplot = fig.add_subplot(1, unit_number, unit_name_ind + 1, projection='3d')
        unit_subplot.grid(False)
        previous_y_upper = None
        for recording_ind, recording_time in enumerate(recording_time_list):
            unit_id = cell_bundle_df.at[recording_time, unit_name]
            unit_ids = waveform_list[recording_ind].sorting.get_unit_ids()
            unit_ind = np.argwhere(unit_ids == unit_id)[0][0]

            templates = waveform_list[recording_ind].get_template(unit_id=unit_id, mode='median')
            templates_std = waveform_list[recording_ind].get_template(unit_id=unit_id, mode='std')
            best_channel = np.argmax(np.max(np.abs(templates), axis=0))
            template = templates[:, best_channel]
            template_std = templates_std[:, best_channel]
            sample_frequency = waveform_list[recording_ind].recording.get_sampling_frequency()
            x = (np.array(range(len(template))) - int(len(template) / 2)) / sample_frequency * 1000
            if previous_y_upper is not None:
                temp = np.max(previous_y_upper - (template - template_std)) + space_factor*np.max(np.abs(template))
                y = template + temp
            else:
                y = template
            y_upper = y + template_std
            y_lower = y - template_std
            previous_y_upper = y_upper
            unit_subplot.plot(x, y, lw=0.5, color=unit_color, zorder=2)
            unit_subplot.fill_between(x, y_lower, y_upper, color=(0.5, 0.5, 0.5, 0.1), zorder=1)
            unit_plots.append(unit_subplot)
        unit_plots[-1].set_xlabel('Time ms')
        unit_plots[-1].set_title(unit_name)
        unit_plots[-1].set_axis_off()
    plt.show()

def trace_waveform_ellipse(root_folder_input, experimenter, acquisition_system, mouse_id, max_spikeform_num=200):
    # time_start = default_timer()
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
    figure_folder = os.path.dirname(os.path.realpath(__file__)) + '/figures/'
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
    common_channel_locations = [list(common_channel_location) for common_channel_location in
                                common_channel_locations]
    common_channel_num = np.array(common_channel_locations).shape[0]
    assert common_channel_num >= 1, 'No common channel found. Bundling failed'
    print('Number of common channels: ' + str(common_channel_num))
    channel_inds_list = list(
        map(lambda waveform: get_channel_inds(waveform, common_channel_locations), waveform_list))
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
    projector = PCA(n_components=2)
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
                template = waveform_list[recording_ind].get_all_templates(unit_ids=[int(unit_id)], mode='median')[:,
                           :,
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
    ellipse_dict = {}
    ellipse_by_waveform_df = pd.DataFrame([], columns=['RecordingTime', 'UnitName', 'Center_PC1', 'Center_PC2', 'a', 'b', 'angle'])
    for recording_time in recording_time_list:
        if recording_time in templates_dict.keys() and recording_time in spikeforms_dict.keys():
            ellipse_dict[recording_time] = {}
            for unit_name in unit_names:
                if unit_name in templates_dict[recording_time].keys() and unit_name in spikeforms_dict[
                    recording_time].keys():
                    new_spikeforms = spikeforms_dict[recording_time][unit_name]
                    ellipse_param = fit_ellipse(new_spikeforms)
                    ellipse_dict[recording_time][unit_name] = {}
                    ellipse_dict[recording_time][unit_name]['center'] = ellipse_param['center']
                    ellipse_dict[recording_time][unit_name]['axes'] = ellipse_param['axes']
                    ellipse_dict[recording_time][unit_name]['angle'] = ellipse_param['angle']
                    new_ellipse_np = np.hstack(
                        (np.array([recording_time, unit_name]).reshape(-1, 2), np.array([ellipse_param['center'][0],
                         ellipse_param['center'][1], ellipse_param['axes'][0], ellipse_param['axes'][1],
                         ellipse_param['angle']]).reshape(-1, 5)))
                    new_ellipse_df = pd.DataFrame(new_ellipse_np,
                                                     columns=['RecordingTime', 'UnitName', 'Center_PC1', 'Center_PC2', 'a', 'b', 'angle'])
                    ellipse_by_waveform_df = pd.concat([ellipse_by_waveform_df, new_ellipse_df])
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    ellipse_by_waveform_df.to_csv(
        result_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_waveform_ellipse.csv',
        index=False)
    # time_space = np.mean(ellipse_by_waveform_df['a'].astype(float)) * time_scale_factor
    del ellipse_by_waveform_df
    print('Plotting PC cluster trace...')
    color_map = matplotlib.cm.get_cmap('gist_rainbow')
    shuffle_color_selector = np.array(range(unit_number)) / unit_number
    random.seed(0)
    random.shuffle(shuffle_color_selector)
    unit_color_map = color_map(shuffle_color_selector)
    fig, ax = plt.subplots(figsize=(8, 8))
    for recording_ind, recording_time in enumerate(recording_time_list):
        alpha = 0.2 + (recording_ind / (len(recording_time_list)-1)) * 0.8  # 浅色(0.2) → 深色(1.0)
        if recording_time in templates_dict.keys() and recording_time in spikeforms_dict.keys():
            for unit_name, unit_color in zip(unit_names, unit_color_map):
                if unit_name in templates_dict[recording_time].keys() and unit_name in spikeforms_dict[
                    recording_time].keys():
                    # plotting ellipse:
                    ellipse = Ellipse(
                        xy=ellipse_dict[recording_time][unit_name]['center'],
                        width=2 * ellipse_dict[recording_time][unit_name]['axes'][0],
                        height=2 * ellipse_dict[recording_time][unit_name]['axes'][1],
                        angle=np.degrees(ellipse_dict[recording_time][unit_name]['angle']),
                        edgecolor=unit_color,
                        facecolor='none',
                        alpha=alpha,
                        linewidth=2
                    )
                    ax.add_patch(ellipse)
    ax.autoscale_view()
    if not os.path.isdir(figure_folder):
        os.makedirs(figure_folder)
    plt.savefig(figure_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_waveform_ellipse.svg')
    # plt.show()

def trace_heatmap_ellipse(root_folder_input, experimenter, acquisition_system, mouse_id, max_spikeform_num=200):
    # time_start = default_timer()
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
    figure_folder = os.path.dirname(os.path.realpath(__file__)) + '/figures/'
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
    common_channel_locations = [list(common_channel_location) for common_channel_location in
                                common_channel_locations]
    common_channel_num = np.array(common_channel_locations).shape[0]
    assert common_channel_num >= 1, 'No common channel found. Bundling failed'
    print('Number of common channels: ' + str(common_channel_num))
    channel_inds_list = list(
        map(lambda waveform: get_channel_inds(waveform, common_channel_locations), waveform_list))
    print('Training PCA projector...')
    training = None
    for recording_ind in range(1, recording_number):
        templates = waveform_list[recording_ind].get_all_templates(mode='median')[:, :,
                    channel_inds_list[recording_ind]]
        n_sample_points = templates.shape[1]
        start_index = int(n_sample_points / 3)
        end_index = int(n_sample_points * 2 / 3)
        templates_sliced = templates[:, start_index:end_index, :]
        abs_templates_sliced = np.abs(templates_sliced)
        max_values = np.max(abs_templates_sliced, axis=1)
        max_indices = np.argmax(abs_templates_sliced, axis=1)
        templates_heatmaps = np.stack([max_values, max_indices.astype(float)], axis=1)
        templates_heatmaps_faltten = templates_heatmaps.reshape(templates_heatmaps.shape[0], -1)
        if training is None:
            training = templates_heatmaps_faltten
        else:
            training = np.vstack([training, templates_heatmaps_faltten])

    projector = PCA(n_components=2)
    projector.fit(training)
    del training
    print('Transforming templates and waveforms...')
    heatmaps_dict = {}
    for recording_ind in range(recording_number):
        recording_time = recording_time_list[recording_ind]
        heatmaps_dict[recording_time] = {}
        for unit_name in unit_names:
            unit_id = cell_bundle_df.loc[recording_time, unit_name]
            if not pd.isna(unit_id):
                spikeforms = waveform_list[recording_ind].get_waveforms(unit_id=int(unit_id))[:, :,
                             channel_inds_list[recording_ind]]
                if spikeforms.shape[0] > max_spikeform_num:
                    random.seed(0)
                    spikeforms = spikeforms[random.sample(range(spikeforms.shape[0]), max_spikeform_num), :, :]

                n_sample_points = spikeforms.shape[1]
                start_index = int(n_sample_points / 3)
                end_index = int(n_sample_points * 2 / 3)
                spikeforms_sliced = spikeforms[:, start_index:end_index, :]
                abs_spikeforms_sliced = np.abs(spikeforms_sliced)
                max_values = np.max(abs_spikeforms_sliced, axis=1)
                max_indices = np.argmax(abs_spikeforms_sliced, axis=1)
                heatmaps = np.stack([max_values, max_indices.astype(float)], axis=1)
                heatmaps_flatten = heatmaps.reshape(heatmaps.shape[0], -1)
                heatmaps_dict[recording_time][unit_name] = projector.transform(heatmaps_flatten)

    print('Saving trace cluster csv...')
    ellipse_dict = {}
    ellipse_by_heatmap_df = pd.DataFrame([], columns=['RecordingTime', 'UnitName', 'Center_PC1', 'Center_PC2', 'a', 'b',
                                                      'angle'])
    for recording_time in recording_time_list:
        if recording_time in heatmaps_dict.keys():
            ellipse_dict[recording_time] = {}
            for unit_name in unit_names:
                if unit_name in heatmaps_dict[recording_time].keys():
                    new_heatmap = heatmaps_dict[recording_time][unit_name]
                    ellipse_param = fit_ellipse(new_heatmap)
                    ellipse_dict[recording_time][unit_name] = {}
                    ellipse_dict[recording_time][unit_name]['center'] = ellipse_param['center']
                    ellipse_dict[recording_time][unit_name]['axes'] = ellipse_param['axes']
                    ellipse_dict[recording_time][unit_name]['angle'] = ellipse_param['angle']
                    new_ellipse_np = np.hstack(
                        (np.array([recording_time, unit_name]).reshape(-1, 2), np.array([ellipse_param['center'][0],
                                                                                         ellipse_param['center'][1],
                                                                                         ellipse_param['axes'][0],
                                                                                         ellipse_param['axes'][1],
                                                                                         ellipse_param[
                                                                                             'angle']]).reshape(-1, 5)))
                    new_ellipse_df = pd.DataFrame(new_ellipse_np,
                                                  columns=['RecordingTime', 'UnitName', 'Center_PC1', 'Center_PC2', 'a',
                                                           'b', 'angle'])
                    ellipse_by_heatmap_df = pd.concat([ellipse_by_heatmap_df, new_ellipse_df])
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    ellipse_by_heatmap_df.to_csv(
        result_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_heatmap_ellipse.csv',
        index=False)
    # time_space = np.mean(ellipse_by_waveform_df['a'].astype(float)) * time_scale_factor
    del ellipse_by_heatmap_df

    print('Plotting PC cluster trace...')
    color_map = matplotlib.cm.get_cmap('gist_rainbow')
    shuffle_color_selector = np.array(range(unit_number)) / unit_number
    random.seed(0)
    random.shuffle(shuffle_color_selector)
    unit_color_map = color_map(shuffle_color_selector)
    fig, ax = plt.subplots(figsize=(8, 8))
    for recording_ind, recording_time in enumerate(recording_time_list):
        alpha = 0.2 + (recording_ind / (len(recording_time_list) - 1)) * 0.8  # 浅色(0.2) → 深色(1.0)
        if recording_time in heatmaps_dict.keys():
            for unit_name, unit_color in zip(unit_names, unit_color_map):
                if unit_name in heatmaps_dict[recording_time].keys():
                    # plotting ellipse:
                    ellipse = Ellipse(
                        xy=ellipse_dict[recording_time][unit_name]['center'],
                        width=2 * ellipse_dict[recording_time][unit_name]['axes'][0],
                        height=2 * ellipse_dict[recording_time][unit_name]['axes'][1],
                        angle=np.degrees(ellipse_dict[recording_time][unit_name]['angle']),
                        edgecolor=unit_color,
                        facecolor='none',
                        alpha=alpha,
                        linewidth=2
                    )
                    ax.add_patch(ellipse)
    ax.autoscale_view()
    if not os.path.isdir(figure_folder):
        os.makedirs(figure_folder)
    plt.savefig(figure_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_heatmap_ellipse.svg')
    plt.show()

def trace_unit_location(root_folder_input, experimenter, acquisition_system, mouse_id):
    print('============================================================================')
    print('UNIT LOCATION TRACE MODULE')
    root_folder = root_folder_input if root_folder_input.endswith('/') else root_folder_input + '/'

    cell_bundle_path = os.path.dirname(os.path.realpath(__file__)) + '/results/' + experimenter + '_' + acquisition_system + '_' + mouse_id + '_cell_bundle_OK.csv'
    assert os.path.isfile(cell_bundle_path), cell_bundle_path + ' does not exist'

    cell_bundle_df = pd.read_csv(cell_bundle_path, index_col=0)
    recording_time_list, unit_names = cell_bundle_df.index, cell_bundle_df.columns
    result_folder = os.path.dirname(os.path.realpath(__file__)) + '/results/'
    location_df = pd.DataFrame(columns=['RecordingTime','UnitName','UnitID','x','y'])
    waveform_list = []

    for recording_time in recording_time_list:
        waveform_folder = root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + '/' + recording_time + '/analysis/sorting_result/waveform_from_phy/'
        assert os.path.isdir(waveform_folder), waveform_folder + ' does not exist'
        waveform_list.append(si.load_waveforms(waveform_folder, with_recording=True))

    for i, recording_time in enumerate(recording_time_list):
        waveform = waveform_list[i]
        unit_locations = pd.DataFrame(spost.compute_unit_locations(waveform), columns=['x','y'])
        unit_ids = waveform.sorting.get_unit_ids()

        for unit_name in unit_names:
            unit_id = cell_bundle_df.loc[recording_time, unit_name]
            if not pd.isna(unit_id):
                unit_id = int(unit_id)
                unit_index = np.where(unit_ids == unit_id)[0][0]
                location_df.loc[len(location_df)] = [
                    recording_time,
                    unit_name,
                    unit_id,
                    unit_locations.iloc[unit_index]['x'],
                    unit_locations.iloc[unit_index]['y']
                ]

    save_path = result_folder + experimenter + '_' + acquisition_system + '_' + mouse_id + '_unit_location_trace.csv'
    location_df.to_csv(save_path, index=False)

    print('Saved:', save_path)
    return location_df

    









