import warnings

warnings.filterwarnings('ignore')
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.exporters as sexp

import probeinterface as pi
import spikeinterface.sorters as ss
from intanutil.read_header import read_header as read_intan_header
import json
import shutil
import time
from phy.apps.template import template_gui
import filecmp
import pprint
from timeit import default_timer
from single_recording_plot_widgets import *
import matplotlib
import re
import ast
import spikeinterface.postprocessing as spost

def sort_spikes(root_folder_input, experimenter, acquisition_system, mouse_id, recording_time, impedance_threshold=3,
                sorter="kilosort2", local=True):
    assert not root_folder_input.startswith('Z:'), 'This is study version code, DO NOT USE REAL DATA ON Z DISK! Change your root to your pseudo dataset.'
    # print(sorter)
    # print(sorter in ss.installed_sorters())
    assert sorter in ss.installed_sorters(), "sorter " + sorter + " is not detected in this computer. If this is a server computer, please make sure input sorter is properly installed. If this is a client computer, DO NOT RUN SORT_SPIKES FUNCTION!!!"
    time_start = default_timer()
    print('============================================================================')
    print('data: ', experimenter, acquisition_system, mouse_id, recording_time)
    print('SORT SPIKES MODULE')
    print('Routing...')
    if not root_folder_input.endswith('/'):
        source_root_folder = root_folder_input + '/'
    else:
        source_root_folder = root_folder_input
    source_probe_layout_file_path = source_root_folder + experimenter + '/probe_layout.csv'
    source_raw_data_folder = source_root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                             '/' + recording_time + '/raw_data/'
    assert os.path.isfile(
        source_probe_layout_file_path), source_probe_layout_file_path + ' does not exist, please create this file under experimenter  folder before sort spikes'
    assert os.path.isdir(
        source_raw_data_folder), source_raw_data_folder + 'does not exist, please put recording data in this path before sorting'
    source_sorting_result_folder = source_root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                                   '/' + recording_time + '/analysis/sorting_result/'
    if not os.path.isdir(source_sorting_result_folder):
        os.makedirs(source_sorting_result_folder)

    if local:
        print('copying data to local...')
        root_folder = os.path.expanduser('~') + '/Documents/Arueruma/'
        probe_layout_file_path = root_folder + '/' + experimenter + '/probe_layout.csv'
        raw_data_folder = root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                          '/' + recording_time + '/raw_data/'
        sorting_result_folder = root_folder + '/' + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                                '/' + recording_time + '/analysis/sorting_result/'
        if os.path.exists(root_folder):
            shutil.rmtree(root_folder)
        os.makedirs(raw_data_folder)
        shutil.copytree(source_raw_data_folder, raw_data_folder, dirs_exist_ok=True)
        os.makedirs(sorting_result_folder)
        # shutil.copytree(source_sorting_result_folder, sorting_result_folder, dirs_exist_ok=True)
        shutil.copyfile(source_probe_layout_file_path, probe_layout_file_path)
    else:
        root_folder = source_root_folder
        probe_layout_file_path = source_probe_layout_file_path
        raw_data_folder = source_raw_data_folder
        sorting_result_folder = source_sorting_result_folder
    print('reading recordings...')

    raw_data_list = os.listdir(raw_data_folder)
    if acquisition_system == 'intan':
        recording_data_list = [i for i in raw_data_list if i.endswith('.rhd')]
        assert len(recording_data_list) >= 1, 'no .rhd files detected in folder: ' + raw_data_folder
        recording = si.concatenate_recordings(
            [se.read_intan(raw_data_folder + recording, stream_id='0') for recording in recording_data_list])
        channel_num = len(recording.get_channel_ids())
        impedance_file = open(raw_data_folder + recording_data_list[0], 'rb')
        intan_header = read_intan_header(impedance_file)
        impedance_list = np.array(
            [intan_header['amplifier_channels'][i]['electrode_impedance_magnitude'] / 1e6 for i in range(channel_num)])
    elif acquisition_system == 'blackrock':
        recording_data_list = [i for i in raw_data_list if i.endswith('.ns6')]
        assert len(recording_data_list) >= 1, 'no .ns6 files detected in folder: ' + raw_data_folder
        recording = si.concatenate_recordings(
            [se.read_blackrock(raw_data_folder + recording) for recording in recording_data_list])
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
    print(recording)
    probe_layout_pd = pd.read_csv(probe_layout_file_path)
    assert ['acquisition_system', 'mouse_id', 'channel_number', 'layout'] == list(
        probe_layout_pd.head()), 'please check header of probe layout file should be [acquisition_system, mouse_id, channel_number, layout]'
    probe_layout_selected_pd = probe_layout_pd[(probe_layout_pd['acquisition_system'] == acquisition_system) &
                                               (probe_layout_pd['mouse_id'] == int(mouse_id))]
    assert probe_layout_selected_pd.shape[
               0] == 1, 'There should be and only be 1 layout for ' + acquisition_system + ' system with mouse ' + mouse_id + ', however, ' + str(
        probe_layout_selected_pd.shape[0]) + ' layout is found'
    assert int(
        probe_layout_selected_pd['channel_number'].values[
            0]) == channel_num, 'channel number from recording' + '(' + str(
        channel_num) + ')' + ' does not match with channel_number' + '(' + probe_layout_selected_pd[
                                    'channel_number'] + ')' + ' in probe_layout file'
    probe_layout = np.array(json.loads(probe_layout_selected_pd['layout'].values[0]))
    assert probe_layout.shape[0] == channel_num, 'channel number from recording' + '(' + str(
        channel_num) + ')' + ' does not match with number of positions' + '(' + str(
        probe_layout.shape[0]) + ')' + ' in probe_layout file'
    probe = pi.Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=probe_layout, shapes='circle', shape_params={'radius': 10})
    probe.set_device_channel_indices(np.array(range(channel_num)))
    recording = recording.set_probe(probe)
    recording_filtered = spre.notch_filter(recording, 50, dtype='float64')
    recording_filtered = spre.bandpass_filter(recording_filtered, freq_min=300, freq_max=5000)
    bad_channels_ids_auto = spre.detect_bad_channels(recording_filtered, method='mad')[0]
    bad_channel_ids_imp = recording_filtered.get_channel_ids()[impedance_list > impedance_threshold]
    bad_channel_ids = list(set(bad_channels_ids_auto) | set(bad_channel_ids_imp))
    recording_removed = recording_filtered.remove_channels(bad_channel_ids)
    print('number of bad channel removed: ' + str(len(bad_channel_ids)) + ' (' + str(
        len(bad_channel_ids_imp)) + ' from impedance and ' + str(len(bad_channels_ids_auto)) + ' from auto detect)')
    if "inter_sample_shift" in recording_removed.get_property_keys():
        recording_removed = spre.phase_shift(recording_removed)
    recording_cmr = spre.common_reference(recording_removed, operator='median', reference='global')
    # print('removing motion noise...')
    # recording_cmr = spre.correct_motion(recording_cmr, preset="nonrigid_accurate")
    print('running sorter...')
    if sorter == 'kilosort':
        hyperparams = ss.KilosortSorter.default_params()
    elif sorter == 'kilosort2':
        hyperparams = ss.Kilosort2Sorter.default_params()
    elif sorter == 'kilosort3':
        hyperparams = ss.Kilosort3Sorter.default_params()
    elif sorter == 'spykingcircus':
        hyperparams = ss.SpykingcircusSorter.default_params()
    else:
        raise Exception('Unknown sorter, currently [kilosort, kilosort2, kilosort3, spykingcircus] are supported')
    print('sorter: ' + sorter)
    print('hyperparams:')
    print(hyperparams)
    try:
        sorting = ss.run_sorter(sorter, recording_cmr, output_folder=sorting_result_folder + 'sorter',
                                remove_existing_folder=True, **hyperparams)
        template_ind = np.load(sorting_result_folder + 'sorter/sorter_output/templates_ind.npy')
        temp_unit_number, _ = template_ind.shape
        if temp_unit_number > 1:
            sorting_success_flag = True
        else:
            print('IMPROTANT: unable to detect more than noe spike! Sorting Failed.')
            sorting_success_flag = False
    except Exception as e:
        print('IMPROTANT: unable to detect any spikes! Sorting Failed.')
        sorting = None
        sorting_success_flag = False
    if sorting_success_flag:
        print('running waveform extractor...')
        waveform = si.WaveformExtractor.create(recording_cmr, sorting, sorting_result_folder + 'waveform',
                                               remove_if_exists=True)
        waveform.set_params(ms_before=3., ms_after=4., return_scaled=True)
        waveform.run_extract_waveforms(n_jobs=-1, chunk_size=30000)
        print('exporting to phy...')
        sexp.export_to_phy(waveform, compute_pc_features=True, compute_amplitudes=True,
                           output_folder=sorting_result_folder + 'phy',
                           remove_if_exists=True, n_jobs=-1, copy_binary=True)

    if os.path.exists(sorting_result_folder + 'sorter'):
        shutil.rmtree(sorting_result_folder + 'sorter')
    if os.path.exists(sorting_result_folder + 'waveform'):
        shutil.rmtree(sorting_result_folder + 'waveform')

    sorting_metadata = {}
    sorting_metadata['sorting_success'] = sorting_success_flag
    sorting_metadata['layout'] = probe_layout
    sorting_metadata['impedance'] = impedance_list
    sorting_metadata['impedance_threshold'] = impedance_threshold
    sorting_metadata['auto_bad_channel_detection'] = True
    sorting_metadata['bad_channel_number'] = len(bad_channel_ids)
    sorting_metadata['sorter'] = sorter
    sorting_metadata['sorter_hyperparameter'] = hyperparams
    sorting_metadata['sorting_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    sorting_metadata['post_curation'] = False
    sorting_metadata['post_curation_time'] = None
    sorting_metadata['post_curator'] = None
    sorting_metadata['good_unit_number'] = None
    sorting_metadata['post_curation_accomplished'] = False
    sorting_metadata['waveform_extraction_accomplished'] = False
    sorting_metadata['single_recording_figure_generation_accomplished'] = False
    sorting_metadata['single_recording_template_layout'] = None
    sorting_metadata['single_recording_unit_position'] = None
    sorting_metadata['single_recording_isi'] = None
    sorting_metadata['single_recording_cross_correlation'] = None
    sorting_metadata['single_recording_template_layout_parameters'] = None
    sorting_metadata['single_recording_unit_position_parameters'] = None
    sorting_metadata['single_recording_isi_parameters'] = None
    sorting_metadata['single_recording_cross_correlation_parameters'] = None
    np.save(sorting_result_folder + 'sorting_metadata.npy', sorting_metadata)
    if local:
        print('copying results to source...')
        shutil.rmtree(source_sorting_result_folder)
        shutil.copytree(sorting_result_folder, source_sorting_result_folder, dirs_exist_ok=True)
    '''
    for process in psutil.Process().children():
        print(process)
        process.terminate()
    '''
    print('done')
    time_end = default_timer()
    print('Module time usage(s): ', time_end - time_start)


def post_curate_spikes(root_folder_input, experimenter, acquisition_system, mouse_id, recording_time, local=True):
    assert not root_folder_input.startswith(
        'Z:'), 'This is study version code, DO NOT USE REAL DATA ON Z DISK! Change your root to your pseudo dataset.'
    time_start = default_timer()
    print('============================================================================')
    print('data: ', experimenter, acquisition_system, mouse_id, recording_time)
    print('POST CURATION MODULE')
    print('Routing...')
    if not root_folder_input.endswith('/'):
        source_root_folder = root_folder_input + '/'
    else:
        source_root_folder = root_folder_input
    source_sorting_result_folder = source_root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                                   '/' + recording_time + '/analysis/sorting_result/'
    source_post_curation_folder = source_sorting_result_folder + 'phy/'
    assert os.path.isdir(
        source_post_curation_folder), source_post_curation_folder + ' does not exist, please run sort_spikes before post curation. Or maybe no spikes are detected.'
    assert os.path.isfile(
        source_post_curation_folder + 'spike_clusters.npy'), 'file not found: ' + source_post_curation_folder + 'spike_clusters.npy'
    assert os.path.isfile(
        source_post_curation_folder + 'cluster_group.tsv'), 'file not found: ' + source_post_curation_folder + 'cluster_group.tsv'
    source_sorting_metadata_path = source_sorting_result_folder + 'sorting_metadata.npy'
    assert os.path.isfile(
        source_sorting_metadata_path), source_sorting_metadata_path + 'does not exist, please run sort_spikes before post curation.'

    if local:
        print('copying data to local...')
        root_folder = os.path.expanduser('~') + '/Documents/Arueruma/'
        sorting_result_folder = root_folder + '/' + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                                '/' + recording_time + '/analysis/sorting_result/'
        post_curation_folder = sorting_result_folder + 'phy/'
        sorting_metadata_path = sorting_result_folder + 'sorting_metadata.npy'
        if os.path.exists(root_folder):
            shutil.rmtree(root_folder)
        os.makedirs(post_curation_folder)
        shutil.copytree(source_post_curation_folder, post_curation_folder, dirs_exist_ok=True)
        shutil.copyfile(source_sorting_metadata_path, sorting_metadata_path)
    else:
        sorting_result_folder = source_sorting_result_folder
        post_curation_folder = source_post_curation_folder
        sorting_metadata_path = source_sorting_metadata_path

    shutil.copyfile(post_curation_folder + 'spike_clusters.npy', post_curation_folder + 'spike_clusters_old.npy')
    shutil.copyfile(post_curation_folder + 'cluster_group.tsv', post_curation_folder + 'cluster_group_old.tsv')
    if os.path.exists(post_curation_folder + 'params_preserve.py'):
        os.remove(post_curation_folder + 'params_preserve.py')
    os.rename(post_curation_folder + 'params.py', post_curation_folder + 'params_preserve.py')
    with open(post_curation_folder + 'params_preserve.py', 'r') as params_preserve:
        content = params_preserve.readlines()
        content[0] = "dat_path = r'" + post_curation_folder + "recording.dat'\n"
        with open(post_curation_folder + 'params.py', 'a') as params:
            params.writelines(content)

    print('post curating...')
    modification_flag = False
    # template_gui(post_curation_folder+'params.py') # , clear_state=True, clear_cache=True
    os.system("phy template-gui " + post_curation_folder + "/params.py")
    if not filecmp.cmp(post_curation_folder + 'spike_clusters.npy',
                       post_curation_folder + 'spike_clusters_old.npy') or not \
            filecmp.cmp(post_curation_folder + 'cluster_group.tsv', post_curation_folder + 'cluster_group_old.tsv'):
        modification_flag = True
        sorting_metadata = np.load(sorting_metadata_path, allow_pickle=True).item()
        cluster_group = pd.read_csv(post_curation_folder + 'cluster_group.tsv', sep='\t')
        units_labels = cluster_group['group'].to_list()
        sorting_metadata['post_curation'] = True
        sorting_metadata['post_curation_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        sorting_metadata['post_curator'] = os.environ['COMPUTERNAME']
        sorting_metadata['good_unit_number'] = units_labels.count('good')
        if 'unsorted' in units_labels:
            sorting_metadata['post_curation_accomplished'] = False
        else:
            sorting_metadata['post_curation_accomplished'] = True
        sorting_metadata['waveform_extraction_accomplished'] = False
        sorting_metadata['single_recording_figure_generation_accomplished'] = False
        sorting_metadata['single_recording_template_layout'] = None
        sorting_metadata['single_recording_unit_position'] = None
        sorting_metadata['single_recording_isi'] = None
        sorting_metadata['single_recording_cross_correlation'] = None
        sorting_metadata['single_recording_template_layout_parameters'] = None
        sorting_metadata['single_recording_unit_position_parameters'] = None
        sorting_metadata['single_recording_isi_parameters'] = None
        sorting_metadata['single_recording_cross_correlation_parameters'] = None
        np.save(sorting_metadata_path, sorting_metadata)
        post_curator = input('Please input your name:')
        if post_curator == '\n' or post_curator == '' or post_curator == None:
            print('no name input, use computer name instead')
        else:
            print('post curator: ' + post_curator)
            sorting_metadata = np.load(sorting_metadata_path, allow_pickle=True).item()
            sorting_metadata['post_curator'] = post_curator
            np.save(sorting_metadata_path, sorting_metadata)
    else:
        modification_flag = False
        print('no modification, nothing changed')
    os.remove(post_curation_folder + 'params.py')
    os.rename(post_curation_folder + 'params_preserve.py', post_curation_folder + 'params.py')
    if os.path.isfile(post_curation_folder + 'spike_clusters_old.npy'):
        os.remove(post_curation_folder + 'spike_clusters_old.npy')
    if os.path.isfile(post_curation_folder + 'cluster_group_old.tsv'):
        os.remove(post_curation_folder + 'cluster_group_old.tsv')
    if local and modification_flag:
        print('copying data to source...')
        shutil.rmtree(source_post_curation_folder)
        shutil.copytree(post_curation_folder, source_post_curation_folder, dirs_exist_ok=True)
        os.remove(source_sorting_metadata_path)
        shutil.copyfile(sorting_metadata_path, source_sorting_metadata_path)
    print('done')
    time_end = default_timer()
    print('Module time usage(s): ', time_end - time_start)


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


def waveform_extraction_after_phy(root_folder_input, experimenter, acquisition_system, mouse_id, recording_time,
                                  local=False):
    assert not root_folder_input.startswith(
        'Z:'), 'This is study version code, DO NOT USE REAL DATA ON Z DISK! Change your root to your pseudo dataset.'
    time_start = default_timer()
    print('============================================================================')
    print('data: ', experimenter, acquisition_system, mouse_id, recording_time)
    print('WAVEFORM EXTRACTION MODULE')
    print('Routing...')
    if not root_folder_input.endswith('/'):
        source_root_folder = root_folder_input + '/'
    else:
        source_root_folder = root_folder_input
    source_raw_data_folder = source_root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                             '/' + recording_time + '/raw_data/'
    assert os.path.isdir(
        source_raw_data_folder), source_raw_data_folder + ' does not exist, please put recording data in this path before sorting'
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
    if not os.path.isdir(source_waveform_folder):
        os.makedirs(source_waveform_folder)
    sorting_metadata = read_sorting_metadata(root_folder_input, experimenter, acquisition_system, mouse_id,
                                             recording_time, display=False)
    assert sorting_metadata[
        'post_curation_accomplished'], 'Post curation is not finished yet. Please label all units before extracting waveforms'
    assert sorting_metadata['good_unit_number'] is not None, 'Please finish post curation before extracting waveforms'
    assert sorting_metadata['good_unit_number'] > 0, '0 good units found, unable to extract waveform'

    if local:
        print('copying data to local...')
        root_folder = os.path.expanduser('~') + '/Documents/Arueruma/'
        raw_data_folder = root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                          '/' + recording_time + '/raw_data/'
        sorting_result_folder = root_folder + '/' + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                                '/' + recording_time + '/analysis/sorting_result/'
        sorting_metadata_path = sorting_result_folder + 'sorting_metadata.npy'
        post_curation_folder = sorting_result_folder + 'phy/'
        waveform_folder = sorting_result_folder + 'waveform_from_phy/'
        if os.path.exists(root_folder):
            shutil.rmtree(root_folder)
        os.makedirs(raw_data_folder)
        shutil.copytree(source_raw_data_folder, raw_data_folder, dirs_exist_ok=True)
        os.makedirs(post_curation_folder)
        shutil.copytree(source_post_curation_folder, post_curation_folder, dirs_exist_ok=True)
        shutil.copyfile(source_sorting_metadata_path, sorting_metadata_path)
        os.makedirs(waveform_folder)
    else:
        root_folder = source_root_folder
        raw_data_folder = source_raw_data_folder
        sorting_result_folder = source_sorting_result_folder
        sorting_metadata_path = source_sorting_metadata_path
        post_curation_folder = source_post_curation_folder
        waveform_folder = source_waveform_folder
    print('reading recordings...')

    sorting_metadata = np.load(sorting_metadata_path, allow_pickle=True).item()
    raw_data_list = os.listdir(raw_data_folder)
    if acquisition_system == 'intan':
        recording_data_list = [i for i in raw_data_list if i.endswith('.rhd')]
        assert len(recording_data_list) >= 1, 'no .rhd files detected in folder: ' + raw_data_folder
        recording = si.concatenate_recordings(
            [se.read_intan(raw_data_folder + recording, stream_id='0') for recording in recording_data_list])
    elif acquisition_system == 'blackrock':
        recording_data_list = [i for i in raw_data_list if i.endswith('.ns6')]
        assert len(recording_data_list) >= 1, 'no .ns6 files detected in folder: ' + raw_data_folder
        recording = si.concatenate_recordings(
            [se.read_blackrock(raw_data_folder + recording) for recording in recording_data_list])
    else:
        recording = None
        raise Exception('acquisition_system currently should either be intan or blackrock')
    print(recording)
    probe_layout = sorting_metadata['layout']
    channel_num = probe_layout.shape[0]
    impedance_list = sorting_metadata['impedance']
    impedance_threshold = sorting_metadata['impedance_threshold']
    auto_bad_channel_detection_flag = sorting_metadata['auto_bad_channel_detection']
    probe = pi.Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=probe_layout, shapes='circle', shape_params={'radius': 10})
    probe.set_device_channel_indices(np.array(range(channel_num)))
    recording = recording.set_probe(probe)

    recording_filtered = spre.notch_filter(recording, 50, dtype='float64')
    recording_filtered = spre.bandpass_filter(recording_filtered, freq_min=300, freq_max=5000)
    if auto_bad_channel_detection_flag:
        bad_channels_ids_auto = spre.detect_bad_channels(recording_filtered, method='mad')[0]
    else:
        bad_channels_ids_auto = []
    bad_channel_ids_imp = recording_filtered.get_channel_ids()[impedance_list > impedance_threshold]
    bad_channel_ids = list(set(bad_channels_ids_auto) | set(bad_channel_ids_imp))
    recording_removed = recording_filtered.remove_channels(bad_channel_ids)
    print('number of bad channel removed: ' + str(len(bad_channel_ids)) + ' (' + str(
        len(bad_channel_ids_imp)) + ' from impedance and ' + str(len(bad_channels_ids_auto)) + ' from auto detect)')
    if "inter_sample_shift" in recording_removed.get_property_keys():
        recording_removed = spre.phase_shift(recording_removed)
    recording_cmr = spre.common_reference(recording_removed, operator='median', reference='global')
    # print('removing motion noise...')
    # recording_cmr = spre.correct_motion(recording_cmr, preset="nonrigid_accurate")
    sorting = se.read_phy(post_curation_folder, exclude_cluster_groups=["noise", 'mua'])
    print('running waveform extractor...')
    waveform = si.WaveformExtractor.create(recording_cmr, sorting, waveform_folder,
                                           remove_if_exists=True)
    waveform.set_params(ms_before=3., ms_after=4., return_scaled=True)
    waveform.run_extract_waveforms(n_jobs=1, chunk_size=30000)
    spost.compute_spike_amplitudes(waveform)
    spost.compute_unit_locations(waveform, method='center_of_mass', radius_um=300)
    spost.compute_isi_histograms(waveform, window_ms=100, bin_ms=1)
    spost.compute_correlograms(waveform, window_ms=100.0, bin_ms=1)
    sorting_metadata['waveform_extraction_accomplished'] = True
    sorting_metadata['single_recording_figure_generation_accomplished'] = False
    sorting_metadata['single_recording_template_layout'] = None
    sorting_metadata['single_recording_unit_position'] = None
    sorting_metadata['single_recording_isi'] = None
    sorting_metadata['single_recording_cross_correlation'] = None
    sorting_metadata['single_recording_template_layout_parameters'] = None
    sorting_metadata['single_recording_unit_position_parameters'] = None
    sorting_metadata['single_recording_isi_parameters'] = None
    sorting_metadata['single_recording_cross_correlation_parameters'] = None
    np.save(sorting_result_folder + 'sorting_metadata.npy', sorting_metadata)

    if local:
        print('copying results to source...')
        shutil.rmtree(source_waveform_folder)
        shutil.copytree(waveform_folder, source_waveform_folder, dirs_exist_ok=True)
        os.remove(source_sorting_metadata_path)
        shutil.copyfile(sorting_metadata_path, source_sorting_metadata_path)
    print('done')
    time_end = default_timer()
    print('Module time usage(s): ', time_end - time_start)


def single_recording_figure_generation(root_folder_input, experimenter, acquisition_system, mouse_id, recording_time,
                                       local=False, template_layout_flag=True, unit_position_flag=True, isi_flag=True,
                                       correlation_flag=True,
                                       template_layout_uv_per_um_scale=1, template_layout_ms_per_um_scale=1 / 20,
                                       template_layout_zoom_scale=1.2, template_layout_scale_bar_ms=1,
                                       template_layout_scale_bar_uv=100, template_layout_scale_bar_margin=40,
                                       unit_position_zoom_scale=1.2, ISI_inter_bar_margin_ratio=0.8, ISI_ncol=5,
                                       correlation_inter_bar_margin_ratio=0.8):
    assert not root_folder_input.startswith(
        'Z:'), 'This is study version code, DO NOT USE REAL DATA ON Z DISK! Change your root to your pseudo dataset.'
    time_start = default_timer()
    print('============================================================================')
    print('data: ', experimenter, acquisition_system, mouse_id, recording_time)
    print('SINGLE RECORDING FIGURE GENERATION MODULE')
    print('Routing...')
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
    source_figure_folder = source_root_folder + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                           '/' + recording_time + '/analysis/figure_result/'
    if not os.path.isdir(source_figure_folder):
        os.makedirs(source_figure_folder)
    if local:
        print('copying data to local...')
        root_folder = os.path.expanduser('~') + '/Documents/Arueruma/'
        sorting_result_folder = root_folder + '/' + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                                '/' + recording_time + '/analysis/sorting_result/'
        sorting_metadata_path = sorting_result_folder + 'sorting_metadata.npy'
        post_curation_folder = sorting_result_folder + 'phy/'
        waveform_folder = sorting_result_folder + 'waveform_from_phy/'
        figure_folder = root_folder + '/' + experimenter + '/' + acquisition_system + '/' + mouse_id + \
                        '/' + recording_time + '/analysis/figure_result/'
        if os.path.exists(root_folder):
            shutil.rmtree(root_folder)
        os.makedirs(post_curation_folder)
        shutil.copytree(source_post_curation_folder, post_curation_folder, dirs_exist_ok=True)
        os.makedirs(waveform_folder)
        shutil.copytree(source_waveform_folder, waveform_folder, dirs_exist_ok=True)
        shutil.copyfile(source_sorting_metadata_path, sorting_metadata_path)
        os.makedirs(figure_folder)
    else:
        root_folder = source_root_folder
        sorting_result_folder = source_sorting_result_folder
        sorting_metadata_path = source_sorting_metadata_path
        post_curation_folder = source_post_curation_folder
        waveform_folder = source_waveform_folder
        figure_folder = source_figure_folder
        if os.path.exists(figure_folder):
            shutil.rmtree(figure_folder)

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
    print('reading sorting and waveform results...')
    sorting = se.read_phy(post_curation_folder, exclude_cluster_groups=["noise", 'mua'])
    waveform = si.load_waveforms(waveform_folder, with_recording=False)
    sorting_metadata = np.load(sorting_metadata_path, allow_pickle=True).item()
    unit_ids = sorting.get_unit_ids()
    unit_number = len(unit_ids)
    color_map = matplotlib.cm.get_cmap('gist_rainbow')
    unit_color_map = color_map(np.array(range(unit_number)) / unit_number)
    if template_layout_flag:
        template_layout_plot(waveform, sorting, sorting_metadata, unit_color_map, figure_folder + 'template_layout/',
                             template_layout_uv_per_um_scale, template_layout_ms_per_um_scale,
                             template_layout_zoom_scale,
                             template_layout_scale_bar_ms, template_layout_scale_bar_uv,
                             template_layout_scale_bar_margin)
    if unit_position_flag:
        unit_position_plot(waveform, sorting_metadata, unit_color_map, figure_folder + 'unit_position/',
                           unit_position_zoom_scale)
    if isi_flag:
        isi_plot(waveform, sorting, unit_color_map, figure_folder + 'isi/', ISI_inter_bar_margin_ratio, ISI_ncol)
    if correlation_flag:
        correlation_plot(waveform, sorting, unit_color_map, figure_folder + 'correlation/',
                         correlation_inter_bar_margin_ratio)

    sorting_metadata['single_recording_figure_generation_accomplished'] = True
    sorting_metadata['single_recording_template_layout'] = template_layout_flag
    sorting_metadata['single_recording_unit_position'] = unit_position_flag
    sorting_metadata['single_recording_isi'] = isi_flag
    sorting_metadata['single_recording_cross_correlation'] = correlation_flag
    sorting_metadata['single_recording_template_layout_parameters'] = {
        "uv_per_um_scale": template_layout_uv_per_um_scale,
        "ms_per_um_scale": template_layout_ms_per_um_scale,
        "zoom_scale": template_layout_zoom_scale,
        "scale_bar_ms": template_layout_scale_bar_ms,
        "scale_bar_uv": template_layout_scale_bar_uv,
        "scale_bar_margin": template_layout_scale_bar_margin}
    sorting_metadata['single_recording_unit_position_parameters'] = {"zoom_scale": unit_position_zoom_scale}
    sorting_metadata['single_recording_isi_parameters'] = {"bar_margin_ratio": ISI_inter_bar_margin_ratio,
                                                           "ncol": ISI_ncol}
    sorting_metadata['single_recording_cross_correlation_parameters'] = {
        "inter_bar_margin_ratio": correlation_inter_bar_margin_ratio}
    np.save(sorting_result_folder + 'sorting_metadata.npy', sorting_metadata)
    os.remove(waveform_folder + 'sorting.json')
    os.rename(waveform_folder + 'sorting_preserve.json', waveform_folder + 'sorting.json')
    if local:
        print('copying results to source...')
        shutil.rmtree(source_figure_folder)
        shutil.copytree(figure_folder, source_figure_folder, dirs_exist_ok=True)
        os.remove(source_sorting_metadata_path)
        shutil.copyfile(sorting_metadata_path, source_sorting_metadata_path)
    print('done')
    time_end = default_timer()
    print('Module time usage(s): ', time_end - time_start)

