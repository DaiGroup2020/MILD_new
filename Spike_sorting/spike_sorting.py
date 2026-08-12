
from module_bank import *

if __name__ == '__main__':
    ss.Kilosort3Sorter.set_kilosort3_path('D:\Github_package\spike_sorters\Kilosort3')
    ss.Kilosort2Sorter.set_kilosort2_path('D:\Github_package\spike_sorters\Kilosort2')
    ss.KilosortSorter.set_kilosort_path('D:\Github_package\spike_sorters\Kilosort')

    root_folder = 'E:/data'  # This should be your own pseudo folder
    experimenter = 'WXY'
    acquisition_system = 'blackrock'
    mouse_id = '130'
    recording_time = '20231228-1346'

    impedance_threshold = 3   # 3
    sorter = "kilosort2"   # "kilosort", "kilosort2", "kilosort3", "spykingcircus"

    sort_spikes(root_folder, experimenter, acquisition_system, mouse_id, recording_time, impedance_threshold, sorter, local=True)
    post_curate_spikes(root_folder, experimenter, acquisition_system, mouse_id, recording_time, local=False)
    # read_sorting_metadata(root_folder, experimenter, acquisition_system, mouse_id, recording_time, display=True)
    # waveform_extraction_after_phy(root_folder, experimenter, acquisition_system, mouse_id, recording_time, local=False)
    # single_recording_figure_generation(root_folder, experimenter, acquisition_system, mouse_id, recording_time, local=False, template_layout_flag=True, unit_position_flag=True, isi_flag=True, correlation_flag=True)






















