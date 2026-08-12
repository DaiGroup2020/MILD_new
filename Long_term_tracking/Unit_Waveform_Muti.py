import warnings
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore')
import spikeinterface as si
import spikeinterface.extractors as se
import json
import re
import ast
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 后台静默出图
import matplotlib.pyplot as plt
import os
import pandas as pd
from timeit import default_timer


def trace_spikes(data_root, output_folder, experimenter, acquisition_system, mouse_id):
    time_start = default_timer()
    print('============================================================================')
    print(f'Animal Target: {experimenter} | Mouse {mouse_id} -> 开始紧凑版波形叠画')

    # 规范化路径
    data_root = data_root if data_root.endswith('/') else data_root + '/'
    os.makedirs(output_folder, exist_ok=True)

    # 🎯 【更新位置】：精准指向你给出的绝对 CSV 文件路径
    cell_bundle_path = f"H:/WXY/MILD-DATA/blackrock/5125/{experimenter}_{acquisition_system}_{mouse_id}_cell_bundle_OK_plot.csv"
    assert os.path.isfile(cell_bundle_path), f"{cell_bundle_path} 不存在，请检查 CSV 文件路径是否正确。"

    cell_bundle_df = pd.read_csv(cell_bundle_path, index_col=0)
    recording_time_list = cell_bundle_df.index.tolist()
    unit_names = cell_bundle_df.columns

    # 🎨 渐变蓝色配置（按录制天数渐变）
    #cmap = plt.get_cmap('Blues')
    #base_color = (252 / 255, 112 / 255, 135 / 255)
    base_color = (23 / 255, 104 / 255, 170 / 255)
    cmap = LinearSegmentedColormap.from_list("custom_color",[(1, 1, 1), base_color])
    color_factors = np.linspace(0.9, 0.2, len(recording_time_list))
    time_color_map = {time_str: cmap(factor) for time_str, factor in zip(recording_time_list, color_factors)}

    for unit_name in unit_names:
        print(f"🎬 正在处理 -> {unit_name}")

        fig_stack = plt.figure(figsize=(6, 9))
        ax_stack = plt.gca()
        has_valid_waveform = False

        for recording_time in recording_time_list:
            unit_id = cell_bundle_df.loc[recording_time, unit_name]
            if pd.isna(unit_id) or str(unit_id).strip() == "" or str(unit_id).strip().lower() == "nan":
                continue

            # 路径对齐：H:/data/XYQ/intan/16/[recording_time]/analysis/...
            source_sorting_result_folder = f"{data_root}{experimenter}/{acquisition_system}/{mouse_id}/{recording_time}/analysis/sorting_result/"
            post_curation_folder = os.path.join(source_sorting_result_folder, 'phy/')
            waveform_folder = os.path.join(source_sorting_result_folder, 'waveform_from_phy/')

            if not os.path.isdir(waveform_folder):
                continue

            # 自动修复 Phy 路径环境中的 json 文件
            try:
                json_path = os.path.join(waveform_folder, 'sorting.json')
                preserve_path = os.path.join(waveform_folder, 'sorting_preserve.json')
                if os.path.exists(preserve_path):
                    os.remove(preserve_path)
                os.rename(json_path, preserve_path)

                with open(preserve_path, 'r') as json_preserve:
                    json_data = json.load(json_preserve)
                    original_address = r"(([A-Z]:)|(\\\\\\\\\d*?\.\d*?\.\d*?\.\d*?))\\\\.*?\\\\analysis\\\\sorting_result\\\\phy"
                    new_address = post_curation_folder.replace("\\", "\\\\\\\\").replace("/", "\\\\\\\\")
                    if new_address.endswith('\\\\\\\\'):
                        new_address = new_address[:-4]
                    json_string_new = re.sub(original_address, new_address, str(json_data))
                    json_data_new = ast.literal_eval(json_string_new)
                    with open(json_path, 'w') as json_file_new:
                        json.dump(json_data_new, json_file_new, indent=4)

                sorting = se.read_phy(post_curation_folder, exclude_cluster_groups=["noise", 'mua'])
                waveform = si.load_waveforms(waveform_folder)
            except Exception:
                continue

            unit_ids = sorting.get_unit_ids()
            unit_id_int = int(float(str(unit_id)))
            if unit_id_int not in unit_ids:
                continue

            # 提取中位数波形并叠加画线
            current_time_color = time_color_map[recording_time]
            sampling_frequency = sorting.get_sampling_frequency()
            preserved_channel_positions = waveform.get_channel_locations().tolist()
            templates = waveform.get_template(unit_id=unit_id_int, mode='median')

            for preserved_channel_ind, preserved_channel_position in enumerate(preserved_channel_positions):
                template_per_channel = templates[:, preserved_channel_ind]
                x = (np.array(range(len(template_per_channel))) - int(
                    len(template_per_channel) / 2)) / sampling_frequency * 1000 / (1 / 20) + preserved_channel_position[
                        0]
                y = template_per_channel / 1 + preserved_channel_position[1]
                ax_stack.plot(x, y, color=current_time_color, linewidth=1.5, alpha=0.85, zorder=2)

            has_valid_waveform = True
            print(f"    ✅ 已叠加时间点: {recording_time}")

        # 所有图层渲染完毕，统一裁剪并直接存入目标文件夹
        if has_valid_waveform:
            finalize_and_save(ax_stack, fig_stack, unit_name, output_folder)
        else:
            plt.close(fig_stack)

    print(f'\n✨ 纯净版长时程叠画全部完成！总耗时: {default_timer() - time_start:.2f} 秒。')


def finalize_and_save(ax, fig, unit_name, output_folder, zoom_scale=1.2, scale_bar_ms=1, scale_bar_uv=100,
                      scale_bar_margin=40):
    xlim = np.array(ax.get_xlim())
    xlim = zoom_scale * xlim + (1 - zoom_scale) * np.average(xlim)
    ylim = np.array(ax.get_ylim())
    ylim = zoom_scale * ylim + (1 - zoom_scale) * np.average(ylim)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # 绘制直角比例尺 (ms_per_um_scale = 1/20)
    ax.plot([xlim[0] + scale_bar_margin, xlim[0] + scale_bar_margin + scale_bar_ms / (1 / 20)],
            [ylim[0] + scale_bar_margin, ylim[0] + scale_bar_margin], zorder=4, color='black', linewidth=2)
    ax.text(xlim[0] + scale_bar_margin + 0.5 * (scale_bar_ms / (1 / 20)), ylim[0] + 0.5 * scale_bar_margin,
            f'{scale_bar_ms}ms', fontsize=7, ha='center', va='center', color='black', zorder=4, weight='bold')

    ax.plot([xlim[0] + scale_bar_margin, xlim[0] + scale_bar_margin],
            [ylim[0] + scale_bar_margin, ylim[0] + scale_bar_margin + scale_bar_uv], zorder=5, color='black',
            linewidth=2)
    ax.text(xlim[0] + 0.5 * scale_bar_margin, ylim[0] + scale_bar_margin + 0.5 * scale_bar_uv,
            f'{scale_bar_uv}uV', fontsize=7, ha='center', va='center', color='black', rotation=90, zorder=4,
            weight='bold')

    ax.tick_params(axis='both', which='both', bottom=True, top=True, right=True, labelbottom=True, labeltop=True,
                   labelright=True)

    # 直接保存到大文件夹下，不建子文件夹
    plt.savefig(os.path.join(output_folder, f'{unit_name}_clean_stacked.svg'), format='svg', dpi=300, transparent=True)
    plt.savefig(os.path.join(output_folder, f'{unit_name}_clean_stacked.png'), format='png', dpi=300, transparent=True)
    plt.close(fig)


# ============================================================================
# 执行入口
# ============================================================================
DATA_ROOT = 'Z:/'
OUTPUT_FOLDER = 'H:/WXY/blackrock/MILD-DATA/5125/Waveform_Multi'  # 所有的图片平铺在这里

experimenter = 'WXY'
acquisition_system = 'blackrock'
mouse_id = '5125'

trace_spikes(
    data_root=DATA_ROOT,
    output_folder=OUTPUT_FOLDER,
    experimenter=experimenter,
    acquisition_system=acquisition_system,
    mouse_id=mouse_id
)