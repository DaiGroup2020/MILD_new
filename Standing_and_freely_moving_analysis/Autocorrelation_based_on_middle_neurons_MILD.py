import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import spikeinterface as si
import spikeinterface.postprocessing as spost

if __name__=="__main__":

    root_folder='H:/data/'
    experimenter='WXY'
    acquisition_system='blackrock'

    csv_path='H:/Moving&Standing/results/unit_summary_MILD_Three_0.1.csv'
    unit_summary=pd.read_csv(csv_path)

    middle_neuron_summary=unit_summary[unit_summary['middle_neuron']]

    inter_bar_margin_ratio=0.8
    group_list=['MILD']
    after_neuron_check_list=[True,False]

    for group in group_list:
        for after_neuron_check in after_neuron_check_list:

            noise_or_neuron_name='neuron' if after_neuron_check else 'noise'
            print('plotting '+group+' '+noise_or_neuron_name)

            unit_summary_selected=middle_neuron_summary[
                (middle_neuron_summary['group']==group)&
                (middle_neuron_summary['after_neuron']==after_neuron_check)
            ].dropna(subset=['mouse','recording','unit'])

            for line in range(len(unit_summary_selected)):

                mouse_id=unit_summary_selected.iloc[line]['mouse']
                recording_time=unit_summary_selected.iloc[line]['recording']
                unit_id=int(unit_summary_selected.iloc[line]['unit'])

                sorting_result_folder=root_folder+experimenter+'/'+acquisition_system+'/'+str(mouse_id)+'/'+recording_time+'/analysis/sorting_result/'
                print('plotting mouse '+str(mouse_id)+' unit '+str(unit_id))

                # middle autocorrelation
                middle_folder=sorting_result_folder+'middle_waveform/'

                if not os.path.isdir(middle_folder):
                    print('Missing:',middle_folder)
                    continue

                middle_waveform=si.load_waveforms(
                    middle_folder,
                    with_recording=False
                )

                if unit_id not in middle_waveform.unit_ids:
                    continue

                middle_unit_ind=middle_waveform.unit_ids.tolist().index(unit_id)

                ccg_middle,edge_middle=spost.compute_correlograms(
                    middle_waveform,
                    load_if_exists=True
                )

                x_middle=[(a+edge_middle[i+1])/2 for i,a in enumerate(edge_middle[:-1])]

                hist_middle=ccg_middle[
                    middle_unit_ind,
                    middle_unit_ind,
                    :
                ]

                if np.max(hist_middle)==0:
                    continue

                hist_middle=hist_middle/np.max(hist_middle)

                plot_data_middle=pd.Series(hist_middle,x_middle)

                width_middle=abs(
                    plot_data_middle.index[1]-
                    plot_data_middle.index[0]
                )

                # after autocorrelation
                after_folder=sorting_result_folder+'after_waveform/'

                if not os.path.isdir(after_folder):
                    print('Missing:',after_folder)
                    continue

                after_waveform=si.load_waveforms(
                    after_folder,
                    with_recording=False
                )

                if unit_id not in after_waveform.unit_ids:
                    continue

                after_unit_ind=after_waveform.unit_ids.tolist().index(unit_id)

                ccg_after,edge_after=spost.compute_correlograms(
                    after_waveform,
                    load_if_exists=True
                )

                x_after=[(a+edge_after[i+1])/2 for i,a in enumerate(edge_after[:-1])]

                hist_after=ccg_after[
                    after_unit_ind,
                    after_unit_ind,
                    :
                ]

                if np.max(hist_after)==0:
                    continue

                hist_after=hist_after/np.max(hist_after)

                plot_data_after=pd.Series(hist_after,x_after)

                width_after=abs(
                    plot_data_after.index[1]-
                    plot_data_after.index[0]
                )

                fig,axes=plt.subplots(1,2,figsize=(8,4))

                axes[0].bar(
                    plot_data_middle.index,
                    plot_data_middle.values,
                    width=inter_bar_margin_ratio*width_middle,
                    align='center'
                )
                axes[0].set_title('middle')

                axes[1].bar(
                    plot_data_after.index,
                    plot_data_after.values,
                    width=inter_bar_margin_ratio*width_after,
                    align='center'
                )
                axes[1].set_title('after')

                result_folder='figures_MILD_Three_0.1/middle_and_after_phase_autocorrelation_on_middle_neurons/'+group+'/'+noise_or_neuron_name+'/'

                os.makedirs(
                    result_folder,
                    exist_ok=True
                )

                plt.savefig(
                    result_folder+
                    'mouse_'+str(mouse_id)+
                    '_unit_'+str(unit_id)+'.svg',
                    bbox_inches='tight'
                )

                plt.close()