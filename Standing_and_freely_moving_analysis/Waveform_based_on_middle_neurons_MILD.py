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
    print(csv_path)

    unit_summary=pd.read_csv(csv_path)

    middle_neuron_summary=unit_summary[unit_summary['middle_neuron']]

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

                # middle waveform
                middle_folder=sorting_result_folder+'middle_waveform/'

                if not os.path.isdir(middle_folder):
                    print('Missing:',middle_folder)
                    continue

                middle_waveform=si.load_waveforms(
                    middle_folder,
                    with_recording=False
                )

                middle_extremum_channels_ids=spost.get_template_extremum_channel(middle_waveform)
                middle_chan_ids=np.array(middle_extremum_channels_ids[unit_id])

                if middle_chan_ids.ndim==0:
                    middle_chan_ids=[middle_chan_ids]

                middle_chan_ind=middle_waveform.channel_ids_to_indices(middle_chan_ids)

                middle_wf=np.squeeze(
                    middle_waveform.get_waveforms(unit_id=unit_id)[:,:,middle_chan_ind]
                ).T

                middle_template=middle_waveform.get_template(unit_id=unit_id)[:,middle_chan_ind]


                # after waveform
                after_folder=sorting_result_folder+'after_waveform/'

                if not os.path.isdir(after_folder):
                    print('Missing:',after_folder)
                    continue

                after_waveform=si.load_waveforms(
                    after_folder,
                    with_recording=False
                )

                if unit_id not in after_waveform.unit_ids:
                    print('Missing unit in after:',unit_id)
                    continue

                after_extremum_channels_ids=spost.get_template_extremum_channel(after_waveform)
                after_chan_ids=np.array(after_extremum_channels_ids[unit_id])

                if after_chan_ids.ndim==0:
                    after_chan_ids=[after_chan_ids]

                after_chan_ind=after_waveform.channel_ids_to_indices(after_chan_ids)

                after_wf=np.squeeze(
                    after_waveform.get_waveforms(unit_id=unit_id)[:,:,after_chan_ind]
                ).T

                after_template=after_waveform.get_template(unit_id=unit_id)[:,after_chan_ind]


                # plot
                fig,axes=plt.subplots(1,2,figsize=(8,4))

                axes[0].plot(middle_wf,c='r')
                axes[0].plot(middle_template,c='k')
                axes[0].set_title('middle')

                axes[1].plot(after_wf,c='r')
                axes[1].plot(after_template,c='k')
                axes[1].set_title('after')


                result_folder='figures_MILD_Three_0.1/middle_neuron_after_'+noise_or_neuron_name+'/'+group+'/'

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