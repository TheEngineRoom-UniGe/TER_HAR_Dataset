import os
import numpy as np
from operator import itemgetter

full_data_path = 'full_sequences'
segmented_data_path = 'dataset'
imu_sensors_names = ['/right_wristPose.txt', '/right_backPose.txt', '/left_wristPose.txt', '/left_backPose.txt']

segmented_data_folders = os.listdir(segmented_data_path)
full_data_folders = os.listdir(full_data_path)

print(f'{len(segmented_data_folders)=}')
print(f'{len(full_data_folders)=}')

for subj_dir in sorted(full_data_folders):
    if subj_dir not in segmented_data_folders:
        print(subj_dir)
for subj_dir in sorted(segmented_data_folders):
    if subj_dir not in full_data_folders:
        print(subj_dir)
        continue
    else:
        segmented_seq_list = []
        subj_dir_full_path = os.path.join(segmented_data_path, subj_dir)
        for action_dir in os.listdir(subj_dir_full_path):
            action_name = action_dir.split('_')[0]
            if 'ASSEMBLY' in action_name:
                action_name = 'ASSEMBLY'
            if 'FAILED' in action_name:
                continue
            if 'IDLE' in action_name:
                continue
            if 'HANDOVER' in action_name or 'DELIVERY' in action_name or 'PICKUP' in action_name:
                action_name = 'PICKUP'

            action_dir_full_path = os.path.join(subj_dir_full_path, action_dir)
            # print('\n')
            for action_id in os.listdir(action_dir_full_path):
                action_id_dir_full_path = os.path.join(action_dir_full_path, action_id)
                action_name_id = action_name + '_' + action_id
                # print(subj_dir, action_name_id)
                imu_files = os.listdir(action_id_dir_full_path)

                if len(imu_files) != 4:
                    print('The number of files is not what we expected, skipping...')
                    continue

                # print(action_id_dir_full_path)
                imu_sensors_data = []
    
                for i in range(4):

                    # print(imu_sensors_names[i])
                    imu_sensors_data.append(np.loadtxt(open(action_id_dir_full_path + imu_sensors_names[i], "rb"), delimiter=",", skiprows=1,usecols=(0,5,6,7,8,9,10)))
                

                '''FIND COMMON START AND END'''
                new_min_stamp = max(imu_sensors_data[0][0,0], imu_sensors_data[1][0,0], imu_sensors_data[2][0,0], imu_sensors_data[3][0,0])
                new_max_stamp = min(imu_sensors_data[0][-1,0], imu_sensors_data[1][-1,0], imu_sensors_data[2][-1,0], imu_sensors_data[3][-1,0])

                '''Print info on the list'''
                # print(f'{new_min_stamp=}')
                # print(f'{new_max_stamp=}')
                # print('\t', action_name_id)
                # print(f'\tlength: {(new_max_stamp-new_min_stamp)/1e9}')
                seq_info = [action_name, new_min_stamp, new_max_stamp]
                # print('\t', seq_info)
                segmented_seq_list.append(seq_info)

    '''Debug list lengths'''             
    # print(subj_dir)
    # print(len(segmented_seq_list))

    '''Order the list by the start time of the actions'''
    segmented_seq_list = sorted(segmented_seq_list, key=itemgetter(1))
 
    print(subj_dir)
    '''Save the list to a file'''
    np.savetxt(os.path.join(full_data_path, subj_dir, 'actions_start_end.txt'), segmented_seq_list, fmt='%s', delimiter=',')
