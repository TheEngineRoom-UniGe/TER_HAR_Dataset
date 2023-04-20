import os
import numpy as np
from scipy.signal import butter, lfilter
from matplotlib import pyplot as plt
from scipy import interpolate
import plotly.express as px
from itertools import compress
import torch
from torch.nn.utils.rnn import pad_sequence

IMU_FREQ = 25 #HZ
TRAIN = True
USE_IDLE = True

# Treshold to remove outliers with too many samples
# threshold = np.mean(len_list)+1*np.std(len_list)        #2*np.std(len_list)
threshold_high = 500#3000
threshold_low = 200#150

def load_set_indexes(path):
    with open(path, 'r') as f:
        content = f.readlines()
        new_content = []
        for line in content:
            new_content.append(line.split('\n')[0])
    return new_content

def remap_categories(labels, dataset):
    action_dict = {}
    for sequence, label in zip (dataset, labels):
        # print(label)
        base_action = label.split('_')[0]
        hand = label.split('_')[1]

        # base_action = label

        if 'FAILED' in label:
            continue

        if not USE_IDLE:
            if 'IDLE' in base_action:
                continue
        
        '''UNCOMMENT THIS TO JOIN ASSEMBLY 1 AND 2'''
        if 'ASSEMBLY' in base_action:
            base_action = 'ASSEMBLY'
        
        # '''UNCOMMENT THIS TO JOIN HANDOVER DELIVERY AND PICKUP'''
        # if 'HANDOVER' in base_action or 'DELIVERY' in base_action or 'PICKUP' in base_action:
        #     base_action = 'PICKUP'

        '''UNCOMMENT THIS TO JOIN HANDOVER DELIVERY AND PICKUP'''
        if 'PICKUP' in base_action or 'DELIVERY' in base_action:
            base_action = 'HANDOVER'

        # if 'BOLT' in base_action:
        #     base_action = 'SCREW'+hand

        # if 'ASSEMBLY1' in base_action:
        #     action_dict['ASSEMBLY1'] = [sequence[:,:3]]
        # if 'ASSEMBLY2' in base_action:
        #     if
        #     action_dict['ASSEMBLY2'] = 1

        add_seq = []

        # if 'ASSEMBLY' in base_action:
        #     continue

        # if 'SCREW' in base_action:
        #     if 'RIGHT' in hand:
        #         add_seq = [sequence[:,12:24]]
        #     if 'LEFT' in hand:
        #         add_seq = [sequence[:,0:12]]
        #     if 'BIMANUAL' in hand:
        #         add_seq = [sequence[:,0:12], sequence[:,12:24]]

        # if 'BOLT' in base_action:
        #     if 'RIGHT' in hand:
        #         add_seq = [sequence[:,12:24]]
        #     if 'LEFT' in hand:
        #         add_seq = [sequence[:,0:12]]

        # if 'DELIVERY' in base_action or 'PICKUP' in base_action or 'HANDOVER' in base_action:
        #     if 'RIGHT' in hand:
        #         add_seq = [sequence[:,12:24]]
        #     if 'LEFT' in hand:
        #         add_seq = [sequence[:,0:12]]
        #     if 'BIMANUAL' in hand:
        #         add_seq = [sequence[:,0:12], sequence[:,12:24]]

        # if 'IDLE' in base_action:
        #     add_seq = [sequence[:,:12], sequence[:,12:24]]

        add_seq = [sequence]

        if base_action not in action_dict.keys():
            action_dict[base_action] = add_seq
        else:
            for seq in add_seq:
                action_dict[base_action].append(seq)

    action_list_np = []
    labels_list = []

    for key in action_dict.keys():
        print(key, '->', len(action_dict[key]), 'sequences')
        for i in range(len(action_dict[key])):
            labels_list.append(key)
            action_list_np.append(np.asarray(action_dict[key][i]))

    # print(f'{len_list=}')
    print(f'{len(labels_list)=}')
    print(f'{len(action_list_np)=}')  

    # exit()
    return labels_list, action_list_np
    

def mergeIntoGeneralActions(labels, actions):
    action_dict = {}
    for seq, action_name in zip(actions, labels):
        # print(seq.shape, action_name)
        general_action_name = action_name.split('_')[0]
        # if "ASSEMBLY" in general_action_name:
        #     general_action_name = "ASSEMBLY"
        if "HANDOVER" in general_action_name:
            general_action_name = "DELIVERY"
        if general_action_name not in action_dict.keys():
            action_dict[general_action_name] = [seq]
        else:
            action_dict[general_action_name].append(seq)

    action_list_np = []
    labels_list = []

    for key in action_dict.keys():
        print(key, '->', len(action_dict[key]), 'sequences')
        for i in range(len(action_dict[key])):
            labels_list.append(key)
            action_list_np.append(np.asarray(action_dict[key][i]))

    # print(f'{len_list=}')
    print(f'{len(labels_list)=}')
    print(f'{len(action_list_np)=}')  

    # exit()
    return labels_list, action_list_np


def zeroPadding(dataset, maxl):
    n_features = dataset[0].shape[-1]
    print(f'{n_features=}')
    if maxl is None:
        maxl = 0
        for sequence in dataset:
            if sequence.shape[0] >= maxl:
                maxl = sequence.shape[0]

    padded_dataset = np.zeros((len(dataset), maxl, n_features), dtype='float32')
    print(f'{padded_dataset.shape=}')
    for i, seq in enumerate(dataset):
        padded_dataset[i, -len(seq):] = seq

    print(maxl)
    # dataset = keras.preprocessing.sequence.pad_sequences( dataset , maxlen=maxl, value=0.0, dtype='float64', padding='pre')
    return padded_dataset

def myReshape(a):
    outer_list = []
    for outer_idx in range(a.shape[0]):
        outer_list.append(a[outer_idx,:]) 
    return np.column_stack(outer_list)

def resample(sequence, tstamp, trgtstamp):
    tstamp = tstamp.squeeze()
    trgtstamp = trgtstamp.squeeze()
    sequence = np.array(sequence)
    newAxes = []
    tries = -1

    # FIX SEQUENCE LENGTH IF THE NEW STAMPS ARE OUTSIDE THE INTERPOLATION INTERVAL
    while tstamp[-1] < trgtstamp[-1]:
        trgtstamp = trgtstamp[:-1]
        tries -=1
    for ax in range(sequence.shape[1]):
        seq = sequence[:,ax]
        f = interpolate.interp1d(tstamp, seq)
        newAxes.append(f(trgtstamp))
    return np.column_stack(newAxes)



def dataResampling(dataset):
    newdata = []
    len_list = []

    for i in range(len(dataset)):
        len_list.append(len(dataset[i]))

    min_len = min(len_list)
    min_stamps = np.array([st for st in range(min_len)])

    for i in range(len(dataset)):
        current_stamps =  np.array([st for st in range(len(dataset[i]))])
        dataset[i] = resample(dataset[i], current_stamps, min_stamps)
    return dataset

def normalize(dataset):
    acceleration_idxs = [0,1,2,6,7,8,12,13,14,18,19,20]
    gyroscope_idxs = [3,4,5,9,10,11,15,16,17,21,22,23]

    # 1g equals 8192. The full range is 2g
    dataset[:,:,acceleration_idxs] = dataset[:,:,acceleration_idxs] / 16384.0
    dataset[:,:,gyroscope_idxs] = dataset[:,:,gyroscope_idxs] / 2000.0

    return dataset


def LowPass(sequence, freq):
    '''UNCOMMENT TO PLOT THE ORIGINAL VS THE FILTERED VERSION'''
    # fig = plt.figure()
    # plt.plot(sequence)
    fs = 120
    w = freq / (fs / 2) # Normalize the frequency
    b, a = butter(5, w, btype='low')
    y = lfilter(b,a,sequence)
    # plt.plot(y)
    # plt.show()
    return y

def frequency_analysis(dataset):
    newdata = []

    '''FOR EACH SENSOR FOR EACH FEATURE FILTER THE DATA AND RETURN THE NEW SEQUENCES'''
    for seq in dataset:
        temp_feature_list = []
        for feature in range(seq.shape[1]):
            temp_feature_list.append(LowPass(seq[:,feature], IMU_FREQ))
        newdata.append(np.column_stack(tuple(temp_feature_list)))

    return newdata


def sequence_bondaries(seq, min, max):
    return seq[np.where((seq[:,0] > min) & (seq[:,0] < max))]

def load_synchronized_sensors(abs_path):
    '''LOAD THE DATA IN NUMPY'''
    print(abs_path)
    imu_sensors_data = []
    imu_sensors_names = ['/right_wristPose.txt', '/right_backPose.txt', '/left_wristPose.txt', '/left_backPose.txt']

    for i in range(4):
        # print(imu_sensors_names[i])
        imu_sensors_data.append(np.loadtxt(open(abs_path + imu_sensors_names[i], "rb"), delimiter=",", skiprows=1,usecols=(0,5,6,7,8,9,10)))
       

    '''FIND COMMON START AND END'''
    new_min_stamp = max(imu_sensors_data[0][0,0], imu_sensors_data[1][0,0], imu_sensors_data[2][0,0], imu_sensors_data[3][0,0])
    new_max_stamp = min(imu_sensors_data[0][-1,0], imu_sensors_data[1][-1,0], imu_sensors_data[2][-1,0], imu_sensors_data[3][-1,0])

    '''SYNCHRONIZE THE SEQUENCES OF THE DIFFERENT SENSORS'''
    for i in range(4):
            imu_sensors_data[i] = sequence_bondaries(imu_sensors_data[i], new_min_stamp, new_max_stamp)
            imu_sensors_data[i] = np.delete(imu_sensors_data[i], 0, 1)

    '''FREQUENCY FILTERING '''
    # print(imu_sensors_data[0].shape)
    imu_sensors_data = frequency_analysis(imu_sensors_data)

    # print(imu_sensors_data[0].shape)
    '''RESAMPLING TO MATCH THE SHORTEST SENSOR SEQUENCE'''
    imu_sensors_data = dataResampling(imu_sensors_data)
    # print(imu_sensors_data[0].shape)
    imu_sensors_data = np.array(imu_sensors_data)
    # print(imu_sensors_data[0].shape)
    '''AT THIS POINT imu_sensors_data CONTAINS THE SEQUENCES OF THE FOUR SENSORS'''
    # print(imu_sensors_data.shape)
    return imu_sensors_data


def window_augmenter(sequences, labels, max_len, min_len):
    for seq, lab in zip(sequences, labels):
        if len(seq) > max_len:
            remaining_len = seq.shape[0]
            while remaining_len >= min_len:

                if remaining_len >= max_len:
                    sequences.append(seq[:max_len])
                    labels.append(lab)
                    seq = seq[max_len:]
                    remaining_len = seq.shape[0]

                else:
                    sequences.append(seq)
                    labels.append(lab)
                    remaining_len = 0

    len_list = [len(seq) for seq in sequences]

    return sequences, labels, np.asarray(len_list)


def main():

    dataset_abs_path = f'./dataset'
    data_dir_list = sorted(os.listdir(dataset_abs_path))
    print(f'{len(data_dir_list)=}')

    test_idxs = load_set_indexes('./test_set.txt')
    # train_idxs = load_set_indexes('./training_set.txt')

    print(f'{len(test_idxs)=}')
    # print(f'{len(train_idxs)=}')

    action_dict = {}
    maxp = 0
    c = 0
    for participant in data_dir_list:
        print(participant)
        ''' COMMENT THIS TO USE THE FULL DATASET '''
        if TRAIN:
            if participant in test_idxs:
                continue
        else:
            if participant not in test_idxs:
                continue
        participant_dir = os.path.join(dataset_abs_path, participant)
        # print(participant_dir)

        list_participant_dir = os.listdir(participant_dir)
        # print(list_participant_dir)
        # if "49" not in participant:
        #     continue
        for action in list_participant_dir:

            action_dir = os.path.join(participant_dir, action)

            list_action_dir = os.listdir(action_dir)
            # print(list_action_dir)
            # print(action)
            # print(len(list_action_dir))
            trial_data_list = []
            for trial_idx in sorted(list_action_dir):

                trial_dir = os.path.join(action_dir, trial_idx)
                list_trial_dir = os.listdir(trial_dir)
                # print(trial_idx)
                if len(list_trial_dir) != 4:
                    # print('Some data file is missing, Skipping this trial!...')
                    continue

                action_name = os.path.split(action_dir)[-1]

                ''' UNCOMMENT THIS TO DETECT POSSIBLE ERRORS IN THE ANNOTATIONS '''
                # kk = myReshape(load_synchronized_sensors(trial_dir))
                # print(kk.shape[0])
                # if kk.shape[0] > 5000:
                #     if kk.shape[0] not in [58905, 5316, 5850, 5628]:
                #         print(participant, action_name, trial_idx)
                #         exit()

                trial_data_list.append(myReshape(load_synchronized_sensors(trial_dir)))

            # print(len(trial_data_list))
            # print(action_dict.get(action))
            old_val = action_dict.get(action)
            # print(action)
            # print(old_val)
            if old_val is None:
                action_dict[action] = [trial_data_list]
            else:
                new_val = old_val.append(trial_data_list)
                if new_val is not None:
                    action_dict[action] = new_val
        maxp +=1
        # if maxp==5:
        #     break
    print('\n\nDebugging the action dictionary!\n')
    for key in action_dict.keys():
        print(key, '->', len(action_dict[key]), 'trials')


    '''TRANSFORMS THE DICTIONARY IN A LIST OF NPARRAYS'''
    action_list = [v for v in action_dict.values()]
    action_names = [k for k in action_dict.keys()]
    print("\naction list length:", len(action_list), '\n')
    print("\naction names list length:", len(action_names), '\n')
    action_list_np = []

    len_list = []
    labels_list = []

    for action, action_name in zip(action_list, action_names):
        # if len(action) < 12:
        #     print(">> ", action_name, " -> ", len(action), "SKIPPED! <<")
        #     print('---------------------------------------------')
        #     continue
        if action_name == 'IDLE':
            print('IDLEEEEEE')
            continue
        print(action_name, " -> ", len(action))
        for seq_list in action:
            # print('\t', len(seq_list))
            for seq in seq_list:
                # print('\t\t',  seq.shape)
                len_list.append(seq.shape[0])
                action_list_np.append(seq)
                labels_list.append(action_name)
        print('---------------------------------------------')

    # action_list_np = np.asarray(action_list_np)


    
    len_list = np.asarray(len_list)
    # labels_list = np.asarray(labels_list)
    # action_list_np = np.asarray(action_list_np)

    print(f'{threshold_high=}')
    print(f'{threshold_low=}')

    print('Number of sequences: ', len(action_list_np))
    print('Number of labels: ', len(labels_list))
    action_list_np, labels_list, len_list = window_augmenter(action_list_np, labels_list, threshold_high, threshold_low)
    print('Number of sequences: ', len(action_list_np))
    print('Number of labels: ', len(labels_list))

    mask_np_high = len_list <= threshold_high
    mask_np_low = len_list >= threshold_low
    mask_np = np.logical_and(mask_np_high, mask_np_low)
    print(f'{np.count_nonzero(mask_np)=}')

    # mask_np1 = len_list < 100.0
    # print(f'{np.count_nonzero(mask_np1)=}')
    # exit()
    # Plotting the histogram of the lengths of the sequences
    font1 = font={
                    'family' : 'Times New Roman',
                    'size' : 18
                    }
    fig = px.histogram(len_list, nbins=400)
    fig.add_vline(x=np.mean(len_list), annotation_text='  <b>Mean', annotation_position='right', annotation=dict(font=font1), line_width=3, line_dash="dash", line_color="black", col=1, row=1)
    fig.add_vline(x=np.median(len_list), annotation_text='<b>Median  ', annotation_position='left', annotation=dict(font=font1), line_width=3, line_dash="dash", line_color="red", col=1, row=1)
    fig.add_vline(x=np.mean(len_list)+2*np.std(len_list), annotation_text='<b>mean+2*std  ', annotation_position='left', annotation=dict(font=font1), line_width=3, line_dash="dash", line_color="green", col=1, row=1)
    fig.add_vline(x=np.mean(len_list)-2*np.std(len_list), annotation_text='<b>mean-2*std  ', annotation_position='left', annotation=dict(font=font1), line_width=3, line_dash="dash", line_color="green", col=1, row=1)
    # fig.show()


    # Apply the mask to the action list to remove the outliers
    print(len(action_list_np))
    masked_list = list(compress(action_list_np, mask_np))
    labels_list = list(compress(labels_list, mask_np))
    print(len(masked_list))
    print(len(masked_list)/len(action_list_np))
    print(len(action_list_np))

    masked_len_list = list(compress(len_list, mask_np))
    fig = px.histogram(masked_len_list, nbins=200)
    # fig.show()
    # exit()

    # This gets the action list and removes the '_left' or '_right' from the action name merging into general actions
    # labels_list, action_list_np = mergeIntoGeneralActions(labels_list, masked_list)
    labels_list, action_list_np = remap_categories(labels_list, masked_list)

    print(np.unique(labels_list), len(labels_list), len(action_list_np))
    unique_labels = np.unique(labels_list)
    len_dict = {lab: val for lab, val in zip(unique_labels, np.zeros((unique_labels.shape[0])))}
    lab_repetitions = {lab: val for lab, val in zip(unique_labels, np.zeros((unique_labels.shape[0])))}
    len_x_action = {lab: [] for lab in unique_labels}
    print(len_dict)
    for lab, seq in zip(labels_list, action_list_np):
        len_x_action[lab].append(seq.shape[0])
        len_dict[lab] += seq.shape[0]
        lab_repetitions[lab] +=1

    print(lab_repetitions)
    print(len_dict)

    avg_seq_len = {lab:len_dict[lab]/lab_repetitions[lab] for lab in unique_labels}
    print(avg_seq_len)
    print(len_x_action)
    plt.boxplot(len_x_action.values(), labels=len_x_action.keys())
    plt.xticks(rotation=45)
    plt.show()

    fig, axs = plt.subplots(len(len_x_action.keys()))
    for i in range(len(len_x_action.keys())):

        axs[i].hist(len_x_action[unique_labels[i]], bins=20)
        axs[i].set_title(unique_labels[i])
        axs[i].set_xlabel('Length')
        axs[i].set_ylabel('N_samples')
        axs[i].set_xlim(threshold_low, threshold_high)

    plt.show()
    print('After removing outliers, with threshold_low = {} and threshold_high = {}'.format(threshold_low, threshold_high))
    print('Number of sequences: ', len(action_list_np))
    print('Number of labels: ', len(labels_list))

    
    exit()
    tensor_list = [torch.tensor(arr) for arr in masked_list]
    tensor = pad_sequence(tensor_list, batch_first=True)

    '''APPLY ZERO PADDING TO HAVE SEQUENCES OF THE SAME SIZES'''
    masked_list_np = zeroPadding(action_list_np, threshold_high)
    labels_list = np.asarray(labels_list).reshape(-1, 1)

    print(f'{masked_list_np.shape=}')
    print(f'{labels_list.shape=}')

    # masked_list_np = normalize(masked_list_np)

    # Save stuff

    filename = "data_shape({}_{}_{}).npy".format(*masked_list_np.shape)

    if not USE_IDLE:
        filename = 'no_idle_' + filename
    if TRAIN:
        filename = 'train_' + filename
    else:
        filename = 'test_' + filename


    np.save(filename, masked_list_np)

    # filename = "data_shape({}_{}_{}).pt".format(*masked_list_np.shape)
    # torch.save(tensor, filename)

    filename = "labels_shape({}_{}).npy".format(*labels_list.shape)

    if not USE_IDLE:
        filename = 'no_idle_' + filename
    if TRAIN:
        filename = 'train_' + filename
    else:
        filename = 'test_' + filename

    np.save(filename, labels_list)
    print("Unique Labels:", np.unique(labels_list).shape[0])

    print(f'{tensor.shape=}')

    # np_action_list = np.array([v for v in action_dict.values()])
    # print(np_action_list.shape)

if __name__ == '__main__':
    main()