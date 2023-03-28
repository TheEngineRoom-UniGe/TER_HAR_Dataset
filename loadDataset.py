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

def mergeIntoGeneralActions(labels, actions):
    action_dict = {}
    for seq, action_name in zip(actions, labels):
        # print(seq.shape, action_name)
        general_action_name = action_name.split('_')[0]
        # if "ASSEMBLY" in general_action_name:
        #     general_action_name = "ASSEMBLY"
        # if "HANDOVER" in general_action_name:
        #     general_action_name = "DELIVERY"
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


def zeroPadding(dataset):
    maxl = 0
    for sequence in dataset:
        if sequence.shape[0] >= maxl:
            maxl = sequence.shape[0]

    padded_dataset = np.zeros((len(dataset), maxl, 24), dtype='float64')

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
    b, a = butter(1, freq,btype='low', fs = 1000)
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
    imu_sensors_data = frequency_analysis(imu_sensors_data)

    '''RESAMPLING TO MATCH THE SHORTEST SENSOR SEQUENCE'''
    imu_sensors_data = dataResampling(imu_sensors_data)
    imu_sensors_data = np.array(imu_sensors_data)

    '''AT THIS POINT imu_sensors_data CONTAINS THE SEQUENCES OF THE FOUR SENSORS'''
    # print(imu_sensors_data.shape)
    return imu_sensors_data


    



dataset_abs_path = f'./dataset'
data_dir_list = sorted(os.listdir(dataset_abs_path))

action_dict = {}
maxp = 0
for participant in data_dir_list:
    participant_dir = os.path.join(dataset_abs_path, participant)
    # print(participant_dir)

    list_participant_dir = os.listdir(participant_dir)
    # print(list_participant_dir)

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
    if len(action) < 12:
        print(">> ", action_name, " -> ", len(action), "SKIPPED! <<")
        print('---------------------------------------------')
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

# Treshold to remove outliers with too many samples
threshold = np.mean(len_list)+1*np.std(len_list)        #2*np.std(len_list)
print(f'{threshold=}')
mask_np = len_list < threshold
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
labels_list, action_list_np = mergeIntoGeneralActions(labels_list, masked_list)

tensor_list = [torch.tensor(arr) for arr in masked_list]
tensor = pad_sequence(tensor_list, batch_first=True)

'''APPLY ZERO PADDING TO HAVE SEQUENCES OF THE SAME SIZES'''
masked_list_np = zeroPadding(masked_list)
labels_list = np.asarray(labels_list).reshape(-1, 1)

print(f'{masked_list_np.shape=}')
print(f'{labels_list.shape=}')

masked_list_np = normalize(masked_list_np)

# Save stuff

filename = "data_shape({}_{}_{}).npy".format(*masked_list_np.shape)
np.save(filename, masked_list_np)

# filename = "data_shape({}_{}_{}).pt".format(*masked_list_np.shape)
# torch.save(tensor, filename)

filename = "labels_shape({}_{}).npy".format(*labels_list.shape)
np.save(filename, labels_list)
print("Unique Labels:", np.unique(labels_list).shape[0])

print(f'{tensor.shape=}')

# np_action_list = np.array([v for v in action_dict.values()])
# print(np_action_list.shape)