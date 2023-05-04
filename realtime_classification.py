import os
import numpy as np
from load_dataset import load_synchronized_sensors, myReshape
from models import CNN_1D_multihead
from torchsummary import summary
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from TER_SLOTH import TER_sloth
from scipy.signal import butter, lfilter
import time as t

FREQ = 25

plot_sequence_labels = False 

full_sequences_path = 'full_sequences'
subj_dir = 's05'
window_size = 500
network_size = 500
stride = 1
imu_sensors_names = ['/right_wristPose.txt', '/right_backPose.txt', '/left_wristPose.txt', '/left_backPose.txt']

action_colors   = {0: 'red',            1: 'green',              2: 'blue',          3: 'pink',           4: 'yellow', -1: 'white'}
cmap = ListedColormap([action_colors[i-1] for i in range(6)])
# action_names   = {0: 'ASSEMBLY',            1: 'BOLT',              2: 'HANDOVER',          3: 'PICKUP',           4: 'SCREW'}
action_names   = {0: 'ASSEMBLY',            1: 'BOLT',              2: 'IDLE',          3: 'PICKUP',           4: 'SCREW'}
thresholds     = {0: 0.98804504,    1: 0.9947729,  2: 0.99488586,  3: 0.97964764, 4: 0.9955418}

gamma = 1
theta = 0.8
Rho = 0.05
Tau = [gamma * value for value in [0.98804504, 0.9947729, 0.99488586, 0.97964764, 0.9955418]]
C = [ window_size/2 for i in range(5)]
# C = [theta * value for value in [742.6095076400679, 1180.1586021505377, 1012.3956989247312, 323.61538461538464, 1106.9691943127962]]


# not_thresholds = {0: 0.19905082751065495,   1: 0.19994731021579357, 2: 0.19927371749654413, 3: 0.1997907838318497, 4: 0.1996165873249993}
not_thresholds = {0: 0.25,   1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}


def show_sequence_labels(actions_start_stop, seq_len):
    sequence_canva = label_visualization(actions_start_stop, seq_len)
    sns.heatmap(sequence_canva, cmap=cmap, yticklabels=False)
    plt.show()

def label_visualization(action_s_e, sequence_len):
    canva = np.zeros((sequence_len, 50)) - 1
    for action, start, end in action_s_e:
        if action == 'IDLE':
            canva[start:end, :] = -1
        else:
            canva[start:end, :] = list(action_names.values()).index(action)
    canva = canva.T
    return canva[::10]

def window_update_offline(stride, window_size, network_size, window_idx, unpad_unscaled_window, synchronized_sequences):
    ''' Slide window of [stride] samples''' 
    unpad_unscaled_window, window_idx = slide_window_offline(stride, unpad_unscaled_window, window_idx, synchronized_sequences)
    pad = np.zeros((window_size + (network_size - window_size), synchronized_sequences.shape[1]))
    pad[(network_size - window_size):, :] = unpad_unscaled_window
    unscaled_window = pad
    
    ''' Normalize on full scale range '''
    window = full_scale_normalize(unscaled_window)

    ''' Frequency analysis '''
    window = frequency_analysis(window)

    ''' Add one dimesnion to match batch size of 1 '''
    window_expanded = np.expand_dims(window, axis=0)

    ''' To tensor '''
    window_tensor = torch.Tensor(window_expanded).cuda()

    return window_tensor, window_idx, unpad_unscaled_window


def full_scale_normalize(data):
    acceleration_idxs = [0,1,2,6,7,8,12,13,14,18,19,20]
    gyroscope_idxs = [3,4,5,9,10,11,15,16,17,21,22,23]

    # 1g equals 8192. The full range is 2g
    data[:,acceleration_idxs] = data[:,acceleration_idxs] / 16384.0
    data[:,gyroscope_idxs] = data[:,gyroscope_idxs] / 1000.0

    return data


def low_pass(sequence, freq):
    '''UNCOMMENT TO PLOT THE ORIGINAL VS THE FILTERED VERSION'''
    # fig = plt.figure()
    # plt.plot(sequence)
    # print(sequence)
    # print(sequence.shape)
    fs = 120
    w = freq / (fs / 2) # Normalize the frequency
    b, a = butter(5, w, btype='low')
    y = lfilter(b,a,sequence)
    # plt.plot(y)
    # plt.show()
    return y

def frequency_analysis(data):
    newdata = data.copy()

    '''FOR EACH SENSOR FOR EACH FEATURE FILTER THE DATA AND RETURN THE NEW SEQUENCES'''
    for j in range(data.shape[1]):
        data[:,j] = low_pass(data[:,j], FREQ)

    return newdata


def initialize_and_fill_window(window_size, synchronized_sequences):
    window_idx = 0
    window = np.zeros((window_size, synchronized_sequences.shape[1]))
    while window_idx < window_size:
        window[window_idx] = synchronized_sequences[window_idx]
        window_idx += 1
    return window, window_idx


def find_idx_from_timestamp(tstamps, actions_start_stop):
    idxs = []
    for action in actions_start_stop:
        idxs.append([action[0], np.argmin(np.abs(tstamps - action[1])), np.argmin(np.abs(tstamps - action[2]))])
    return idxs


def slide_window_offline(stride, window, window_idx, synchronized_sequences):
    window = np.roll(window, -stride, axis=0)
    window[-1] = synchronized_sequences[window_idx]
    window_idx += 1
    return window, window_idx


subj_dir_full_path = os.path.join(full_sequences_path, subj_dir)
with open(os.path.join(subj_dir_full_path, 'actions_start_end.txt'), 'r') as f:
    actions = f.read().splitlines()

actions_start_stop = [[action.split(',')[0], float(action.split(',')[1]), float(action.split(',')[2])]  for action in actions]
print(actions_start_stop)

imu_sensors_data = []

tstamps = np.loadtxt(open(os.path.join(subj_dir_full_path + imu_sensors_names[0]), "rb"), delimiter=",", skiprows=1,usecols=(0))
print(f'{tstamps.shape=}')

actions_start_stop = find_idx_from_timestamp(tstamps, actions_start_stop)
print(actions_start_stop)

synchronized_sequences = myReshape(load_synchronized_sensors(subj_dir_full_path))
print(f'{synchronized_sequences.shape=}')
print(f'{synchronized_sequences[:3,:6]}')

if plot_sequence_labels:
    show_sequence_labels(actions_start_stop, synchronized_sequences.shape[0])

unpad_unscaled_window, window_idx = initialize_and_fill_window(window_size, synchronized_sequences)

print(f'{window_idx=}')

output_dim = len(action_names.keys())
input_dim = unpad_unscaled_window.shape[1]
print(f'{output_dim=}')
print(f'{input_dim=}')

model = CNN_1D_multihead(input_dim, output_dim).cuda()
model.load_state_dict(torch.load("best_model908.pth"))
# summary(model, (window_size, input_dim), 1, device='cuda')
model.eval()

# print(window[:3,:3])
# print(window_idx)
# print(synchronized_sequences[:5, :3])

in_action = False
current_action = -1 
action_time = 0
start_time = -1
# synchronized_sequences = full_scale_normalize(synchronized_sequences)

sloth = TER_sloth(model, 
                  window_size=window_size, 
                  class_size=output_dim, 
                  feature_size=synchronized_sequences.shape[1], 
                  rho=Rho, 
                  tau=Tau, 
                  c=C, 
                  action_names=action_names, 
                  action_colors=action_colors)

# rospy.init_node('realtime_classification', anonymous=True)
# r = rospy.Rate(10)



''' Loop over all the samples '''
while window_idx < synchronized_sequences.shape[0]:
    tic = t.time()
    # print(f'{window_idx=}')
    ''' Slide Window, padding, scaling, adding dummy dimension to match batch size of 1'''
    window_tensor, window_idx, unpad_unscaled_window = window_update_offline(stride, window_size, network_size, window_idx, unpad_unscaled_window, synchronized_sequences)
    # print(f'{window_tensor[0,0,:]}')
    # model(window_tensor)
    ''' Classify current window '''

    # print(window_tensor.shape)
    # print(window_tensor[0, -3:, :3])
    pred, time = sloth.classify(window_tensor)
    # print("=====================================")
    # print([f'{s:.2f}' for s in window_tensor[0, -1, :]][:6])
    # print([f'{s:.2f}' for s in window_tensor[0, -1, :]][6:12])
    # print([f'{s:.2f}' for s in window_tensor[0, -1, :]][12:18])
    # print([f'{s:.2f}' for s in window_tensor[0, -1, :]][18:24])

    # sloth.update_plot(pred, time)
    sloth.update_terminal_stats(pred, time)
    current_action = np.argmax(pred)
    current_prob = pred[current_action]
    # print(action_names[current_action], current_prob, '----', time)
    # if not in_action:
    #     if current_prob > Tau[current_action]:
    #         in_action = True
    #         start_time = time
    # if in_action:
    #     if current_prob < Tau[current_action]:
    #         if action_time > C[current_action]:
    #             print(action_names[current_action], start_time, time)
    #         in_action = False
    #         action_time = 0
    #         start_time = -1
    #     else:
    #         action_time +=1


    # sloth.detect()
    # print(f'FPS = {(t.time() - tic)}')

    # ''' Predict '''
    # prediction = model(window_tensor)
    # predicted_action_idx = torch.argmax(prediction).item()

    # current_prediction = prediction[0, predicted_action_idx].item()
    # current_threshold = thresholds[predicted_action_idx]

    # # print('Current prediction: ', action_names[predicted_action_idx])
    # # print('current_threshold: ', current_threshold)
    # # print('current_prediction: ', current_prediction)
    

    # # if not in_action:
    # if current_prediction > current_threshold:
    #     print(f'Action: {action_names[predicted_action_idx]} at {window_idx - window_size} ___ {current_prediction}')
    #     in_action = True
    #     current_action = predicted_action_idx
    # else:
    #     print(f'Action: IDLE at {window_idx - window_size} ___ {current_prediction}')

    # # else:
    # #     current_prediction = prediction[0, current_action].item()
    # #     # print('current_threshold: ', not_thresholds[predicted_action_idx])
    # #     # print('current_prediction: ', prediction[0, current_action].item())
    # #     # if current_prediction == predicted_action_idx:
    # #     if current_prediction < not_thresholds[predicted_action_idx]:
    # #         print(f'Action: {action_names[predicted_action_idx]} finished at {window_idx - window_size}')
    # #         in_action = False
    # #         current_action = -1

    # ''' Delete variables '''
    # del window_tensor
    # del prediction

# print(window[:3,:3])
# print(window_idx)

