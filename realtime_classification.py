import os
import numpy as np
from load_dataset import load_synchronized_sensors, myReshape

full_sequences_path = 'full_sequences'
subj_dir = 's01'
window_size = 3000
imu_sensors_names = ['/right_wristPose.txt', '/right_backPose.txt', '/left_wristPose.txt', '/left_backPose.txt']

def find_idx_from_timestamp(tstamps, actions_start_stop):
    idxs = []
    for action in actions_start_stop:
        idxs.append([action[0], np.argmin(np.abs(tstamps - action[1])), np.argmin(np.abs(tstamps - action[2]))])
    return idxs

def slide_window_offline(window, window_idx, synchronized_sequences):
    window = np.roll(window, 1, axis=0)
    window[0] = synchronized_sequences[window_idx]
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

window_idx = 0
window = np.zeros((window_size, synchronized_sequences.shape[1]))
while window_idx < window_size:
    window[window_idx] = synchronized_sequences[window_idx]
    window_idx += 1

print(f'{window_idx=}')

print(window[:5,:3])
print(window_idx)
window, window_idx = slide_window_offline(window, window_idx, synchronized_sequences)
print(window[:5,:3])
print(window_idx)

