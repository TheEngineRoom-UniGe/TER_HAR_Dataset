import os
import numpy as np
from operator import itemgetter

full_data_path = 'full_sequences'
segmented_data_path = 'dataset'

segmented_data_folders = os.listdir(segmented_data_path)
full_data_folders = os.listdir(full_data_path)

print(f'{len(segmented_data_folders)=}')
print(f'{len(full_data_folders)=}')

for subj_dir in sorted(full_data_folders):
    if subj_dir not in segmented_data_folders:
        full_data_folders.remove(subj_dir)

for subj_dir in sorted(segmented_data_folders):
    if subj_dir not in full_data_folders:
        segmented_data_folders.remove(subj_dir)

print(f'{len(segmented_data_folders)=}')
print(f'{len(full_data_folders)=}')

test_percentage = len(full_data_folders) * 0.2
test_percentage = int(test_percentage)
print(f'{test_percentage=}')

test_set = np.random.choice(full_data_folders, test_percentage, replace=False)
training_set = np.setdiff1d(full_data_folders, test_set)
print(f'{test_set=}')
print(f'{training_set=}')
np.savetxt('test_set.txt', test_set, fmt='%s')
np.savetxt('training_set.txt', training_set, fmt='%s')


for item in test_set:
    if item in training_set:
        print('ERROR')

for item in training_set:
    if item in test_set:
        print('ERROR')