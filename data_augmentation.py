import numpy as np
import utils.augmentation as aug
import utils.helper as hlp
import matplotlib.pyplot as plt
import random 
import os 
from enum import Enum

class Augmentation(Enum):
    jitter = 1
    scaling = 2
    magnitude_warp = 3
    time_warp = 4
    window_slice = 5
    discriminative_guided_warp = 6
    random_guided_warp = 7
    wdba = 8

def save_augmented_data(balanced_data, path):
    values = balanced_data.values()
    keys = balanced_data.keys()
    balanced_data_list = [val for val in values]
    np_balanced_data = np.row_stack(balanced_data_list)
    n_seqXaction = len(list(values)[0])
    n_actions = len(balanced_data.keys())
    np_balanced_labels = np.zeros((n_seqXaction*n_actions, 1), dtype=object)
    for i, key in enumerate(keys):
        print(i, key)
        np_balanced_labels[i*n_seqXaction:(i+1)*n_seqXaction] = key
    np.save(os.path.join(path, "train_balanced_data({}_{}_{}).npy".format(*np_balanced_data.shape)), np_balanced_data)
    np.save(os.path.join(path, "train_balanced_labels({}_{}).npy".format(*np_balanced_labels.shape)), np_balanced_labels)

def switch(augmentation):
    if augmentation == Augmentation.jitter:
        return aug.jitter
    elif augmentation == Augmentation.scaling:
        return aug.scaling
    elif augmentation == Augmentation.magnitude_warp:
        return aug.magnitude_warp
    elif augmentation == Augmentation.time_warp:
        return aug.time_warp
    elif augmentation == Augmentation.window_slice:
        return aug.window_slice
    # elif augmentation == Augmentation.discriminative_guided_warp:
    #     return aug.discriminative_guided_warp
    # elif augmentation == Augmentation.random_guided_warp:
    #     return aug.random_guided_warp
    # elif augmentation == Augmentation.wdba:
    #     return aug.wdba
    else:
        print('Invalid augmentation')
        return None

def augment_action(action):
    ac_sh1, ac_sh2 = action.shape[0], action.shape[1]
    tmp_action = action.reshape((-1, ac_sh1, ac_sh2)) 
    # print(tmp_action.shape)
    augmentation = Augmentation(random.randint(1, 5))
    # print(augmentation)

    augmented_action = switch(augmentation)(tmp_action)[0]

    # fig, axs = plt.subplots(1,8)
    # for i in range(8):
    #     for j in range(3):
    #         k=(i*3)+j
    #         axs[i].plot(action[:,k])
    #         axs[i].plot(augmented_action[:,k])

    # plt.show()
    return augmented_action

def pick_rand_action(action_list):
    return random.choice(action_list)

def balance_dataset(dataset, labels):
    random.seed(42)

    action_dict = build_action_dict(dataset, labels)
    print_action_dict(action_dict)
    action_distribution = build_action_distribution(action_dict)
    action_distribution = sorted(action_distribution.items(), key=lambda x: x[1], reverse=False)
    # print(action_distribution)
    maxl = action_distribution[-1][1]
    print('Max length:', maxl)

    for key, length in action_distribution:
        if length >= maxl:
            continue
        # print(key, length)
        while length < maxl:
            action_list = action_dict[key]
            action_list.append(augment_action(pick_rand_action(action_list)))
            length += 1
        # print(key, length)

    print_action_dict(action_dict)
    return action_dict

def build_action_dict(dataset, labels):
    action_dictionary = {}
    for action, label in zip(dataset, labels):
        label = label[0]
        if label not in action_dictionary:
            action_dictionary[label] = [action]
        else:
            action_dictionary[label].append(action)
    return action_dictionary

def build_action_distribution(action_dict):
    action_distribution = {}
    for key in action_dict:
        action_distribution[key] = len(action_dict[key])
    return action_distribution

def print_action_dict(action_dict):
    for key in action_dict:
        print(key, '---->', len(action_dict[key]))

dataset = np.load('train_data_shape(4950_500_24).npy').astype('float64')
labels = np.load('train_labels_shape(4950_1).npy')

print(dataset.shape)
print('Unique labels:')
print(np.unique(labels), '\n')

balanced_ds = balance_dataset(dataset, labels)
save_augmented_data(balanced_ds, 'balanced_datasets')

exit()
action_dict = {}
for sequence, label in zip (dataset, labels):
    # print(label[0])
    base_action = label[0].split('_')[0]
    hand = label[0].split('_')[1]

    # if 'ASSEMBLY1' in base_action:
    #     action_dict['ASSEMBLY1'] = [sequence[:,:3]]
    # if 'ASSEMBLY2' in base_action:
    #     if
    #     action_dict['ASSEMBLY2'] = 1

    add_seq = []

    if 'ASSEMBLY' in base_action:
        continue

    if 'SCREW' in base_action:
        if 'RIGHT' in hand:
            add_seq = [sequence[:,12:24]]
        if 'LEFT' in hand:
            add_seq = [sequence[:,0:12]]
        if 'BIMANUAL' in hand:
            add_seq = [sequence[:,0:12], sequence[:,12:24]]

    if 'BOLT' in base_action:
        if 'RIGHT' in hand:
            add_seq = [sequence[:,12:24]]
        if 'LEFT' in hand:
            add_seq = [sequence[:,0:12]]

    if 'DELIVERY' in base_action or 'PICKUP' in base_action or 'HANDOVER' in base_action:
        if 'RIGHT' in hand:
            add_seq = [sequence[:,12:24]]
        if 'LEFT' in hand:
            add_seq = [sequence[:,0:12]]
        if 'BIMANUAL' in hand:
            add_seq = [sequence[:,0:12], sequence[:,12:24]]

    if 'IDLE' in base_action:
        add_seq = [sequence[:,:12], sequence[:,12:24]]

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
    
# print(action_dict)