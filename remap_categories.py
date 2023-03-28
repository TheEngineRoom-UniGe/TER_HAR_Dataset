import numpy as np


dataset = np.load('data_shape(2699_2981_24).npy').astype('float64')
labels = np.load('labels_shape(2699_1).npy')

print(dataset.shape)
print(np.unique(labels))

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