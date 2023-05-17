import os 
import numpy as np
import plotly.express as px
import pandas as pd

set = 'all'
set_dir = []
full_sequences_path = 'full_sequences'

if set == 'test':
    with open('test_set.txt', 'r') as f:
        set_dir = f.read().splitlines()
        print(set_dir)

elif set == 'training':
    with open('training_set.txt', 'r') as f:
        set_dir = f.read().splitlines()
        print(set_dir)

elif set == 'all':
    set_dir = os.listdir(full_sequences_path)
    print(set_dir)

# set_dir = ['s42', 's49', 's38_nh', 's45_nh']
print(set_dir)
action_lengths = {}
action_total_length = {}
for subj_dir in os.listdir(full_sequences_path):
    if subj_dir in set_dir:
        if os.path.isfile(os.path.join(full_sequences_path, subj_dir, 'actions_start_end.txt')):
            with open(os.path.join(full_sequences_path, subj_dir, 'actions_start_end.txt'), 'r') as f:
                actions = f.read().splitlines()
                # print(actions[0])
                for action in actions:
                    action = action.split(',')
                    if action[0] not in action_lengths.keys():
                        action_lengths[action[0]] = [(float(action[2]) - float(action[1])) * 1e-9]
                    else:
                        action_lengths[action[0]].append((float(action[2]) - float(action[1])) * 1e-9)
                    
                    if action[0] not in action_total_length.keys():
                        action_total_length[action[0]] = 0
                        action_total_length[action[0]] += (float(action[2]) - float(action[1])) * 1e-9
                    else:
                        action_total_length[action[0]] += (float(action[2]) - float(action[1])) * 1e-9


class_distribution = [len(action_lengths[key]) for key in sorted(action_lengths.keys())]
class_length_distribution = [action_total_length[key]/len(action_lengths[key])*100 for key in sorted(action_lengths.keys())]

print(class_length_distribution)
print('Keys:', sorted(action_lengths.keys()))

global_lengths = [item for sublist, action in zip(action_lengths.values(), action_lengths.keys()) for item in sublist if 'SCREW' in action]

font1 = font={
                    'family' : 'Times New Roman',
                    'size' : 18
                    }

fig = px.histogram(global_lengths, nbins=100)
fig.add_vline(x=np.mean(global_lengths), annotation_text='  <b>Mean', annotation_position='right', annotation=dict(font=font1), line_width=3, line_dash="dash", line_color="black", col=1, row=1)
fig.add_vline(x=np.median(global_lengths), annotation_text='<b>Median  ', annotation_position='left', annotation=dict(font=font1), line_width=3, line_dash="dash", line_color="red", col=1, row=1)
fig.add_vline(x=np.mean(global_lengths)+2*np.std(global_lengths), annotation_text='<b>mean+2*std  ', annotation_position='left', annotation=dict(font=font1), line_width=3, line_dash="dash", line_color="green", col=1, row=1)
# fig.add_vline(x=np.mean(global_lengths)-2*np.std(global_lengths), annotation_text='<b>mean-2*std  ', annotation_position='left', annotation=dict(font=font1), line_width=3, line_dash="dash", line_color="green", col=1, row=1)
fig.add_vline(x=np.mean(global_lengths)+np.std(global_lengths), annotation_text='<b>mean+std  ', annotation_position='left', annotation=dict(font=font1), line_width=3, line_dash="dash", line_color="green", col=1, row=1)
# fig.add_vline(x=np.mean(global_lengths)-np.std(global_lengths), annotation_text='<b>mean-std  ', annotation_position='left', annotation=dict(font=font1), line_width=3, line_dash="dash", line_color="blue", col=1, row=1)
fig.show()

action_dist_dict = {'number_of_actions': class_distribution, 'action_name': sorted(action_lengths.keys())}
action_dist_dict = pd.DataFrame.from_dict(action_dist_dict)
print(action_dist_dict)
print('Total number of actions:', sum(class_distribution))
fig = px.histogram(action_dist_dict, nbins=6, x='number_of_actions', y='action_name', color='number_of_actions', color_discrete_sequence=px.colors.qualitative.Pastel)
fig.show()



