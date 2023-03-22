#!/usr/bin/env python3
from posixpath import abspath
from site import abs_paths
from PIL import Image
from PIL.ExifTags import TAGS
import os 
import numpy as np

def getImageDescription(file_absolute_path):
    '''Given the full path of an image it return its description
        if no description is available None is returned'''

    image = Image.open(file_absolute_path)
    exifdata = image.getexif()

    for tag_id in exifdata:
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id, tag_id)
        if tag != 'ImageDescription':
            print('skipping this tag')
            continue
        data = exifdata.get(tag_id)
        # decode bytes 
        if isinstance(data, bytes):
            data = data.decode()
        # print(f"{tag:25}: {data}")
        return data

def getIndex(filename):
    return int(filename.split('.')[0][6:])

def listJPEG(full_path_directory):
    JPEGlist = []
    for file in os.listdir(abs_path):
        if file.endswith('.jpeg'):
            JPEGlist.append(file)
    return JPEGlist

def getTimestamps(img2ros_filepath):
    file = open(img2ros_filepath, 'r')
    lines = file.readlines()
    idx2ros_stamp = np.empty([len(lines), 2])

    for i in range(len(lines)):
        idx2ros_stamp[i][0] = lines[i].split(',')[0]
        idx2ros_stamp[i][1] = getIndex(lines[i].split(',')[1].split('/')[-1])
    return idx2ros_stamp

def ROSIdxConversion(description_list, ros_idx_conversion_file):
    new_description_list = []
    for entry in description_list:

        #   Action begin
        file_idx_start = np.where(ros_idx_conversion_file[:,1] == entry[1])
        file_line_start = ros_idx_conversion_file[file_idx_start]
        ros_start = file_line_start[0,0]
        # print(file_line_start)
        # print(ros_start)

        #   Action end
        file_idx_end = np.where(ros_idx_conversion_file[:,1] == entry[2])
        file_line_end = ros_idx_conversion_file[file_idx_end]
        ros_end = file_line_end[0,0]
        # print(file_line_end)
        # print(ros_end)

        entry = entry + (ros_start, ros_end)
        new_description_list.append(entry)
    return new_description_list

def trimData(imu_file_full_path, action_stamps, dataset_full_path):
    file = open(imu_file_full_path, 'r')
    lines = file.readlines()
    action_dict = {}   
    sensor_name = os.path.split(imu_file_full_path)[1]

    volunteer = imu_file_full_path.split('/')[-2]
    vol_dir = os.path.join(dataset_full_path, volunteer)
    if not os.path.exists(vol_dir):
        print('Creating dir...', vol_dir)
        os.mkdir(vol_dir)
    
    for entry in action_stamps:
        print(entry)
        action_dir = os.path.join(vol_dir, entry[0])
        if not os.path.exists(action_dir):
            print('Creating dir...', action_dir)
            os.mkdir(action_dir)
        list_action_dir = os.listdir(action_dir)
        
        name = entry[0]

        if name in action_dict:
            new_file_full_path = os.path.join(action_dir, str(action_dict[name]))
            action_dict[name] = action_dict[name] +1

        else:
            new_file_full_path = os.path.join(action_dir, str(0))
            action_dict[name] = 1

        if not os.path.exists(new_file_full_path):
            print('Creating dir...', new_file_full_path)
            os.mkdir(new_file_full_path)

        new_file_full_path = os.path.join(new_file_full_path, sensor_name)

        if not os.path.exists(new_file_full_path):
            print(f'Creating file {len(list_action_dir)}')
            new_imu_file = open(new_file_full_path, 'w+')
        
        for line in lines:
            if int(line.split(',')[0]) >= entry[3] and int(line.split(',')[0]) <= entry[4]:
                new_imu_file.writelines(line)
   
            else:
                continue
    return


def findEndAction(description_list):
    new_description_list = []
    already_checked = []
    for i in range(len(image_descriptions_list)):
        desc = image_descriptions_list[i][0]
        num = image_descriptions_list[i][1]
        if num in already_checked:
            continue
        action_name = desc.split('-')[0]
        delimiter = desc.split('-')[1]
        # print(i, action_name)
        if delimiter == 'START':
            # print(len(image_descriptions_list) - i)
            for j in range(len(image_descriptions_list) - i):
                next_desc = image_descriptions_list[i+j][0]
                next_num = image_descriptions_list[i+j][1]
                next_action_name = next_desc.split('-')[0]
                next_delimiter = next_desc.split('-')[1]

                if next_action_name == action_name and next_delimiter == 'END':
                    new_description_list.append((action_name, num, next_num))
                    already_checked.append(num)
                    already_checked.append(next_num)
                    break
                else:
                    if i == len(image_descriptions_list)-1:
                        print('non trovato')
                        print((action_name, num, next_action_name, next_num))
    # print(already_checked)
    return new_description_list

for idx in range(10,16):
    abs_path = f'/newSSD/baxter_unity/data/s{idx}_nh/frames'
    image_list = sorted(listJPEG(abs_path), key=getIndex)

    image_descriptions_list = []

    for image in image_list:
        num = getIndex(image)
        filename = f'frame_{num}.jpeg'

        file_abs_path = os.path.join(abs_path, filename)
        description = getImageDescription(file_abs_path)
        
        if description is not None and description != ' ':
            image_descriptions_list.append((description, num))
        
    nlist1=[]
    for l in image_descriptions_list:
        nlist1.append(l[1]) 
    print(image_descriptions_list)
    print(len(image_descriptions_list))
    image_descriptions_list = findEndAction(image_descriptions_list)
    print(image_descriptions_list)
    print(len(image_descriptions_list))

    nlist2=[]
    for l in image_descriptions_list:
        nlist2.append(l[1]) 
        nlist2.append(l[2]) 

    for l in nlist1:
        if l not in nlist2:
            print(l)


    # continue

    img2ros_file_path = os.path.join(os.path.split(abs_path)[0],'frames_log.txt')
    print(img2ros_file_path)

    ros_idx_stamp = getTimestamps(img2ros_file_path)
    # print(ros_idx_stamp[:,1])
    # exit()
    image_descriptions_list = ROSIdxConversion(image_descriptions_list, ros_idx_stamp)
    # print(image_descriptions_list)

    new_dataset_path = '/newSSD/baxter_unity/dataset'

    #['left_backPose.txt', 'right_backPose.txt', 'left_wristPose.txt', 'right_wristPose.txt']
    for sensor_name in ['left_backPose.txt', 'right_backPose.txt', 'left_wristPose.txt', 'right_wristPose.txt']:
        imu_file_path = os.path.join(os.path.split(abs_path)[0], sensor_name)
        print(imu_file_path)
        trimData(imu_file_path, image_descriptions_list, new_dataset_path)

