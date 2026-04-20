import os
import shutil
import glob
import numpy as np
import tqdm
import time
import re


# Functions to sort file names correctly
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# checking to see what experiments are written
folders = glob.glob("C:/Users/daq2/niegil_codes/data/channel/*")
exp_numbers = [i.split("\\")[1].split('_')[1] for i in folders]
exp_numbers.sort(key=natural_keys)
last_exp_no_in_C_drive = int(exp_numbers[-1])

# checking to see what experiments are written
folders = glob.glob("D:/big_setup/*")
folders = [i for i in folders if 'experiment_' in i]
exp_numbers = [i.split("\\")[1].split('_')[1] for i in folders]
exp_numbers.sort(key=natural_keys)
last_exp_no_in_D_drive = int(exp_numbers[-1])

if last_exp_no_in_D_drive>last_exp_no_in_C_drive:
    experiment_no = last_exp_no_in_D_drive
else:
    experiment_no = last_exp_no_in_C_drive

print("***********************************************************************************")
print("THE EXPERIMENT NUMBER IS: ", experiment_no)
print("***********************************************************************************")

while True:
    usr_in = input("Is the experiment number correct? y/n \n")
    if usr_in == 'y' or usr_in == 'Y':
        break
    elif usr_in == 'n' or usr_in == 'N':
        try:
            experiment_no = int(input("Enter the experiment no.: "))
            break
        except:
            print("Invalid input, try again")
    else:
        print("Invalid input, try again")
        
print("Data transfer started")

#---------------------------------------------------------------------------------pre run check (number of channels) ------------------------------------------------------------
no_channels = 13
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create a new folder in the destination folder for the experiment
path_D_name = "D:/big_setup/experiment_{}/".format(experiment_no)
try:
    os.makedirs(path_D_name)
except:
    pass

# Creating separate folders for the nidaq and video data
path_D_vid = "D:/big_setup/experiment_{}/videos/".format(experiment_no)
path_D_ni = "D:/big_setup/experiment_{}/nidaq/".format(experiment_no)
path_D_log = "D:/big_setup/experiment_{}/".format(experiment_no)

try:
    os.makedirs(path_D_vid)
except:
    pass

try:
    os.makedirs(path_D_ni)
except:
    pass

# Path to C drive video data
path_C_vid =  "./v2.0.0-win64/v2.0.0-win64/data/defaultsession/"

# Path to C drive nidaq data
path_C_ni =  "./data/channel/experiment_{}/".format(experiment_no)


#---------------------------------------------------------------------------------pre run check (cameras) ------------------------------------------------------------
# Creating a list for the camera names
cam_names = ["gily_center"]
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Functions to sort file names correctly
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]



# Moving files continuously from C to D drive 

try:
    while True:
        # Getting the videos in the folder and moving the video if it not currently being written to
        for i in cam_names:
            path_C_cam = path_C_vid+i+"*"
            all_vids = np.sort(glob.glob(path_C_cam))
            if len(all_vids) > 1:
                print("Moving Video data")
                for j in tqdm.tqdm(all_vids[:-1]):
                        file_name = j.split("\\")[-1]
                        shutil.move(path_C_vid + file_name, path_D_vid + file_name)

        # Getting all the nidaq files in the folder and moving the files if not currently being written to
        all_nidaq = glob.glob(path_C_ni+"*.wav")
        all_nidaq.sort(key=natural_keys)

        if len(all_nidaq) > no_channels:
            time.sleep(2)
            print("Moving NIDAQ data")
            for j in tqdm.tqdm(all_nidaq[:no_channels]):
                file_name = j.split("\\")[-1]
                shutil.move(path_C_ni + file_name, path_D_ni + file_name)
        
        time.sleep(60)


except KeyboardInterrupt:
    print("Moving the rest of the videos")
    for i in cam_names:
        path_C_cam = path_C_vid+i+"*"
        all_vids = np.sort(glob.glob(path_C_cam))
        for j in tqdm.tqdm(all_vids):
                file_name = j.split("\\")[-1]
                shutil.move(path_C_vid + file_name, path_D_vid + file_name)

    print("Moving the rest of the nidaq data")
    all_nidaq = glob.glob(path_C_ni+"*.wav")
    for j in tqdm.tqdm(all_nidaq):
        file_name = j.split("\\")[-1]
        shutil.move(path_C_ni + file_name, path_D_ni + file_name)


    # Checking for log files
    print("Moving log files if any ")
    all_log = glob.glob(path_C_ni+"*.txt")
    for j in tqdm.tqdm(all_log):
        file_name = j.split("\\")[-1]
        shutil.move(path_C_ni + file_name, path_D_log + file_name)
