import glob
from scipy.io import wavfile
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import re
import tqdm
import cv2
from collections import defaultdict
from imutils.video import count_frames
from itertools import chain

# Functions to sort file names correctly
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# Path to the nidaq data for a particular experiment
experiment_no = 432
clk_channel = 12
crop_len = 1 # length (in seconds) of audio and video to cut off from the end
camera_fps = 30
camera_expected_no_frames = 9000-1
# camera_corrected_sample_time = 0.03338542
sampling_rate = 125000
path_nidaq = "D:/big_setup/experiment_{}/nidaq/".format(experiment_no)


# Path to the video data for a particular experiment
path_video = "D:/big_setup/experiment_{}/videos/".format(experiment_no)


# Deleting the trunc files if it is already present
trunc_files = glob.glob(path_nidaq+"*_trunc*.*")
for i in trunc_files:
    os.remove(i) 

trunc_files = glob.glob(path_video+"*_trunc*.*")
for i in trunc_files:
    os.remove(i) 

# Get all clock channels for the experiment
clock_chs_names = glob.glob(path_nidaq + f"acquisition_data_*_{clk_channel}.wav")
clock_chs_names.sort(key=natural_keys)

# Getting all the rising and falling edges in the clock channels
threshold_rf = 1 # To detect all rising and falling edges

# Storing the indices of the rising and falling edges in a list
indices_r = []
indices_f = []

# To store the length of each clock channel
length_clks = [] 

print("Getting the rising and falling edges in the clock signal")
for name in tqdm.tqdm(clock_chs_names):
    temp_r = []
    temp_f = []

    # Loading the file
    sampling_rate,clk_ch = wavfile.read(name)

    diff = clk_ch[1:]-clk_ch[0:-1]
    temp_r = np.where(diff>threshold_rf)[0]
    temp_f = np.where(diff<-threshold_rf)[0]
    length_clks.append(len(clk_ch))
    indices_r.append(temp_r)
    indices_f.append(temp_f)

print("Clock channel lengths")
print(length_clks)


# Removing multiple detections of the same rising edge
camera_sampling_rate = 30 #FPS - frames per second. This is used to estimate when the next rising edge is to be detected
# window_samples = int((1/camera_sampling_rate)*sampling_rate) # Number of samples present in the window
window_samples = 3000

temp_r_0 = []
for i in indices_r:
    prev_sample = i[0]
    temp_r_1 = []
    temp_r_1.append(i[0])
    for j in i[1:]:
        if not(abs(j - prev_sample) < window_samples ):
            temp_r_1.append(j)
        prev_sample = j
    temp_r_0.append(temp_r_1)


indices_r = temp_r_0

# Getting the moving average of the first and last clock channel
MA = []

# Setting the file number for which we expect to see the start and stop record
start_record_file_no = 0 # 0 indicates the first saved clock channel
stop_record_file_no = -1 # -1 indicates the last saved clock channel

start_record_file_no = int(input("Enter the saved clock channel index where the start record can be found (file name format: acquisition_data_(index)_(channel).wav ):  "))
stop_record_file_no = int(input("Enter the saved clock channel index where the stop record can be found (file name format: acquisition_data_(index)_(channel).wav ):  "))

if start_record_file_no<-len(clock_chs_names)+1 or start_record_file_no> len(clock_chs_names)-1:
    start_record_file_no = 0
    print("Error in start record index input")

if stop_record_file_no<-len(clock_chs_names)+1 or stop_record_file_no> len(clock_chs_names)-1:
    stop_record_file_no = -1
    print("Error in stop record index input")

if stop_record_file_no < start_record_file_no:
    stop_record_file_no = -1
    start_record_file_no = 0
    print("Error in either stop record or start record index input")


print("Getting the moving average of the first and last clock signal")
for name in tqdm.tqdm([clock_chs_names[start_record_file_no],clock_chs_names[stop_record_file_no]]):

    # Loading the file
    sampling_rate,clk_ch = wavfile.read(name)

    camera_sampling_rate = 30 #FPS - frames per second. This is used to estimate the window for the moving average
    averaging_window_samples = int((1/camera_sampling_rate)*sampling_rate) # Number of samples present in the window

    # Converting the numpy array to a series for faster processing
    df_clock = pd.Series(clk_ch) 

    # Getting the moving average 
    temp_MA = df_clock.rolling(averaging_window_samples).mean()
    MA.append(temp_MA.fillna(0))
# Using the moving average and rising/falling edges to detect start and stop

threshold_ma_start = -1 # To detect start of recording (threshold on the moving average)
threshold_ma_stop = 0.25 # change to 1 if it's too sensitive a threshold



# Getting the indices of the start and stop record
print("Getting the index for start record")
indices_start = []
for idx1,idx2 in tqdm.tqdm(zip(indices_r[start_record_file_no][:-1],indices_r[start_record_file_no][1:])):
    diff = MA[0][idx2]-MA[0][idx1]
    if diff < threshold_ma_start:
        indices_start.append(idx2)

# **************************************************hard coding a the start record index if there is an issue*****************************************************
# indices_start = [] 
# indices_start.append(8020601) 
# ***********************************************************************************************************************************************

if len(indices_start) == 0:
    print("No start index found, defaulting to the first rising edge in the file")
    indices_start.append(indices_r[start_record_file_no][0])



#---------------------------------------------------manual start position input-----------------------------------------------------------------------------
# target = input("Enter the start second: ")
# def closest_value(lst, target):
#     return min(lst, key=lambda x: abs(x - target))
# indices_start = [closest_value(indices_r[start_record_file_no], float(target)*192000)]
# print("The reworked start indices are:",indices_start)
#---------------------------------------------------------------------------------------------------------------------------------------


print("Getting the index for stop record")
indices_stop = []
for idx1,idx2 in tqdm.tqdm(zip(indices_r[stop_record_file_no][:-1],indices_r[stop_record_file_no][1:])):
    diff = MA[1][idx2]-MA[1][idx1]
    if diff > threshold_ma_stop:
        indices_stop.append(idx1)

# **************************************************hard coding the stop record index if there is an issue*****************************************************
# indices_stop = [] 
# indices_stop.append(18473553) 
# ***********************************************************************************************************************************************

if len(indices_stop) == 0:
    print("No stop index found, defaulting to the last rising edge in the file")
    indices_stop.append(indices_r[stop_record_file_no][-1])


#---------------------------------------------------manual stop position input-----------------------------------------------------------------------------
# target = input("Enter the stop second: ")
# def closest_value(lst, target):
#     return min(lst, key=lambda x: abs(x - target))
# indices_stop = [closest_value(indices_r[stop_record_file_no], float(target)*192000)]
# print("The reworked stop indices are:",indices_stop)
#---------------------------------------------------------------------------------------------------------------------------------------

# Plotting to see the start and stop record indices

sorted_indices_start = np.sort(indices_start)
sorted_indices_stop = np.sort(indices_stop)

print("The indices are:",sorted_indices_start)
print("The indices are:",sorted_indices_stop)



# getting the indices to plot for the start record clock channel
first_sample_0 = sorted_indices_start[0] - int(sampling_rate/2)
last_sample_0 = sorted_indices_start[-1] + int(sampling_rate/2)


# getting the indices to plot for the stop record clock channel
first_sample_1 = sorted_indices_stop[0] - int(sampling_rate/2)
last_sample_1 = sorted_indices_stop[-1] + int(sampling_rate/2)


# Loading the files
sampling_rate,start_clk_ch = wavfile.read(clock_chs_names[start_record_file_no])
sampling_rate, stop_clk_ch = wavfile.read(clock_chs_names[stop_record_file_no])


time_axis_0 = np.array(range(len(start_clk_ch)))/sampling_rate
time_axis_1 = np.array(range(len(stop_clk_ch)))/sampling_rate


# Getting the first and last index to find the range to plot
# Plotting the clock channel data with the moving average and the rising edge/start record detection
fig = plt.figure(figsize=(15,5))
plt.subplot(211)

plt.plot(time_axis_0[first_sample_0:last_sample_0],start_clk_ch[first_sample_0:last_sample_0],label = "Clock+AUX")
plt.plot(time_axis_0[first_sample_0:last_sample_0],MA[0][first_sample_0:last_sample_0],'r', label = "Moving Average")



# Converting the rising  edges list to an array 
rising_0 = np.array(indices_r[start_record_file_no])
# Uncomment below line to show all the rising edges
plt.vlines(time_axis_0[rising_0[(rising_0<=last_sample_0) & (rising_0>=first_sample_0)]],0,5,'g')


# # Uncomment below line to show start record point
plt.vlines(time_axis_0[sorted_indices_start],0,5,'y', label = "Start Record Point")

plt.xlabel("Time (s)")
plt.ylabel("Signal Amplitude (V)")




plt.subplot(212)
plt.plot(time_axis_1[first_sample_1:last_sample_1],stop_clk_ch[first_sample_1:last_sample_1])
plt.plot(time_axis_1[first_sample_1:last_sample_1],MA[1][first_sample_1:last_sample_1],'r')

# Converting the rising  edges list to an array 
rising_1 = np.array(indices_r[stop_record_file_no])
# Uncomment below line to show all the rising edges
plt.vlines(time_axis_1[rising_1[(rising_1<=last_sample_1) & (rising_1>=first_sample_1)]],0,5,'g')


# Uncomment below line to show start record point
plt.vlines(time_axis_1[sorted_indices_stop],0,5,'y')

plt.xlabel("Time (s)")
plt.ylabel("Signal Amplitude (V)")
fig.legend()
fig.tight_layout()
plt.show()

# Finalizing the start and stop record indices
if len(sorted_indices_start) > 1:
    idx_entry = input("Enter which index number is start record (starts from index 0, 1 and so on)\n")
    print(f"You have entered {int(idx_entry)} corresponding to {sorted_indices_start[int(idx_entry)]}")
    start_record = sorted_indices_start[int(idx_entry)]
else:
    start_record = sorted_indices_start[0]

if len(sorted_indices_stop) > 1:
    idx_entry = input("Enter which index number is stop record (starts from index 0, 1 and so on)\n")
    print(f"You have entered {int(idx_entry)} corresponding to {sorted_indices_stop[int(idx_entry)]}")
    stop_record = sorted_indices_stop[int(idx_entry)]
else:
    stop_record = sorted_indices_stop[0]





# Get all video names for the experiment
video_names = glob.glob(path_video + "*.mp4")

video_names.sort(key=natural_keys)
# Getting the camera names
temp_cam_name = []
for i in video_names:
    temp_cam_name.append(i.split("\\")[1].split("-")[0])
camera_names = list(set(temp_cam_name))
print("The camera names are:")
print(camera_names)
# Creating a dictionary of all videos of a single camera
video_by_camera = defaultdict(lambda:[])
for idx,name in enumerate(temp_cam_name):
    video_by_camera[name].append(video_names[idx])

# Creating a dictionary with the length,timestamps and fps of each video
video_deets = defaultdict(lambda:[])

# max_length = 125000
# for item in tqdm.tqdm(video_by_camera.items()):
#     for vid in item[1]:
#         cap = cv2.VideoCapture(vid)
#         length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         if length<max_length:
#             max_length = length

# print(max_length)
vid_lengths = defaultdict(lambda:[])
for item in tqdm.tqdm(video_by_camera.items()):
    tot_length = 0
    for idx,vid in enumerate(item[1]):
        cap = cv2.VideoCapture(vid)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Number of frames using opencv: {length}")
        if length<camera_expected_no_frames:
            length = count_frames(vid, override=True)
            print(f"Number of frames using imutils: {length}")
        if idx == len(item[1])-1:   # we overrite the length with the imutils length for the last video because it is more accurate
            # length = count_frames(vid, override=True)
            # print(f"Number of frames using imutils: {length}")
            print(f"The total number of frames are: {tot_length+length}")
        tot_length+=length
        fps = cap.get(cv2.CAP_PROP_FPS)
        for frame in range(length):
            # video_deets[item[0]].append({"file_name":vid,"frame_no":frame,"time_from_start":frame/fps})
            video_deets[item[0]+"_file_name"].append(vid)
            video_deets[item[0]+"_frame_idx"].append(frame)
            video_deets[item[0]+"_time_from_vid_start"].append(frame/camera_fps)
        cap.release()
    for frame in range(tot_length):
        video_deets["concat_"+item[0]+"_frame_idx"].append(frame)
        video_deets["concat_"+item[0]+"_time_from_vid_start"].append(frame/camera_fps)
    vid_lengths[item[0]] = tot_length

# print("Number of rising edges in each clock signal")

# Converting to a dataframe
try:
    video_data =  pd.DataFrame.from_dict(video_deets)
except Exception as e:
    print(e)
    print("The camera lengths are as follows")
    print(vid_lengths)
    video_data = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in video_deets.items() ]))
mic_data = defaultdict(lambda:[])
for idx,edges in enumerate(indices_r):
    edges = np.array(edges)
    if idx < start_record_file_no:
        continue
    if idx == start_record_file_no:
        edges = edges[edges>=start_record]
    if idx == stop_record_file_no:
        edges = edges[edges<=stop_record]
    if idx > stop_record_file_no:
        break
    
    print(f"Clock channel index :{idx} has {len(edges)} number of rising edges")
    for sample in edges:
        mic_data["clk_ch_file_name"].append(clock_chs_names[idx])
        mic_data["clk_ch_sample_idx"].append(sample)
        mic_data["time_from_clk_ch_start"].append(sample/sampling_rate)
        mic_data["mics_file_idx"].append(idx)
        if idx>0:
            mic_data["concat_clk_ch_sample_idx"].append(sample+np.sum(length_clks[0:idx],dtype=np.int64))
            mic_data["concat_time_from_clk_ch_start"].append((sample+np.sum(length_clks[0:idx],dtype=np.float64))/sampling_rate)
        else:
            mic_data["concat_clk_ch_sample_idx"].append(sample)
            mic_data["concat_time_from_clk_ch_start"].append(sample/sampling_rate)


        
# Converting to a dataframe
nidaq_data =  pd.DataFrame.from_dict(mic_data)
# print(nidaq_data.head())

# Making the video data the same length as the mic data
if len(nidaq_data) < len(video_data):
    print(f"Number of video frames is longer and truncated by : {len(video_data)-len(nidaq_data)}")
    video_data = video_data.iloc[:len(nidaq_data),:]

if len(video_data) < len(nidaq_data):
    print(f"Number of mic samples is longer and truncated by : {len(nidaq_data)-len(video_data)}")
    nidaq_data = nidaq_data.iloc[:len(video_data),:]

# Joining the two dataframes
combined_data = pd.concat([video_data, nidaq_data], axis = 1)

# Removing the last 1 second of data - video corruption (software issue for white matter)
combined_data = combined_data.iloc[:len(combined_data)-camera_fps*crop_len,:]
print(f"Removed the last {crop_len} seconds")

path_timestamps = "D:/big_setup/experiment_{}/camera_timestamps.csv".format(experiment_no)
combined_data.to_csv(path_timestamps, index= False)