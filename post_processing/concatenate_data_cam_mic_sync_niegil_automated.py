import glob
import os
import tqdm
from collections import defaultdict
import re
import numpy as np
import pandas as pd
from scipy.io import wavfile
import shutil
import subprocess
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# from moviepy.video.io.VideoFileClip import VideoFileClip
from datetime import datetime, timedelta
import json
import sys


# in order to find the path, run shutil.which after activating the environment with ffmpeg in miniforge or on the cluster
ffmpeg = shutil.which("ffmpeg") or 'C:\\Users\\DAQ3\\miniforge3\\envs\\daq\\Library\\bin\\ffmpeg.EXE'



# def seconds_to_hms(seconds):
#     # Calculate hours, minutes, and seconds
#     hours, remainder = divmod(seconds, 3600)
#     minutes, seconds = divmod(remainder, 60)
#     # Format the time in HH:MM:SS
#     return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(remainder%1*1000):03}"


def ffmpeg_extract_subclip(filename, t1, t2=None, targetname=None):
    """
    Fast + accurate subclip extraction using hybrid seeking.

    If t2 is None, extracts from t1 to end of file.
    """

    # Fast seek goes slightly BEFORE target (improves accuracy)
    fast_seek = max(t1 - 1, 0)
    precise_offset = t1 - fast_seek

    cmd = [
        ffmpeg,
        "-y",
        "-ss", str(fast_seek),
        "-i", filename,
        "-ss", str(precise_offset),
    ]

    # Only include duration if t2 is provided
    if t2 is not None:
        duration = t2 - t1
        if duration <= 0:
            raise ValueError("t2 must be greater than t1")
        cmd += ["-t", str(duration)]

    cmd += [
        "-c:v", "h264_nvenc",
        "-cq", "18", # similar to CRF (lower = better quality)
        "-preset", "medium", # NVENC preset (p1 fastest → p7 best quality)
        "-c:a", "copy",
        targetname
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"{targetname} created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")




# Functions to sort file names correctly
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def process_experiment(experiment_no):
        print(f"======================================================== Working on experiment: {experiment_no} ========================================================")

        camera_fps = 30
        samples_per_cam_frame = 4166.5
        precise_cam_fps = 125000/samples_per_cam_frame
        # camera_corrected_sample_time = 0.03338542
        file_length = 120 # in seconds (360 sec - 6 mins)
        # nidaq_folder = "D:/big_setup/experiment_{}/nidaq/".format(experiment_no)
        nidaq_folder = "D:/big_setup/experiment_{}/nidaq/".format(experiment_no)
        video_folder = "D:/big_setup/experiment_{}/videos/".format(experiment_no)


        # checking if the timestamp file is present
        if len(glob.glob(f"D:/big_setup/experiment_{experiment_no}/camera_timestamps*csv")) == 0:
            print(f"No timetsamp file found for experiment {experiment_no}, skipping it")
            return
        # checking whether concatenation files are present
        elif len(glob.glob(f"D:/big_setup/experiment_{experiment_no}/concatenated_data_cam_mic_sync/temp/*.txt")) > 0:
            print(f"Skipping experiment {experiment_no}, delete concatenation data if you'd like to run this experiment")
            return 
            # val = input(f"Do you want to rerun concatenation for experiment {experiment_no}? (y/n): ")
            # if val == 'y' or val == "Y":
            #     print("Deleting the already present concatenated data")
            #     shutil.rmtree(f"D:/big_setup/experiment_{experiment_no}/concatenated_data_cam_mic_sync")
            # else:
            #     print(f"Skipping experiment {experiment_no}")
            #     continue

        # Deleting the trunc files if it is already present
        trunc_files = glob.glob(nidaq_folder+"*_trunc*")
        for i in trunc_files:
            os.remove(i) 

        trunc_files = glob.glob(video_folder+"*_trunc*")
        for i in trunc_files:
            os.remove(i) 

        # Automatically find the number of channels
        all_nidaq_files = glob.glob(nidaq_folder + "acquisition_data_*_*.wav")
        all_nidaq_files.sort(key=natural_keys)

        channels = []
        for f in all_nidaq_files:
            ch = int(re.findall(r'acquisition_data_\d+_(\d+)\.wav', os.path.basename(f))[0])
            channels.append(ch)

        no_channels = max(channels) + 1



        nidaq_files = glob.glob(nidaq_folder+"*")
        video_files = glob.glob(video_folder+"*")

        # Creating separate folders for the concatenated data
        path_concat = "D:/big_setup/experiment_{}/concatenated_data_cam_mic_sync/temp/".format(experiment_no)

        try:
            os.makedirs(path_concat)
        except:
            pass

        path_final = "D:/big_setup/experiment_{}/concatenated_data_cam_mic_sync/".format(experiment_no)



        nidaq_files.sort(key=natural_keys)
        video_files.sort(key=natural_keys)

        nidaq_files = [i.replace("\\",'/') for i in nidaq_files]
        video_files = [i.replace("\\",'/') for i in video_files]

        ### For Video concatenation ---------------------------------------------------------------------------------------------------------

        # Getting the camera names
        temp_cam_name = []
        for i in video_files:
            temp_cam_name.append(i.split("/")[-1].split("-")[0])
        camera_names = list(set(temp_cam_name))
        print("Camera names: \n",camera_names)


        # Creating a dictionary of all videos of a single camera
        video_by_camera = defaultdict(lambda:[])
        for idx,name in enumerate(temp_cam_name):
            video_by_camera[name].append(video_files[idx])
        

        cam_clk_data = pd.read_csv(f"D:/big_setup/experiment_{experiment_no}/camera_timestamps.csv", low_memory=False)
        file_breaks  = np.arange(0,cam_clk_data.tail(1).index[0]/camera_fps,file_length)
        timestamp_idx = []
        for val in file_breaks:
            timestamp_idx.append((np.abs(cam_clk_data[f'concat_{camera_names[0]}_time_from_vid_start'] - val)).argmin())
        for cam in camera_names:

            print(f"Working on camera: {cam}")
            file_break_idx = 1
            file_concat_index = 0 
            str_1 = "%03d"%(file_concat_index)
            filename_txt = path_concat + f"video_{cam}_{str_1}.txt"
            f = open(filename_txt, 'w')
            no_videos = len(video_by_camera[cam])

            for index, vid_name in tqdm.tqdm(enumerate(video_by_camera[cam])):

                
                try:
                    temp = cam_clk_data.iloc[timestamp_idx[file_break_idx]][f'{cam}_file_name']
                    temp = temp.replace("\\","/")
                except:
                    print("Index from timestamp not found")
                    pass
                
                if temp == vid_name and index != no_videos -1 :
                    # break_time = cam_clk_data.iloc[timestamp_idx[file_break_idx]][f'{cam}_time_from_vid_start']
                    break_frame = cam_clk_data.iloc[timestamp_idx[file_break_idx]][f'{cam}_frame_idx']
                    break_time = break_frame/precise_cam_fps
                    target_file = temp.split(".")[0]+"_trunc_0.mp4"
                    ffmpeg_extract_subclip(vid_name, 0.0, break_time, targetname=target_file)
                    
                    # with VideoFileClip(vid_name) as video:
                    #     new = video.subclip(0.0, break_time)
                    #     new.write_videofile(target_file, audio_codec='aac')

                    f.write("file \'{}\'\n".format(target_file))
                    f.close()


                    target_file = temp.split(".")[0]+"_trunc_1.mp4"
                    # with VideoFileClip(vid_name) as video:
                        # end_time = video.duration
                        # end_frame = video.reader.nframes

                    ffmpeg_extract_subclip(vid_name, break_time, None, targetname=target_file)



                    file_concat_index+=1
                    file_break_idx+=1
                    str_1 = "%03d"%(file_concat_index)
                    filename_txt = path_concat + f"video_{cam}_{str_1}.txt"
                    f = open(filename_txt, 'w')
                    f.write("file \'{}\'\n".format(target_file))


                elif temp != vid_name and index == no_videos -1:
                    #cam_stop_rec_video_name = cam_clk_data[f"{cam}_file_name"].iloc[-1]
                    # try:
                    #     cam_stop_rec_frame = int(cam_clk_data[f"{cam}_frame_idx"].iloc[-1])
                    # except Exception as e: # trying to catch exceptions that arise when the video is not long enough (camera drop/frame drop)
                    #     print(e)
                    #     cam_stop_rec_frame = None
                    
                    # target_file = vid_name.split(".")[0]+"_trunc_0.mp4"

                    # if cam_stop_rec_frame == None:
                    #     target_file = cam_stop_rec_video_name
                    # else:
                    ffmpeg_extract_subclip(vid_name, 0.0, None, targetname=target_file)
                        

                    f.write("file \'{}\'\n".format(target_file))
                    f.close()

                elif temp == vid_name and index == no_videos -1:
                    cam_stop_rec_video_name = cam_clk_data[f"{cam}_file_name"].iloc[-1]
                    # cam_stop_rec_frame = int(cam_clk_data[f"{cam}_frame_idx"].iloc[-1])

                    try:
                        cam_stop_rec_frame = int(cam_clk_data[f"{cam}_frame_idx"].iloc[-1])
                    except Exception as e: # trying to catch exceptions that arise when the video is not long enough (camera drop/frame drop)
                        print(e)
                        cam_stop_rec_frame = None

                    #break_time = cam_clk_data.iloc[timestamp_idx[file_break_idx]][f'{cam}_time_from_vid_start']
                    break_frame = cam_clk_data.iloc[timestamp_idx[file_break_idx]][f'{cam}_frame_idx']
                    break_time = break_frame/precise_cam_fps
                    target_file = temp.split(".")[0]+"_trunc_0.mp4"
                    if cam_stop_rec_frame == None:
                        target_file = os.path.dirname(vid_name)+"/"+os.path.basename(vid_name).split(".")[0]+"_trunc_111.mp4"
                        ffmpeg_extract_subclip(vid_name, 0, None, targetname=target_file)
                    else:
                        ffmpeg_extract_subclip(vid_name, 0, break_time, targetname=target_file)


                    f.write("file \'{}\'\n".format(target_file))
                    f.close()

                    # if cam_stop_rec_frame != None:
                    #     target_file = temp[0]+"/"+temp[1].split(".")[0]+"_trunc_1.mp4"
                    #     #end_time = (cam_stop_rec_frame+1)/camera_fps
                    #     # end_frame = cam_stop_rec_frame
                    ffmpeg_extract_subclip(vid_name, break_time, None, targetname=target_file)
                    
                    file_concat_index+=1
                    str_1 = "%03d"%(file_concat_index)
                    filename_txt = path_concat + f"video_{cam}_{str_1}.txt"
                    f = open(filename_txt, 'w')
                    f.write("file \'{}\'\n".format(target_file))
                    f.close()

                elif index != no_videos -1:
                    target_file = os.path.dirname(vid_name)+"/"+os.path.basename(vid_name).split(".")[0]+"_trunc_111.mp4"
                    ffmpeg_extract_subclip(vid_name, 0.0, None, targetname=target_file)
                    f.write("file \'{}\'\n".format(target_file))

                
        ### For NIDAQ ---------------------------------------------------------------------------------------------------------------------

        # Reading the timestamps data for the camera clock channel 
        #audio_clk_data = pd.read_csv(f"D:/big_setup/experiment_{experiment_no}/camera_timestamps.csv",low_memory=False)
        audio_start_rec_idx = int(cam_clk_data["clk_ch_file_name"][0].split("/")[-1].split("_")[2])
        audio_stop_rec_idx = int(cam_clk_data["clk_ch_file_name"].iloc[-1].split("/")[-1].split("_")[2])
        start_sample_index =  int(cam_clk_data["clk_ch_sample_idx"][0])
        end_sample_index =  int(cam_clk_data["clk_ch_sample_idx"].iloc[-1])

        # Concatenating the microphone channels
        for i in range(no_channels):
            file_break_idx = 1
            file_concat_index = 0 # index for the text files used to concatenate data
            flag = 0 # to indicate camera start record index has reached
            str_1 = "%02d"%(i)
            str_2 = "%03d"%(file_concat_index)
            filename_txt = path_concat + f"channel_{str_1}_file_{str_2}.txt"
            f = open(filename_txt, 'w')
            for idx,j in tqdm.tqdm(enumerate(nidaq_files[i::no_channels])):
                    
                    try:
                        break_channel_idx = int(cam_clk_data.iloc[timestamp_idx[file_break_idx]][f'clk_ch_file_name'].split('_')[-2])
                    except:
                        pass

                    if idx == audio_start_rec_idx:            
                        # Loading the file
                        sampling_rate,ch = wavfile.read(j)
                        ch = ch[start_sample_index:]
                        target_file = j.split(".")[0]+"_trunc_0.wav"
                        wavfile.write(target_file,sampling_rate,ch)
                        flag = 1
                        f.write("file \'{}\'\n".format(target_file))

                    elif idx == break_channel_idx and idx!= audio_stop_rec_idx:

                        break_sample_idx = cam_clk_data.iloc[timestamp_idx[file_break_idx]][f'clk_ch_sample_idx']
            
                        # Loading the file
                        sampling_rate,ch = wavfile.read(j)
                        ch = ch[:break_sample_idx]
                        target_file = j.split(".")[0]+"_trunc_0.wav"
                        wavfile.write(target_file,sampling_rate,ch)

                        f.write("file \'{}\'\n".format(target_file))
                        f.close()

                        
                        target_file = j.split(".")[0]+"_trunc_1.wav"
                        # Loading the file
                        sampling_rate,ch = wavfile.read(j)
                        ch = ch[break_sample_idx:]
                        wavfile.write(target_file,sampling_rate,ch)

                        file_concat_index+=1
                        file_break_idx+=1
                        str_1 = "%02d"%(i)
                        str_2 = "%03d"%(file_concat_index)
                        filename_txt = path_concat + f"channel_{str_1}_file_{str_2}.txt"
                        f = open(filename_txt, 'w')
                        f.write("file \'{}\'\n".format(target_file))
            
                    elif idx != break_channel_idx and idx == audio_stop_rec_idx:
                            
                    
                        target_file = j.split(".")[0]+"_trunc_0.wav"

                        # Loading the file
                        sampling_rate,ch = wavfile.read(j)
                        ch = ch[:end_sample_index]
                        wavfile.write(target_file,sampling_rate,ch)
                        f.write("file \'{}\'\n".format(target_file))
                        f.close()
                        break

                    elif idx == break_channel_idx and idx == audio_stop_rec_idx:

                        break_sample_idx = cam_clk_data.iloc[timestamp_idx[file_break_idx]][f'clk_ch_sample_idx']


                        target_file = j.split(".")[0]+"_trunc_0.wav"
                        # Loading the file
                        sampling_rate,ch = wavfile.read(j)
                        ch = ch[:break_sample_idx]
                        wavfile.write(target_file,sampling_rate,ch)
                        f.write("file \'{}\'\n".format(target_file))
                        f.close()

                    
                        target_file = j.split(".")[0]+"_trunc_1.wav"
                        # Loading the file
                        sampling_rate,ch = wavfile.read(j)
                        ch = ch[break_sample_idx:end_sample_index]
                        wavfile.write(target_file,sampling_rate,ch)

                        file_concat_index+=1
                        str_1 = "%02d"%(i)
                        str_2 = "%03d"%(file_concat_index)
                        filename_txt = path_concat + f"channel_{str_1}_file_{str_2}.txt"
                        f = open(filename_txt, 'w')
                        f.write("file \'{}\'\n".format(target_file))
                        f.close()
                        break

                    elif flag == 1:
                        f.write("file \'{}\'\n".format(j))



        ## Steps done below ---------------------------------------------------------------------------------------------------------------------------------------------
        # - open anaconda prompt
        # - Enter the following commands:
        #     - For video concatenation:
        #         - ```conda activate daq```
        #         - ```ffmpeg -f concat -safe 0 -i {path_to_text_file} -c copy {path_to_output_file}```
                    
        #             eg: ```ffmpeg -f concat -safe 0 -i D:\big_setup\experiment_9\concatenated_data_cam_mic_sync\video_e3v83b3.txt -c copy D:\big_setup\experiment_9\concatenated_data_cam_mic_sync\e3v83b3_ffmpeg.mp4```
        #     - For NIDAQ concatenation:
        #         - - ```conda activate daq```
        #         - ```ffmpeg -f concat -safe 0 -i {path_to_text_file} -c copy {path_to_output_file}```
                    
        #             eg: ```ffmpeg -f concat -safe 0 -i D:\big_setup\experiment_9\concatenated_data_cam_mic_sync\channel_0.txt -c copy D:\big_setup\experiment_9\concatenated_data_cam_mic_sync\channel_0_ffmpeg.wav```

                    
        no_audio_files = int(len(glob.glob(path_concat + "*channel_*.txt"))/no_channels)
        no_video_files = int(len(glob.glob(path_concat + "*video_*.txt"))/len(camera_names))
        timestamp_record_data = cam_clk_data.iloc[timestamp_idx]
        time_starts = []
        time_ends = []
        column_names = [i for i in timestamp_record_data.columns if i.endswith("file_name") and not(i.startswith('clk'))]

        starting_timestamp = datetime.strptime(timestamp_record_data[column_names[0]][0].split('/')[-1].split('-')[1], '%Y%m%dT%H%M%S')

        matched_csv = []
        timestamp_write = starting_timestamp
        f = open(path_final+"README.txt", 'w')
        f.write("The files that are synced are as follows:\n")
        f.write("Video file names - corresponding audio file names - timestamp range of file \n")
        for i in range(no_audio_files):
            temp_0 = []
            temp_1 = []
            temp_2 = [timestamp_write.strftime("%Y-%m-%d %H:%M:%S"),(timestamp_write + timedelta(minutes=file_length/60)).strftime("%Y-%m-%d %H:%M:%S")]
            timestamp_write += timedelta(minutes=file_length/60)
            for j in range(no_channels):
                var_1 = "%03d"%(i)
                var_2 = "%02d"%(j)
                temp_0.append(f"channel_{var_2}_file_{var_1}")
            for j in camera_names:
                var_1 = "%03d"%(i)
                temp_1.append(f"video_{j}_{var_1}")
            f.write(f"[{', '.join(temp_1)}] - [{', '.join(temp_0)}] - [{', '.join(temp_2)}]\n")
            matched_csv.append([temp_1,temp_0,temp_2])
        pd.DataFrame(matched_csv,columns=["video","audio","timestamp"]).to_csv(path_final+"sync.csv", index = False)
        f.close()
        filename_txt = path_concat + "*.txt"
        txt_files = glob.glob(filename_txt)
        video_files = []
        audio_files = []

        for file in txt_files:
            name = file.split('/')[-1].split(".")[0]
            full_path_txt = path_concat+name+".txt"
            # full_path_txt = full_path_txt.replace("/","\\")
            if name[0] == "c":
                full_path_wav = path_final+name+".wav"
                # full_path_wav = full_path_wav.replace("/","\\")
                cmd = [ffmpeg,
                        "-f", "concat",
                        "-safe", "0",
                        "-i",full_path_txt,
                        "-c", "copy",
                        full_path_wav]
                try:
                    result = subprocess.run(cmd, check=True, capture_output = True, text= True)
                    print(f"{full_path_wav} ran successfully")

                except subprocess.CalledProcessError as e:
                    print(f"An error occurred in {full_path_wav}: Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
                # !ffmpeg -f concat -safe 0 -i {full_path_txt} -c copy  {full_path_wav}
                # -rf64 auto
            else:
                full_path_mp4 = path_final+name+".mp4"
                # full_path_mp4 = full_path_mp4.replace("/","\\")
                cmd = [ffmpeg,
                        "-f", "concat",
                        "-safe", "0",
                        "-i",full_path_txt,
                        "-c", "copy",
                        full_path_mp4]
                # !ffmpeg -f concat -safe 0 -i {full_path_txt} -c copy {full_path_mp4}
                try:
                    result = subprocess.run(cmd, check=True, capture_output = True, text= True)
                    print(f"{full_path_mp4} ran successfully.")

                except subprocess.CalledProcessError as e:
                    print(f"An error occurred in {full_path_mp4}: Command '{e.cmd}' returned non-zero exit status {e.returncode}.")

        # Deleting the trunc files if it is already present
        trunc_files = glob.glob(nidaq_folder+"*_trunc*")
        for i in trunc_files:
            os.remove(i) 

        trunc_files = glob.glob(video_folder+"*_trunc*")
        for i in trunc_files:
            os.remove(i) 

# sample run script: python concatenate_data_cam_mic_sync_gily_automated_flatiron.py 493 494 495

if __name__ == "__main__":
    exps = list(map(int, sys.argv[1:]))

    from multiprocessing import Pool, cpu_count
    n_workers = min(len(exps), 4)

    with Pool(n_workers) as pool:
        pool.map(process_experiment, exps)