import nidaqmx.system
from scipy.io import wavfile
import numpy as np
import nidaqmx as nidaq
import os
import pandas as pd
import pickle
from multiprocessing import Process, Value, Array
import ctypes as ct
import time
from nidaqmx.constants import Level, LineGrouping, TerminalConfiguration


import matplotlib
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, freqz
import math
import scipy
import cv2
from collections import deque


import tkinter as Tk
from tkinter import simpledialog
from datetime import datetime
import glob
import re


plt.style.use('dark_background')



# TO DO if experiment folder exists, iterate to the next folder number

# clock terminal
clk_term = "/PXI1Slot2/PFI3"

# counter to generate the clock
clk_cntr = "PXI1Slot2/ctr0"

time_of_recording = int(time.time()) # To have the time for the experiment



# Initializing global variables

# Configure the acquisition parameters
sampling_rate = 125000  # Hz
duration_read_buffer = 60  # seconds - how long the read buffer needs to store data for WHI
duration_store_buffer = 5*60 # seconds - how long the store buffer needs to store data for

# old 
# channels = ["PXI1Slot2/ai3", "PXI1Slot3/ai1", # "PXI1Slot2/ai3","PXI1Slot3/ai1" - Ralph Mic 11 and 17
#             "PXI1Slot3/ai4","PXI1Slot3/ai5", # "PXI1Slot3/ai4","PXI1Slot3/ai5" - Gily Mic 4 and 5 
#             "PXI1Slot3/ai6", # "PXI1Slot3/ai6" - Nest Audio - Mic 6
#             "PXI1Slot4/ai0", # "PXI1Slot4/ai0" - Burrow Audio - Mic 21
#             "PXI1Slot4/ai5", 
#             "PXI1Slot4/ai6","PXI1Slot4/ai7"]

# new 
#--------------------------------------------------------------------------------Pre run check (channels included)--------------------------------------------------------
# SSL project channels
channels = ["PXI1Slot2/ai6",   #  mic 10
            "PXI1Slot2/ai7", # mic 11
            "PXI1Slot3/ai0",  # mic 12
            "PXI1Slot3/ai1",  # mic 13
            "PXI1Slot3/ai2", # mic 14
            "PXI1Slot3/ai3", # mic 15
            "PXI1Slot3/ai4", # mic 16
            "PXI1Slot3/ai5",  # mic 17
            "PXI1Slot3/ai6", # mic 18
            "PXI1Slot3/ai7", # mic 19
            "PXI1Slot4/ai7"] # Camera TTL

# Digital input channels
channels_di_slot_2 = [] # No white matter TTL
# channels_di_slot_2 = ["PXI1Slot2/port0/line0"] # digital inputs --don't use this for white matter transceiver accuracy will be low due to read_idx mismatch
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

 
# Variables to store data in chunks for the read buffer
num_samples = sampling_rate * duration_read_buffer
chunk_size = 1000 # num_samples needs to be a multiple of chunk_size
num_chunks = int(num_samples / chunk_size)

# Variables to store data for the store buffer
num_samples_store = int(sampling_rate * duration_store_buffer)
num_chunks_store = int(num_samples_store / chunk_size)


# Parameters for Spectrogram plot
sr = sampling_rate # config['microphone_sample_rate']
nfft =  512 #config['spectrogram_nfft']
n_overlap = 0 #config['spectrogram_noverlap']
nperseg = nfft # config['spectrogram_nfft']
# spec_lower_cutoff =  10e-20 # config['spectrogram_lower_cutoff']
spec_lower_cutoff =  -250 # config['spectrogram_lower_cutoff']
spec_upper_cutoff =  -120 # config['spectrogram_upper_cutoff']
spec_red_color = np.array([87, 66, 206]).reshape((1, 1, 3)),  # BGR order
spec_blue_color = np.array([218, 214, 109]).reshape((1, 1, 3)),  # BGR order
spec_white_color = np.array([255, 255, 255]).reshape((1, 1, 3)),  # BGR order
spec_black_color = np.array([0, 0, 0]).reshape((1, 1, 3)),  # BGR order
spec_mic_diff_thresh = 450e-11
n_channels = len(channels)-1
spec_buffer_len = 576 # number of chunks - 576 is 3 seconds worth
mic_deque = deque(maxlen=spec_buffer_len) # 3 seconds worth



# Functions to sort file names correctly
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def write_log(log_path):
    tkinstance = Tk.Tk()
    tkinstance.withdraw()
    log = simpledialog.askstring("Logger", "Enter the log, type in q/Q if done by mistake",parent=tkinstance)
    if log == 'q' or log == "Q":
        print("Didn't save log")
    else:
        with open(log_path, 'a') as file:
            file.write(datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S")+">"+log +"\n")
        print("Saved log")


#---------------------------------------------------------------pre run check (spectrogram plot)-------------------------------------------------------------------------------------
def calc_spec_frame_segment_mono(all_audio):
    avg_audio_gily = np.mean(all_audio[:10,:], axis=0)


    try:

        _, _, spec = scipy.signal.spectrogram(avg_audio_gily,
            fs= sr,
            nfft= nfft,
            noverlap= n_overlap,
            nperseg= nperseg)
        
    except UserWarning as e:    
        print(e)
        return []
        
    
    
    # spec = 20*np.log10(spec+1e-12)
    minavg, maxavg = spec_lower_cutoff, spec_upper_cutoff

    spec = 20*np.log10(spec+1e-12)
    # print("Before cutoff")
    # print(np.min(spec),np.max(spec))


    spec = np.clip(spec, minavg, maxavg)
    spec = (spec - minavg) * 255 / (maxavg - minavg)

    # print("After clip")
    # print(np.min(spec),np.max(spec))

    return spec[::-1].astype(np.uint8)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def logging(flag_end, log_path, experiment_no):
    print("Opening logging functionality")
    with open(log_path, 'w') as file:
            t = datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S")
            file.write(f"Time>Log for experiment number {experiment_no}\n")
    win = Tk.Tk()
    win.title('Logging & Stopping experiment')
    win.geometry('500x100')
    try:
        # Logging button
        log = Tk.Button(win, text = 'Log', command = lambda: write_log(log_path))
        # Quit button
        quit_B = Tk.Button(win, text = 'Quit Experiment', command = win.quit)

        log.pack(side = Tk.BOTTOM, fill = Tk.X, expand = True)
        quit_B.pack(side = Tk.BOTTOM, fill = Tk.X, expand = True)
        win.mainloop()
        flag_end.value = 1  
    except KeyboardInterrupt:
        flag_end.value = 1
        print("Logging stopped")
    except Exception as e:
        print(e)
    



def spec_plot(read_buffer,read_idx,flag_end):
    print("Spectrogram Plot Started")
    np_arr = np.frombuffer(read_buffer.get_obj()) # making the read buffer array a numpy array
    final_arr = np_arr.reshape(len(channels+channels_di_slot_2),num_samples) # make it two-dimensional


    start = time.time()
    last_printed = 0
    while flag_end.value == 0:
        try:
                # Update timer
                elapsed = int(time.time() - start)
                hours = elapsed // 3600
                minutes = elapsed // 60 - 60 * hours
                seconds = elapsed % 60
                if hours > 0:
                    timer_string = 'Timer: {}:{:>02}:{:>02}'.format(hours, minutes, seconds)
                else:
                    timer_string = 'Timer: {}:{:>02}'.format(minutes, seconds)
                if elapsed >= last_printed + 5:
                    print(timer_string)
                    last_printed = elapsed
                
                if read_idx.value == 0:
                    continue
                else:
        #-----------------------------------------------------------------pre run check (spectrogram plotting) -------------------------------------------------------------------------
                    color_frame  = calc_spec_frame_segment_mono(final_arr[:-1,(read_idx.value-1)*chunk_size:read_idx.value*chunk_size])
                    
                    if type(color_frame) == list:
                        continue
                

                mic_deque.append(color_frame)
                
                complete_image = np.ascontiguousarray(np.concatenate(mic_deque, axis=1), dtype=np.uint8)

            

                # Display timer on spectrogram window
                text_color =  255
                cv2.putText(
                    complete_image,
                    timer_string,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    text_color,
                    2)
                cv2.imshow('Gily Rig Spectrogram', complete_image)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                cv2.waitKey(1)
        except KeyboardInterrupt:
            flag_end.value = 1
            print("Spectrogram plotting stopped")
            break
        except Exception as e:
            print(e)
            continue

    print("Exited out of Spectrogram plotting")
    



def read_NIDAQ(read_buffer,flag_end,read_idx,flag_reset):
    print("Recording Started")
    np_arr = np.frombuffer(read_buffer.get_obj()) # making the read buffer array a numpy array
    final_arr = np_arr.reshape(len(channels+channels_di_slot_2),num_samples) # make it two-dimensional
    
    # Data is continually read from the NIDAQ until keyboard interrupt is pressed

    # Create the task and configure the acquisition
    with nidaqmx.Task() as task:
        for idx,channel in enumerate(channels):

            task.ai_channels.add_ai_voltage_chan(channel)


        task.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        # Start the acquisition
        task.start()

        try:
            while flag_end.value == 0: 
                chunks = task.read(number_of_samples_per_channel=chunk_size)
                for j in range(len(channels)):
                    final_arr[j][read_idx.value*chunk_size:(read_idx.value+1)*chunk_size] = chunks[j]

                # Resetting the index if the last sample is reached (it's a circular buffer)
                if (read_idx.value+1)*chunk_size == num_samples:
                    read_idx.value = 0

                    # letting the other process know that the index has been reset to change the condition to store data
                    flag_reset.value = 1
                else:
                    # updating the read index
                    read_idx.value = read_idx.value + 1
            
            task.stop()
            print("Recording stopped")

        except KeyboardInterrupt:
            flag_end.value = 1
            task.stop()
            print("Recording stopped")
            

# Function to generate clock signal on a PFI terminal
def gen_clock(flag_end):
    print("Clock Started")
    
    # Clock is continually generated from the NIDAQ until keyboard interrupt is pressed

    # Create the task and configure the generation
    with nidaqmx.Task() as task:
        
        channel = task.co_channels.add_co_pulse_chan_freq(clk_cntr, idle_state=Level.LOW, initial_delay=0.0, freq=sampling_rate, duty_cycle=0.5)

        channel.co_pulse_term = clk_term
        task.timing.cfg_implicit_timing(sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        task.start()
        try:
            while flag_end.value == 0:
                continue
            
            task.stop()
            print("Recording stopped from clk loop")
        except KeyboardInterrupt:
            flag_end.value = 1
            task.stop()
            print("Recording stopped from clk loop")


            
def digital_in_slot_2(read_buffer,flag_end,read_idx,flag_reset):
    print("Digital Input Slot 2 Recording Started")
    np_arr = np.frombuffer(read_buffer.get_obj()) # making the read buffer array a numpy array
    final_arr = np_arr.reshape(len(channels+channels_di_slot_2),num_samples) # make it two-dimensional
    
    # Data is continually read from the NIDAQ until keyboard interrupt is pressed

    # Create the task and configure the acquisition
    with nidaqmx.Task() as tsk:

        if len(channels_di_slot_2) > 0:

            for idx,channel in enumerate(channels_di_slot_2):

                tsk.di_channels.add_di_chan(channel, line_grouping=LineGrouping.CHAN_PER_LINE)


            # Start the acquisition
            tsk.start()

            try:
                while flag_end.value == 0: 
                    chunks = tsk.read(number_of_samples_per_channel=chunk_size)
                    
                    # for j in range(len(channels),len(channels+channels_di_slot_2)):
                    #     final_arr[j][read_idx.value*chunk_size:(read_idx.value+1)*chunk_size] = chunks[j]
                    read_idx_ = read_idx.value # we're doing this because the read_idx value gets updated very quick - causes lower accuracy based on chunk size
                    final_arr[len(channels):len(channels+channels_di_slot_2),read_idx_*chunk_size:(read_idx_+1)*chunk_size] = chunks

                tsk.stop()
                print("Recording stopped for digital input slot 2")


            except KeyboardInterrupt:
                flag_end.value = 1
                tsk.stop()
                print("Recording stopped for digital input slot 2")


def store_data(read_buffer,flag_end,read_idx,flag_reset,path_name):
    
    
    np_arr = np.frombuffer(read_buffer.get_obj()) # making the array a numpy array
    final_arr = np_arr.reshape(len(channels+channels_di_slot_2),num_samples) # make it two-dimensional
    store_buffer = np.zeros((len(channels+channels_di_slot_2),num_samples_store)) # Buffer to store 30 mins of data
    store_idx = 0 # To store the index value to save to in the store buffer
    store_read_buffer_idx = 0 # To store the last read buffer value saved to the store buffer
    file_idx = 0 # index to store multiple files 

    try:
        while flag_end.value == 0:
            flag_store_ = 0 # To check whether there is new data to store to the store buffer

            if flag_reset.value == 0 and store_read_buffer_idx < read_idx.value:
                    flag_store_ = 1
            elif flag_reset.value == 1 and store_read_buffer_idx > read_idx.value: 
                    flag_store_ = 1


            if flag_store_ and store_idx<num_chunks_store:
            
                for j in range(len(channels+channels_di_slot_2)): 
                    store_buffer[j][store_idx*chunk_size:(store_idx+1)*chunk_size] = final_arr[j][store_read_buffer_idx*chunk_size:(store_read_buffer_idx+1)*chunk_size]
                
                store_read_buffer_idx+=1
                store_idx+=1

                # going back to the start of the read buffer if the last chunk of the read buffer is stored
                if (store_read_buffer_idx+1)*chunk_size == num_samples:
                    time.sleep(0.5)
                    store_read_buffer_idx = 0
                    flag_reset.value = 0 # Resetting the flag to 0 

            elif store_idx==num_chunks_store: # Checking to see if the store buffer is full
                t0 = time.time()
                store_idx = 0

                # Saving the file
                # Save the data as a pickle file in the data folder
                # filename = os.path.expanduser(path_name + "acquisition_data_"+str(file_idx)+".pkl")
                # with open(filename, 'wb') as file: 
                #     pickle.dump(store_buffer,file)
                # print(f"Data saved to {filename}")

                # Save the data of each channel in a separate wav file
                for j in range(len(channels+channels_di_slot_2)):
                    filename_wav = os.path.expanduser(path_name + "acquisition_data_"+str(file_idx)+"_"+str(j)+".wav")
                    wavfile.write(filename_wav, sampling_rate, store_buffer[j].astype("float32"))
                    print(f"Data saved to {filename_wav}")
                
                file_idx+=1
                t1 = time.time()
                print("Time to write data in seconds:", t1-t0)

            else:
                continue

        
        t0 = time.time()
        print("Saving the remainder data")
        for j in range(len(channels+channels_di_slot_2)):
            store_buffer[j][store_idx*chunk_size+1] = 10.0



        # Save the data of each channel in a separate wav file
        for j in range(len(channels+channels_di_slot_2)):
            filename_wav = os.path.expanduser(path_name + "acquisition_data_"+str(file_idx)+"_"+str(j)+".wav")
            wavfile.write(filename_wav, sampling_rate, store_buffer[j].astype("float32"))
            print(f"Data saved to {filename_wav}")


        print("Checking the end flag: ",flag_end.value)
        t1 = time.time()
        print("Time to write data in seconds:", t1-t0)
                

    except KeyboardInterrupt:
        t0 = time.time()
        flag_end.value = 1
        print("Saving the remainder data")
        for j in range(len(channels+channels_di_slot_2)):
            store_buffer[j][store_idx*chunk_size+1] = 10.0



        # Save the data of each channel in a separate wav file
        for j in range(len(channels+channels_di_slot_2)):
            filename_wav = os.path.expanduser(path_name + "acquisition_data_"+str(file_idx)+"_"+str(j)+".wav")
            wavfile.write(filename_wav, sampling_rate, store_buffer[j].astype("float32"))
            print(f"Data saved to {filename_wav}")


        print("Checking the end flag: ",flag_end.value)
        t1 = time.time()
        print("Time to write data in seconds:", t1-t0)
        


 # Main function to call the processes       
if __name__ == '__main__':

    

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
        experiment_no = last_exp_no_in_D_drive+1
    else:
        experiment_no = last_exp_no_in_C_drive+1

    print("***********************************************************************************")
    print("THE EXPERIMENT NUMBER IS: ", experiment_no)
    print("***********************************************************************************")

    # Creating a folder for the experiment
    path_name = "./data/channel/experiment_{}/".format(experiment_no)
    try:
        os.makedirs(path_name)
    except:
        pass

    experiment_time = datetime.now().strftime(f'%Y-%m-%d_%H-%M-%S')
    log_name = f"experiment_{experiment_no}_log_{experiment_time}.txt"
    log_path = os.path.join(path_name,log_name)


    read_buffer = Array(ct.c_double, len(channels+channels_di_slot_2)*num_samples) # Creating a shared "read buffer" to store the data read from the NIDAQ 


    flag_end = Value('i', 0) # Shared memory flag to indicate when recording stops
    read_idx = Value('i', 0) # Shared memory read index to indicate which index value is being written
    flag_reset = Value('i', 0) # Shared memory reset index to indicate whether the circular "read buffer" index has been reset



    p1 = Process(target=read_NIDAQ, args = (read_buffer,flag_end,read_idx,flag_reset,))
    p2 = Process(target=store_data, args = (read_buffer,flag_end,read_idx,flag_reset,path_name))
    p3 = Process(target=spec_plot, args = (read_buffer,read_idx,flag_end,))
    p4 = Process(target=logging, args = (flag_end,log_path,experiment_no))
    p5 = Process(target=gen_clock, args = (flag_end,))
    p6 = Process(target=digital_in_slot_2, args = (read_buffer,flag_end,read_idx,flag_reset,))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    
   

    
