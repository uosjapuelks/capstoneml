import numpy as np
import pandas as pd
import scipy.stats as stats
# from scipy.signal import find_peaks
# from scipy.interpolate import interp1d
# from scipy.fftpack import fft
# import matplotlib.pyplot as plt

def scale_255(x):
    return (x)/255

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

def scale_vals(data):
    data['gx'] = data['gx'].apply(scale_255)
    data['gy'] = data['gy'].apply(scale_255)
    data['gz'] = data['gz'].apply(scale_255)
    data['ax'] = data['ax'].apply(scale_255)
    data['ay'] = data['ay'].apply(scale_255)
    data['az'] = data['az'].apply(scale_255)
    return data

def scaled_data(data, labelled=True):
    ds_X = data[['ax','ay','az','gx','gy','gz']]
    if labelled:
        ds_Y = data['label']

    scaled_X = pd.DataFrame(data=ds_X, columns=['ax','ay','az','gx','gy','gz'])
    if labelled:
        scaled_X['Activity_code'] = ds_Y.values

    return scaled_X

def get_frames(filt_df, frame_size, hop_size, labelled=True):
    N_FEATURES = 6

    frames = []
    labels = []
    for i in range(0, len(filt_df) - frame_size, hop_size):
        gx = filt_df['gx'].values[i: i + frame_size]
        gy = filt_df['gy'].values[i: i + frame_size]
        gz = filt_df['gz'].values[i: i + frame_size]
        ax = filt_df['ax'].values[i: i + frame_size]
        ay = filt_df['ay'].values[i: i + frame_size]
        az = filt_df['az'].values[i: i + frame_size]

        # Retrieve most often used label in this segment
        if labelled:
            label = stats.mode(filt_df['Activity_code'][i: i + frame_size])[0][0]
        frames.append([gx,gy,gz,ax,ay,az])
        if labelled:
            labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    if labelled:
        labels = np.asarray(labels)

    return frames, labels

def extract_frames(raw_df, frame_size, hop_size, start_idx=0):
    N_FEATURES = 6
    frames = []
    for i in range(start_idx, len(raw_df)-frame_size, hop_size):
        gx = raw_df['gx'].values[i: i + frame_size]
        gy = raw_df['gy'].values[i: i + frame_size]
        gz = raw_df['gz'].values[i: i + frame_size]
        ax = raw_df['ax'].values[i: i + frame_size]
        ay = raw_df['ay'].values[i: i + frame_size]
        az = raw_df['az'].values[i: i + frame_size]
        frames.append([gx,gy,gz,ax,ay,az])
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    return frames
