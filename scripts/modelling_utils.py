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


def calculate_emd(res, fs, tAxis, fAxis, kind='quadratic'):
    upper_peaks, _ = find_peaks(res)
    lower_peaks, _ = find_peaks(res)

    f1 = interp1d(upper_peaks/fs,res[upper_peaks], kind=kind, fill_value = 'extrapolate')
    f2 = interp1d(lower_peaks/fs,res[lower_peaks], kind=kind, fill_value = 'extrapolate')

    y1 = f1(tAxis)
    y2 = f2(tAxis)
    y1[0:5] = 0
    y1[-5:] = 0
    y2[0:5] = 0
    y2[-5:] = 0

    avg_envelope = (y1 + y2) / 2

    res1 = avg_envelope
    imf2 = res - avg_envelope
    # Calculate Fast Fourier Transform
    xfft1 = np.abs(fft(res1,1024))
    xfft1 = xfft1[0:512]

    plt.figure(figsize = (20,8))
    plt.subplot(1,2,1)
    plt.plot(tAxis,res1)
    plt.xlabel('Time [s]')
    plt.title('Signal residual in the second iteration')
    plt.subplot(1,2,2)
    plt.plot(fAxis,xfft1)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('Signal residual spectrum in the second iteration')

    return res1, imf2, xfft1
