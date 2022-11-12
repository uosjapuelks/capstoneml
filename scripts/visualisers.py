import matplotlib.pyplot as plt
import numpy as np

def plot_history(history, epochs):
    # Plot training & validation accuracy values
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & Validation loss values
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=6, figsize=(18,12), sharex=True)
    plot_axis(ax0, data['time'], data['gx'], 'GX-AXIS')
    plot_axis(ax1, data['time'], data['gy'], 'GY-AXIS')
    plot_axis(ax2, data['time'], data['gz'], 'GZ-AXIS')
    plot_axis(ax3, data['time'], data['ax'], 'AX-AXIS')
    plot_axis(ax4, data['time'], data['ay'], 'AY-AXIS')
    plot_axis(ax5, data['time'], data['az'], 'AZ-AXIS')
    
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_activity_v2(activity, data):
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(18,6), sharex=True)
    plot_axis(ax0, data['time'], data['gx'], 'GX-AXIS', colour='r')
    plot_axis(ax0, data['time'], data['gy'], 'GY-AXIS', colour='g')
    plot_axis(ax0, data['time'], data['gz'], 'GZ-AXIS', colour='b')
    plot_axis(ax1, data['time'], data['ax'], 'AX-AXIS', colour='r')
    plot_axis(ax1, data['time'], data['ay'], 'AY-AXIS', colour='g')
    plot_axis(ax1, data['time'], data['az'], 'AZ-AXIS', colour='b')
    
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_activity_v3(activity, data, figsize=(6,4)):
    fig, (ax0) = plt.subplots(nrows=1, figsize=figsize, sharex=True)
    plot_axis(ax0, data['time'], data['gx'], label='GX-AXIS', colour='red')
    plot_axis(ax0, data['time'], data['gy'], label='GY-AXIS', colour='green')
    plot_axis(ax0, data['time'], data['gz'], label='GZ-AXIS', colour='blue')
    plot_axis(ax0, data['time'], data['ax'], label='AX-AXIS', colour='pink')
    plot_axis(ax0, data['time'], data['ay'], label='AY-AXIS', colour='cyan')
    plot_axis(ax0, data['time'], data['az'], label='AZ-AXIS', colour='orange')
    
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_full(data):
    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=6, figsize=(18,12), sharex=True)
    plot_axis(ax0, data['time'], data['gx'], 'GX-AXIS')
    plot_axis(ax1, data['time'], data['gy'], 'GY-AXIS')
    plot_axis(ax2, data['time'], data['gz'], 'GZ-AXIS')
    plot_axis(ax3, data['time'], data['ax'], 'AX-AXIS')
    plot_axis(ax4, data['time'], data['ay'], 'AY-AXIS')
    plot_axis(ax5, data['time'], data['az'], 'AZ-AXIS')
    
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle("DATAFRAME")
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_full_v2(data):
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(18,6), sharex=True)
    plot_axis(ax0, data['time'], data['gx'], 'GX-AXIS', label='gx', colour='g')
    plot_axis(ax0, data['time'], data['gy'], 'GY-AXIS', label='gy', colour='r')
    plot_axis(ax0, data['time'], data['gz'], 'GZ-AXIS', label='gz', colour='b')
    plt.legend()
    plt.grid()

    plot_axis(ax1, data['time'], data['ax'], 'AX-AXIS', label='ax', colour='g')
    plot_axis(ax1, data['time'], data['ay'], 'AY-AXIS', label='ay', colour='r')
    plot_axis(ax1, data['time'], data['az'], 'AZ-AXIS', label='az', colour='b')
    plt.legend()
    plt.grid()
    
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle("DATAFRAME")
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_full_v3(data):
    fig, (ax0) = plt.subplots(nrows=2, figsize=(18,6), sharex=True)
    plot_axis(ax0, data['time'], data['gx'], 'GX-AXIS', label='gx', colour='red')
    plot_axis(ax0, data['time'], data['gy'], 'GY-AXIS', label='gy', colour='blue')
    plot_axis(ax0, data['time'], data['gz'], 'GZ-AXIS', label='gz', colour='green')
    plot_axis(ax0, data['time'], data['ax'], 'AX-AXIS', label='ax', colour='pink')
    plot_axis(ax0, data['time'], data['ay'], 'AY-AXIS', label='ay', colour='cyan')
    plot_axis(ax0, data['time'], data['az'], 'AZ-AXIS', label='az', colour='orange')
    plt.legend()
    plt.grid()
    
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle("DATAFRAME")
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_features(activity, data):
    fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(nrows=10, figsize=(18,12), sharex=True)
    plot_axis(ax0, data['time'], data['max_a'], 'MAX-A')
    plot_axis(ax1, data['time'], data['min_a'], 'MIN-A')
    plot_axis(ax2, data['time'], data['range_a'], 'RANGE-A')
    plot_axis(ax3, data['time'], data['mean_a'], 'MEAN-A')
    plot_axis(ax4, data['time'], data['std_a'], 'STD-A')
    plot_axis(ax5, data['time'], data['max_g'], 'MAX-G')
    plot_axis(ax6, data['time'], data['min_g'], 'MIN-G')
    plot_axis(ax7, data['time'], data['range_g'], 'RANGE-G')
    plot_axis(ax8, data['time'], data['mean_g'], 'MEAN-G')
    plot_axis(ax9, data['time'], data['std_g'], 'STD-G')
    # plot_axis(ax10, data['time'], data['accel_z_peaks'], 'AZ-PEAK')
    
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_axis(ax, x, y, title="", label='', colour='g'):
    if label == '':
        ax.plot(x,y,colour)
    else:
        ax.plot(x, y, colour,label=label)
        ax.legend(loc='upper left')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    # ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    # ax.set_xlim([min(x), max(x)])
    ax.set_ylim([0, 255])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

