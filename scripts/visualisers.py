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
    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=6, figsize=(15,7), sharex=True)
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

def plot_axis(ax, x, y, title):
    ax.plot(x, y,'g')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

