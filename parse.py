import os, sys
import pyedflib
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Load the EDF file
def read_file(path):
    edf_file = pyedflib.EdfReader(path) #"data/S001/S001R03.edf")

    num_channels = edf_file.signals_in_file
    channel_labels = edf_file.getSignalLabels()
    sampling_rate = edf_file.getSampleFrequency(0)

    data = []
    for i in range(num_channels):
        channel_data = edf_file.readSignal(i)
        data.append(channel_data)

    annotations = edf_file.readAnnotations()

    edf_file.close()

    return data, annotations, channel_labels, sampling_rate

path = sys.argv[1]
meta_labels = [sys.argv[2]] if len(sys.argv) > 2 else ['T0', 'T1', 'T2']
data, annotations, channel_labels, sampling_rate = read_file(path)

def extract_segments():
    segments = []
    labels = []

    for onset, duration, description in zip(annotations[0], annotations[1], annotations[2]):
        if description in meta_labels:
            start_index = int(onset * sampling_rate)
            end_index = int((onset + duration) * sampling_rate)
            segment_data = np.array(data)[:, start_index:end_index]
            segments.append(segment_data)
            labels.append(description)

    return segments, labels

segments, labels = extract_segments()

def preprocess_data():
    order = 4
    lowcut = 1.0  # Low cutoff frequency in Hz
    highcut = 30.0  # High cutoff frequency in Hz

    # Apply bandpass filter to each channel
    filtered_segments = []
    for segment in segments:
        # Create bandpass filter
        b, a = butter(order, [lowcut, highcut], fs=sampling_rate, btype='band')
        # Apply filter to segment
        filtered_segment = filtfilt(b, a, segment)
        filtered_segments.append(filtered_segment)

    return filtered_segments

filtered_segments = preprocess_data()

def display_data(i, ax, data=segments):
    segment = data[i]
    time = np.arange(segment.shape[1]) / sampling_rate
    for seg_i in range(segment.shape[0]):
        ax.plot(time, segment[seg_i, :], label=channel_labels[seg_i])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")


def plot():
    n_segs_display = 3  # number of segments to display
    fig, axs = plt.subplots(n_segs_display, 1, figsize=(8, 6))  # Set the figure size

    if os.getenv('FILTERED'):
        print("Filtered segments")
        data = filtered_segments
    else:
        print("Unfiltered segments")
        data = segments
    
    fig.suptitle("Segments with label: " + ", ".join(set(labels)))  # Set the figure title once

    for i in range(n_segs_display):
        ax = axs[i] if n_segs_display > 1 else axs  # Access the correct subplot
        display_data(i, ax, data)

    plt.tight_layout()
    plt.show()

plot()

