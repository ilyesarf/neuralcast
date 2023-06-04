import os, sys
import pyedflib
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Load the EDF file
class Load:
    def __init__(self, path, meta_labels=['T0', 'T1', 'T2']):
        self.path = path 
        self.meta_labels = [sys.argv[2]] if len(sys.argv) > 2 else meta_labels

        self.data, self.annotations, self.channel_labels, self.sampling_rate = self.read_file(path)

        self.segments = self.extract_segments()
        
    def read_file(self, path):
        edf_file = pyedflib.EdfReader(self.path) #"data/S001/S001R03.edf")

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


    def extract_segments(self):
        segments = {k: [] for k in self.meta_labels}

        for onset, duration, description in zip(self.annotations[0], self.annotations[1], self.annotations[2]):
            if description in self.meta_labels:
                start_index = int(onset * self.sampling_rate)
                end_index = int((onset + duration) * self.sampling_rate)
                segment_data = np.array(self.data)[:, start_index:end_index]

                segments[description].append(segment_data)

        return segments

    def preprocess_data(self):
        order = 4
        lowcut = 1.0  # Low cutoff frequency in Hz
        highcut = 30.0  # High cutoff frequency in Hz

        # Apply bandpass filter to each channel
        filtered_segments = {k: [] for k in self.meta_labels}
        for label in self.meta_labels:
            for segment in self.segments[label]:
                # Create bandpass filter
                b, a = butter(order, [lowcut, highcut], fs=self.sampling_rate, btype='band')
                # Apply filter to segment
                filtered_segment = filtfilt(b, a, segment)
                filtered_segments[label].append(filtered_segment)

        return filtered_segments


class Plot:
    def __init__(self, path):
        self.load = Load(path)

    def display_data(self, i, ax, data):
        segment = data[i]
        time = np.arange(segment.shape[1]) / self.load.sampling_rate
        for seg_i in range(segment.shape[0]):
            ax.plot(time, segment[seg_i, :], label=self.load.channel_labels[seg_i])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

    def plot(self):
        filtered_segments = self.load.preprocess_data()

        if os.getenv('FILTERED'):
            print("Filtered segments")
            data = filtered_segments
        else:
            print("Unfiltered segments")
            data = self.load.segments
        
        for label in self.load.meta_labels:
            n_segs_display = 3  # number of segments to display
            fig, axs = plt.subplots(n_segs_display, 1, figsize=(8, 6))  # Set the figure size
            fig.suptitle("Segments with label: " + label)  # Set the figure title once
            for i in range(n_segs_display):
                ax = axs[i] if n_segs_display > 1 else axs  # Access the correct subplot
                self.display_data(i, ax, data[label])

            plt.show()

if __name__ == '__main__':
    path = sys.argv[1]
    Plot(path).plot()

