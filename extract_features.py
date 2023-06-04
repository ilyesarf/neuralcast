from parse import Load
from scipy.signal import welch
from scipy.stats import kurtosis, skew
import numpy as np

load = Load()

filtered_segments = load.preprocess_data()

def extract_features(segments):
    features = {k: [] for k in load.meta_labels}
    
    for label in load.meta_labels:
        for segment in segments[label]:
            #segment = segment.ravel()
            # Compute Power Spectral Density (PSD)
            f, psd = welch(segment, fs=load.sampling_rate)
            
            # Statistical features
            mean = np.mean(segment, axis=1)
            std = np.std(segment, axis=1)
            min_val = np.min(segment, axis=1)
            max_val = np.max(segment, axis=1)
            median = np.median(segment, axis=1)
            kurt = kurtosis(segment, axis=1)
            skewness = skew(segment, axis=1)
            
            # Concatenate features
            segment_features = np.concatenate((mean, std, min_val, max_val, median, kurt, skewness, psd), axis=None)
            features[label].append(segment_features)
    
    return features

# Example usage:
extracted_features = extract_features(filtered_segments)

# Print the extracted features
for label in load.meta_labels:
    for i, segment_features in enumerate(extracted_features[label]):
        print(f"Segment {i+1} features:")
        print(segment_features)
        print("--------")
