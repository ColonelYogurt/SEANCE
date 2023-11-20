import numpy as np
from scipy.signal import welch

# EEG data is in `eeg_data` with shape (num_samples, time_points, num_channels)
# Where num_channels = 16


def compute_PSD(data):
    freqs, psd = welch(data, fs=250)  # sample frequency of 250 Hz
    return psd


def compute_sum(data):
    return np.sum(data, axis=1)


def compute_RMS(data):
    return np.sqrt(np.mean(np.square(data), axis=1))


def compute_energy(data):
    return np.sum(data**2, axis=1)


def compute_SD(data):
    return np.std(data, axis=1)


# Feature extraction
features = []
for i in range(16):  # 16 channels
    channel_data = eeg_data[:, :, i]
    features.append(compute_sum(channel_data))
    features.append(compute_RMS(channel_data))
    features.append(compute_PSD(channel_data))
    features.append(compute_energy(channel_data))
    features.append(compute_SD(channel_data))

# Convert to numpy array
features = np.array(features).T
