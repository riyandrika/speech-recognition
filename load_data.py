import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

HALFPATH = r"C:\Users\ASUS\Downloads\tensorflow-speech-recognition-challenge\train\train\audio"

path_data = pd.DataFrame()
audio_series = []
labels_series = []
labels_list = os.listdir(HALFPATH)
labels_list = [label for label in labels_list if "background" not in label]
for label in labels_list:
    audio_files = os.listdir(HALFPATH + "/" + label)
    labels = [label] * len(audio_files)
    audio_series.extend(audio_files)
    labels_series.extend(labels)
path_data["audio"] = pd.Series(audio_series)
path_data["labels"] = pd.Series(labels_series)
sampled_data = []

def padding(time_array, total_width = 16000):
    time_array = np.pad(time_array, pad_width = (0,total_width - len(time_array)), mode = "constant", constant_values = 0)
    return time_array

def get_time(audio_file, label):
    fullpath = HALFPATH + "/" + label + "/" + audio_file
    sample_rate, time_sample = wavfile.read(fullpath)
    time_sample = padding(time_sample)
    return time_sample

def get_spectro(audio_file, label, sample_rate = 16000):
    freq_array, time_array, spectrogram_sample = spectrogram(get_time(audio_file,label),
                                                             sample_rate,
                                                             nperseg = 20,
                                                             noverlap = 2)
    return spectrogram_sample

for index, row in path_data.iterrows():
    sampled_data.append(get_spectro(row["audio"], row["labels"]))

sampled_data = np.dstack(sampled_data)

encoder = LabelEncoder()
labels = encoder.fit_transform(path_data["labels"])

def save_to_npz(array, labels, filename):
    np.savez(filename, array, labels)

save_to_npz(array = sampled_data, labels = labels, filename = "spectrogram.npz")