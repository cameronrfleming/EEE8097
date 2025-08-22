import librosa
import os
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Paths to the data
audio_files_path = 'env_data/ESC-50-master/audio/'  # Path to audio files
meta_data_path = 'env_data/ESC-50-master/meta/esc50.csv'  # Path to metadata CSV

def extract_features(audio_file_name):
    audio_amplitude, sample_rate = librosa.load(audio_file_name, res_type='kaiser_fast')

    # Apply frequency filters
    b, a = signal.butter(2, 4000 / (sample_rate / 2), btype='low')
    audio_amplitude = signal.filtfilt(b, a, audio_amplitude)
    b, a = signal.butter(2, 300 / (sample_rate / 2), btype='high')
    audio_amplitude = signal.filtfilt(b, a, audio_amplitude)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_amplitude, sr=sample_rate, n_mfcc=13)
    delta_mfccs = librosa.feature.delta(mfccs.T)
    delta2_mfccs = librosa.feature.delta(mfccs.T, order=2)
    mfccs_combined = np.mean(np.concatenate([mfccs, delta_mfccs.T, delta2_mfccs.T], axis=0).T, axis=0)

    #Log-Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_amplitude, sr=sample_rate, n_mels=64)
    mel_scaled = np.mean(mel_spectrogram.T, axis=0)

    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_amplitude, sr=sample_rate)
    spectral_contrast_scaled = np.mean(spectral_contrast.T, axis=0)

    # Additional spectral and zero crossing rate features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_amplitude, sr=sample_rate))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_amplitude, sr=sample_rate))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_amplitude, sr=sample_rate))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_amplitude))

    # Wrap scalar features as arrays
    # Combine all features
    features = np.concatenate([
        mfccs_combined,
        mel_scaled,
        spectral_contrast_scaled,
        [spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate],
    ])
    return features

def load_filenames_and_encoded_classlabels():
   # Load metadata
    meta_data = pd.read_csv(meta_data_path)
    meta_data = meta_data[meta_data.esc10 == True]

    # Extract raw features and labels
    total_files = len(meta_data)
    filenames = []
    labels = []
    for index, row in tqdm(meta_data.iterrows(), total=total_files, desc='Processing files'):
        audio_file_name = os.path.join(audio_files_path, row['filename'])
        filenames.append(audio_file_name)
        class_label = row['category']
        labels.append(class_label)

    # Convert to numpy arrays
    labels = np.array(labels)

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    return filenames, labels_encoded

def extract_features_from_files(filenames):
    # Extract raw features
    total_files = len(filenames)
    features_list = []
    for filename in tqdm(filenames, total=total_files, desc='Processing files'):
        # audio_file_name = os.path.join(audio_files_path, filename)
        features = extract_features(filename)
        features_list.append(features)

    # Convert to numpy arrays
    features = np.array(features_list)
    return features


def split_test_and_training_data(filenames, classname):
    # Split dataset
    X_train_filenames, X_test_filenames, y_train_classnames, y_test_classnames = train_test_split(
        filenames, classname, test_size=0.2, random_state=42, stratify=classname
    )
    return X_train_filenames, X_test_filenames, y_train_classnames, y_test_classnames

def load_data_and_split():
    filenames, classnames = load_filenames_and_encoded_classlabels()
    X_train_filenames, X_test_filenames, y_train_classnames, y_test_classnames = split_test_and_training_data(filenames, classnames)
    return X_train_filenames, X_test_filenames, y_train_classnames, y_test_classnames
