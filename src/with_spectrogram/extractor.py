# extractor_augmented_sliding.py
import librosa
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cv2

N_JOBS = 8
AUDIO_FOLDER = "./data"
WINDOW_DURATION = 10.0  # segundos
STEP_DURATION = 5       # segundos
SR = 22050
N_FFT = 2048
HOP_LENGTH = 1024
N_MFCC = 128  # agora 128 coeficientes
TARGET_FRAMES = 128  # queremos 128 frames no tempo

def extract_spectrogram(y, sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, target_frames=TARGET_FRAMES):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc_resized = cv2.resize(mfcc, (target_frames, n_mfcc), interpolation=cv2.INTER_LINEAR)
    mfcc_resized = (mfcc_resized - mfcc_resized.mean()) / (mfcc_resized.std() + 1e-9)
    return mfcc_resized.astype(np.float32)

def augment_audio(y, sr):
    augmented = [y]
    # Pitch shift
    for n_steps in [-2, 2]:
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps))

    # Time stretch
    for rate in [0.9, 1.1]:
        augmented.append(librosa.effects.time_stretch(y, rate=rate))

    # White noise
    wn = 0.005 * np.random.randn(len(y))
    augmented.append(y + wn)
    return augmented

def sliding_windows(y, sr, window_duration=WINDOW_DURATION, step_duration=STEP_DURATION):
    window_len = int(sr * window_duration)
    step_len = int(sr * step_duration)
    windows = []
    for start in range(0, len(y) - window_len + 1, step_len):
        windows.append(y[start:start+window_len])
    return windows

def process_file(file_path):
    label = os.path.basename(os.path.dirname(file_path))
    y, sr = librosa.load(file_path, sr=SR, mono=True)

    windows = sliding_windows(y, sr)
    mfccs = []
    labels = []

    for win in windows:
        augmented_segments = augment_audio(win, sr)
        for seg in augmented_segments:
            mfcc = extract_spectrogram(seg, sr)
            mfccs.append(mfcc)
            labels.append(label)

    mfccs = np.array(mfccs)
    labels = np.array(labels)
    return mfccs, labels


def process_spectrograms_for_all_files():
    genres_folders = [os.path.join(AUDIO_FOLDER, f) for f in os.listdir(AUDIO_FOLDER) if os.path.isdir(os.path.join(AUDIO_FOLDER, f))]
    all_files = []
    for folder in genres_folders:
        all_files.extend([os.path.join(folder, f) for f in os.listdir(folder)])

    X_all = []
    Y_all = []

    with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {executor.submit(process_file, f): f for f in all_files}
        for future in tqdm(as_completed(futures), total=len(all_files)):
            mfccs, labels = future.result()
            X_all.extend(mfccs)
            Y_all.extend(labels)

    X_all = np.array(X_all)
    Y_all = np.array(Y_all)
    np.save("src/with_spectrogram/X.npy", X_all)
    np.save("src/with_spectrogram/Y.npy", Y_all)

if __name__ == "__main__":
    process_spectrograms_for_all_files()
