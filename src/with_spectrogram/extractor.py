"""
Módulo responsável pela extração de espectrogramas com janelas deslizantes
e aplicação de técnicas de aumento de dados em arquivos de áudio.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa
import numpy as np
import cv2
from tqdm import tqdm


N_JOBS = 8
AUDIO_FOLDER = "./data"
WINDOW_DURATION = 10.0  # segundos
STEP_DURATION = 5       # segundos
SR = 22050
N_FFT = 2048
HOP_LENGTH = 1024
N_MFCC = 128            # número de coeficientes MFCC
TARGET_FRAMES = 128     # número de frames no tempo


def extract_spectrogram(y, sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH,
                        target_frames=TARGET_FRAMES):
    """
    Extrai e normaliza o espectrograma MFCC de um sinal de áudio.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc_resized = cv2.resize(mfcc, (target_frames, n_mfcc) # pylint: disable=no-member
,
                              interpolation=cv2.INTER_LINEAR)  # pylint: disable=no-member
    mfcc_resized = (mfcc_resized - mfcc_resized.mean()) / (mfcc_resized.std() + 1e-9)
    return mfcc_resized.astype(np.float32)


def augment_audio(y, sr):
    """
    Aplica técnicas de aumento de dados no sinal de áudio:
    - Pitch shift
    - Time stretch
    - Adição de ruído branco
    """
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


def sliding_windows(y, sr, window_duration=WINDOW_DURATION,
                    step_duration=STEP_DURATION):
    """
    Gera janelas deslizantes do áudio para segmentação temporal.
    """
    window_len = int(sr * window_duration)
    step_len = int(sr * step_duration)
    windows = []

    for start in range(0, len(y) - window_len + 1, step_len):
        windows.append(y[start:start + window_len])

    return windows


def process_file(file_path):
    """
    Processa um arquivo de áudio:
    - Divide em janelas
    - Aplica aumentação
    - Extrai espectrogramas MFCC
    """
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

    return np.array(mfccs), np.array(labels)


def process_spectrograms_for_all_files():
    """
    Processa todos os arquivos de áudio na pasta de dados,
    gerando arrays X (features) e Y (labels) e salvando em disco.
    """
    genres_folders = [
        os.path.join(AUDIO_FOLDER, f)
        for f in os.listdir(AUDIO_FOLDER)
        if os.path.isdir(os.path.join(AUDIO_FOLDER, f))
    ]

    all_files = []
    for folder in genres_folders:
        all_files.extend([os.path.join(folder, f) for f in os.listdir(folder)])

    x_all = []
    y_all = []

    with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {executor.submit(process_file, f): f for f in all_files}
        for future in tqdm(as_completed(futures), total=len(all_files)):
            mfccs, labels = future.result()
            x_all.extend(mfccs)
            y_all.extend(labels)

    x_all = np.array(x_all)
    y_all = np.array(y_all)

    np.save("src/with_spectrogram/X.npy", x_all)
    np.save("src/with_spectrogram/Y.npy", y_all)


if __name__ == "__main__":
    process_spectrograms_for_all_files()
