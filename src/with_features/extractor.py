"""
Módulo para extração de features de arquivos de áudio usando Librosa.
As features incluem RMS, MFCC, Chroma, Spectral Contrast, Tonnetz e Spectral Centroid.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import librosa
from tqdm import tqdm

N_JOBS = 8
AUDIO_FOLDER = "./data"


def extract_features(y, sr):
    """
    Extrai um conjunto de features a partir do sinal de áudio.

    Args:
        y (np.ndarray): Sinal de áudio.
        sr (int): Taxa de amostragem.

    Returns:
        np.ndarray: Vetor concatenado de features.
    """
    # RMS
    rms = librosa.feature.rms(y=y)
    rms_stats = [np.mean(rms, axis=1)]

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_stats = [np.mean(mfcc, axis=1), np.var(mfcc, axis=1)]

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stats = [np.mean(chroma, axis=1), np.var(chroma, axis=1)]

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_stats = [np.mean(contrast, axis=1), np.var(contrast, axis=1)]

    # Tonnetz
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    tonnetz_stats = [np.mean(tonnetz, axis=1), np.var(tonnetz, axis=1)]

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_stats = [np.mean(spectral_centroid, axis=1)]

    # Concatenar tudo
    return np.concatenate(
        rms_stats
        + mfcc_stats
        + chroma_stats
        + contrast_stats
        + tonnetz_stats
        + spectral_stats
    )


def extract_features_from_file(file_path):
    """
    Extrai features de um arquivo de áudio e retorna junto com o rótulo (nome da pasta).
    """
    label = os.path.basename(os.path.dirname(file_path))
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    features = extract_features(y, sr)
    return features, label


def process_features_for_all_files():
    """
    Processa todos os arquivos de áudio em AUDIO_FOLDER, extrai features
    e salva nos arquivos X.npy e Y.npy.
    """
    genres_folders = [
        os.path.join(AUDIO_FOLDER, f)
        for f in os.listdir(AUDIO_FOLDER)
        if os.path.isdir(os.path.join(AUDIO_FOLDER, f))
    ]
    print("Iniciando o processamento das features, por favor não interrompa o processo...")
    print(f"Gêneros para processar: {list(map(os.path.basename, genres_folders))}\n")

    all_files = []
    all_features = []

    for genre_folder in genres_folders:
        files = [
            os.path.join(genre_folder, f)
            for f in os.listdir(genre_folder)
            if f.endswith(".wav")
        ]
        all_files.extend(files)

    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {executor.submit(extract_features_from_file, f): f for f in all_files}
        for future in tqdm(
            as_completed(futures), desc="Extraindo as features", total=len(all_files)
        ):
            all_features.append(future.result())

    x_data = np.array([features for features, _ in all_features], dtype=np.float32)
    y_labels = np.array([label for _, label in all_features])

    np.save("./src/with_features/X.npy", x_data)
    np.save("./src/with_features/Y.npy", y_labels)


if __name__ == "__main__":
    process_features_for_all_files()
