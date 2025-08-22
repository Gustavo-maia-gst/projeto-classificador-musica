import librosa
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
from tqdm import tqdm
import re

N_JOBS = 8
AUDIO_FOLDER = "./data"

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var = np.var(mfcc, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    tonnetz_var = np.var(tonnetz, axis=1)
    
    return np.concatenate(
        [mfcc_mean, mfcc_var, chroma_mean, contrast_mean, tonnetz_mean, tonnetz_var]
    )

def extract_features_from_file(file_path):
    label = re.search(r'data/(\w+)/\w+\.\d{5}\.wav', file_path).group(1)

    """Extrai features de uma música .wav"""
    y, sr = librosa.load(file_path, sr=22050, mono=True)

    features = extract_features(y, sr)

    return features, label

def process_features_for_all_files():
    genres_folders = [
        os.path.join(AUDIO_FOLDER, f)
        for f in os.listdir(AUDIO_FOLDER)
        if os.path.isdir(os.path.join(AUDIO_FOLDER, f))
    ]
    print(f"Inciando o processamento das features, por favor não interrompa o processo...")
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


    with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {executor.submit(extract_features_from_file, f): f for f in all_files}
        for future in tqdm(as_completed(futures), desc="Extraindo as features", total=len(all_files)):
            all_features.append(future.result())

    X = np.array([features for features, _ in all_features])
    Y = np.array([label for _, label in all_features])

    np.save("./src/with_features/X.npy", X)
    np.save("./src/with_features/Y.npy", Y)


if __name__ == "__main__":
    process_features_for_all_files()