import librosa
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
from tqdm import tqdm
import re

N_JOBS = 8
TARGET_LENGTH = 128
AUDIO_FOLDER = "./data"

def extract_spectrogram(y, sr, n_mels=128, n_fft=2048, hop_length=512):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=n_mels, 
        n_fft=n_fft, 
        hop_length=hop_length
    )
    
    spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if spectrogram.shape[1] < TARGET_LENGTH:
        pad_width = TARGET_LENGTH - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        spectrogram = spectrogram[:, :TARGET_LENGTH]
    
    return spectrogram

def extract_spectrogram_from_file(file_path):
    label = re.search(r'data/(\w+)/\w+\.\d{5}\.wav', file_path).group(1)

    y, sr = librosa.load(file_path, sr=22050, mono=True)

    spectrogram = extract_spectrogram(y, sr)
    
    return spectrogram, label

def process_spectrograms_for_all_files():
    genres_folders = [
        os.path.join(AUDIO_FOLDER, f)
        for f in os.listdir(AUDIO_FOLDER)
        if os.path.isdir(os.path.join(AUDIO_FOLDER, f))
    ]
    print(f"Iniciando o processamento dos espectrogramas, por favor não interrompa o processo...")
    print(f"Gêneros para processar: {list(map(os.path.basename, genres_folders))}\n")

    all_files = []
    all_spectrograms = []

    for genre_folder in genres_folders:
        files = [
            os.path.join(genre_folder, f)
            for f in os.listdir(genre_folder)
            if f.endswith(".wav")
        ]
        all_files.extend(files)

    with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {executor.submit(extract_spectrogram_from_file, f): f for f in all_files}
        for future in tqdm(as_completed(futures), desc="Extraindo espectrogramas", total=len(all_files)):
            all_spectrograms.append(future.result())

    X = np.array([spectrogram for spectrogram, _ in all_spectrograms])
    Y = np.array([label for _, label in all_spectrograms])

    np.save("./src/with_spectrogram/X.npy", X)
    np.save("./src/with_spectrogram/Y.npy", Y)
    
if __name__ == "__main__":
    process_spectrograms_for_all_files()