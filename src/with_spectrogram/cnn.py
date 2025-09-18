"""
Módulo CNN para classificação de gêneros musicais usando espectrogramas.
"""

import os
import warnings
import joblib
import numpy as np
import librosa
import tensorflow as tf
from keras import Sequential, optimizers  # pylint: disable=import-error,no-name-in-module
from keras.layers import (  # pylint: disable=import-error,no-name-in-module
    Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense
)
from keras.callbacks import (  # ✅ callbacks direto do keras (corrige E1101)
    EarlyStopping, ReduceLROnPlateau
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from with_spectrogram.extractor import extract_spectrogram  # import absoluto

# Constantes
TEST_SIZE = 0.2
RANDOM_STATE = 42

warnings.filterwarnings("ignore")
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class CNNGenreClassifier:  # pylint: disable=too-many-instance-attributes
    """Classificador CNN para espectrogramas de gêneros musicais."""

    def __init__(self, input_shape=(128, 128, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.x_data = None
        self.y_data = None
        self.y_encoded = None
        self.classes = None
        self.model = self._build_model()

    def _build_model(self):
        """Constrói a arquitetura da CNN."""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Flatten(),

            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),

            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),

            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_data(
        self,
        x_path='./src/with_spectrogram/X.npy',
        y_path='./src/with_spectrogram/Y.npy'
    ):
        """Carrega os dados e prepara para treinamento."""
        print("Carregando dados...")
        self.x_data = np.load(x_path)
        self.y_data = np.load(y_path)

        self.x_data = self.x_data.reshape(-1, 128, 128, 1)
        self.x_data = (self.x_data - self.x_data.min()) / (
            self.x_data.max() - self.x_data.min()
        )

        self.y_encoded = self.label_encoder.fit_transform(self.y_data)
        self.classes = self.label_encoder.classes_

        print(f"Dados carregados: {self.x_data.shape[0]} amostras")
        print(f"Dimensão: {self.x_data.shape[1:]} ")
        print(f"Classes: {self.classes}")

    def train(self, epochs=50, batch_size=32, validation_split=0.2):
        """Treina o modelo CNN com os dados carregados."""
        print("Treinando modelo CNN...")

        x_train, x_test, y_train, y_test = train_test_split(
            self.x_data, self.y_encoded,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=self.y_encoded
        )

        callbacks_list = [
            EarlyStopping(
                monitor='val_accuracy', patience=10, restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
            )
        ]

        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks_list,
            verbose=1
        )

        self.is_trained = True

        y_pred = np.argmax(self.model.predict(x_test), axis=1)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nAcurácia: {accuracy:.4f}")
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=self.classes))

        return accuracy, history

    def predict(self, file_path):
        """Prediz o gênero de um arquivo de áudio WAV."""
        if not self.is_trained:
            raise RuntimeError("Modelo não treinado!")

        if not os.path.exists(file_path):
            print(f"Erro: Arquivo {file_path} não encontrado!")
            return None

        y, sr = librosa.load(file_path, sr=22050, mono=True)
        spectrogram = extract_spectrogram(y, sr)

        if len(spectrogram.shape) == 2:
            spectrogram = spectrogram[np.newaxis, ..., np.newaxis]

        probs = self.model.predict(spectrogram)[0]
        idx = np.argmax(probs)

        return self.classes[idx], probs

    def save_model(self, filepath='./src/with_spectrogram/cnn.model'):
        """Salva o modelo treinado no disco."""
        if not self.is_trained:
            print("Erro: Modelo não foi treinado!")
            return

        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'classes': self.classes
        }, filepath)
        print(f"Modelo salvo em: {filepath}")

    def load_model(self, filepath='./src/with_spectrogram/cnn.model'):
        """Carrega o modelo treinado do disco."""
        if not os.path.exists(filepath):
            print(f"Erro: Arquivo {filepath} não encontrado!")
            return

        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.classes = model_data['classes']
        self.is_trained = True
        print(f"Modelo carregado de: {filepath}")


def main():
    """Função principal de demonstração do classificador."""
    print("=== CLASSIFICADOR CNN DE GÊNEROS MUSICAIS ===")
    classifier = CNNGenreClassifier()
    classifier.load_model()

    sample_file = "./data/blues/blues.00015.wav"
    if os.path.exists(sample_file):
        prediction, probabilities = classifier.predict(sample_file)
        print("\nExemplo de predição:")
        print(f"Arquivo: {sample_file}")
        print(f"Gênero predito: {prediction}")
        print(f"Probabilidade: {max(probabilities):.4f}")
    else:
        print("Arquivo de exemplo não encontrado para teste.")


if __name__ == "__main__":
    main()
