"""
Módulo CRNN para classificação de gêneros musicais a partir de espectrogramas.
"""

import os
import warnings
import joblib
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks  # pylint: disable=import-error,no-name-in-module
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Importação absoluta para resolver problema de import relativo
try:
    from extractor import extract_spectrogram
except ImportError:
    try:
        # Tentativa alternativa para estrutura de pacotes
        from src.with_spectrogram.extractor import extract_spectrogram
    except ImportError:
        # Fallback para desenvolvimento local com import relativo
        from .extractor import extract_spectrogram  # pylint: disable=relative-beyond-top-level

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)


class CRNNGenreClassifier: # pylint: disable=too-many-instance-attributes
    """
    Classe que implementa uma CRNN (CNN + RNN) para classificação de gêneros musicais.
    """

    def __init__(self, input_shape=(128, 128, 1), num_classes=10):
        """Inicializa o classificador CRNN.

        Args:
            input_shape: Tupla com formato de entrada dos espectrogramas
            num_classes: Número de classes de gêneros musicais
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.model = self._build_model()
        # Atributos de dados serão inicializados posteriormente
        self.x_data = None
        self.y_data = None
        self.y_encoded = None
        self.classes = None

    def _build_model(self):
        """Constrói o modelo CRNN com CNN + GRU bidirecional.

        Returns:
            models.Sequential: Modelo CRNN compilado
        """

        model = models.Sequential([
            layers.Conv2D(
                32, (3, 3), activation='relu', padding='same',
                input_shape=self.input_shape
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Converte em sequência: frames × features
            layers.Reshape((-1, 128)),

            layers.Bidirectional(layers.GRU(64, return_sequences=True)),
            layers.Dropout(0.2),
            layers.Bidirectional(layers.GRU(32)),
            layers.Dropout(0.2),

            layers.Dense(128, activation='relu'),
            layers.LayerNormalization(),
            layers.Dropout(0.2),

            layers.Dense(64, activation='relu'),
            layers.LayerNormalization(),

            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_data(self, x_path='src/with_spectrogram/X.npy', y_path='src/with_spectrogram/Y.npy'):
        """Carrega os dados de espectrogramas e labels.

        Args:
            x_path: Caminho para arquivo com espectrogramas
            y_path: Caminho para arquivo com labels
        """

        print("Carregando dados...")
        self.x_data = np.load(x_path)
        self.y_data = np.load(y_path)

        # Adiciona canal para CNN
        self.x_data = self.x_data[..., np.newaxis]

        # Codifica labels
        self.y_encoded = self.label_encoder.fit_transform(self.y_data)
        self.classes = self.label_encoder.classes_

        print(
            f"{self.x_data.shape[0]} amostras carregadas, "
            f"shape: {self.x_data.shape[1:]}"
        )
        print(f"Classes: {self.classes}")

    def train(self, epochs=10, batch_size=32, validation_split=0.2):
        """Treina o modelo com divisão treino/teste.

        Args:
            epochs: Número de épocas de treinamento
            batch_size: Tamanho do batch
            validation_split: Proporção de dados para validação

        Returns:
            tuple: (acurácia, histórico de treinamento)
        """

        x_train, x_test, y_train, y_test = train_test_split(
            self.x_data, self.y_encoded, test_size=0.2, random_state=42
        )

        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss', patience=2, restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
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

        y_pred_probs = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        acc = accuracy_score(y_test, y_pred)
        print(f"Acurácia: {acc:.4f}")
        print("\nRelatório de Classificação:")
        print(
            classification_report(
                [self.classes[idx] for idx in y_test],
                [self.classes[idx] for idx in y_pred]
            )
        )

        return acc, history

    def predict(self, file_path):
        """Realiza predição em um arquivo de áudio WAV.

        Args:
            file_path: Caminho para o arquivo de áudio

        Returns:
            tuple: (gênero predito, probabilidades) ou None se erro
        """

        if not self.is_trained:
            raise RuntimeError("O modelo ainda não foi treinado!")

        if not os.path.exists(file_path):
            print(f"Erro: Arquivo {file_path} não encontrado!")
            return None

        y, sample_rate = librosa.load(file_path, sr=22050, mono=True)
        spectrogram = extract_spectrogram(y, sample_rate)

        if len(spectrogram.shape) == 2:
            spectrogram = spectrogram[np.newaxis, ..., np.newaxis]

        probs = self.model.predict(spectrogram, verbose=0)[0]
        idx = np.argmax(probs)

        return prediction, probabilities

    def save_model(self, path='src/with_spectrogram/crnn.model'):
        """Salva o modelo e encoder em disco.

        Args:
            path: Caminho onde salvar o modelo
        """
        joblib.dump(
            {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'classes': self.classes
            },
            path
        )
        print(f"Modelo salvo em {path}")

    def load_model(self, path='src/with_spectrogram/crnn.model'):
        """Carrega modelo e encoder do disco.

        Args:
            path: Caminho de onde carregar o modelo
        """
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data['model']
            self.label_encoder = data['label_encoder']
            self.classes = data['classes']
            self.is_trained = True
            print(f"Modelo carregado de {path}")
        else:
            print(f"Arquivo {path} não encontrado!")


def main():
    """Exemplo de uso do classificador CRNN."""

    classifier = CRNNGenreClassifier()
    classifier.load_model()
    # classifier.train()
    # classifier.save_model()

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
