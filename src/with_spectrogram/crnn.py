import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import librosa
import joblib
import warnings
import os
from .extractor import extract_spectrogram

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

class CRNNGenreClassifier:
    def __init__(self, input_shape=(128, 128, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(128,128,1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            # Transformar para sequência: frames x features
            layers.Reshape((-1, 128)),  # 128 filtros finais → features

            # GRU bidirecional leve
            layers.Bidirectional(layers.GRU(64, return_sequences=True)),
            layers.Dropout(0.2),
            layers.Bidirectional(layers.GRU(32)),
            layers.Dropout(0.2),

            # Dense final
            layers.Dense(128, activation='relu'),
            layers.LayerNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.LayerNormalization(),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_data(self, X_path='src/with_spectrogram/X.npy', Y_path='src/with_spectrogram/Y.npy'):
        print("Carregando dados...")
        self.X = np.load(X_path)
        self.y = np.load(Y_path)

        # Adiciona canal para CNN
        self.X = self.X[..., np.newaxis]  # (n_samples, 128, 128, 1)

        # Codificar labels
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        self.classes = self.label_encoder.classes_

        print(f"{self.X.shape[0]} amostras carregadas, shape: {self.X.shape[1:]}")
        print(f"Classes: {self.classes}")

    def train(self, epochs=10, batch_size=32, validation_split=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.2, random_state=42
        )

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        self.is_trained = True

        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Acurácia: {acc:.4f}")
        print("\nRelatório de Classificação:")
        print(classification_report(list(map(lambda x: self.classes[x], y_test)), list(map(lambda x: self.classes[x], y_pred))))

        return acc, history

    def predict(self, file_path):
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

    def save_model(self, path='src/with_spectrogram/crnn.model'):
        joblib.dump({'model': self.model, 'label_encoder': self.label_encoder, 'classes': self.classes}, path)
        print(f"Modelo salvo em {path}")

    def load_model(self, path='src/with_spectrogram/crnn.model'):
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data['model']
            self.label_encoder = data['label_encoder']
            self.classes = data['classes']
            self.is_trained = True
            print(f"Modelo carregado de {path}")

def main():
    classifier = CRNNGenreClassifier()
    classifier.load_model()
    # classifier.train()
    # classifier.save_model()

    sample_file = "./data/blues/blues.00015.wav"
    if os.path.exists(sample_file):
        prediction, probabilities = classifier.predict(sample_file)
        print(f"\nExemplo de predição:")
        print(f"Arquivo: {sample_file}")
        print(f"Gênero predito: {prediction}")
        print(f"Probabilidade: {max(probabilities):.4f}")
    else:
        print("Arquivo de exemplo não encontrado para teste")


if __name__ == "__main__":
    main()
