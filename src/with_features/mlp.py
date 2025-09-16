"""
Módulo que implementa um classificador de gêneros musicais usando MLP (scikit-learn).
Inclui funções para treino, predição, salvamento e carregamento do modelo.
"""

import os
import warnings
import joblib

import numpy as np
import librosa

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.exceptions import ConvergenceWarning
from with_features.extractor import extract_features
TEST_SIZE = 0.2
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class MLPGenreClassifier:
    """Classificador de gêneros musicais baseado em MLP."""

    def __init__(self, hidden_layer_sizes=(256, 128, 64, 32), random_state=42):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=500,
            alpha=0.001,
            learning_rate_init=0.001,
            solver="adam",
            activation="relu",
            early_stopping=True,
            n_iter_no_change=15,
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False

        # Atributos inicializados para evitar warnings
        self.x_data = None
        self.y_data = None
        self.classes = None

    def load_data(self, x_path="./src/with_features/X.npy", y_path="./src/with_features/Y.npy"):
        """Carrega os dados salvos em numpy arrays."""
        self.x_data = np.load(x_path)
        self.y_data = np.load(y_path)
        self.classes = np.unique(self.y_data)
        self.y_data = self.label_encoder.fit_transform(self.y_data)

        print(f"Dados carregados: {self.x_data.shape[0]} amostras, {self.x_data.shape[1]} features")

    def train(self):
        """Treina o classificador MLP com aumento de dados e retorna a acurácia."""
        x_train, x_test, y_train, y_test = train_test_split(
            self.x_data, self.y_data, test_size=TEST_SIZE, random_state=42, stratify=self.y_data
        )

        rng = np.random.default_rng(42)
        x_aug = x_train + rng.normal(0, 0.25, x_train.shape)
        x_train = np.vstack([x_train, x_aug])
        y_train = np.concatenate([y_train, y_train])

        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        self.model.fit(x_train_scaled, y_train)
        self.is_trained = True

        y_pred = self.model.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Acurácia: {accuracy:.4f}")
        print("\nRelatório de Classificação:")
        print(
            classification_report(
                [self.classes[i] for i in y_test],
                [self.classes[i] for i in y_pred],
            )
        )

        return accuracy

    def predict(self, file_path):
        """Prediz o gênero musical de um arquivo de áudio."""
        if not self.is_trained:
            print("Erro: Modelo não foi treinado!")
            return None

        if not os.path.exists(file_path):
            print(f"Erro: Arquivo {file_path} não encontrado!")
            return None

        y, sr = librosa.load(file_path, sr=22050, mono=True)
        features = extract_features(y, sr).reshape(1, -1)

        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]

        return prediction, probabilities

    def save_model(self, filepath="./src/with_features/mlp.model"):
        """Salva o modelo em disco."""
        if not self.is_trained:
            print("Erro: Modelo não foi treinado!")
            return

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "classes": self.classes,
        }
        joblib.dump(model_data, filepath)
        print(f"Modelo salvo em: {filepath}")

    def load_model(self, filepath="./src/with_features/mlp.model"):
        """Carrega o modelo salvo em disco."""
        if not os.path.exists(filepath):
            print(f"Erro: Arquivo {filepath} não encontrado!")
            return

        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.classes = model_data["classes"]
        self.is_trained = True
        print(f"Modelo carregado de: {filepath}")


def main():
    """Função principal para treinar e salvar o modelo."""
    print("=== CLASSIFICADOR MLP DE GÊNEROS MUSICAIS (OTIMIZADO) ===")

    classifier = MLPGenreClassifier()
    classifier.load_data()
    classifier.train()

    sample_file = "./data/blues/blues.00000.wav"
    if os.path.exists(sample_file):
        prediction, probabilities = classifier.predict(sample_file)
        print("\nExemplo de predição:")
        print(f"Arquivo: {sample_file}")
        print(f"Gênero predito: {prediction}")
        print(f"Probabilidade: {max(probabilities):.4f}")
    else:
        print("Arquivo de exemplo não encontrado para teste")

    classifier.save_model()


if __name__ == "__main__":
    main()
