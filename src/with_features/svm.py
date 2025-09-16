"""Classificador de gêneros musicais usando SVM."""

import os
import warnings
import joblib
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.exceptions import ConvergenceWarning
from with_features.extractor import extract_features

TEST_SIZE = 0.2
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class SVMGenreClassifier:
    """Classificador de gêneros musicais com SVM."""

    def __init__(self, kernel="rbf", c_value=2.5, random_state=42):
        """
        Inicializa o modelo SVM e variáveis auxiliares.

        Args:
            kernel (str): Kernel a ser usado (ex: 'rbf').
            c_value (float): Parâmetro de regularização C.
            random_state (int): Semente de reprodutibilidade.
        """
        self.model = SVC(
            kernel=kernel,
            C=c_value,
            random_state=random_state,
            probability=True,
            gamma=0.035,
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.x_data = None
        self.y_data = None
        self.classes = None

    def load_data(
        self,
        x_path: str = "./src/with_features/X.npy",
        y_path: str = "./src/with_features/Y.npy",
    ):
        """
        Carrega os arrays de features (X) e rótulos (y) a partir dos arquivos .npy.

        Args:
            x_path (str): Caminho para o arquivo de features.
            y_path (str): Caminho para o arquivo de rótulos.
        """
        self.x_data = np.load(x_path)
        self.y_data = np.load(y_path)
        self.classes = np.unique(self.y_data)
        self.y_data = self.label_encoder.fit_transform(self.y_data)

        print(
            f"Dados carregados: {self.x_data.shape[0]} amostras, "
            f"{self.x_data.shape[1]} features"
        )

    def train(self) -> float:
        """
        Treina o modelo SVM com data augmentation e normalização.

        Returns:
            float: Acurácia no conjunto de teste.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            self.x_data,
            self.y_data,
            test_size=TEST_SIZE,
            random_state=42,
            stratify=self.y_data,
        )

        # Data augmentation com ruído gaussiano
        rng = np.random.default_rng(42)
        x_aug = x_train + rng.normal(0, 0.15, x_train.shape)
        x_train = np.vstack([x_train, x_aug])
        y_train = np.concatenate([y_train, y_train])

        # Normalizar dados
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        print("Treinando modelo SVM...")
        self.model.fit(x_train_scaled, y_train)
        self.is_trained = True

        # Avaliar
        y_pred = self.model.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Acurácia: {accuracy:.4f}")
        print("\nRelatório de Classificação:")
        print(
            classification_report(
                [self.classes[idx] for idx in y_test],
                [self.classes[idx] for idx in y_pred],
            )
        )

        return accuracy

    def predict(self, file_path: str):
        """
        Prediz o gênero musical de um arquivo de áudio.

        Args:
            file_path (str): Caminho do arquivo .wav.

        Returns:
            tuple: (gênero predito, probabilidades por classe) ou None se erro.
        """
        if not self.is_trained:
            print("Erro: Modelo não foi treinado!")
            return None

        if not os.path.exists(file_path):
            print(f"Erro: Arquivo {file_path} não encontrado!")
            return None

        y_audio, sr = librosa.load(file_path, sr=22050, mono=True)
        features = extract_features(y_audio, sr).reshape(1, -1)

        # Normalizar features
        features_scaled = self.scaler.transform(features)

        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]

        return prediction, probabilities

    def save_model(self, filepath: str = "./src/with_features/svm.model"):
        """
        Salva o modelo treinado em disco.

        Args:
            filepath (str): Caminho para salvar o arquivo .model.
        """
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

    def load_model(self, filepath: str = "./src/with_features/svm.model"):
        """
        Carrega um modelo previamente salvo.

        Args:
            filepath (str): Caminho do arquivo .model.
        """
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
    """Função principal para treinar e salvar o classificador SVM."""
    print("=== CLASSIFICADOR SVM DE GÊNEROS MUSICAIS ===")

    classifier = SVMGenreClassifier()
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
