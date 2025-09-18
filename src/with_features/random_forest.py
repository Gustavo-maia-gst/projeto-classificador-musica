"""Classificador de gêneros musicais usando Random Forest."""

import os
import joblib
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from with_features.extractor import extract_features

TEST_SIZE = 0.2


class RandomForestGenreClassifier:
    """Classificador de gêneros musicais com Random Forest."""

    def __init__(self, n_estimators=500, random_state=42):
        """
        Inicializa o modelo RandomForest e variáveis auxiliares.

        Args:
            n_estimators (int): Número de árvores.
            random_state (int): Semente para reprodutibilidade.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=15,
            min_samples_leaf=5,
            bootstrap=True,
            n_jobs=1,
        )
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
        Treina o modelo RandomForest com cross-validation e split de teste.

        Returns:
            float: acurácia no conjunto de teste.
        """
        print("Treinando modelo com validação cruzada...")

        # Avaliação cross-validation
        scores = cross_val_score(
            self.model, self.x_data, self.y_data, cv=5, scoring="accuracy", n_jobs=1
        )
        print(f"Acurácia média (CV): {scores.mean():.4f} ± {scores.std():.4f}\n")

        # Split para relatório detalhado
        x_train, x_test, y_train, y_test = train_test_split(
            self.x_data,
            self.y_data,
            test_size=TEST_SIZE,
            random_state=42,
            stratify=self.y_data,
        )

        rng = np.random.default_rng(42)
        x_aug = x_train + rng.normal(0, 0.2, x_train.shape)
        x_train = np.vstack([x_train, x_aug])
        y_train = np.concatenate([y_train, y_train])

        # Treina modelo final
        self.model.fit(x_train, y_train)
        self.is_trained = True

        # Avaliar e gerar relatório
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Acurácia no conjunto de teste: {accuracy:.4f}")
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
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        return prediction, probabilities

    def save_model(self, filepath: str = "./src/with_features/random_forest.model"):
        """
        Salva o modelo treinado em disco.

        Args:
            filepath (str): Caminho para salvar o arquivo .model.
        """
        if not self.is_trained:
            print("Erro: Modelo não foi treinado!")
            return

        model_data = {"model": self.model, "classes": self.classes}
        joblib.dump(model_data, filepath)
        print(f"Modelo salvo em: {filepath}")

    def load_model(self, filepath: str = "./src/with_features/random_forest.model"):
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
        self.classes = model_data["classes"]
        self.is_trained = True
        print(f"Modelo carregado de: {filepath}")


def main():
    """Função principal para treinar e salvar o classificador."""
    print("=== CLASSIFICADOR DE GÊNEROS MUSICAIS ===")

    classifier = RandomForestGenreClassifier()
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
