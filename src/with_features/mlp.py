import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
import joblib
import os
import librosa
import scipy.stats
from .extractor import extract_features
from sklearn.preprocessing import LabelEncoder

TEST_SIZE = 0.2
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class MLPGenreClassifier:
    def __init__(self, hidden_layer_sizes=(256, 128, 64, 32), random_state=42):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=500,
            alpha=0.001,
            learning_rate_init=0.001,
            solver='adam',
            activation='relu',
            early_stopping=True,
            n_iter_no_change=15
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def load_data(self, X_path='./src/with_features/X.npy', Y_path='./src/with_features/Y.npy'):
        self.X = np.load(X_path)
        self.y = np.load(Y_path)
        self.classes = np.unique(self.y)
        self.y = self.label_encoder.fit_transform(self.y)

        print(f"Dados carregados: {self.X.shape[0]} amostras, {self.X.shape[1]} features")

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=TEST_SIZE, random_state=42, stratify=self.y
        )

        rng = np.random.default_rng(42)
        X_aug = X_train + rng.normal(0, 0.25, X_train.shape)
        X_train = np.vstack([X_train, X_aug])
        y_train = np.concatenate([y_train, y_train])

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Avaliar
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Acurácia: {accuracy:.4f}")
        print("\nRelatório de Classificação:")
        print(classification_report(list(map(lambda x: self.classes[x], y_test)), list(map(lambda x: self.classes[x], y_pred))))

        return accuracy

    def predict(self, file_path):
        if not self.is_trained:
            print("Erro: Modelo não foi treinado!")
            return None

        if not os.path.exists(file_path):
            print(f"Erro: Arquivo {file_path} não encontrado!")
            return None

        y, sr = librosa.load(file_path, sr=22050, mono=True)
        features = extract_features(y, sr)
        features = features.reshape(1, -1)
        
        # Normalizar features
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        return prediction, probabilities

    def save_model(self, filepath='./src/with_features/mlp.model'):
        if not self.is_trained:
            print("Erro: Modelo não foi treinado!")
            return

        model_data = {'model': self.model, 'scaler': self.scaler, 'classes': self.classes}
        joblib.dump(model_data, filepath)
        print(f"Modelo salvo em: {filepath}")

    def load_model(self, filepath='./src/with_features/mlp.model'):
        if not os.path.exists(filepath):
            print(f"Erro: Arquivo {filepath} não encontrado!")
            return

        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.classes = model_data['classes']
        self.is_trained = True
        print(f"Modelo carregado de: {filepath}")


def main():
    print("=== CLASSIFICADOR MLP DE GÊNEROS MUSICAIS (OTIMIZADO) ===")

    classifier = MLPGenreClassifier()
    classifier.load_data()
    accuracy = classifier.train()

    sample_file = "./data/blues/blues.00000.wav"
    if os.path.exists(sample_file):
        prediction, probabilities = classifier.predict(sample_file)
        print(f"\nExemplo de predição:")
        print(f"Arquivo: {sample_file}")
        print(f"Gênero predito: {prediction}")
        print(f"Probabilidade: {max(probabilities):.4f}")
    else:
        print("Arquivo de exemplo não encontrado para teste")

    classifier.save_model()


if __name__ == "__main__":
    main()
