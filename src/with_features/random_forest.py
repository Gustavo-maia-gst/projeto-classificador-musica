import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import librosa
from .extractor import extract_features

TEST_SIZE = 0.2

class RandomForestGenreClassifier:
    def __init__(self, n_estimators=500, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=15,
            min_samples_leaf=5,
            bootstrap=True,
            n_jobs=1
        )
        self.is_trained = False
        
    def load_data(self, X_path='./src/with_features/X.npy', Y_path='./src/with_features/Y.npy'):
        X_raw = np.load(X_path)
        y_raw = np.load(Y_path)

        X_aug = X_raw + np.random.normal(0, 0.01, X_raw.shape)
        self.X = np.vstack([X_raw, X_aug])
        self.y = np.concatenate([y_raw, y_raw])
        
        print(f"Dados carregados: {self.X.shape[0]} amostras, {self.X.shape[1]} features")
        
    def train(self):
        print("Treinando modelo com CV...")
        
        # Avaliação cross-validation
        scores = cross_val_score(self.model, self.X, self.y, cv=5, scoring='accuracy', n_jobs=1)
        print(f"Accuracy média (CV): {scores.mean():.4f} ± {scores.std():.4f}\n")
        
        # Split para relatório detalhado
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=TEST_SIZE, random_state=42, stratify=self.y
        )
        
        # Treina modelo final
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Avaliar e gerar relatório como antes
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Acurácia no split de teste: {accuracy:.4f}")
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred))
        
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
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        return prediction, probabilities
        
    def save_model(self, filepath='./src/with_features/random_forest.model'):
        if not self.is_trained:
            print("Erro: Modelo não foi treinado!")
            return
            
        model_data = {
            'model': self.model,
        }
        joblib.dump(model_data, filepath)
        print(f"Modelo salvo em: {filepath}")
        
    def load_model(self, filepath='./src/with_features/random_forest.model'):
        if not os.path.exists(filepath):
            print(f"Erro: Arquivo {filepath} não encontrado!")
            return
            
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.is_trained = True
        print(f"Modelo carregado de: {filepath}")


def main():
    """Função principal"""
    print("=== CLASSIFICADOR DE GÊNEROS MUSICAIS ===")
    
    classifier = RandomForestGenreClassifier()
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
