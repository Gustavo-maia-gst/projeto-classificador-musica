import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
import joblib
import os

TEST_SIZE = 0.2

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class MLPGenreClassifier:
    def __init__(self, hidden_layer_sizes=(100, 50), random_state=42):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=300,
            alpha=0.001,
            learning_rate_init=0.001,
            solver='adam',
            activation='relu'
        )
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def load_data(self, X_path='./src/with_features/X.npy', Y_path='./src/with_features/Y.npy'):
        print("Carregando dados...")
        self.X = np.load(X_path)
        self.y = np.load(Y_path)
        print(f"Dados carregados: {self.X.shape[0]} amostras, {self.X.shape[1]} features")
        
    def train(self):
        print("Treinando modelo MLP...")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=TEST_SIZE, random_state=42, stratify=self.y
        )
        
        # Normalizar dados
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Avaliar
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Acurácia: {accuracy:.4f}")
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
        
    def predict(self, features):
        if not self.is_trained:
            print("Erro: Modelo não foi treinado!")
            return None
            
        features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return prediction, probabilities
        
    def save_model(self, filepath='./src/with_features/mlp.model'):
        if not self.is_trained:
            print("Erro: Modelo não foi treinado!")
            return
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, filepath)
        print(f"Modelo salvo em: {filepath}")
        
    def load_model(self, filepath='./src/with_features/mlp.model'):
        if not os.path.exists(filepath):
            print(f"Erro: Arquivo {filepath} não encontrado!")
            return
            
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = True
        print(f"Modelo carregado de: {filepath}")


def main():
    """Função principal"""
    print("=== CLASSIFICADOR MLP DE GÊNEROS MUSICAIS ===")
    
    classifier = MLPGenreClassifier()
    classifier.load_data()
    accuracy = classifier.train()
    
    # Exemplo de predição
    sample_features = classifier.X[0]  # Primeira amostra
    prediction, probabilities = classifier.predict(sample_features)
    
    print(f"\nExemplo de predição:")
    print(f"Gênero predito: {prediction}")
    print(f"Probabilidade: {max(probabilities):.4f}")
    
    classifier.save_model()


if __name__ == "__main__":
    main()
