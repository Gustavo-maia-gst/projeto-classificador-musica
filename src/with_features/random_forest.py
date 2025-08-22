import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

TEST_SIZE = 0.2

class RandomForestGenreClassifier:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False
        
    def load_data(self, X_path='./src/with_features/X.npy', Y_path='./src/with_features/Y.npy'):
        print("Carregando dados...")
        self.X = np.load(X_path)
        self.y = np.load(Y_path)
        print(f"Dados carregados: {self.X.shape[0]} amostras, {self.X.shape[1]} features")
        
    def train(self):
        print("Treinando modelo...")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=TEST_SIZE, random_state=42, stratify=self.y
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Avaliar
        y_pred = self.model.predict(X_test)
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
    
    # Exemplo de predição
    sample_features = classifier.X[0]  # Primeira amostra
    prediction, probabilities = classifier.predict(sample_features)
    
    print(f"\nExemplo de predição:")
    print(f"Gênero predito: {prediction}")
    print(f"Probabilidade: {max(probabilities):.4f}")
    
    classifier.save_model()


if __name__ == "__main__":
    main()
