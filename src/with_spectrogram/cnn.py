import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
import joblib
import os

TEST_SIZE = 0.2

warnings.filterwarnings("ignore")

class CNNGenreClassifier:
    def __init__(self, input_shape=(128, 128, 1), num_classes=10, random_state=42):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Configurar seed para reprodutibilidade
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Constrói a arquitetura da CNN para classificação de espectrogramas
        """
        model = keras.Sequential([
            # Primeira camada convolucional
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Segunda camada convolucional
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Terceira camada convolucional
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Quarta camada convolucional
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten para conectar com camadas densas
            layers.Flatten(),
            
            # Camadas densas
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Camada de saída
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compilar o modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def load_data(self, X_path='./src/with_spectrogram/X.npy', Y_path='./src/with_spectrogram/Y.npy'):
        print("Carregando dados...")
        self.X = np.load(X_path)
        self.y = np.load(Y_path)
        
        # Preparar dados para CNN
        # Adicionar dimensão do canal (batch_size, height, width, channels)
        self.X = self.X.reshape(-1, 128, 128, 1)
        
        # Normalizar dados para [0, 1]
        self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())
        
        # Codificar labels
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        self.classes = self.label_encoder.classes_
        
        print(f"Dados carregados: {self.X.shape[0]} amostras, {self.X.shape[1:]} dimensões")
        print(f"Classes: {self.classes}")
        
    def train(self, epochs=50, batch_size=32, validation_split=0.2):
        print("Treinando modelo CNN...")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_encoded, test_size=TEST_SIZE, random_state=self.random_state, stratify=self.y_encoded
        )
        
        # Callbacks para melhor treinamento
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Treinar o modelo
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Avaliar
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nAcurácia: {accuracy:.4f}")
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=self.classes))
        
        return accuracy, history
        
    def predict(self, spectrogram):
        if not self.is_trained:
            print("Erro: Modelo não foi treinado!")
            return None
            
        # Preparar entrada
        if len(spectrogram.shape) == 2:
            spectrogram = spectrogram.reshape(1, 128, 128, 1)
        
        # Normalizar
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        
        # Predição
        probabilities = self.model.predict(spectrogram)[0]
        prediction_idx = np.argmax(probabilities)
        prediction = self.classes[prediction_idx]
        
        return prediction, probabilities
        
    def save_model(self, filepath='./src/with_spectrogram/cnn.model'):
        if not self.is_trained:
            print("Erro: Modelo não foi treinado!")
            return
            
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'classes': self.classes
        }
        joblib.dump(model_data, filepath)
        print(f"Modelo salvo em: {filepath}")
        
    def load_model(self, filepath='./src/with_spectrogram/cnn.model'):
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
    """Função principal"""
    print("=== CLASSIFICADOR CNN DE GÊNEROS MUSICAIS ===")
    
    classifier = CNNGenreClassifier()
    classifier.load_data()
    accuracy, history = classifier.train()
    
    # Exemplo de predição
    sample_spectrogram = classifier.X[0]  # Primeira amostra
    sample_spectrogram = sample_spectrogram.reshape(1, 128, 128, 1)  # agora (1,128,128,1)
    prediction, probabilities = classifier.predict(sample_spectrogram)
    
    print(f"\nExemplo de predição:")
    print(f"Gênero predito: {prediction}")
    print(f"Probabilidade: {max(probabilities):.4f}")
    
    classifier.save_model()


if __name__ == "__main__":
    main()
