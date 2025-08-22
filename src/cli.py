import sys
import os
import numpy as np
import os
import sys
import librosa

if os.name == "nt":
    import msvcrt
else:
    import tty
    import termios

from with_features.extractor import extract_features, process_features_for_all_files
from with_features.random_forest import RandomForestGenreClassifier
from with_features.mlp import MLPGenreClassifier

from with_spectrogram.extractor import process_spectrograms_for_all_files
from with_spectrogram.cnn import CNNGenreClassifier

MLP_NAME = "MLP"
RF_NAME = "Random Forest"
CNN_NAME = "CNN"


class SubmenuCLI:
    def __init__(self):
        self.rf_classifier = RandomForestGenreClassifier()
        self.mlp_classifier = MLPGenreClassifier()
        self.cnn_classifier = CNNGenreClassifier()
        self.current_option = 0
        self.current_menu = "main"

        # Menus disponíveis
        self.menus = {
            "main": {
                "title": "🎵 CLASSIFICADOR DE GÊNEROS MUSICAIS",
                "options": [
                    "🚀 Treinar modelo",
                    "🧪 Testar modelo",
                    "💾 Salvar modelo",
                    "📊 Reprocessar dados",
                    "❌ Sair",
                ],
            },
            "train": {
                "title": "🚀 TREINAR MODELO",
                "options": ["🌲 Treinar Random Forest", "🧠 Treinar MLP", "🖼️ Treinar CNN", "⬅️  Voltar"],
            },
            "test": {
                "title": "🧪 TESTAR MODELO",
                "options": ["🌲 Testar Random Forest", "🧠 Testar MLP", "🖼️ Testar CNN", "⬅️  Voltar"],
            },
            "save": {
                "title": "💾 SALVAR MODELO",
                "options": ["🌲 Salvar Random Forest", "🧠 Salvar MLP", "🖼️ Salvar CNN", "⬅️  Voltar"],
            },
            "reprocess": {
                "title": "📊 REPROCESSAR DADOS",
                "options": ["🔄 Reprocessar features", "🔄 Reprocessar espectrogramas", "⬅️  Voltar"],
            },
        }

    def clear_screen(self):
        """Limpa a tela"""
        os.system("cls" if os.name == "nt" else "clear")

    def print_menu(self):
        """Imprime o menu atual"""
        self.clear_screen()

        menu = self.menus[self.current_menu]
        print(f"\n{menu['title']}\n")
        print("Selecione uma opção:\n")

        for i, option in enumerate(menu["options"]):
            if i == self.current_option:
                print(f"  [X] {option}")
            else:
                print(f"  [ ] {option}")

        print("\nUse ↑↓ para navegar, ENTER para selecionar")

    def get_key(self):
        """Captura uma tecla pressionada"""
        if os.name == "nt":  # Windows
            return msvcrt.getch()
        else:  # Linux/Mac
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                if ch == "\x1b":  # ESC
                    ch = sys.stdin.read(1)
                    if ch == "[":
                        ch = sys.stdin.read(1)
                        if ch == "A":
                            return "UP"
                        elif ch == "B":
                            return "DOWN"
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def get_model(self, model_type):
        """Retorna o modelo atual"""
        if model_type == RF_NAME:
            self.try_load_model(self.rf_classifier)
            return self.rf_classifier
        elif model_type == MLP_NAME:
            self.try_load_model(self.mlp_classifier)
            return self.mlp_classifier
        elif model_type == CNN_NAME:
            self.try_load_model(self.cnn_classifier)
            return self.cnn_classifier
        else:
            return None

    def try_load_model(self, model):
        if model.is_trained:
            return

        try:
            model.load_model()
        except Exception as e:
            pass

    def train_model(self, model_type):
        model = self.get_model(model_type)

        try:
            print(f"🔄 Carregando dados para {model_type}...")
            model.load_data()

            print(f"🚀 Treinando {model_type}...")
            accuracy = model.train()

            print(f"✅ {model_type} treinado com acurácia: {accuracy:.4f}")
            self.current_model = model
            input("Pressione ENTER para continuar...")

        except Exception as e:
            print(f"❌ Erro ao treinar {model_type}: {str(e)}")
            input("Pressione ENTER para continuar...")

    def test_specific(self, model_type):
        model = self.get_model(model_type)

        if model is None or not model.is_trained:
            print("❌ Nenhum modelo treinado. Treine um modelo primeiro")
            input("Pressione ENTER para continuar...")
            return

        try:
            file_path = input("Digite o caminho do arquivo de amostra: ")

            if not os.path.exists(file_path):
                print("❌ Arquivo não encontrado")
                input("Pressione ENTER para continuar...")
                return

            y, sr = librosa.load(file_path, sr=22050, mono=True)
            features = extract_features(y, sr)

            prediction, probabilities = model.predict(features)

            print(f"\n🎯 PREDIÇÃO DA AMOSTRA:")
            print(f"Gênero predito: {prediction}")
            print("Top3 Probabilidades:")

            # Mostrar top 3 probabilidades
            sorted_probs = sorted(
                enumerate(probabilities), key=lambda x: x[1], reverse=True
            )
            for i, (class_idx, prob) in enumerate(sorted_probs[:3]):
                genre = self.current_model.model.classes_[class_idx]
                print(f"  {genre}: {prob:.4f}")

            input("Pressione ENTER para continuar...")

        except Exception as e:
            print(f"❌ Erro na predição: {str(e)}")
            input("Pressione ENTER para continuar...")
    
    def save_model(self, model_type):
        model = self.get_model(model_type)
        model.save_model()
        input("Pressione ENTER para continuar...")

    def handle_menu_selection(self):
        """Processa a seleção do menu atual"""
        if self.current_menu == "main":
            if not self.handle_main_submenu():
                return False

        elif self.current_menu == "reprocess":
            self.handle_reprocess_submenu()

        elif self.current_menu == "train":
            self.handle_select_model_submenu(self.train_model)

        elif self.current_menu == "test":
            self.handle_select_model_submenu(self.test_specific)

        elif self.current_menu == "save":
            self.handle_select_model_submenu(self.save_model)

        return True

    def handle_main_submenu(self):
        if self.current_option == 0:
            self.current_menu = "train"
            self.current_option = 0
        elif self.current_option == 1:
            self.current_menu = "test"
            self.current_option = 0
        elif self.current_option == 2:
            self.current_menu = "save"
            self.current_option = 0
        elif self.current_option == 3:
            self.current_menu = "reprocess"
            self.current_option = 0
        elif self.current_option == 4:
            return False

        return True

    def handle_select_model_submenu(self, callback):
        if self.current_option == 0:
            callback(RF_NAME)
        elif self.current_option == 1:
            callback(MLP_NAME)
        elif self.current_option == 2:
            callback(CNN_NAME)
        elif self.current_option == 3:
            self.current_menu = "main"
            self.current_option = 0

    def handle_reprocess_submenu(self):
        if self.current_option == 0:
            process_features_for_all_files()
        elif self.current_option == 1:
            process_spectrograms_for_all_files()
        elif self.current_option == 2:
            self.current_menu = "main"
            self.current_option = 0

    def run(self):
        while True:
            self.print_menu()

            key = self.get_key()

            if key == "UP" or key == b"H":  # Seta para cima
                max_options = len(self.menus[self.current_menu]["options"]) - 1
                self.current_option = (self.current_option - 1) % (max_options + 1)
            elif key == "DOWN" or key == b"P":  # Seta para baixo
                max_options = len(self.menus[self.current_menu]["options"]) - 1
                self.current_option = (self.current_option + 1) % (max_options + 1)
            elif key == "\r" or key == b"\r":  # ENTER
                if not self.handle_menu_selection():
                    break


def main():
    cli = SubmenuCLI()
    cli.run()
    print("👋 Até logo!")


if __name__ == "__main__":
    main()
