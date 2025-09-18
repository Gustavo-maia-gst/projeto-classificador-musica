"""
Interface de linha de comando para treinar, testar e salvar modelos
de classificação de gêneros musicais.
"""

import os
import sys

if os.name != "nt":
    import tty
    import termios

from with_features.extractor import process_features_for_all_files
from with_features.random_forest import RandomForestGenreClassifier
from with_features.mlp import MLPGenreClassifier
from with_features.svm import SVMGenreClassifier

from with_spectrogram.extractor import process_spectrograms_for_all_files
from with_spectrogram.cnn import CNNGenreClassifier
from with_spectrogram.crnn import CRNNGenreClassifier

# Constantes com os nomes dos modelos
MLP_NAME = "MLP"
RF_NAME = "Random Forest"
SVM_NAME = "SVM"
CNN_NAME = "CNN"
CRNN_NAME = "CRNN"


class SubmenuCLI:  # pylint: disable=too-many-instance-attributes
    """Interface de linha de comando para seleção e execução de modelos."""

    def __init__(self):
        self.rf_classifier = RandomForestGenreClassifier()
        self.mlp_classifier = MLPGenreClassifier()
        self.svm_classifier = SVMGenreClassifier()
        self.cnn_classifier = CNNGenreClassifier()
        self.crnn_classifier = CRNNGenreClassifier()
        self.current_option = 0
        self.current_menu = "main"
        self.current_model = None

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
                "options": [
                    "🌲 Treinar Random Forest",
                    "🧠 Treinar MLP",
                    "📐 Treinar SVM",
                    "🖼️ Treinar CNN",
                    "🖼️ Treinar CRNN",
                    "⬅️  Voltar",
                ],
            },
            "test": {
                "title": "🧪 TESTAR MODELO",
                "options": [
                    "🌲 Testar Random Forest",
                    "🧠 Testar MLP",
                    "📐 Testar SVM",
                    "🖼️ Testar CNN",
                    "🖼️ Testar CRNN",
                    "⬅️  Voltar",
                ],
            },
            "save": {
                "title": "💾 SALVAR MODELO",
                "options": [
                    "🌲 Salvar Random Forest",
                    "🧠 Salvar MLP",
                    "📐 Salvar SVM",
                    "🖼️ Salvar CNN",
                    "🖼️ Salvar CRNN",
                    "⬅️  Voltar",
                ],
            },
            "reprocess": {
                "title": "📊 REPROCESSAR DADOS",
                "options": [
                    "🔄 Reprocessar features",
                    "🔄 Reprocessar espectrogramas",
                    "⬅️  Voltar",
                ],
            },
        }

    def clear_screen(self):
        """Limpa a tela."""
        os.system("cls" if os.name == "nt" else "clear")

    def print_menu(self):
        """Imprime o menu atual."""
        self.clear_screen()

        menu = self.menus[self.current_menu]
        print(f"\n{menu['title']}\n")
        print("Selecione uma opção:\n")

        for i, option in enumerate(menu["options"]):
            prefix = "[X]" if i == self.current_option else "[ ]"
            print(f"  {prefix} {option}")

        print("\nUse ↑↓ para navegar, ENTER para selecionar")

    def get_key(self):
        """Captura uma tecla pressionada."""
        if os.name == "nt":  # Windows
            import msvcrt
            return msvcrt.getch()

        # Linux/Mac
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":  # ESC
                if sys.stdin.read(1) == "[":
                    ch = sys.stdin.read(1)
                    if ch == "A":
                        return "UP"
                    if ch == "B":
                        return "DOWN"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def get_model(self, model_type):
        """Retorna o modelo correspondente ao tipo informado."""
        model_map = {
            RF_NAME: self.rf_classifier,
            MLP_NAME: self.mlp_classifier,
            SVM_NAME: self.svm_classifier,
            CNN_NAME: self.cnn_classifier,
            CRNN_NAME: self.crnn_classifier,
        }
        model = model_map.get(model_type)
        if model:
            self.try_load_model(model)
        return model

    def try_load_model(self, model):
        """Carrega o modelo do disco se ele ainda não estiver treinado."""
        if model.is_trained:
            return
        try:
            model.load_model()
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    def train_model(self, model_type):
        """Treina um modelo do tipo especificado."""
        model = self.get_model(model_type)
        try:
            print(f"🔄 Carregando dados para {model_type}...")
            model.load_data()

            print(f"🚀 Treinando {model_type}...")
            accuracy = model.train()

            print(f"✅ {model_type} treinado com acurácia: {accuracy:.4f}")
            self.current_model = model
        except Exception as err:  # pylint: disable=broad-exception-caught
            print(f"❌ Erro ao treinar {model_type}: {err}")
        input("Pressione ENTER para continuar...")

    def test_specific(self, model_type):
        """Testa um modelo com um arquivo de áudio informado pelo usuário."""
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

            prediction, probabilities = model.predict(file_path)
            if prediction is None:
                print("❌ Erro na predição")
                input("Pressione ENTER para continuar...")
                return

            print("\n🎯 PREDIÇÃO DA AMOSTRA:")
            print(f"Gênero predito: {prediction}")
            print("Top 3 Probabilidades:")
            sorted_probs = sorted(
                enumerate(probabilities), key=lambda x: x[1], reverse=True
            )
            for class_idx, prob in sorted_probs[:3]:
                genre = model.classes[class_idx]
                print(f"  {genre}: {prob:.4f}")
        except Exception as err:  # pylint: disable=broad-exception-caught
            print(f"❌ Erro na predição: {err}")
        input("Pressione ENTER para continuar...")

    def save_model(self, model_type):
        """Salva o modelo do tipo especificado."""
        model = self.get_model(model_type)
        model.save_model()
        input("Pressione ENTER para continuar...")

    def handle_menu_selection(self):
        """Processa a seleção do menu atual."""
        if self.current_menu == "main":
            return self.handle_main_submenu()
        if self.current_menu == "reprocess":
            self.handle_reprocess_submenu()
        elif self.current_menu == "train":
            self.handle_select_model_submenu(self.train_model)
        elif self.current_menu == "test":
            self.handle_select_model_submenu(self.test_specific)
        elif self.current_menu == "save":
            self.handle_select_model_submenu(self.save_model)
        return True

    def handle_main_submenu(self):
        """Gerencia a navegação do menu principal."""
        if self.current_option == 0:
            self.current_menu = "train"
        elif self.current_option == 1:
            self.current_menu = "test"
        elif self.current_option == 2:
            self.current_menu = "save"
        elif self.current_option == 3:
            self.current_menu = "reprocess"
        elif self.current_option == 4:
            return False
        self.current_option = 0
        return True

    def handle_select_model_submenu(self, callback):
        """Executa a ação do submenu de seleção de modelos."""
        model_types = [RF_NAME, MLP_NAME, SVM_NAME, CNN_NAME, CRNN_NAME]
        if self.current_option < 5:
            callback(model_types[self.current_option])
        else:
            self.current_menu = "main"
            self.current_option = 0

    def handle_reprocess_submenu(self):
        """Executa as ações do submenu de reprocessamento."""
        if self.current_option == 0:
            process_features_for_all_files()
        elif self.current_option == 1:
            process_spectrograms_for_all_files()
        else:
            self.current_menu = "main"
        self.current_option = 0

    def run(self):
        """Loop principal da interface CLI."""
        while True:
            self.print_menu()
            key = self.get_key()
            max_options = len(self.menus[self.current_menu]["options"]) - 1

            if key in ("UP", b"H"):
                self.current_option = (self.current_option - 1) % (max_options + 1)
            elif key in ("DOWN", b"P"):
                self.current_option = (self.current_option + 1) % (max_options + 1)
            elif key in ("\r", b"\r"):
                if not self.handle_menu_selection():
                    break


def main():
    """Ponto de entrada da CLI."""
    cli = SubmenuCLI()
    cli.run()
    print("👋 Até logo!")


if __name__ == "__main__":
    main()
