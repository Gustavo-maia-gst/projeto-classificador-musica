import pickle
from pathlib import Path

from with_features.mlp import MLPGenreClassifier as MLP
from with_features.random_forest import RandomForestGenreClassifier as RF
from with_features.svm import SVMGenreClassifier as SVM
from with_spectrogram.cnn import CNNGenreClassifier as CNN
from inspect import getfullargspec

CACHE_DIR = Path(".pytest_model_cache")
CACHE_DIR.mkdir(exist_ok=True)

def training_model(model):
    if "epochs" in getfullargspec(model.train).args:
        val = model.train(epochs = 1)[0]
    else:
        val = model.train()
    return model

# Lista de modelos que queremos pré-treinar
MODEL_CLASSES = [MLP, RF, SVM, CNN]

def pytest_configure(config):
    """Executa no processo master (não nos workers)."""
    if not hasattr(config, "workerinput"):
        # Só roda no master ou sem -n
        for model_class in MODEL_CLASSES:
            model = model_class()
            model.load_data()
            model = training_model(model)

            path = CACHE_DIR / f"{model_class.__name__}.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)

def pytest_configure_node(node):
    """Executa no master para configurar cada worker."""
    node.workerinput["cache_dir"] = str(CACHE_DIR)
