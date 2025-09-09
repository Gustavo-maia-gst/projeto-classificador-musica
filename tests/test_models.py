from with_features.mlp import MLPGenreClassifier as MLP
from with_features.random_forest import RandomForestGenreClassifier as RF
from with_features.svm import SVMGenreClassifier as SVM
from with_spectrogram.cnn import CNNGenreClassifier as CNN
from unittest.mock import patch
import builtins
import copy
import pytest
from inspect import getfullargspec
import pickle
from pathlib import Path

rprint = print
model_classes = [MLP, RF, SVM, CNN]

def training_model(model):
    if "epochs" in getfullargspec(model.train).args:
        val = model.train(epochs = 1)[0]
    else:
        val = model.train()
    return (model, val)

@pytest.fixture(params=model_classes)
def model_class(request):
    cls = request.param
    return cls

@pytest.fixture
def untrained_model(model_class):
    return model_class()

trained_cache = {}

@pytest.fixture
def trained_model(request, model_class):
    cache_dir = Path(request.config.workerinput["cache_dir"])
    path = cache_dir / f"{model_class.__name__}.pkl"
    if not path.exists():
        # fallback se o master não treinou esse modelo ainda
        m = model_class()
        m.load_data()
        m = training_model(m)[0]
        with open(path, "wb") as f:
            pickle.dump(m, f)
    with open(path, "rb") as f:
        return pickle.load(f)

def test(model_class):
    model = model_class()
    assert  isinstance(model, model_class) 

def test_load_data(untrained_model):
    model = untrained_model
    with patch("builtins.print") as mock_print:
        model.load_data()
        
        saida = [" ".join(map(str, args)) for args, _ in mock_print.call_args_list][0]
        assert "Dados carregados" in saida

def test_train(untrained_model):
    model = untrained_model
    with patch("builtins.print") as mock_print:
        model.load_data()
        model, return_value = training_model(model)

        assert isinstance(model.is_trained,bool)
        assert model.is_trained

        chamadas = mock_print.call_args_list

        saidas = [" ".join(map(str, args)) for args, _ in mock_print.call_args_list]

        assert any("Acurácia" in s for s in saidas)
        assert any("Relatório de Classificação:" in s for s in saidas)

        assert isinstance(return_value,float)
        assert 0 <= return_value <= 1
        
def test_predict_without_training(untrained_model):
    model = untrained_model

    with patch("builtins.print") as mock_print:
        return_value = model.predict("./data/blues/blues.00000.wav")

        saida = [" ".join(map(str, args)) for args, _ in mock_print.call_args_list][0]

        assert "Erro: Modelo não foi treinado!" in saida
        assert return_value is None

def test_predict_file_path_wrong(trained_model):
    model = trained_model

    with patch("builtins.print") as mock_print:
        file_path = "./pasta_nao_existente/musica_nao_existente.wav"
        return_value = model.predict(file_path)

        saidas = [" ".join(map(str, args)) for args, _ in mock_print.call_args_list]

        assert any(f"Erro: Arquivo {file_path} não encontrado!" in s for s in saidas)
        assert return_value is None

def test_predict(trained_model):
    model = trained_model

    file_path = "./data/blues/blues.00000.wav"
    predict, probabilities = model.predict(file_path)

    probabilities = list(probabilities)
    assert predict in range(0, 10)
    assert len(probabilities) == 10
    assert probabilities.index(max(probabilities)) == predict

def test_save_model_without_training(untrained_model):
    model = untrained_model

    with patch("builtins.print") as mock_print:
        return_value = model.save_model()

        saida = [" ".join(map(str, args)) for args, _ in mock_print.call_args_list][0]

        assert "Erro: Modelo não foi treinado!" in saida
        assert return_value is None

def test_save_model(trained_model):
    model = trained_model

    with patch("builtins.print") as mock_print:
        model.save_model()

        saidas = [" ".join(map(str, args)) for args, _ in mock_print.call_args_list]

        assert any(f"Modelo salvo em:" in s for s in saidas)

def test_load_model_file_path_wrong(trained_model):
    model = trained_model

    with patch("builtins.print") as mock_print:
        file_path = "./pasta_nao_existente/arquivo_nao_existente"
        model.load_model(file_path)

        saidas = [" ".join(map(str, args)) for args, _ in mock_print.call_args_list]

        assert any(f"Erro: Arquivo {file_path} não encontrado" in s for s in saidas)

    
def test_load_model(untrained_model):
    model = untrained_model

    with patch("builtins.print") as mock_print:
        model.load_model()

        saidas = [" ".join(map(str, args)) for args, _ in mock_print.call_args_list]

        assert any(f"Modelo carregado" in s for s in saidas)