from with_features.mlp import MLPGenreClassifier as MLP
from unittest.mock import patch
import builtins

def test():
    model = MLP()
    assert  isinstance(model, MLP) 

def test_load_data():
    model = MLP()

    with patch("builtins.print") as mock_print:
        model.load_data()
        
        saida = [" ".join(map(str, args)) for args, _ in mock_print.call_args_list][0]
        assert "Dados carregados" in saida

def test_train():
    model = MLP()


    with patch("builtins.print") as mock_print:
        model.load_data()
        return_value = model.train()

        assert model.is_trained

        chamadas = mock_print.call_args_list

        saidas = [" ".join(map(str, args)) for args, _ in mock_print.call_args_list]

        assert any("Acurácia:" in s for s in saidas)
        assert any("Relatório de Classificação:" in s for s in saidas)

        assert isinstance(return_value,float)
        assert 0 <= return_value <= 1
        
def test_predict_without_training():
    model = MLP()

    with patch("builtins.print") as mock_print:
        return_value = model.predict("./data/blues/blues.00000.wav")

        saida = [" ".join(map(str, args)) for args, _ in mock_print.call_args_list][0]

        assert "Erro: Modelo não foi treinado!" in saida
        assert return_value is None

def test_predict_file_path_wrong():
    model = MLP()

    with patch("builtins.print") as mock_print:
        model.load_data()
        model.train()
        file_path = "./pasta_nao_existente/musica_nao_existente.wav"
        return_value = model.predict(file_path)

        saidas = [" ".join(map(str, args)) for args, _ in mock_print.call_args_list]

        assert any(f"Erro: Arquivo {file_path} não encontrado!" in s for s in saidas)
        assert return_value is None

def test_predict():
    model = MLP()

    real_print = print
    with patch("builtins.print") as mock_print:
        model.load_data()
        model.train()
    file_path = "./data/blues/blues.00000.wav"
    predict, probabilities = model.predict(file_path)

    probabilities = list(probabilities)
    assert predict in range(0, 10)
    assert len(probabilities) == 10
    assert probabilities.index(max(probabilities)) == predict

def test_save_model_without_training():
    model = MLP()

    with patch("builtins.print") as mock_print:
        return_value = model.save_model()

        saida = [" ".join(map(str, args)) for args, _ in mock_print.call_args_list][0]

        assert "Erro: Modelo não foi treinado!" in saida
        assert return_value is None

def test_save_model():
    model = MLP()

    with patch("builtins.print") as mock_print:
        model.load_data()
        model.train()
        return_value = model.save_model()

        saidas = [" ".join(map(str, args)) for args, _ in mock_print.call_args_list]

        assert any(f"Modelo salvo em:" in s for s in saidas)

def test_load_model():
    model = MLP()

    with patch("builtins.print") as mock_print:
        model.load_model()

        saidas = [" ".join(map(str, args)) for args, _ in mock_print.call_args_list]

        assert any(f"Modelo carregado" in s for s in saidas)