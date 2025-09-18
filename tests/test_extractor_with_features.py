import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from with_features import extractor
import pathlib


# Fixture para mock do librosa

# Mocka todas as funções do librosa usadas em extract_features para evitar crashes no Windows
# e tornar o teste independente de arquivos de áudio reais.
@pytest.fixture(autouse=True)
def mock_librosa_features():
    with patch("with_features.extractor.librosa.feature.rms", return_value=np.ones((1, 10))), \
            patch("with_features.extractor.librosa.feature.mfcc", return_value=np.ones((20, 10))), \
            patch("with_features.extractor.librosa.feature.chroma_stft", return_value=np.ones((12, 10))), \
            patch("with_features.extractor.librosa.feature.spectral_contrast", return_value=np.ones((7, 10))), \
            patch("with_features.extractor.librosa.effects.harmonic", side_effect=lambda y: y), \
            patch("with_features.extractor.librosa.feature.tonnetz", return_value=np.ones((6, 10))), \
            patch("with_features.extractor.librosa.feature.spectral_centroid", return_value=np.ones((1, 10))):
        yield  # mocks ativos durante todos os testes automaticamente


# Testa extract_features diretamente

def test_extract_features_shape():
    # Sinal sintético simples
    y = np.zeros(22050)
    sr = 22050

    # Chama a função a ser testada
    features = extractor.extract_features(y, sr)

    # Verifica tipo e dimensionalidade do array retornado
    assert isinstance(features, np.ndarray)
    assert features.ndim == 1

    # Verifica se o tamanho do vetor de features corresponde ao esperado:
    # rms + mfcc_mean+var + chroma_mean+var + contrast_mean+var + tonnetz_mean+var + spectral_centroid
    expected_size = 1 + 20 + 20 + 12 + 12 + 7 + 7 + 6 + 6 + 1
    assert features.shape[0] == expected_size


# Testa extract_features_from_file

@patch("with_features.extractor.librosa.load", return_value=(np.zeros(22050), 22050))
def test_extract_features_from_file(mock_load, tmp_path):
    # Cria arquivo falso de áudio em diretório de gênero
    fake_file = tmp_path / "rock" / "song.wav"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("fake-wav")  # conteúdo irrelevante

    # Chama a função
    features, label = extractor.extract_features_from_file(str(fake_file))

    # Verifica tipo e rótulo
    assert isinstance(features, np.ndarray)
    assert label == "rock"

    # Verifica se librosa.load foi chamado corretamente
    mock_load.assert_called_once_with(str(fake_file), sr=22050, mono=True)



# Testa process_features_for_all_files

def fake_extract_features_from_file(path):
    return np.ones(66), "rock"

def test_process_features_for_all_files(monkeypatch, tmp_path):
    # captura a função np.save original ANTES de monkeypatchar
    original_save = extractor.np.save

    # substitui extract_features_from_file por função top-level (picklable)
    monkeypatch.setattr(extractor, "extract_features_from_file", fake_extract_features_from_file)

    # fake para os.listdir / isdir (simula 1 gênero e 1 arquivo .wav)
    def fake_listdir(path):
        cnt = getattr(fake_listdir, "cnt", 0)
        fake_listdir.cnt = cnt + 1
        return ["rock"] if cnt == 0 else ["song1.wav"]

    monkeypatch.setattr(extractor.os, "listdir", fake_listdir)
    monkeypatch.setattr(extractor.os.path, "isdir", lambda p: True)

    # prepara tmp dir
    extractor.AUDIO_FOLDER = str(tmp_path)
    (tmp_path / "rock").mkdir()

    # fake_np_save grava os .npy no tmp_path usando a função original_save (não recursiva)
    def fake_np_save(file, arr, *args, **kwargs):
        file_name = pathlib.Path(file).name
        dest = tmp_path / file_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        # chama a função original que salvava arquivos (não será recursiva)
        original_save(str(dest), arr, *args, **kwargs)

    # aplica o monkeypatch na função save do módulo extractor (substitui por nossa fake)
    monkeypatch.setattr(extractor.np, "save", fake_np_save)

    # executa
    extractor.process_features_for_all_files()

    # asserções
    assert (tmp_path / "X.npy").exists(), "X.npy não foi criado em tmp_path"
    assert (tmp_path / "Y.npy").exists(), "Y.npy não foi criado em tmp_path"