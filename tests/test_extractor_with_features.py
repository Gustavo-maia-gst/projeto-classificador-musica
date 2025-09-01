import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from with_features import extractor


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

@patch("with_features.extractor.extract_features_from_file", return_value=(np.ones(66), "rock"))
@patch("with_features.extractor.os.listdir")
@patch("with_features.extractor.os.path.isdir", return_value=True)
def test_process_features_for_all_files(mock_isdir, mock_listdir, mock_extract, tmp_path):
    # Simula diretórios de gêneros e arquivos .wav
    mock_listdir.side_effect = [
        ["rock"],  # gênero
        ["song1.wav"],  # arquivos
    ]

    # Define a pasta base de áudio para o teste
    extractor.AUDIO_FOLDER = str(tmp_path)
    (tmp_path / "rock").mkdir()

    # Mock np.save para não criar arquivos reais
    with patch("with_features.extractor.np.save") as mock_save:
        extractor.process_features_for_all_files()

        # Verifica se X.npy e Y.npy foram "salvos"
        saved_files = [call.args[0] for call in mock_save.call_args_list]
        assert any("X.npy" in f for f in saved_files)
        assert any("Y.npy" in f for f in saved_files)

        # Verifica se pelo menos um arquivo foi processado
        mock_extract.assert_called_once()
