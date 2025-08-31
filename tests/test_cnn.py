from with_spectrogram.cnn import CNNGenreClassifier
import numpy as np
import pytest
import joblib
from unittest.mock import patch, MagicMock
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


@pytest.fixture
def fake_data(tmp_path):
    """Cria dados falsos para treino"""
    # tem que haver muitas amostras
    X = np.random.rand(30, 128, 128).astype(np.float32)
    y = np.array([
        "rock", "jazz", "rock", "pop", "pop", "jazz", "rock", "rock", "jazz", "pop",
        "rock", "jazz", "rock", "pop", "pop", "jazz", "rock", "rock", "jazz", "pop", 
        "rock", "jazz", "rock", "pop", "pop", "jazz", "rock", "rock", "jazz", "pop"
    ])
    X_path = tmp_path / "X.npy"
    Y_path = tmp_path / "Y.npy"
    np.save(X_path, X)
    np.save(Y_path, y)
    return str(X_path), str(Y_path), X, y


@pytest.fixture
def trained_classifier(fake_data):
    """Cria um classificador com dados carregados."""
    X_path, Y_path, _, _ = fake_data
    clf = CNNGenreClassifier(num_classes=3)
    clf.load_data(X_path, Y_path)
    clf.is_trained = True
    return clf


def test_build_model():
    clf = CNNGenreClassifier(num_classes=3)
    assert clf.model is not None
    assert clf.model.input_shape[1:] == (128, 128, 1)
    assert not clf.is_trained


def test_load_data(fake_data):
    X_path, Y_path, X_original, y_original = fake_data
    clf = CNNGenreClassifier(num_classes=3)
    clf.load_data(X_path, Y_path)
    
    assert clf.X.shape[1:] == (128, 128, 1)
    assert clf.X.shape[0] == 30  # 30 amostras
    assert len(clf.y_encoded) == 30
    assert hasattr(clf, "classes")
    assert np.array_equal(np.unique(y_original), clf.classes)
    
    # Verifica normalização dos dados
    assert clf.X.min() >= 0
    assert clf.X.max() <= 1


def test_label_encoding(fake_data):
    X_path, Y_path, _, y_original = fake_data
    clf = CNNGenreClassifier(num_classes=3)
    clf.load_data(X_path, Y_path)
    
    # Verifica se as labels foram codificadas corretamente
    unique_labels = np.unique(y_original)
    assert len(clf.classes) == len(unique_labels)
    
    # Verifica se todas as labels originais estão nas classes
    for label in y_original:
        assert label in clf.classes


def test_data_normalization(fake_data):
    X_path, Y_path, X_original, _ = fake_data
    clf = CNNGenreClassifier(num_classes=3)
    clf.load_data(X_path, Y_path)
    
    # Verifica se os dados foram normalizados para [0, 1]
    assert clf.X.min() >= 0
    assert clf.X.max() <= 1
    assert not np.array_equal(clf.X, X_original.reshape(-1, 128, 128, 1))


def test_train_and_predict(fake_data):
    X_path, Y_path, _, _ = fake_data
    clf = CNNGenreClassifier(num_classes=3)
    clf.load_data(X_path, Y_path)

    # Mock para não treinar de verdade
    with patch.object(clf.model, "fit", return_value=MagicMock(history={"accuracy": [1.0]})):
        with patch.object(clf.model, "predict", return_value=np.array([[0.1, 0.7, 0.2]])):
            # Mock para evitar problema de tamanho
            with patch('with_spectrogram.cnn.train_test_split') as mock_split:
                # Configura mock para retornar dados válidos
                mock_X_train = np.random.rand(24, 128, 128, 1)
                mock_X_test = np.random.rand(6, 128, 128, 1)
                mock_y_train = np.random.randint(0, 3, 24)
                mock_y_test = np.random.randint(0, 3, 6)
                mock_split.return_value = (mock_X_train, mock_X_test, mock_y_train, mock_y_test)
                
                acc, hist = clf.train(epochs=1, batch_size=2)
                assert isinstance(acc, float)
                assert clf.is_trained

                # Testa predict com espectrograma 2D ((128, 128) provavelmente)
                spectrogram = clf.X[0]
                pred, probs = clf.predict(spectrogram)
                assert pred in clf.classes
                assert len(probs) == 3
                assert abs(sum(probs) - 1.0) < 1e-6  # Probabilidades somam =~ 1


def test_predict_different_formats(trained_classifier):
    """Testa predição com diferentes formatos de entrada."""
    clf = trained_classifier
    
    # Mock predict
    mock_probs = np.array([0.1, 0.7, 0.2])
    with patch.object(clf.model, "predict", return_value=np.array([mock_probs])):
        
        # Teste com array 2D 
        spectrogram_2d = np.random.rand(128, 128)
        pred, probs = clf.predict(spectrogram_2d)
        assert pred in clf.classes
        assert np.array_equal(probs, mock_probs)
        
        # Teste com array 3D 
        spectrogram_3d = np.random.rand(1, 128, 128)
        pred, probs = clf.predict(spectrogram_3d)
        assert pred in clf.classes
        
        # Teste com array 4D 
        spectrogram_4d = np.random.rand(1, 128, 128, 1)
        pred, probs = clf.predict(spectrogram_4d)
        assert pred in clf.classes


def test_predict_without_training():
    clf = CNNGenreClassifier(num_classes=3)
    result = clf.predict(np.random.rand(128, 128))
    assert result is None


def test_predict_not_trained_after_loading(tmp_path):
    clf = CNNGenreClassifier(num_classes=3)
    
    # Tenta predict sem treinar nem carregar
    result = clf.predict(np.random.rand(128, 128))
    assert result is None


def test_train_callbacks(fake_data):
    # callbacks corretos
    X_path, Y_path, _, _ = fake_data
    clf = CNNGenreClassifier(num_classes=3)
    clf.load_data(X_path, Y_path)
    
    with patch.object(clf.model, "fit") as mock_fit:
        with patch.object(clf.model, "predict", return_value=np.array([[0.1, 0.7, 0.2]])):
            # Mock para evitar problema de tamanho
            with patch('with_spectrogram.cnn.train_test_split') as mock_split:
                # Configura mock para retornar dados válidos
                mock_X_train = np.random.rand(24, 128, 128, 1)
                mock_X_test = np.random.rand(6, 128, 128, 1)
                mock_y_train = np.random.randint(0, 3, 24)
                mock_y_test = np.random.randint(0, 3, 6)
                mock_split.return_value = (mock_X_train, mock_X_test, mock_y_train, mock_y_test)
                
                clf.train(epochs=1, batch_size=2)
                
                # Verifica se fit foi chamado com callbacks
                call_args = mock_fit.call_args
                assert call_args is not None
                assert "callbacks" in call_args[1]
                assert len(call_args[1]["callbacks"]) > 0


def test_save_and_load_model(fake_data, tmp_path):
    X_path, Y_path, _, _ = fake_data
    clf = CNNGenreClassifier(num_classes=3)
    clf.load_data(X_path, Y_path)
    clf.is_trained = True

    filepath = tmp_path / "cnn.model"

    with patch("joblib.dump") as mock_dump:
        clf.save_model(filepath)
        mock_dump.assert_called_once()

    # Mocka load
    mock_model_data = {
        "model": clf.model,
        "label_encoder": clf.label_encoder,
        "classes": clf.classes
    }
    
    with patch("joblib.load", return_value=mock_model_data):
        with patch("os.path.exists", return_value=True):
            clf2 = CNNGenreClassifier(num_classes=3)
            clf2.load_model(filepath)
            assert clf2.is_trained is True
            assert clf2.model is not None
            assert clf2.label_encoder is not None
            assert clf2.classes is not None


def test_save_model_without_training(tmp_path):
    clf = CNNGenreClassifier(num_classes=3)
    filepath = tmp_path / "cnn.model"
    
    # Não deve salvar se não estiver treinado
    with patch("joblib.dump") as mock_dump:
        clf.save_model(filepath)
        mock_dump.assert_not_called()


def test_load_model_file_not_found(tmp_path):
    clf = CNNGenreClassifier()
    
    with patch("os.path.exists", return_value=False):
        clf.load_model(tmp_path / "inexistente.model")
        assert clf.is_trained is False


def test_model_compilation():
    """Testa se o modelo foi compilado corretamente."""
    clf = CNNGenreClassifier(num_classes=3)
    
    assert hasattr(clf.model, "optimizer")
    assert hasattr(clf.model, "loss")
    assert hasattr(clf.model, "metrics")
    assert clf.model.loss == "sparse_categorical_crossentropy"
    
    # Verifica métricas de forma flexível
    metrics_names = []
    if hasattr(clf.model, 'metrics'):
        for metric in clf.model.metrics:
            if hasattr(metric, 'name'):
                metrics_names.append(metric.name)
            elif hasattr(metric, '__name__'):
                metrics_names.append(metric.__name__)
    
    # Verifica se tem accuracy
    has_accuracy = any('accuracy' in str(name).lower() or 'acc' in str(name).lower() 
                      for name in metrics_names)
    assert has_accuracy, f"Métricas encontradas: {metrics_names}"


def test_predict_proba_sum(trained_classifier):
    """Testa se as probabilidades somam =~ 1."""
    clf = trained_classifier
    
    mock_probs = np.array([0.2, 0.5, 0.3])
    with patch.object(clf.model, "predict", return_value=np.array([mock_probs])):
        pred, probs = clf.predict(np.random.rand(128, 128))
        assert abs(sum(probs) - 1.0) < 1e-6


def test_classes_attribute_consistency(fake_data):
    """Testa consistência entre classes e label_encoder."""
    X_path, Y_path, _, y_original = fake_data
    clf = CNNGenreClassifier(num_classes=3)
    clf.load_data(X_path, Y_path)
    
    # Verifica se classes e label_encoder estão sincronizados
    assert len(clf.classes) == len(clf.label_encoder.classes_)
    assert np.array_equal(clf.classes, clf.label_encoder.classes_)
    
    # Verifica se a codificação é reversível
    for i, class_name in enumerate(clf.classes):
        encoded = clf.label_encoder.transform([class_name])[0]
        decoded = clf.label_encoder.inverse_transform([encoded])[0]
        assert decoded == class_name


def test_train_with_insufficient_data():
    """Testa comportamento com dados insuficientes."""
    clf = CNNGenreClassifier(num_classes=3)
    
    # Dados com poucas amostras
    X = np.random.rand(5, 128, 128).astype(np.float32)
    y = np.array(["rock", "jazz", "rock", "pop", "pop"])
    
    # Mock load
    with patch('with_spectrogram.cnn.np.load') as mock_load:
        mock_load.side_effect = [X, y]
        with patch('with_spectrogram.cnn.os.path.exists', return_value=True):
            clf.load_data('X.npy', 'Y.npy')
            
            # Deve falhar 
            with pytest.raises(ValueError):
                clf.train(epochs=1, batch_size=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])