from with_spectrogram import extractor
from with_spectrogram.extractor import extract_spectrogram, extract_spectrogram_from_file, process_spectrograms_for_all_files, TARGET_LENGTH
import pytest
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock
import librosa

class TestSpectrogramExtractor:
    
    def test_extract_spectrogram_shape(self):
        """Testa se o espectrograma tem a forma correta 2D (128, 128)"""
        # Cria um sinal de áudio sintético
        sample_rate = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        y = 0.5 * np.sin(2 * np.pi * 440 * t)  # tom 440hz
        
        spectrogram = extract_spectrogram(y, sample_rate)
        
        # Verifica se tem shape correto
        assert spectrogram.shape == (128, 128)
        assert isinstance(spectrogram, np.ndarray)
    
    def test_extract_spectrogram_values_range(self):
        """Testa se os valores do espectrograma estão em uma faixa razoável em decibéis"""
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        y = 0.8 * np.sin(2 * np.pi * 1000 * t)  
        
        spectrogram = extract_spectrogram(y, sample_rate)
        
        # Verifica se os valores estão em decíbeis (normalmente os valores são < 0)
        assert spectrogram.max() <= 0  
        assert spectrogram.min() >= -80  # Valor mínimo razoável para áudio
    
    def test_extract_spectrogram_padding_short_audio(self):
        """Testa se os audios tem mesmo tamanho quando muito pequenos (padding)"""
        sample_rate = 22050
        duration = 0.5  # audio curto
        t = np.linspace(0, duration, int(sample_rate * duration))
        y = np.sin(2 * np.pi * 440 * t)
        
        spectrogram = extract_spectrogram(y, sample_rate)
        
        # Deve ter esse formato mesmo muito curto 
        assert spectrogram.shape == (128, 128)
    
    def test_extract_spectrogram_cropping_long_audio(self):
        """Testa se os cortes no tamanho de um sinal grande funcionam corretamente (cropping)"""
        sample_rate = 22050
        duration = 10.0  # audio longo
        t = np.linspace(0, duration, int(sample_rate * duration))
        y = np.sin(2 * np.pi * 440 * t)
        
        spectrogram = extract_spectrogram(y, sample_rate)
        
        # Deve ter esse shape (com corte aplicado)
        assert spectrogram.shape == (128, 128)
    
    def test_extract_spectrogram_from_file_invalid_path(self):
        """Testa comportamento com caminho de arquivo inválido"""
        with pytest.raises(FileNotFoundError):
            extract_spectrogram_from_file("caminho/inexistente/arquivo.wav")
    
    def test_extract_spectrogram_from_file_mock(self, monkeypatch):
        # mock librosa.load
        mock_y = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 66150))
        mock_sr = 22050
        monkeypatch.setattr(extractor.librosa, "load", lambda path, sr=22050, mono=True: (mock_y, mock_sr))

        # monkeypatch apenas dirname/basename (não todo os.path)
        monkeypatch.setattr(extractor.os.path, "dirname", lambda p: "/caminho/para/blues")
        monkeypatch.setattr(extractor.os.path, "basename", lambda p: "blues")

        spectrogram, label = extract_spectrogram_from_file("/caminho/para/blues/arquivo.wav")

        assert spectrogram.shape == (128, 128)
        assert label == "blues"
    
    def test_extract_spectrogram_empty_signal(self):
        """Testa comportamento com sinal vazio"""
        empty_signal = np.array([])
        sample_rate = 22050
        
        with pytest.raises(ValueError):
            extract_spectrogram(empty_signal, sample_rate)
    
    def test_extract_spectrogram_constant_tone(self):
        """Testa com tom constante"""
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        constant_tone = 0.5 * np.sin(2 * np.pi * 1000 * t) 
        
        spectrogram = extract_spectrogram(constant_tone, sample_rate)
        
        # Shape correto
        assert spectrogram.shape == (128, 128)
        # Tom constante deve mostrar uma linha horizontal no espectrograma (quase sem variação vertical)
        vertical_variation = np.std(spectrogram, axis=0).mean()
        assert vertical_variation < 10  
    
    def test_process_spectrograms_for_all_files_mock(self,monkeypatch):
        # 1) controla listdir/isdir de forma simples
        def fake_listdir(path):
            # primeira chamada: pasta base -> retorna um único genero "blues"
            # chamada posterior para a pasta blues -> retorna um único arquivo .wav
            if path == extractor.AUDIO_FOLDER:
                return ['blues']
            if path.endswith('blues'):
                return ['blues.00000.wav']
            return []

        monkeypatch.setattr(extractor.os, 'listdir', fake_listdir)
        monkeypatch.setattr(extractor.os.path, 'isdir', lambda p: True)

        # 2) DummyFuture / DummyExecutor simples (ThreadPoolExecutor substituído)
        class DummyFuture:
            def __init__(self, res):
                self._res = res
            def result(self):
                return self._res

        class DummyExecutor:
            def __init__(self, *a, **k):
                pass
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def submit(self, fn, arg):
                # ignora fn/arg e retorna uma future com resultado previsível
                return DummyFuture((np.zeros((128, 128)), 'blues'))

        monkeypatch.setattr(extractor, 'ThreadPoolExecutor', DummyExecutor)

        # 3) Simplifica as_completed e tqdm para iteração direta
        monkeypatch.setattr(extractor, 'as_completed', lambda futures, **kw: iter(futures))
        monkeypatch.setattr(extractor, 'tqdm', lambda it, **kw: it)

        # 4) Captura chamadas de np.save para verificar gravação de X.npy / Y.npy
        saved = []
        def fake_save(path, arr, *a, **kw):
            saved.append(path)
        monkeypatch.setattr(extractor.np, 'save', fake_save)

        # 5) executar (usa extractor.AUDIO_FOLDER padrão "./data" — não precisa existir)
        extractor.AUDIO_FOLDER = './data'
        process_spectrograms_for_all_files()

        # 6) asserts simples: garantem que X.npy e Y.npy foram "salvos"
        assert any('X.npy' in str(p) for p in saved), f"X.npy não gravado. saved={saved}"
        assert any('Y.npy' in str(p) for p in saved), f"Y.npy não gravado. saved={saved}"
    
    @patch('with_spectrogram.extractor.ThreadPoolExecutor')  
    @patch('with_spectrogram.extractor.os.listdir')  
    def test_process_spectrograms_empty_folder(self, mock_listdir, mock_executor):
        """Testa processamento com array vazio"""
        mock_listdir.return_value = []  
        mock_executor.return_value.__enter__.return_value = MagicMock()
        mock_executor.return_value.__exit__.return_value = None
        
        with patch('with_spectrogram.extractor.np.save') as mock_save: 
            process_spectrograms_for_all_files()
            
            # Deve salvar arrays vazios
            assert mock_save.call_count == 2
    
    def test_extract_spectrogram_very_short_audio(self):
        """Testa com áudio muito curto"""
        sample_rate = 22050
        duration = 0.1  
        t = np.linspace(0, duration, int(sample_rate * duration))
        y = np.sin(2 * np.pi * 440 * t)
        
        spectrogram = extract_spectrogram(y, sample_rate)
        
        # Verifica se padding foi aplicado
        assert spectrogram.shape == (128, 128)
    
    def test_extract_spectrogram_different_sample_rates(self):
        """Testa com diferentes amostragens"""
        for sample_rate in [11025, 22050, 44100]:
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            y = 0.5 * np.sin(2 * np.pi * 440 * t)
            
            spectrogram = extract_spectrogram(y, sample_rate)
            
            # esse deve ser o shape independentemente do rate
            assert spectrogram.shape == (128, 128)
    
    def test_target_length_constant(self):
        """Testa se TARGET_LENGTH está definido corretamente"""
        assert TARGET_LENGTH == 128
        assert isinstance(TARGET_LENGTH, int)
    
    def test_extract_spectrogram_from_file_different_labels(self, monkeypatch):
        mock_y = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 66150))
        mock_sr = 22050
        monkeypatch.setattr('with_spectrogram.extractor.librosa.load',
                            lambda path, sr=22050, mono=True: (mock_y, mock_sr))

        test_cases = [
            ('/caminho/para/blues/song.wav', 'blues'),
            ('/caminho/para/rock/song.wav', 'rock'),
            ('/caminho/para/jazz/song.wav', 'jazz'),
            ('/caminho/para/classical/song.wav', 'classical')
        ]

        for file_path, expected_label in test_cases:
            monkeypatch.setattr('with_spectrogram.extractor.os.path.dirname',
                                lambda p, expected=expected_label: f'/caminho/para/{expected}')
            monkeypatch.setattr('with_spectrogram.extractor.os.path.basename',
                                lambda p, expected=expected_label: expected)

            _, label = extract_spectrogram_from_file(file_path)
            assert label == expected_label

            # desfaz os patchings para o próximo caso (monkeypatch faz isso automaticamente ao final do teste,
            # mas como aqui estamos sobrescrevendo em loop, restaure o original manualmente se necessário)
            monkeypatch.setattr('with_spectrogram.extractor.os.path.dirname', lambda p: os.path.dirname(p))
            monkeypatch.setattr('with_spectrogram.extractor.os.path.basename', lambda p: os.path.basename(p))
    
    def test_extract_spectrogram_high_frequency(self):
        """Testa altas frequencias"""
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Testa várias frequências altas
        for freq in [5000, 8000, 10000]:
            y = 0.5 * np.sin(2 * np.pi * freq * t)
            spectrogram = extract_spectrogram(y, sample_rate)
            
            assert spectrogram.shape == (128, 128)
            assert np.any(spectrogram > -40)  # Alguma frequencia detectada
    
    @patch('with_spectrogram.extractor.librosa.feature.melspectrogram')  
    @patch('with_spectrogram.extractor.librosa.power_to_db')  
    def test_extract_spectrogram_parameters(self, mock_power_to_db, mock_melspectrogram):
        """Testa se os parâmetros do melspectrogram estão corretos"""
        # Configura mocks
        mock_melspectrogram.return_value = np.random.rand(128, 100)
        mock_power_to_db.return_value = np.random.rand(128, 100)
        
        sample_rate = 22050
        y = np.sin(2 * np.pi * 440 * np.linspace(0, 2, sample_rate * 2))
        
        extract_spectrogram(y, sample_rate)
        
        # Verifica se melspectrogram foi chamado com parâmetros corretos
        mock_melspectrogram.assert_called_once()
        args, kwargs = mock_melspectrogram.call_args
        assert kwargs['sr'] == sample_rate
        assert kwargs['n_mels'] == 128
        assert kwargs['n_fft'] == 2048
        assert kwargs['hop_length'] == 512
    
    def test_process_spectrograms_for_all_files_no_audio_files(self, tmp_path):
        """Testa processamento quando não há arquivos de áudio"""
        # Cria estrutura de diretórios sem arquivos .wav
        blues_dir = tmp_path / "data" / "blues"
        blues_dir.mkdir(parents=True)
        
        rock_dir = tmp_path / "data" / "rock" 
        rock_dir.mkdir(parents=True)
        
        # Cria arquivos que não são .wav
        (blues_dir / "readme.txt").write_text("info")
        (rock_dir / "metadata.json").write_text("{}")
        
        with patch('with_spectrogram.extractor.AUDIO_FOLDER', str(tmp_path / "data")):
            with patch('with_spectrogram.extractor.np.save') as mock_save:
                process_spectrograms_for_all_files()
                
                # Deve salvar arrays vazios
                assert mock_save.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])