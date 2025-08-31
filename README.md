# 🎵 Classificador de Gêneros Musicais

Sistema de classificação automática de gêneros musicais usando machine learning. O projeto implementa 4 algoritmos diferentes: MLP, CNN, SVM e Random Forest.

## Como Usar

### 1. Instalação
```bash
pip install -r requirements.txt
```

### 2. Executar o Sistema
```bash
python src/cli.py # é importante ser da pasta root
```

### 3. Fluxo de Uso

#### Navegação

- **↑↓**: Navegar no menu
- **ENTER**: Selecionar opção

#### Treinar e Testar:
1. **"🚀 Treinar modelo"** → Escolha um modelo
2. **"🧪 Testar modelo"** → Escolha o modelo treinado → Digite caminho do arquivo .wav/.mp3
3. **"💾 Salvar modelo"** → Para salvar o modelo treinado

#### Modelos Disponíveis

- **🖼️ CNN**
- **🧠 MLP**
- **📐 SVM**
- **🌲 Random Forest**

#### Para reprocessar o dataset:
1. **"📊 Reprocessar dados"** → **"🔄 Reprocessar features"**
2. **"📊 Reprocessar dados"** → **"🔄 Reprocessar espectrogramas"**

## Dados

- **10 gêneros**: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- **100 músicas por gênero** (1000 total)
- **Formato**: Arquivos .wav

- As features são extraídas e concatenadas em um vetor, elas são: 
  - MFCC
  - Chroma features
  - Spectral Contrast
  - Tonnetz
  - RMS 
  - Spectral Centroid 

- Os espectrogramas são gerados a partir do mel spectrogram

## Estrutura

```
projeto-classificador-musica/
├── data/                   # Dataset com as músicas
├── src/
│   ├── cli.py              # Interface
│   ├── with_features/      # Modelos treinados a partir das features
│   └── with_spectrogram/   # CNN treinada a partir do espectrograma
└── requirements.txt
```