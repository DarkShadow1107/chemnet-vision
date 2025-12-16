# ChemNet-Vision v4.0

## Sistem Inteligent de Recunoaștere Molecule din Imagini

**Student:** Alexandru Gabriel  
**Proiect:** Rețele Neuronale - Sistem Inteligent Autonom  
**Data:** 03.12.2025

---

## 📋 Cuprins

1. [Descrierea Proiectului](#descrierea-proiectului)
2. [Structura Proiectului](#structura-proiectului)
3. [Arhitectura Rețelei Neuronale](#arhitectura-rețelei-neuronale)
4. [Dual Mode (AI/Fallback)](#dual-mode-aifallback)
5. [Rezultate Antrenament](#rezultate-antrenament)
6. [Instalare și Rulare](#instalare-și-rulare)
7. [API Endpoints](#api-endpoints)
8. [Tehnologii Utilizate](#tehnologii-utilizate)

---

## 🎯 Descrierea Proiectului

**ChemNet-Vision** este un sistem inteligent autonom care recunoaște molecule chimice din imagini 2D și generează reprezentarea lor SMILES (Simplified Molecular Input Line Entry System).

### Caracteristici principale:

-   🧠 **Rețea neuronală custom** - CNN+MLP+GNN+LSTM (15.3M parametri, antrenată de la zero)
-   🔄 **Dual Mode** - Toggle între AI și Fallback (database + RAG)
-   🧪 **Vizualizare 3D** - Molecule interactive cu 3Dmol.js
-   💬 **Chat conversațional** - Căutare molecule prin limbaj natural
-   📊 **42,149 molecule** - Dataset procesat din ChEMBL

### Problema rezolvată:

| Nevoie Reală                                        | Soluție SIA                                                 | Modul                 |
| --------------------------------------------------- | ----------------------------------------------------------- | --------------------- |
| Recunoașterea automată a moleculelor din imagini 2D | CNN custom extrage features vizuale → LSTM generează SMILES | ai_model/model.py     |
| Căutare molecule prin descriere în limbaj natural   | RAG cu FAISS + sentence-transformers                        | backend/rag_helper.py |

---

## 📁 Structura Proiectului

```
chemnet-vision/
│
├── 📂 ai_model/                     # 🧠 MODULUL 2: Rețea Neuronală
│   ├── model.py                     # Arhitectura completă (639 linii)
│   │   ├── ConvBlock                # Bloc convoluțional cu BN + ReLU
│   │   ├── ResidualBlock            # Bloc rezidual pentru skip connections
│   │   ├── CNNEncoder               # Encoder vizual (512 dim output)
│   │   ├── MLPEncoder               # Encoder numeric (128 dim output)
│   │   ├── GNNEncoder               # Encoder graf molecular (128 dim output)
│   │   ├── FusionLayer              # Fuziune multimodală (768→256)
│   │   ├── LSTMDecoder              # Decoder autoregresiv SMILES
│   │   └── ChemNetVisionModel       # Model complet end-to-end
│   │
│   ├── train.py                     # Pipeline antrenament
│   ├── train_model.py               # Script antrenament alternativ
│   └── inference.py                 # Inferență și predicție
│
├── 📂 backend/                      # 🌐 MODULUL 3: Web Service
│   ├── app.py                       # Flask API cu dual mode
│   │   ├── /api/status              # GET - Status server
│   │   ├── /api/mode                # GET/POST - Mod curent (AI/Fallback)
│   │   ├── /predict                 # POST - Predicție SMILES din imagine
│   │   └── /chat                    # POST - Căutare conversațională
│   │
│   └── rag_helper.py                # Sistem RAG pentru căutare semantică
│
├── 📂 src/
│   ├── 📂 app/                      # 🎨 Frontend Next.js
│   │   ├── page.tsx                 # Pagina principală
│   │   ├── layout.tsx               # Layout global
│   │   ├── globals.css              # Stiluri CSS
│   │   └── api/conversations/       # API routes
│   │
│   ├── 📂 components/               # Componente React
│   │   ├── ChatInterface.tsx        # Interfață chat
│   │   ├── MoleculeViewer.tsx       # Vizualizator 3D cu 3Dmol.js
│   │   └── MessageBubble.tsx        # Componente mesaje
│   │
│   └── 📂 preprocessing/            # 📊 MODULUL 1: Data Acquisition
│       └── data_preprocessing.py    # Preprocesare + extragere descriptori
│
├── 📂 scripts/                      # 📊 MODULUL 1: Generare Date
│   ├── generate_molecule_images.py  # Generare imagini 2D din SMILES
│   ├── csv_to_json.py               # Conversie format date
│   ├── process_pdfs_for_rag.py      # Procesare PDFs pentru RAG
│   └── wiki_pdf_downloader.py       # Download knowledge base
│
├── 📂 saved_models/                 # Modele antrenate
│   ├── checkpoint_best.pth          # Best model (~58 MB, val_loss: 0.0001)
│   ├── checkpoint_latest.pth        # Latest checkpoint
│   └── vocab.json                   # Vocabular SMILES (65 tokens)
│
├── 📂 data/
│   ├── 2d_images/                   # 42,037 imagini PNG 300×300
│   ├── processed/                   # Dataset procesat (67 features)
│   │   └── molecules_processed.csv  # 42,149 molecule
│   ├── train/                       # 29,503 molecule (70%)
│   ├── validation/                  # 6,323 molecule (15%)
│   ├── test/                        # 6,323 molecule (15%)
│   ├── faiss_index/                 # Index FAISS pentru RAG
│   ├── chunks.json                  # Chunks text pentru RAG
│   ├── embeddings.npy               # Embeddings semantice
│   └── molecules.csv                # Dataset original
│
├── 📂 docs/                         # Documentație
│   ├── datasets/                    # Grafice EDA
│   └── PROGRESS.md                  # Progres proiect
│
├── 📂 config/                       # Configurări
├── 📂 public/                       # Assets statice Next.js
│
├── README.md                        # README principal
├── README_V2.md                     # Documentație v2
├── README_V4.md                     # Acest fișier
├── README_Etapa4_Arhitectura_SIA_03.12.2025.md  # Documentație Etapa 4
│
├── package.json                     # Dependențe Node.js
├── requirements.txt                 # Dependențe Python
├── start-servers.bat                # Script pornire servere
├── next.config.ts                   # Configurare Next.js
├── tsconfig.json                    # Configurare TypeScript
└── chemnet-vision.code-workspace    # VS Code workspace
```

---

## 🧠 Arhitectura Rețelei Neuronale

### Diagrama vizuală a rețelei:

![ChemNet-Vision Network Architecture](docs/network_architecture.png)

_Fișiere diagramă:_

-   **PNG:** `docs/network_architecture.png`
-   **SVG:** `docs/network_architecture.svg`
-   **Script generare:** `scripts/generate_network_diagram.py`

### Diagrama ASCII a rețelei:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          ChemNet-Vision Neural Network                               │
│                           (15,300,290 parametri trainable)                           │
│                                                                                      │
│    ┌────────────────────────────────────────────────────────────────────────────┐    │
│    │                              INPUT LAYER                                    │    │
│    │   [IMAGE: 224×224×3]    [FEATURES: 23 numeric]    [SMILES: graph data]     │    │
│    └────────────┬───────────────────┬───────────────────────┬───────────────────┘    │
│                 │                   │                       │                        │
│    ┌────────────▼───────────┐ ┌─────▼─────────┐ ┌───────────▼───────────┐            │
│    │     CNN ENCODER        │ │  MLP ENCODER  │ │      GNN ENCODER      │            │
│    │  (Custom - NO PRETRAIN)│ │               │ │                       │            │
│    │                        │ │  23 → 64      │ │  GCNConv Layer 1      │            │
│    │  ConvBlock: 3→64      │ │  64 → 128     │ │  (atom_features→64)   │            │
│    │  ResBlock ×2: 64→64   │ │  ReLU         │ │                       │            │
│    │  ConvBlock: 64→128    │ │  Dropout(0.3) │ │  GCNConv Layer 2      │            │
│    │  ResBlock ×2: 128→128 │ │               │ │  (64→64)              │            │
│    │  ConvBlock: 128→256   │ │  Output: 128  │ │                       │            │
│    │  ResBlock ×2: 256→256 │ │               │ │  GCNConv Layer 3      │            │
│    │  ConvBlock: 256→512   │ │               │ │  (64→128)             │            │
│    │  ResBlock ×2: 512→512 │ │               │ │                       │            │
│    │  AdaptiveAvgPool2d    │ │               │ │  Global Mean Pool     │            │
│    │                        │ │               │ │                       │            │
│    │  Output: 512 dim      │ │  Output: 128  │ │  Output: 128 dim      │            │
│    └────────────┬───────────┘ └─────┬─────────┘ └───────────┬───────────┘            │
│                 │                   │                       │                        │
│    ┌────────────▼───────────────────▼───────────────────────▼───────────┐            │
│    │                         FUSION LAYER                                │            │
│    │                                                                     │            │
│    │   Concatenate: [CNN:512] + [MLP:128] + [GNN:128] = 768 dim         │            │
│    │   Linear: 768 → 256                                                 │            │
│    │   ReLU activation                                                   │            │
│    │   Dropout(0.3)                                                      │            │
│    │                                                                     │            │
│    │   Output: 256 dim (unified molecular representation)               │            │
│    └────────────────────────────────────┬────────────────────────────────┘            │
│                                         │                                             │
│    ┌────────────────────────────────────▼────────────────────────────────┐            │
│    │                         LSTM DECODER                                 │            │
│    │                                                                      │            │
│    │   Token Embedding: 65 → 256 dim (vocabular SMILES)                  │            │
│    │   LSTM: 2 layers, hidden=512, dropout=0.2                           │            │
│    │   Linear: 512 → 65 (output logits)                                  │            │
│    │                                                                      │            │
│    │   Generare autoregressivă:                                          │            │
│    │   <SOS> → token₁ → token₂ → ... → tokenₙ → <EOS>                   │            │
│    │                                                                      │            │
│    │   Output: SMILES string (ex: "CCO", "c1ccccc1")                     │            │
│    └─────────────────────────────────────────────────────────────────────┘            │
│                                                                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

### Detalii componente:

#### 1. CNN Encoder (Custom - FĂRĂ pretraining)

```python
class CNNEncoder(nn.Module):
    """
    Encoder vizual custom pentru imagini moleculare.
    Arhitectură ResNet-style antrenată de la zero.

    Input:  (batch, 3, 224, 224) - Imagine RGB
    Output: (batch, 512) - Vector features

    Blocuri:
    - ConvBlock: Conv2d + BatchNorm2d + ReLU + MaxPool2d
    - ResidualBlock: 2× Conv2d cu skip connection
    """
    def __init__(self):
        # Bloc inițial: 3 → 64 canale
        self.initial = ConvBlock(3, 64)

        # Stage 1: 64 → 64 (2 ResidualBlocks)
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        # Stage 2: 64 → 128 (downsample + 2 ResidualBlocks)
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )

        # Stage 3: 128 → 256
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )

        # Stage 4: 256 → 512
        self.stage4 = nn.Sequential(
            ConvBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512)
        )

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
```

#### 2. MLP Encoder

```python
class MLPEncoder(nn.Module):
    """
    Encoder pentru features numerice moleculare.

    Input:  (batch, 23) - 23 proprietăți numerice
    Output: (batch, 128) - Vector features

    Proprietăți procesate:
    - MolWeight_RDKit, LogP_RDKit, TPSA_RDKit
    - NumHDonors, NumHAcceptors, NumRotatableBonds
    - NumAromaticRings, FractionCSP3, etc.
    """
    def __init__(self, input_dim=23, hidden_dim=64, output_dim=128):
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
```

#### 3. GNN Encoder

```python
class GNNEncoder(nn.Module):
    """
    Encoder pentru structura graf a moleculei.
    Folosește Graph Convolutional Networks (GCN).

    Input:  Graph data (node features, edge index)
    Output: (batch, 128) - Vector features
    """
    def __init__(self, node_features=9, hidden_dim=64, output_dim=128):
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
```

#### 4. Fusion Layer

```python
class FusionLayer(nn.Module):
    """
    Fuzionează reprezentările din cele 3 encodere.

    Input:  [CNN:512 + MLP:128 + GNN:128] = 768 dim
    Output: (batch, 256) - Unified representation
    """
    def __init__(self, cnn_dim=512, mlp_dim=128, gnn_dim=128, output_dim=256):
        total_dim = cnn_dim + mlp_dim + gnn_dim  # 768
        self.fc = nn.Linear(total_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
```

#### 5. LSTM Decoder

```python
class LSTMDecoder(nn.Module):
    """
    Decoder autoregresiv pentru generarea SMILES.

    Input:  (batch, 256) - Unified representation
    Output: (batch, max_len, vocab_size) - Token probabilities

    Vocabular: 65 tokens (caractere SMILES + <PAD>, <SOS>, <EOS>)
    """
    def __init__(self, vocab_size=65, embed_dim=256, hidden_dim=512, num_layers=2):
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)
```

### Vocabular SMILES (65 tokens):

```json
{
    "<PAD>": 0, "<SOS>": 1, "<EOS>": 2,
    "C": 3, "c": 4, "N": 5, "n": 6, "O": 7, "o": 8,
    "S": 9, "s": 10, "F": 11, "Cl": 12, "Br": 13, "I": 14,
    "(": 15, ")": 16, "[": 17, "]": 18,
    "=": 19, "#": 20, "-": 21, "+": 22,
    "1": 23, "2": 24, "3": 25, "4": 26, "5": 27, "6": 28, "7": 29, "8": 30, "9": 31,
    "@": 32, "@@": 33, "/": 34, "\\": 35,
    ...
}
```

---

## 🔄 Dual Mode (AI/Fallback)

Sistemul suportă două moduri de operare:

### 1. AI Mode (USE_AI_MODEL = True)

```
User Upload Image → Preprocess → CNN Encode → Fusion → LSTM Decode → SMILES
```

-   Folosește rețeaua neuronală pentru predicție
-   Generare autoregressivă SMILES
-   Validare cu RDKit
-   Fallback automat dacă SMILES invalid

### 2. Fallback Mode (USE_AI_MODEL = False)

```
User Query → RAG Search → FAISS Index → Top-K Results → Display
```

-   Căutare în baza de date cu FAISS
-   Embedding-uri semantice cu sentence-transformers
-   Nu folosește rețeaua neuronală
-   Răspunsuri bazate pe date existente

### API pentru schimbarea modului:

```bash
# Get current mode
curl http://localhost:5000/api/mode

# Set AI mode
curl -X POST http://localhost:5000/api/mode \
     -H "Content-Type: application/json" \
     -d '{"mode": "ai"}'

# Set Fallback mode
curl -X POST http://localhost:5000/api/mode \
     -H "Content-Type: application/json" \
     -d '{"mode": "fallback"}'
```

---

## 📊 Rezultate Antrenament

### Configurație antrenament:

| Parametru     | Valoare                    |
| ------------- | -------------------------- |
| Epochs        | 2 (din 50)                 |
| Batch Size    | 32                         |
| Learning Rate | 0.001                      |
| Optimizer     | Adam                       |
| Loss Function | CrossEntropyLoss           |
| GPU           | NVIDIA GeForce RTX 3050 Ti |

### Rezultate:

| Epoch | Train Loss | Val Loss | Status  |
| ----- | ---------- | -------- | ------- |
| 1     | 0.0453     | 0.0001   | ✅      |
| 2     | 0.0002     | 0.0001   | ✅ Best |

### Modele salvate:

```
saved_models/
├── checkpoint_best.pth      # ~58 MB (epoch 2, val_loss: 0.0001)
├── checkpoint_latest.pth    # ~58 MB
└── vocab.json               # 65 tokens
```

---

## 🚀 Instalare și Rulare

### Prerequisites:

-   Python 3.11+
-   Node.js 18+
-   NVIDIA GPU (opțional, pentru antrenament)

### 1. Clonare repository:

```bash
git clone https://github.com/[username]/chemnet-vision.git
cd chemnet-vision
```

### 2. Setup Python environment:

```bash
# Creare virtual environment
python -m venv .venv

# Activare (Windows)
.venv\Scripts\activate

# Instalare dependențe Python
pip install -r requirements.txt
```

### 3. Setup Node.js:

```bash
npm install
```

### 4. Rulare servere:

```bash
# Metoda 1: Script automat (Windows)
start-servers.bat

# Metoda 2: Manual (2 terminale)
# Terminal 1 - Backend
python backend/app.py

# Terminal 2 - Frontend
npm run dev
```

### 5. Accesare:

-   **Frontend:** http://localhost:3000
-   **Backend API:** http://localhost:5000

---

## 🔗 API Endpoints

| Endpoint         | Method | Descriere                             |
| ---------------- | ------ | ------------------------------------- |
| `/api/status`    | GET    | Status server și mod curent           |
| `/api/mode`      | GET    | Returnează modul curent (ai/fallback) |
| `/api/mode`      | POST   | Setează modul (ai/fallback/auto)      |
| `/predict`       | POST   | Predicție SMILES din imagine          |
| `/chat`          | POST   | Căutare molecule prin conversație     |
| `/conversations` | GET    | Lista conversații                     |
| `/conversations` | POST   | Creează conversație nouă              |

### Exemplu predicție:

```bash
curl -X POST http://localhost:5000/predict \
     -F "image=@molecule.png"
```

### Răspuns:

```json
{
	"smiles": "CCO",
	"name": "Ethanol",
	"confidence": 0.95,
	"mode": "ai",
	"image_2d": "base64...",
	"structure_3d": "mol_data..."
}
```

---

## 🛠 Tehnologii Utilizate

### Backend:

-   **Python 3.11** - Limbaj principal
-   **PyTorch** - Framework deep learning
-   **Flask** - API REST
-   **RDKit** - Procesare moleculară
-   **FAISS** - Index vectorial pentru RAG
-   **sentence-transformers** - Embeddings semantice

### Frontend:

-   **Next.js 16** - Framework React
-   **TypeScript** - Type safety
-   **3Dmol.js** - Vizualizare 3D molecule
-   **Tailwind CSS** - Stilizare

### Model:

-   **CNN** - Extragere features vizuale
-   **MLP** - Procesare features numerice
-   **GNN** - Procesare structură graf
-   **LSTM** - Generare SMILES

---

## 📜 Scripturi Utilizate

### 1. Generarea Imaginilor 2D Moleculare

**Script:** `scripts/generate_molecule_images.py`

```bash
python scripts/generate_molecule_images.py
```

**Ce face:**

-   Citește SMILES-uri din `data/molecules.csv`
-   Generează imagini 2D PNG 300×300 pentru fiecare moleculă
-   Folosește RDKit pentru desenare moleculară
-   Salvează imaginile în `data/2d_images/`

**Output:** 42,037 imagini PNG

**Parametri:**

-   Dimensiune imagine: 300×300 pixeli
-   Format: PNG cu fundal alb
-   Atomi colorați conform convenției CPK

---

### 2. Procesarea PDF-urilor pentru RAG

**Script:** `scripts/process_pdfs_for_rag.py`

```bash
python scripts/process_pdfs_for_rag.py
```

**Ce face:**

1. Citește PDF-urile din `data/pdfs/`
2. Extrage textul din fiecare pagină
3. Împarte textul în chunks de ~500 caractere
4. Generează embeddings cu sentence-transformers (`all-MiniLM-L6-v2`)
5. Creează index FAISS pentru căutare semantică
6. Salvează:
    - `data/chunks.json` - Chunks de text
    - `data/embeddings.npy` - Vectori embedding
    - `data/faiss_index/index.faiss` - Index FAISS

**Dependențe:**

-   PyPDF2 sau pdfplumber pentru extragere text
-   sentence-transformers pentru embeddings
-   FAISS pentru indexare vectorială

---

### 3. Descărcare PDFs Wikipedia

**Script:** `scripts/wiki_pdf_downloader.py`

```bash
python scripts/wiki_pdf_downloader.py
```

**Ce face:**

-   Descarcă articole Wikipedia despre molecule chimice
-   Salvează ca PDF în `data/pdfs/`
-   Folosit pentru a construi knowledge base-ul RAG

---

### 4. Conversie CSV la JSON

**Script:** `scripts/csv_to_json.py`

```bash
python scripts/csv_to_json.py
```

**Ce face:**

-   Convertește `data/molecules.csv` în `data/molecules.json`
-   Format JSON pentru acces rapid în backend

---

## 📝 Changelog v4.0

### Adăugări:

-   ✅ Dual Mode (AI/Fallback) cu toggle în UI
-   ✅ **Toggle frontend** pentru selectare mod (AI/Auto/Fallback)
-   ✅ Indicator "AI Ready" / "AI Offline" în sidebar
-   ✅ Endpoint `/mode` GET/POST pentru schimbarea modului
-   ✅ Documentație completă arhitectură rețea
-   ✅ Documentație scripturi utilizate

### Modificări:

-   📝 README_Etapa4 actualizat cu checklist completat
-   📝 README_V4 creat cu structura detaliată
-   🎨 ChatInterface.tsx actualizat cu selector de mod

### Fixuri:

-   🔧 Fallback automat când SMILES generat invalid
-   🔧 Verificare disponibilitate AI înainte de selectare mod

---

## 🖼️ Screenshots

### Interfață principală cu toggle mode:

```
┌─────────────────────────────────────────────┐
│  ChemNet-Vision                             │
├─────────────────────────────────────────────┤
│  [+ New Chat]                               │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │ Inference Mode          [AI Ready] │    │
│  │ ┌─────┬───────┬──────┐             │    │
│  │ │ 🧠AI │ ⚡Auto │ 📚DB │             │    │
│  │ └─────┴───────┴──────┘             │    │
│  │ AI first, then database fallback   │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  Recent                                     │
│  • Aspirin query...                         │
│  • Caffeine analysis...                     │
└─────────────────────────────────────────────┘
```

---

**© 2025 Alexandru Gabriel - ChemNet-Vision**
