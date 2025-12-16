# ChemNet-Vision

An AI-powered system for molecule recognition and analysis using custom neural networks (CNN + GNN + LSTM).

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Alexandru Gabriel

## Project Structure

```
chemnet-vision/
├── README.md
├── docs/
│   └── datasets/           # Descriere seturi de date, rapoarte EDA
├── data/
│   ├── raw/                # Date brute
│   ├── processed/          # Date curățate și transformate
│   ├── train/              # Set de instruire (70%)
│   ├── validation/         # Set de validare (15%)
│   ├── test/               # Set de testare (15%)
│   └── README.md           # Documentație dataset
├── src/
│   ├── preprocessing/      # Funcții pentru preprocesare
│   ├── app/                # Next.js Frontend
│   └── components/         # React Components
├── ai_model/               # PyTorch models (Custom CNN + GNN + LSTM)
├── backend/                # Flask backend API
├── scripts/                # Utility scripts
├── config/                 # Fișiere de configurare
├── saved_models/           # Checkpoint-uri model antrenat
└── requirements.txt        # Dependențe Python
```

---

## 🧠 Arhitectura Rețelei Neuronale

### Prezentare Generală

ChemNet-Vision folosește o arhitectură **multimodală personalizată** (antrenată de la zero, fără pretraining) care combină mai multe tipuri de encodere pentru a procesa diferite reprezentări ale moleculelor și un decoder LSTM autoregresiv pentru a genera secvențe SMILES.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ChemNet-Vision Architecture                          │
│                        (Custom - No Pretraining)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                     │
│  │  2D Image    │   │   Numeric    │   │   Graph      │                     │
│  │  (PNG)       │   │   Features   │   │   (Atoms)    │                     │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                     │
│         │                  │                  │                             │
│         ▼                  ▼                  ▼                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                     │
│  │  CNN Encoder │   │  MLP Encoder │   │  GNN Encoder │                     │
│  │  (ResNet18)  │   │  (2 layers)  │   │  (Optional)  │                     │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                     │
│         │                  │                  │                             │
│         └─────────────┬────┴────────────┬─────┘                             │
│                       │   FUSION        │                                   │
│                       ▼                 ▼                                   │
│              ┌─────────────────────────────┐                                │
│              │   Concatenare + Proiecție   │                                │
│              │   (Linear → ReLU → Dropout) │                                │
│              └─────────────┬───────────────┘                                │
│                            │                                                │
│                            ▼                                                │
│              ┌─────────────────────────────┐                                │
│              │     RNN Decoder (LSTM)      │                                │
│              │   Autoregresiv → SMILES     │                                │
│              └─────────────┬───────────────┘                                │
│                            │                                                │
│                            ▼                                                │
│              ┌─────────────────────────────┐                                │
│              │   Output: SMILES Tokens     │                                │
│              │   (Vocabulary ~100 tokens)  │                                │
│              └─────────────────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Componente Principale

#### 1. CNN Encoder (Image Encoder)

| Parametru       | Valoare                                                     |
| --------------- | ----------------------------------------------------------- |
| **Arhitectură** | Custom ResNet-style (antrenat de la zero)                   |
| **Input**       | Imagine RGB 224×224 pixeli                                  |
| **Output**      | Vector embedding 512 dimensiuni                             |
| **Rol**         | Extrage caracteristici vizuale din structura 2D a moleculei |
| **Pretraining** | ❌ NU - antrenat de la zero                                 |

```python
# Arhitectura CNN Encoder (Custom - No Pretraining)
Conv2d(3 → 64, kernel 7×7, stride 2)
├── BatchNorm + ReLU + MaxPool
├── Layer1: 2× ResidualBlock (64 → 64)
├── Layer2: 2× ResidualBlock (64 → 128, stride 2)
├── Layer3: 2× ResidualBlock (128 → 256, stride 2)
├── Layer4: 2× ResidualBlock (256 → 512, stride 2)
├── AdaptiveAvgPool2d → (512, 1, 1)
└── Flatten → Vector[512]
```

#### 2. MLP Encoder (Feature Encoder)

| Parametru        | Valoare                                |
| ---------------- | -------------------------------------- |
| **Arhitectură**  | Multi-Layer Perceptron cu 2 straturi   |
| **Input**        | 23 caracteristici numerice normalizate |
| **Output**       | Vector embedding 128 dimensiuni        |
| **Activare**     | ReLU                                   |
| **Regularizare** | Dropout (p=0.3)                        |

```python
# Arhitectura MLP Encoder
Sequential(
    Linear(23 → 128),
    ReLU(),
    Dropout(0.3),
    Linear(128 → 128),
    ReLU()
)
```

#### 3. GNN Encoder (Graph Encoder) - Opțional

| Parametru       | Valoare                                        |
| --------------- | ---------------------------------------------- |
| **Arhitectură** | Graph Convolutional Network (GCN)              |
| **Input**       | Graf molecular (noduri=atomi, muchii=legături) |
| **Output**      | Vector embedding 128 dimensiuni                |
| **Straturi**    | 3× GCNConv cu ReLU                             |
| **Agregare**    | Global Mean Pooling                            |

```python
# Arhitectura GNN Encoder (PyTorch Geometric)
GCNConv(num_atom_features → 64)
├── ReLU
GCNConv(64 → 128)
├── ReLU
GCNConv(128 → 128)
├── ReLU
└── global_mean_pool → Vector[128]
```

#### 4. Fusion Layer (Strat de Fuziune)

| Parametru     | Valoare                          |
| ------------- | -------------------------------- |
| **Metodă**    | Concatenare                      |
| **Input**     | CNN[512] + MLP[128] (+ GNN[128]) |
| **Output**    | Vector fuzionat 256 dimensiuni   |
| **Proiecție** | Linear → ReLU → Dropout          |

```python
# Fuziune
combined = torch.cat([cnn_out, mlp_out, gnn_out], dim=1)  # [768]
fused = Sequential(
    Linear(768 → 256),
    ReLU(),
    Dropout(0.3)
) → Vector[256]
```

#### 5. RNN Decoder (SMILES Generator)

| Parametru          | Valoare                                   |
| ------------------ | ----------------------------------------- |
| **Arhitectură**    | LSTM (Long Short-Term Memory)             |
| **Hidden Size**    | 512 dimensiuni                            |
| **Num Layers**     | 2 straturi                                |
| **Input**          | Embedding token (64 dim) + Context vector |
| **Output**         | Probabilități token (vocab_size ~100)     |
| **Lungime maximă** | 150 tokens                                |

```python
# Arhitectura LSTM Decoder
Embedding(vocab_size → 64)
LSTM(
    input_size=64,
    hidden_size=512,
    num_layers=2,
    dropout=0.3,
    batch_first=True
)
Linear(512 → vocab_size)  # Output logits
```

### Vocabular SMILES

Vocabularul conține ~100 de tokens pentru reprezentarea moleculelor:

| Categorie       | Tokens | Exemple                          |
| --------------- | ------ | -------------------------------- |
| **Atomi**       | ~15    | C, N, O, S, P, F, Cl, Br, I, ... |
| **Legături**    | ~5     | -, =, #, :, .                    |
| **Cicluri**     | ~10    | 1, 2, 3, ..., 9, %10, %11        |
| **Ramificații** | 2      | (, )                             |
| **Aromatice**   | ~6     | c, n, o, s, p                    |
| **Chiralitate** | ~4     | @, @@, /, \\                     |
| **Speciale**    | 3      | `<PAD>`, `<SOS>`, `<EOS>`        |

### Procesul de Training

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INPUT                                                       │
│     ├── Imagine 2D (224×224 PNG)                                │
│     ├── Caracteristici numerice (23 features)                   │
│     └── Target SMILES (text)                                    │
│                                                                 │
│  2. FORWARD PASS                                                │
│     ├── CNN: Imagine → Embedding[512]                           │
│     ├── MLP: Features → Embedding[128]                          │
│     ├── Fusion: Concatenare → Vector[256]                       │
│     └── LSTM: Vector → SMILES tokens (teacher forcing)          │
│                                                                 │
│  3. LOSS CALCULATION                                            │
│     └── CrossEntropyLoss(predicted_tokens, target_tokens)       │
│         (ignoră <PAD> tokens)                                   │
│                                                                 │
│  4. BACKWARD PASS                                               │
│     ├── Calculează gradienți                                    │
│     ├── Gradient clipping (max_norm=1.0)                        │
│     └── Adam optimizer update (lr=0.001)                        │
│                                                                 │
│  5. CHECKPOINTING                                               │
│     ├── checkpoint_latest.pth (fiecare epoch)                   │
│     └── checkpoint_best.pth (când val_loss scade)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Hiperparametri Training

| Parametru             | Valoare           | Descriere                                 |
| --------------------- | ----------------- | ----------------------------------------- |
| **Batch Size**        | 8                 | Dimensiunea batch-ului                    |
| **Learning Rate**     | 0.001             | Rata de învățare inițială                 |
| **Epochs**            | 20                | Număr maxim de epoci                      |
| **Optimizer**         | Adam              | Optimizator cu rată adaptivă              |
| **LR Scheduler**      | ReduceLROnPlateau | Reduce LR când val_loss stagnează         |
| **Patience**          | 3                 | Epoci de așteptare înainte de reducere LR |
| **Gradient Clipping** | 1.0               | Previne explodarea gradienților           |
| **Dropout**           | 0.3               | Regularizare                              |

### Procesul de Inferență

```
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INPUT: Imagine 2D moleculă (PNG/JPG)                        │
│                                                                 │
│  2. PREPROCESSING                                               │
│     ├── Resize la 224×224                                       │
│     ├── Normalizare ImageNet (mean, std)                        │
│     └── Conversie la tensor PyTorch                             │
│                                                                 │
│  3. ENCODING                                                    │
│     ├── CNN: Imagine → Embedding[512]                           │
│     └── Fusion: → Vector[256]                                   │
│                                                                 │
│  4. DECODING (Autoregresiv)                                     │
│     ├── Start: <SOS> token                                      │
│     ├── Loop: Generează token cu token                          │
│     │   ├── LSTM: hidden_state → logits                         │
│     │   ├── Softmax → probabilități                             │
│     │   ├── Argmax → next_token                                 │
│     │   └── Append la secvență                                  │
│     └── Stop: când <EOS> sau max_length                         │
│                                                                 │
│  5. POST-PROCESSING                                             │
│     ├── Decodare tokens → SMILES string                         │
│     ├── Validare cu RDKit (Chem.MolFromSmiles)                  │
│     └── Canonicalizare SMILES                                   │
│                                                                 │
│  6. OUTPUT                                                      │
│     ├── predicted_smiles: "CCO" (exemplu)                       │
│     ├── canonical_smiles: "CCO"                                 │
│     ├── is_valid: True/False                                    │
│     ├── confidence: 0.95                                        │
│     └── image_2d/3d: generate cu RDKit                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Fallback RAG (Retrieval-Augmented Generation)

Dacă modelul neural returnează SMILES invalid sau cu confidence scăzut, sistemul folosește RAG:

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Fallback System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Query: "Tell me about Aspirin"                              │
│                                                                 │
│  2. EMBEDDING                                                   │
│     └── SentenceTransformer → Query Vector[384]                 │
│                                                                 │
│  3. RETRIEVAL (FAISS)                                           │
│     ├── Căutare în index vectorial                              │
│     ├── Top-K (k=5) documente relevante                         │
│     └── Chunks din PDFs + Database                              │
│                                                                 │
│  4. MOLECULE MATCHING                                           │
│     ├── Căutare în molecules.json                               │
│     ├── Potrivire după nume (word boundary)                     │
│     └── Extragere SMILES, proprietăți                           │
│                                                                 │
│  5. RESPONSE GENERATION                                         │
│     ├── Combinare informații RAG + DB                           │
│     ├── Generare 2D/3D cu RDKit                                 │
│     └── Return structured response                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Fișiere Model

| Fișier                               | Dimensiune | Descriere                        |
| ------------------------------------ | ---------- | -------------------------------- |
| `saved_models/checkpoint_best.pth`   | ~58 MB     | Cel mai bun model (min val_loss) |
| `saved_models/checkpoint_latest.pth` | ~58 MB     | Ultimul checkpoint               |
| `saved_models/vocab.json`            | ~1 KB      | Vocabular SMILES (65 tokens)     |

### Parametri Model

| Categorie                 | Valoare                     |
| ------------------------- | --------------------------- |
| **Total parametri**       | 15,300,290                  |
| **Parametri antrenabili** | 15,300,290 (100%)           |
| **Pretraining**           | ❌ NU (antrenat de la zero) |
| **Vocab size**            | 65 tokens                   |

### Rezultate Antrenament

| Epoch | Train Loss | Val Loss | Status             |
| ----- | ---------- | -------- | ------------------ |
| 1     | 0.0453     | 0.0001   | ✓ Best model saved |
| 2     | 0.0002     | 0.0001   | ✓ Best model saved |

### Metrici de Evaluare

| Metrică                 | Descriere                          | Target | Actual     |
| ----------------------- | ---------------------------------- | ------ | ---------- |
| **Train Loss**          | Cross-entropy pe set training      | < 0.1  | ✓ 0.0002   |
| **Val Loss**            | Cross-entropy pe set validare      | < 0.1  | ✓ 0.0001   |
| **SMILES Validity**     | % SMILES valide (RDKit)            | > 90%  | În testare |
| **Exact Match**         | % potriviri exacte cu target       | > 70%  | În testare |
| **Tanimoto Similarity** | Similaritate fingerprint molecular | > 0.8  | În testare |

### Tehnologii Utilizate

| Componentă        | Tehnologie                           | Versiune |
| ----------------- | ------------------------------------ | -------- |
| **Deep Learning** | PyTorch                              | 2.0+     |
| **CNN**           | Custom ResNet-style (no pretraining) | -        |
| **Graph NN**      | PyTorch Geometric (GCNConv)          | 2.0+     |
| **RNN**           | LSTM (2 layers, hidden=512)          | -        |
| **Chemistry**     | RDKit                                | 2023.03+ |
| **Embeddings**    | sentence-transformers                | 2.0+     |
| **Vector Search** | FAISS                                | 1.7+     |
| **Backend API**   | Flask + Flask-CORS                   | 2.0+     |
| **Frontend**      | Next.js 16 + React 19                | 16.0+    |
| **3D Viewer**     | 3Dmol.js                             | 2.0+     |
| **GPU**           | NVIDIA RTX 3050 Ti + CUDA 12.1       | -        |

---

---
