# ChemNet-Vision

An AI-powered system for molecule recognition and analysis using custom neural network architectures (CNN + GNN + LSTM).

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Alexandru Gabriel

---

## 🎯 Obiectivul Proiectului

Sistemul ChemNet-Vision este conceput pentru **recunoașterea și analiza moleculelor** utilizând rețele neuronale profunde. Modelul primește ca input o imagine 2D a unei molecule și generează reprezentarea SMILES corespunzătoare.

**Caracteristici principale:**

-   ✅ **Model custom antrenat de la zero** (fără pretraining/transfer learning)
-   ✅ **Arhitectură multimodală**: CNN + MLP + GNN + LSTM
-   ✅ **Dataset ChEMBL**: 42,149 molecule validate
-   ✅ **Interfață web interactivă** cu vizualizare 2D/3D

---

## 📁 Structura Proiectului

```
chemnet-vision/
├── README.md                   # Documentația principală
├── requirements.txt            # Dependențe Python
├── package.json                # Dependențe Node.js
│
├── ai_model/                   # 🧠 Rețeaua Neuronală
│   ├── model.py                # Arhitectura modelului (CNN + MLP + GNN + LSTM)
│   ├── train_model.py          # Script de antrenament
│   └── inference.py            # Predicție/inferență
│
├── backend/                    # 🖥️ Flask API Server
│   ├── app.py                  # Endpoints REST API
│   └── rag_helper.py           # Retrieval-Augmented Generation
│
├── src/                        # 💻 Frontend Next.js
│   ├── app/                    # Next.js App Router
│   └── components/             # React Components
│       ├── ChatInterface.tsx   # Interfața chat
│       ├── MessageBubble.tsx   # Mesaje cu vizualizare molecule
│       └── MoleculeViewer.tsx  # Vizualizator 3D (3Dmol.js)
│
├── data/                       # 📊 Date și Dataset-uri
│   ├── train/                  # Set de antrenament (70%)
│   ├── validation/             # Set de validare (15%)
│   ├── test/                   # Set de testare (15%)
│   ├── 2d_images/              # Imagini moleculare PNG
│   └── processed/              # Date preprocesate
│
├── saved_models/               # 💾 Checkpoint-uri model
│   ├── checkpoint_best.pth     # Cel mai bun model
│   └── vocab.json              # Vocabular SMILES
│
├── scripts/                    # 🔧 Scripturi utilitare
│   ├── csv_to_json.py
│   ├── generate_molecule_images.py
│   └── process_pdfs_for_rag.py
│
└── config/                     # ⚙️ Configurări
```

---

## 🧠 Etapa 4: Arhitectura Rețelei Neuronale

### Model Custom (Fără Pretraining)

ChemNet-Vision folosește o arhitectură **multimodală personalizată**, antrenată complet de la zero:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ChemNet-Vision Architecture                              │
│                    (Custom - No Pretrained Weights)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                     │
│  │  2D Image    │   │   Numeric    │   │   Graph      │                     │
│  │  (224×224)   │   │  Features    │   │   (Atoms)    │                     │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                     │
│         │                  │                  │                             │
│         ▼                  ▼                  ▼                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                     │
│  │ CNN Encoder  │   │ MLP Encoder  │   │ GNN Encoder  │                     │
│  │ Custom [512] │   │    [128]     │   │    [128]     │                     │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                     │
│         │                  │                  │                             │
│         └─────────────┬────┴────────────┬─────┘                             │
│                       │   FUSION        │                                   │
│                       ▼                 ▼                                   │
│              ┌─────────────────────────────┐                                │
│              │   Concatenate + Project     │                                │
│              │        [768 → 256]          │                                │
│              └─────────────┬───────────────┘                                │
│                            │                                                │
│                            ▼                                                │
│              ┌─────────────────────────────┐                                │
│              │     LSTM Decoder            │                                │
│              │   [256 → 512 → vocab]       │                                │
│              └─────────────┬───────────────┘                                │
│                            │                                                │
│                            ▼                                                │
│              ┌─────────────────────────────┐                                │
│              │   Output: SMILES Tokens     │                                │
│              └─────────────────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Componente ale Modelului

#### 1. CNN Encoder (Custom ResNet-style)

| Parametru        | Valoare                                   |
| ---------------- | ----------------------------------------- |
| **Arhitectură**  | Custom ResNet-style (antrenat de la zero) |
| **Input**        | Imagine RGB 224×224 pixeli                |
| **Output**       | Vector embedding 512 dimensiuni           |
| **Straturi**     | Conv1 → 4× ResidualBlock layers           |
| **Inițializare** | Kaiming Normal                            |

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

#### 3. GNN Encoder (Graph Encoder)

| Parametru       | Valoare                                        |
| --------------- | ---------------------------------------------- |
| **Arhitectură** | Graph Convolutional Network (GCN)              |
| **Input**       | Graf molecular (noduri=atomi, muchii=legături) |
| **Output**      | Vector embedding 128 dimensiuni                |
| **Straturi**    | 3× GCNConv cu ReLU + Dropout                   |
| **Agregare**    | Global Mean Pooling                            |

```python
# Arhitectura GNN Encoder
GCNConv(9 → 64) + ReLU + Dropout
GCNConv(64 → 128) + ReLU + Dropout
GCNConv(128 → 128) + ReLU
└── global_mean_pool → Vector[128]
```

#### 4. Fusion Layer

| Parametru     | Valoare                                  |
| ------------- | ---------------------------------------- |
| **Input**     | CNN[512] + MLP[128] + GNN[128] = 768 dim |
| **Output**    | Vector fuzionat 256 dimensiuni           |
| **Proiecție** | Linear → ReLU → Dropout                  |

#### 5. LSTM Decoder

| Parametru       | Valoare                       |
| --------------- | ----------------------------- |
| **Arhitectură** | LSTM (Long Short-Term Memory) |
| **Hidden Size** | 512 dimensiuni                |
| **Num Layers**  | 2 straturi                    |
| **Embedding**   | 64 dimensiuni                 |
| **Output**      | vocab_size (~65 tokens)       |

### Parametri Model

| Categorie                 | Valoare                     |
| ------------------------- | --------------------------- |
| **Total parametri**       | 15,300,290                  |
| **Parametri antrenabili** | 15,300,290 (100%)           |
| **Pretraining**           | ❌ NU (antrenat de la zero) |
| **Vocab size**            | 65 tokens                   |

---

## 📊 Etapa 3: Analiza și Pregătirea Setului de Date

### Sursa Datelor

| Atribut                  | Valoare                                             |
| ------------------------ | --------------------------------------------------- |
| **Origine**              | ChEMBL Database (European Bioinformatics Institute) |
| **Domeniu**              | Compuși chimici și molecule bioactive               |
| **Format original**      | CSV cu separator punct-virgulă (;)                  |
| **Dimensiune originală** | 48,960 molecule × 29 caracteristici                 |

### Preprocesare Aplicată

1. ✅ **Eliminarea duplicatelor**
2. ✅ **Validarea și filtrarea SMILES** cu RDKit
3. ✅ **Imputarea valorilor lipsă** (mediană)
4. ✅ **Tratarea outlierilor** (IQR capping)
5. ✅ **Encoding variabile categoriale**
6. ✅ **Extragerea descriptorilor moleculari** (10 RDKit)
7. ✅ **Normalizare Min-Max** [0, 1]
8. ✅ **Corelarea cu imaginile 2D** moleculare

### Împărțirea Seturilor de Date

| Set            | Molecule   | Procent | Imagini 2D | Acoperire | Fișiere                       |
| -------------- | ---------- | ------- | ---------- | --------- | ----------------------------- |
| **Train**      | 29,503     | 70%     | 27,989     | 94.9%     | `train.csv`, `X_train.npy`    |
| **Validation** | 6,323      | 15%     | 6,020      | 95.2%     | `validation.csv`, `X_val.npy` |
| **Test**       | 6,323      | 15%     | 6,009      | 95.0%     | `test.csv`, `X_test.npy`      |
| **Total**      | **42,149** | 100%    | **40,018** | 94.9%     |                               |

### Caracteristici Utilizate (23)

**Originale din ChEMBL (13):**

-   Molecular Weight, Targets, Bioactivities, AlogP
-   Polar Surface Area, HBA, HBD, #RO5 Violations
-   #Rotatable Bonds, QED Weighted, Aromatic Rings
-   Heavy Atoms, Np Likeness Score

**Extrase cu RDKit (10):**

-   MolWeight_RDKit, LogP_RDKit, TPSA_RDKit
-   NumHDonors_RDKit, NumHAcceptors_RDKit
-   NumRotatableBonds_RDKit, NumAromaticRings_RDKit
-   FractionCSP3, NumHeteroatoms, RingCount

---

## 📈 Rezultate Antrenament

### Configurație Training

| Parametru             | Valoare                        |
| --------------------- | ------------------------------ |
| **Batch Size**        | 8                              |
| **Learning Rate**     | 0.001                          |
| **Optimizer**         | Adam                           |
| **LR Scheduler**      | ReduceLROnPlateau (patience=3) |
| **Gradient Clipping** | 1.0                            |
| **Max Epochs**        | 20                             |
| **Device**            | NVIDIA GeForce RTX 3050 Ti     |

### Evoluția Antrenamentului

| Epoch | Train Loss | Val Loss | Status             |
| ----- | ---------- | -------- | ------------------ |
| 1     | 0.0453     | 0.0001   | ✓ Best model saved |
| 2     | 0.0002     | 0.0001   | ✓ Best model saved |

### Observații

-   **Convergență rapidă**: Modelul converge în primele 2 epoci
-   **Overfitting minimal**: Val Loss ≈ Train Loss după stabilizare
-   **Loss foarte mic**: Indicație că modelul învață pattern-urile SMILES

---

## 🛠️ Setup și Instalare

### 1. Dependențe Python

```bash
# Creează virtual environment
python -m venv .venv

# Activează (Windows)
.venv\Scripts\activate

# Instalează dependențe cu suport CUDA
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```

### 2. Dependențe Node.js

```bash
npm install
```

### 3. Preprocesare Date

```bash
python src/preprocessing/data_preprocessing.py
```

### 4. Antrenare Model

```bash
python ai_model/train_model.py
```

### 5. Rulare Aplicație

```bash
# Backend Flask (port 5000)
python backend/app.py

# Frontend Next.js (port 3000)
npm run dev -- --turbo

# SAU folosește script-ul batch
start-servers.bat
```

---

## 🖥️ Funcționalități

### 1. Recunoaștere Molecule

-   Input: Imagine 2D PNG a unei molecule
-   Output: Reprezentare SMILES generată de model

### 2. Vizualizare 2D/3D

-   **2D**: Generate cu RDKit (imagine base64)
-   **3D**: Vizualizator interactiv cu 3Dmol.js

### 3. Chat Interface

-   Căutare molecule după nume
-   Afișare proprietăți moleculare
-   Fallback RAG pentru informații suplimentare

### 4. API REST

| Endpoint           | Metodă | Descriere                    |
| ------------------ | ------ | ---------------------------- |
| `/chat`            | POST   | Procesare mesaje chat        |
| `/predict`         | POST   | Predicție SMILES din imagine |
| `/molecule/<name>` | GET    | Info despre o moleculă       |

---

## 📚 Tehnologii Utilizate

| Componentă        | Tehnologie         | Versiune |
| ----------------- | ------------------ | -------- |
| **Deep Learning** | PyTorch            | 2.0+     |
| **Graph NN**      | PyTorch Geometric  | 2.0+     |
| **Chemistry**     | RDKit              | 2023.03+ |
| **Backend**       | Flask + Flask-CORS | 2.0+     |
| **Frontend**      | Next.js + React    | 16.0+    |
| **3D Viewer**     | 3Dmol.js           | 2.0+     |
| **Vector Search** | FAISS              | 1.7+     |
| **GPU**           | CUDA               | 12.1     |

---

## 📄 Licență

Proiect educațional pentru disciplina Rețele Neuronale, POLITEHNICA București.
