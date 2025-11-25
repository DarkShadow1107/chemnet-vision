# ChemNet-Vision

An AI-powered system for molecule recognition and analysis using GNN and RNN architectures.

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR

## Project Structure

```
chemnet-vision/
├── README.md
├── docs/
│   └── datasets/           # Descriere seturi de date, rapoarte EDA
├── data/
│   ├── raw/                # Date brute
│   ├── processed/          # Date curățate și transformate
│   ├── train/              # Set de instruire
│   ├── validation/         # Set de validare
│   ├── test/               # Set de testare
│   └── README.md           # Documentație dataset
├── src/
│   ├── preprocessing/      # Funcții pentru preprocesare
│   ├── app/                # Next.js Frontend
│   └── components/         # React Components
├── ai_model/               # PyTorch models (GNN + RNN)
├── backend/                # Flask backend API
├── scripts/                # Utility scripts
├── config/                 # Fișiere de configurare
└── requirements.txt        # Dependențe Python
```

## Etapa 3: Analiza și Pregătirea Setului de Date

### Preprocesare Date

```bash
python src/preprocessing/data_preprocessing.py
```

### Rezultate Preprocesare:

-   **Dataset original:** 48,960 molecule (ChEMBL)
-   **Dataset final:** 42,149 molecule cu SMILES valid
-   **Împărțire:** Train 70% | Validation 15% | Test 15%

### Pași de preprocesare aplicați:

1. ✅ Eliminarea duplicatelor
2. ✅ Validarea și filtrarea SMILES
3. ✅ Imputarea valorilor lipsă (mediană)
4. ✅ Tratarea outlierilor (IQR capping)
5. ✅ Encoding variabile categoriale
6. ✅ Extragerea descriptorilor moleculari (10 RDKit)
7. ✅ Normalizare Min-Max
8. ✅ Corelarea cu imaginile 2D moleculare

---

## 📊 Descrierea Detaliată a Datelor

### Sursa Datelor

| Atribut | Valoare |
|---------|---------|
| **Origine** | ChEMBL Database (European Bioinformatics Institute) |
| **Domeniu** | Compuși chimici și molecule bioactive |
| **Format original** | CSV cu separator punct-virgulă (;) |
| **Dimensiune originală** | 48,960 molecule × 29 caracteristici |

### Structura Folderului `data/`

```
data/
├── molecules.csv                  # 📄 Dataset original ChEMBL (36.3 MB)
├── molecules.json                 # 📄 Dataset în format JSON pentru API (69.4 MB)
├── chunks.json                    # 📝 Fragmente text pentru RAG (1.5 MB)
├── conversations.json             # 💬 Istoric conversații chatbot (10 KB)
├── embeddings.npy                 # 🔢 Vectori embedding semantic (3.3 MB)
│
├── 2d_images/                     # 🖼️ Imagini 2D moleculare (42,037 fișiere PNG)
│   └── [MOLECULE_NAME].png        # Structuri generate cu RDKit
│
├── pdfs/                          # 📚 Documentație PDF (121 fișiere)
│   └── [MOLECULE_NAME].pdf        # Informații Wikipedia
│
├── faiss_index/                   # 🔍 Index pentru căutare vectorială
│   ├── index.faiss                # Index binar (1.6 MB)
│   └── index.pkl                  # Metadata (1.1 MB)
│
├── raw/                           # Date brute (copie)
│   └── molecules_raw.csv          # 33.5 MB
│
├── processed/                     # Date preprocesate
│   └── molecules_processed.csv    # 51 MB, 67 caracteristici
│
├── train/                         # 🏋️ Set de antrenament (70%)
│   ├── train.csv                  # 29,503 molecule
│   ├── X_train.npy                # Caracteristici normalizate (29503 × 23)
│   └── train_images.json          # Căi către 27,989 imagini
│
├── validation/                    # 📊 Set de validare (15%)
│   ├── validation.csv             # 6,323 molecule
│   ├── X_val.npy                  # Caracteristici normalizate (6323 × 23)
│   └── validation_images.json     # Căi către 6,020 imagini
│
├── test/                          # 🧪 Set de testare (15%)
│   ├── test.csv                   # 6,323 molecule
│   ├── X_test.npy                 # Caracteristici normalizate (6323 × 23)
│   └── test_images.json           # Căi către 6,009 imagini
│
└── README.md                      # Documentație detaliată dataset
```

### Caracteristici din Dataset Original (29 coloane)

| Caracteristică | Tip | Descriere |
|----------------|-----|-----------|
| `ChEMBL ID` | String | Identificator unic (ex: CHEMBL25) |
| `Name` | String | Numele moleculei |
| `Synonyms` | String | Nume alternative |
| `Type` | Categoric | Tipul compusului (Small molecule, Antibody, etc.) |
| `Max Phase` | Numeric | Faza clinică maximă (0-4) |
| `Molecular Weight` | Numeric | Masa moleculară (Da) |
| `Targets` | Numeric | Număr de ținte biologice |
| `Bioactivities` | Numeric | Număr de activități biologice înregistrate |
| `AlogP` | Numeric | Coeficient de partiție octanol-apă |
| `Polar Surface Area` | Numeric | Suprafața polară topologică (Å²) |
| `HBA` | Numeric | Număr acceptori de hidrogen |
| `HBD` | Numeric | Număr donori de hidrogen |
| `#RO5 Violations` | Numeric | Încălcări ale Regulii lui Lipinski |
| `#Rotatable Bonds` | Numeric | Număr legături rotabile |
| `Passes Ro3` | Boolean | Respectă Regula lui 3 (Y/N) |
| `QED Weighted` | Numeric | Scor de drug-likeness (0-1) |
| `Aromatic Rings` | Numeric | Număr inele aromatice |
| `Structure Type` | Categoric | Tip structură (MOL, SEQ) |
| `Inorganic Flag` | Boolean | Este compus anorganic |
| `Heavy Atoms` | Numeric | Număr atomi grei (non-H) |
| `Np Likeness Score` | Numeric | Similaritate cu produse naturale |
| `Molecular Formula` | String | Formula moleculară |
| `Smiles` | String | Reprezentare SMILES a structurii |
| `Inchi Key` | String | Identificator InChI |
| `Inchi` | String | Reprezentare InChI completă |
| `Withdrawn Flag` | Boolean | Compus retras de pe piață |
| `Orphan` | Boolean | Medicament orfan |
| `Records Key` | String | Cheie înregistrare |
| `Records Name` | String | Nume înregistrare |

### Descriptori Moleculari Extrași cu RDKit (10 noi)

| Descriptor | Descriere | Interval tipic |
|------------|-----------|----------------|
| `MolWeight_RDKit` | Masă moleculară recalculată | 50-1000 Da |
| `LogP_RDKit` | Coeficient de partiție | -5 to 10 |
| `TPSA_RDKit` | Suprafață polară topologică | 0-300 Å² |
| `NumHDonors_RDKit` | Donori de hidrogen | 0-15 |
| `NumHAcceptors_RDKit` | Acceptori de hidrogen | 0-20 |
| `NumRotatableBonds_RDKit` | Legături rotabile | 0-20 |
| `NumAromaticRings_RDKit` | Inele aromatice | 0-8 |
| `FractionCSP3` | Fracție carbon sp³ | 0-1 |
| `NumHeteroatoms` | Număr heteroatomi | 0-30 |
| `RingCount` | Număr total inele | 0-10 |

### Împărțirea Seturilor de Date

| Set | Molecule | Procent | Imagini 2D | Acoperire |
|-----|----------|---------|------------|-----------|
| **Train** | 29,503 | 70% | 27,989 | 94.9% |
| **Validation** | 6,323 | 15% | 6,020 | 95.2% |
| **Test** | 6,323 | 15% | 6,009 | 95.0% |
| **Total** | 42,149 | 100% | 40,018 | 94.9% |

### Formatul Datelor pentru Training

#### 1. Caracteristici Numerice (`.npy`)
```python
import numpy as np

X_train = np.load('data/train/X_train.npy')  # Shape: (29503, 23)
X_val = np.load('data/validation/X_val.npy')  # Shape: (6323, 23)
X_test = np.load('data/test/X_test.npy')      # Shape: (6323, 23)
```

#### 2. Imagini 2D Moleculare (`.json` + `.png`)
```python
import json
from PIL import Image

with open('data/train/train_images.json', 'r') as f:
    train_data = json.load(f)

print(f"Imagini disponibile: {train_data['count']}")  # 27,989

# Încarcă o imagine
img = Image.open(train_data['images'][0])
```

#### 3. Date Complete (`.csv`)
```python
import pandas as pd

train_df = pd.read_csv('data/train/train.csv')
# Coloane: ChEMBL ID, Name, Smiles, image_path, has_image, 
#          + toate caracteristicile + *_normalized
```

### Calitatea Datelor

#### Valori Lipsă (în datasetul original)
| Caracteristică | Lipsă | Procent |
|----------------|-------|---------|
| Synonyms | 29,720 | 60.7% |
| Max Phase | 29,736 | 60.7% |
| Smiles | 6,811 | 13.9% |
| Molecular Weight | 4,610 | 9.4% |
| AlogP, PSA, HBA, HBD | 9,208 | 18.8% |

#### Tratament Aplicat
- **Valori lipsă numerice:** Imputare cu mediană
- **SMILES invalide:** Molecule eliminate (6,811)
- **Outlieri:** IQR capping (1.5 × IQR)
- **Normalizare:** Min-Max scaling [0, 1]

---

## Setup

1.  **Install Dependencies:**

    ```bash
    # Python
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121

    # Node.js
    npm install
    ```

2.  **Run Scripts:**

    -   Preprocesare date: `python src/preprocessing/data_preprocessing.py`
    -   Convert CSV to JSON: `python scripts/csv_to_json.py`
    -   Download Wikipedia PDFs: `python scripts/wiki_pdf_downloader.py`
    -   Generate Images: `python scripts/generate_molecule_images.py`

3.  **Train Model:**

    ```bash
    python ai_model/train.py
    ```

4.  **Run Application:**
    -   **Backend:**
        ```bash
        python backend/app.py
        ```
    -   **Frontend:**
        ```bash
        npm run dev -- --turbo
        ```

## Features

-   **AI System:** Uses Graph Neural Networks (GNN) and Recurrent Neural Networks (RNN) for molecule analysis.
-   **Data Processing:** Automated EDA, preprocessing, and train/val/test splitting.
-   **Frontend:** Next.js 15+ with React Compiler and Tailwind CSS.
-   **Backend:** Flask API.
